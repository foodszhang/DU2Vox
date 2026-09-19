"""Voxel-detail recovery with an exact fixed FEM-complement operator."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


DETAIL_MODES = ("unconstrained", "soft", "hard")


def normalize_detail_mode(mode: str | bool) -> str:
    """Map current and legacy checkpoint flags to the three canonical modes."""

    if isinstance(mode, bool):
        return "hard" if mode else "unconstrained"
    normalized = str(mode).lower()
    if normalized == "constrained":
        return "hard"
    if normalized not in DETAIL_MODES:
        raise ValueError(f"Unknown voxel-detail mode {mode!r}; expected {DETAIL_MODES}")
    return normalized


class _ExactComplementFunction(torch.autograd.Function):
    """CPU sparse forward/backward for the fixed self-adjoint linear operator Q."""

    @staticmethod
    def forward(ctx, values: torch.Tensor, operator: "ExactVoxelComplement") -> torch.Tensor:
        ctx.operator = operator
        ctx.input_dtype = values.dtype
        cpu_values = values.detach().cpu()
        if cpu_values.dtype in (torch.bfloat16, torch.float16):
            cpu_values = cpu_values.float()
        result = operator.apply_numpy(cpu_values.numpy())
        # Keep the structural result in FP32 even when the proposal MLP uses BF16.
        # Casting Qz back to BF16 measurably reintroduces coarse-space leakage.
        output_dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
        return torch.from_numpy(result).to(device=values.device, dtype=output_dtype)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        # Q is self-adjoint on this uniform sampled domain, hence d(Qz)/dz = Q.
        cpu_gradient = gradient.detach().cpu()
        if cpu_gradient.dtype in (torch.bfloat16, torch.float16):
            cpu_gradient = cpu_gradient.float()
        result = ctx.operator.apply_numpy(cpu_gradient.numpy())
        return torch.from_numpy(result).to(device=gradient.device, dtype=ctx.input_dtype), None


class ExactVoxelComplement(nn.Module):
    """Apply Q = I - P(P.T P)^-1 P.T exactly, with an exact custom backward.

    The sparse factorization and products intentionally run in float64 on CPU. The
    returned tensor preserves the input device; FP64 stays FP64, while lower-precision
    proposals return FP32 to preserve the contract. This is not a learned or detached
    approximation: backward applies the exact adjoint, which equals Q for W=wI.
    """

    def __init__(
        self, prolongation: sp.spmatrix, *, quadrature_weight: float = 1.0
    ) -> None:
        super().__init__()
        p = prolongation.tocsr().astype(np.float64)
        active = np.flatnonzero(np.asarray(p.getnnz(axis=0)).ravel() > 0)
        if len(active) != p.shape[1]:
            raise ValueError("Exact complement requires every FEM column to be active")
        self.p = p
        self.pt = p.T.tocsr()
        self._solve = spla.splu((self.pt @ self.p).tocsc(), permc_spec="COLAMD").solve
        self.quadrature_weight = float(quadrature_weight)
        if self.quadrature_weight <= 0:
            raise ValueError("quadrature_weight must be positive")

    @property
    def n_voxels(self) -> int:
        return int(self.p.shape[0])

    @property
    def n_fem_nodes(self) -> int:
        return int(self.p.shape[1])

    def project_numpy(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        flattened = array.reshape(-1, self.n_voxels)
        projected = np.empty_like(flattened)
        for index, row in enumerate(flattened):
            coefficients = self._solve(np.asarray(self.pt @ row).ravel())
            projected[index] = np.asarray(self.p @ coefficients).ravel()
        return projected.reshape(array.shape)

    def apply_numpy(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        return array - self.project_numpy(array)

    def coefficients_numpy(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        flattened = array.reshape(-1, self.n_voxels)
        coefficients = [self._solve(np.asarray(self.pt @ row).ravel()) for row in flattened]
        return np.stack(coefficients).reshape(*array.shape[:-1], self.n_fem_nodes)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != self.n_voxels:
            raise ValueError(f"Expected {self.n_voxels} canonical voxels, got {values.shape[-1]}")
        return _ExactComplementFunction.apply(values, self)


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.layers = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.layers(self.norm(hidden))


class ComplementConstrainedVoxelDetail(nn.Module):
    """Lightweight local MLP producing z, followed by an optional fixed exact Q."""

    def __init__(
        self,
        *,
        input_dim: int,
        complement: ExactVoxelComplement,
        hidden_dim: int = 160,
        n_hidden_layers: int = 3,
        mode: str | None = None,
        constrained: bool | None = None,
    ) -> None:
        super().__init__()
        if not 128 <= hidden_dim <= 192:
            raise ValueError("First-pass voxel detail width must be in [128, 192]")
        if not 3 <= n_hidden_layers <= 4:
            raise ValueError("First-pass voxel detail must use 3 or 4 hidden layers")
        self.complement = complement
        if mode is not None and constrained is not None:
            raise ValueError("Specify mode or legacy constrained, not both")
        self.mode = normalize_detail_mode(
            mode if mode is not None else (True if constrained is None else constrained)
        )
        # Retain this attribute for callers/checkpoints that inspect the old flag.
        self.constrained = self.mode == "hard"
        self.input = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(ResidualMLPBlock(hidden_dim) for _ in range(n_hidden_layers))
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.activation = nn.GELU()
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def proposal(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.input(features))
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.output_norm(hidden)).squeeze(-1)

    def proposal_chunked(self, chunks: Iterable[torch.Tensor]) -> torch.Tensor:
        return torch.cat([self.proposal(chunk) for chunk in chunks], dim=-1)

    def proposal_checkpointed(self, features: torch.Tensor) -> torch.Tensor:
        """Run one proposal chunk with activation recomputation in backward."""

        if not torch.is_grad_enabled():
            return self.proposal(features)
        return checkpoint(self.proposal, features, use_reentrant=False)

    def constrain(self, proposal: torch.Tensor) -> torch.Tensor:
        return self.complement(proposal) if self.mode == "hard" else proposal

    def soft_coarse_penalty(
        self, proposal: torch.Tensor, *, eps: float = 1e-12
    ) -> torch.Tensor:
        """Exact differentiable ``||P_h z||_W^2 / (||z||_W^2 + eps)``.

        The returned reconstruction remains the unconstrained proposal ``z``;
        this method is only a loss term.
        """

        qz = self.complement(proposal)
        coarse_component = proposal.float() - qz.float()
        weight = self.complement.quadrature_weight
        numerator = weight * coarse_component.square().sum(dim=-1)
        denominator = weight * proposal.float().square().sum(dim=-1) + float(eps)
        return (numerator / denominator).mean()

    def forward(self, features: torch.Tensor, coarse: torch.Tensor) -> dict[str, torch.Tensor]:
        proposal = self.proposal(features)
        detail = self.constrain(proposal)
        # The only final path is fixed coarse P1 plus either Qz or matched-control z.
        final = coarse + detail
        return {"proposal": proposal, "detail": detail, "final_prediction": final}


class HierarchicalFEMVoxelRefiner(nn.Module):
    """Thin composition wrapper that cannot bypass the fixed detail branch."""

    def __init__(self, detail_model: ComplementConstrainedVoxelDetail) -> None:
        super().__init__()
        self.detail_model = detail_model

    def forward(self, features: torch.Tensor, coarse: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.detail_model(features, coarse)
