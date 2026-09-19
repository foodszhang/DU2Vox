"""
Core building blocks for MS-GDUN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sparse_left_mm_batched(matrix: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Apply one sparse ``[N, N]`` matrix to a ``[B, N, C]`` tensor.

    Folding the batch and channel axes into the dense right-hand side avoids one
    sparse kernel launch per sample while preserving the per-sample operation.
    """
    if values.ndim != 3:
        raise ValueError("values must have shape [B, N, C]")
    batch, n_nodes, channels = values.shape
    if matrix.shape != (n_nodes, n_nodes):
        raise ValueError("matrix and values node dimensions differ")
    rhs = values.permute(1, 0, 2).reshape(n_nodes, batch * channels)
    result = torch.sparse.mm(matrix, rhs)
    return result.reshape(n_nodes, batch, channels).permute(1, 0, 2)


class GCNBlock(nn.Module):
    """Standard spectral GCN: out = LeakyReLU(L @ X @ W + b)."""

    def __init__(self, L: torch.Tensor, in_dim: int, out_dim: int):
        super().__init__()
        self.L = L
        self.weight = nn.Parameter(
            nn.init.kaiming_normal_(
                torch.empty(in_dim, out_dim, dtype=torch.float32),
                mode="fan_out",
            )
        )
        self.bias = nn.Parameter(torch.zeros(out_dim, dtype=torch.float32))
        self.act = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, in_dim]
        x_w = torch.matmul(x, self.weight)  # [B, N, out_dim]
        if self.L.is_sparse:
            out = sparse_left_mm_batched(self.L, x_w)
        else:
            out = torch.matmul(self.L, x_w)
        out = out + self.bias
        return self.act(out)


class InputBlock(nn.Module):
    """Input module: concat(x, L^TL x, A^TA x - A^T b).

    x:  [B, N, 1]
    b:  [B, S, 1]
    L:  [N, N] sparse Laplacian
    A:  [S, N] sparse forward matrix

    Returns: [B, N, 3]
    Uses L.T @ (L @ x) and A.T @ (A @ x) to avoid dense intermediate storage.
    """

    def __init__(
        self,
        L: torch.Tensor,
        A: torch.Tensor,
        LTL=None,
        ATA=None,
        physics_evidence: str = "raw",
        evidence_eps: float = 1e-8,
        profiled_evidence_rms: float = 0.05,
    ):
        super().__init__()
        if physics_evidence not in {"raw", "profiled_normalized"}:
            raise ValueError("physics_evidence must be raw or profiled_normalized")
        self.L = L
        self.A = A
        self.physics_evidence = physics_evidence
        self.evidence_eps = float(evidence_eps)
        self.profiled_evidence_rms = float(profiled_evidence_rms)
        if self.profiled_evidence_rms <= 0:
            raise ValueError("profiled_evidence_rms must be positive")

    def forward(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # x: [B, N, 1], b: [B, S, 1]
        # L.t() @ (L @ x), with batch samples folded into one sparse RHS.
        Lx = sparse_left_mm_batched(self.L, x)
        LTLx = sparse_left_mm_batched(self.L.t(), Lx)

        # A.t() @ (A @ x) - A is dense
        # x: [B, N] -> A @ x: [B, S]
        x_b = x.squeeze(-1)  # [B, N]
        Ax = torch.mm(x_b, self.A.t())  # [B, S]
        b_b = b.squeeze(-1)  # [B, S]
        if self.physics_evidence == "raw":
            gradient = torch.mm(Ax - b_b, self.A).unsqueeze(-1)
        else:
            # Profile out the unknown positive measurement/state amplitude. At
            # the exact cold start Ax=0 the profiled gradient is zero, so use
            # the raw adjoint until a nonzero response exists. Per-case RMS
            # normalization removes the arbitrary evidence magnitude.
            denominator = Ax.square().sum(dim=1, keepdim=True)
            numerator = (Ax * b_b).sum(dim=1, keepdim=True).clamp_min(0.0)
            alpha = numerator / (denominator + self.evidence_eps)
            profiled_residual = alpha * Ax - b_b
            profiled = alpha * torch.mm(profiled_residual, self.A)
            raw_residual = Ax - b_b
            raw = torch.mm(raw_residual, self.A)
            usable = (denominator > self.evidence_eps) & (numerator > self.evidence_eps)
            gradient = torch.where(usable, profiled, raw)
            residual = torch.where(usable, profiled_residual, raw_residual)
            relative_residual = torch.linalg.vector_norm(
                residual, dim=1, keepdim=True
            ) / torch.linalg.vector_norm(b_b, dim=1, keepdim=True).clamp_min(self.evidence_eps)
            rms = gradient.square().mean(dim=1, keepdim=True).sqrt()
            gradient = (
                gradient
                / (rms + self.evidence_eps)
                * self.profiled_evidence_rms
                * relative_residual
            ).unsqueeze(-1)

        return torch.cat([x, LTLx, gradient], dim=-1)


class AdaptiveThreshold(nn.Module):
    """Node-wise adaptive sparse threshold — gradient-friendly version.

    Replaces sign(u) * softplus(|u| - λ) with soft shrinkage:
    out = u * sigmoid(k * (|u| - λ))
    - When |u| >> λ: sigmoid → 1, out ≈ u (preserve signal)
    - When |u| << λ: sigmoid → 0, out ≈ 0 (sparse suppression)
    - Gradients flow through both u and gate paths, never zeroed by sign()
    """

    def __init__(self, feat_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_dim // 2, 1),
            nn.Softplus(),
        )
        self.k = 10.0  # sigmoid sharpness — higher = closer to hard threshold
        nn.init.constant_(self.net[2].bias, -3.0)  # init λ ≈ 0.05 at start

    def forward(self, u: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """u: [B, N, 1], feat: [B, N, C]"""
        lam = self.net(feat)  # [B, N, 1], always positive
        gate = torch.sigmoid(self.k * (torch.abs(u) - lam))
        return u * gate  # gradients flow through both u and gate


class UpdateBlock(nn.Module):
    """Wraps AdaptiveThreshold with u = x - grad computation."""

    def __init__(self, feat_dim: int):
        super().__init__()
        self.adaptive_thresh = AdaptiveThreshold(feat_dim)

    def forward(self, x: torch.Tensor, grad: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        u = x - grad
        return self.adaptive_thresh(u, feat)


class SparseUpdate(nn.Module):
    """Sparse update: sign(u) * softplus(|u| - theta).

    Simple version with fixed theta, or adaptive via AdaptiveThreshold.
    """

    def __init__(self, theta: float = 0.0):
        super().__init__()
        self.theta = theta

    def forward(
        self,
        x: torch.Tensor,
        grad: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        u = x - alpha * grad
        return torch.sign(u) * F.softplus(torch.abs(u) - self.theta)
