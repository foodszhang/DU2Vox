"""Isocapacity plain-residual controls for CQR innovation falsification."""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn

from du2vox.models.stage2.cqr_residual_inr import PositionalEncoding
from du2vox.models.stage2.transport_cqr_lifter import TransportConsistentCQRLifter


@contextmanager
def component_seed(offset: int):
    """Isolate component initialization so paired models share common weights."""

    base_seed = torch.initial_seed()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(base_seed + offset)
        yield


def normalized_measurement_residual(
    measured_modes: torch.Tensor,
    stage1_modes: torch.Tensor,
    mode: str = "least_squares_stage1_scale",
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the relative-task residual and Stage 1 scale used by all controls."""

    measured = measured_modes.float()
    stage1 = stage1_modes.float()
    if mode == "least_squares_stage1_scale":
        scale = torch.sum(measured * stage1, dim=-1) / (torch.sum(stage1.square(), dim=-1) + eps)
        scale = scale.clamp_min(0.0)
        return measured - scale.unsqueeze(-1) * stage1, scale
    if mode == "l2":
        measured_n = measured / (torch.linalg.vector_norm(measured, dim=-1, keepdim=True) + eps)
        stage1_n = stage1 / (torch.linalg.vector_norm(stage1, dim=-1, keepdim=True) + eps)
        return measured_n - stage1_n, torch.ones_like(measured[:, 0])
    if mode == "max":
        measured_n = measured / (measured.abs().amax(dim=-1, keepdim=True) + eps)
        stage1_n = stage1 / (stage1.abs().amax(dim=-1, keepdim=True) + eps)
        return measured_n - stage1_n, torch.ones_like(measured[:, 0])
    raise ValueError(f"Unknown measurement normalization mode: {mode}")


class CQRResidualBackbone(nn.Module):
    """Shared trunk used by plain and partitioned residual decoders."""

    def __init__(
        self,
        n_freqs: int,
        hidden_dim: int,
        n_hidden_layers: int,
        prior_dim: int,
        view_feat_dim: int,
        use_band_embedding: bool,
        band_embed_dim: int,
        num_bands: int,
    ) -> None:
        super().__init__()
        self.pe = PositionalEncoding(n_freqs=n_freqs, include_input=True)
        self.prior_dim = int(prior_dim)
        self.view_feat_dim = int(view_feat_dim)
        self.use_band_embedding = bool(use_band_embedding)
        self.band_embed_dim = int(band_embed_dim)
        self.num_bands = int(num_bands)

        input_dim = self.pe.out_dim + prior_dim + self.view_feat_dim + 2
        if self.use_band_embedding:
            self.band_embedding = nn.Embedding(self.num_bands, self.band_embed_dim)
            input_dim += self.band_embed_dim

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(max(0, n_hidden_layers - 1)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        self.net = nn.Sequential(*layers)
        self.out_dim = int(hidden_dim)

    def forward(
        self,
        coords: torch.Tensor,
        prior_lift: torch.Tensor,
        correction_band: torch.Tensor,
        view_feat: torch.Tensor | None,
        adjoint_evidence: torch.Tensor,
        rho0: torch.Tensor,
    ) -> torch.Tensor:
        features = [self.pe(coords), prior_lift]
        if view_feat is not None:
            features.append(view_feat)
        elif self.view_feat_dim:
            features.append(
                torch.zeros(
                    (*coords.shape[:2], self.view_feat_dim),
                    device=coords.device,
                    dtype=coords.dtype,
                )
            )
        features.extend([adjoint_evidence.unsqueeze(-1), rho0.unsqueeze(-1)])
        if self.use_band_embedding:
            features.append(
                self.band_embedding(correction_band.long().clamp(0, self.num_bands - 1))
            )
        return self.net(torch.cat(features, dim=-1))


class CQRPhysicsEvidenceMixin:
    """Common operator/evidence calculations without correction projection."""

    measurement_normalization: str
    measurement_basis: torch.Tensor
    projected_forward_modes: torch.Tensor

    @staticmethod
    def build_query_operator(
        query_green: torch.Tensor,
        candidate_cell_weight: torch.Tensor | None,
        n_valid_candidate_pool: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, n_query = query_green.shape[:2]
        if candidate_cell_weight is None:
            weight = torch.ones(
                (batch, n_query), device=query_green.device, dtype=query_green.dtype
            )
        else:
            weight = candidate_cell_weight.to(query_green.dtype)
        if n_valid_candidate_pool is not None:
            pool_size = n_valid_candidate_pool.to(query_green.dtype).reshape(batch, 1)
            weight = weight * pool_size / max(n_query, 1)
        return query_green.transpose(-1, -2) * weight.unsqueeze(-2)

    def measurement_residual_modes(
        self,
        measurement_b: torch.Tensor | None,
        coarse_d: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            measurement_b is None
            or coarse_d is None
            or self.measurement_basis.numel() == 0
            or self.projected_forward_modes.numel() == 0
        ):
            return (
                torch.zeros((batch_size, rank), device=device, dtype=torch.float32),
                torch.ones(batch_size, device=device, dtype=torch.float32),
            )
        measurement = measurement_b.reshape(batch_size, -1).float()
        coarse = coarse_d.reshape(batch_size, -1).float()
        with torch.amp.autocast(device.type, enabled=False):
            measured_modes = measurement @ self.measurement_basis.float()
            stage1_modes = coarse @ self.projected_forward_modes.float().transpose(0, 1)
            return normalized_measurement_residual(
                measured_modes, stage1_modes, mode=self.measurement_normalization
            )


class PlainCQRResidualControl(nn.Module, CQRPhysicsEvidenceMixin):
    """Strong CQR residual control with either P1 or learned transport lifting."""

    def __init__(
        self,
        green_node_modes: torch.Tensor,
        elements: torch.Tensor,
        measurement_basis: torch.Tensor | None = None,
        projected_forward_modes: torch.Tensor | None = None,
        n_freqs: int = 10,
        hidden_dim: int = 256,
        n_hidden_layers: int = 4,
        prior_dim: int = 15,
        view_feat_dim: int = 0,
        use_band_embedding: bool = True,
        band_embed_dim: int = 8,
        num_bands: int = 5,
        lifting_mode: str = "p1",
        max_logit_delta: float = 2.0,
        mu_relative: float = 1e-3,
        jitter: float = 1e-6,
        measurement_normalization: str = "least_squares_stage1_scale",
    ) -> None:
        super().__init__()
        if prior_dim != 15:
            raise ValueError("PlainCQRResidualControl requires prior_lift dim 15")
        if lifting_mode not in {"p1", "transport"}:
            raise ValueError(f"Unknown lifting_mode: {lifting_mode}")
        self.lifting_mode = lifting_mode
        self.measurement_normalization = str(measurement_normalization)
        self.mu_relative = float(mu_relative)
        self.jitter = float(jitter)
        self.register_buffer(
            "green_node_modes", torch.as_tensor(green_node_modes, dtype=torch.float32)
        )
        self.register_buffer("elements", torch.as_tensor(elements, dtype=torch.long))
        if measurement_basis is None:
            measurement_basis = torch.empty((0, self.green_node_modes.shape[0]))
        if projected_forward_modes is None:
            projected_forward_modes = torch.empty((self.green_node_modes.shape[0], 0))
        self.register_buffer(
            "measurement_basis", torch.as_tensor(measurement_basis, dtype=torch.float32)
        )
        self.register_buffer(
            "projected_forward_modes",
            torch.as_tensor(projected_forward_modes, dtype=torch.float32),
        )
        if lifting_mode == "transport":
            with component_seed(101):
                self.lifter = TransportConsistentCQRLifter(
                    self.green_node_modes,
                    self.elements,
                    hidden_dim=min(hidden_dim, 128),
                    max_logit_delta=max_logit_delta,
                )
        else:
            self.lifter = None
        with component_seed(202):
            self.backbone = CQRResidualBackbone(
                n_freqs=n_freqs,
                hidden_dim=hidden_dim,
                n_hidden_layers=n_hidden_layers,
                prior_dim=prior_dim,
                view_feat_dim=view_feat_dim,
                use_band_embedding=use_band_embedding,
                band_embed_dim=band_embed_dim,
                num_bands=num_bands,
            )
        with component_seed(303):
            self.residual_head = nn.Linear(self.backbone.out_dim, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def _p1_lift(self, prior_lift: torch.Tensor, tet_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        valid = (tet_ids >= 0) & (tet_ids < len(self.elements))
        safe = tet_ids.clamp(0, max(len(self.elements) - 1, 0))
        vertices = self.elements[safe]
        node_green = self.green_node_modes.transpose(0, 1)[vertices]
        barycentric = prior_lift[..., 4:8]
        node_values = prior_lift[..., :4]
        fem_interp = torch.sum(node_values * barycentric, dim=-1)
        query_green = torch.sum(barycentric.unsqueeze(-1) * node_green, dim=-2)
        query_green = query_green * valid.unsqueeze(-1)
        return {
            "rho0": fem_interp,
            "fem_interp": fem_interp,
            "alpha": barycentric,
            "delta_logits": torch.zeros_like(barycentric),
            "query_green": query_green,
        }

    def _regularized_adjoint(
        self, residual_modes: torch.Tensor, a_query: torch.Tensor
    ) -> torch.Tensor:
        with torch.amp.autocast(a_query.device.type, enabled=False):
            a = a_query.float()
            gram = a @ a.transpose(-1, -2)
            trace = gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            mu = self.mu_relative * trace / max(gram.shape[-1], 1)
            eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
            solved = torch.linalg.solve(
                gram + (mu + self.jitter).view(-1, 1, 1) * eye,
                residual_modes.float().unsqueeze(-1),
            )
            return (a.transpose(-1, -2) @ solved).squeeze(-1).to(residual_modes.dtype)

    def forward(
        self,
        coords: torch.Tensor,
        prior_lift: torch.Tensor,
        view_feat: torch.Tensor | None = None,
        *,
        correction_band: torch.Tensor,
        tet_ids: torch.Tensor,
        role: torch.Tensor | None = None,
        query_src_tag: torch.Tensor | None = None,
        candidate_cell_weight: torch.Tensor | None = None,
        n_valid_candidate_pool: torch.Tensor | None = None,
        measurement_b: torch.Tensor | None = None,
        coarse_d: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.lifter is None:
            lifted = self._p1_lift(prior_lift, tet_ids)
        else:
            lifted = self.lifter(
                prior_lift,
                tet_ids,
                correction_band,
                role=role,
                query_src_tag=query_src_tag,
            )
        a_query = self.build_query_operator(
            lifted["query_green"], candidate_cell_weight, n_valid_candidate_pool
        )
        residual_modes, measurement_scale = self.measurement_residual_modes(
            measurement_b,
            coarse_d,
            coords.shape[0],
            coords.device,
            rank=self.green_node_modes.shape[0],
        )
        adjoint = self._regularized_adjoint(residual_modes, a_query)
        features = self.backbone(
            coords,
            prior_lift,
            correction_band,
            view_feat,
            adjoint,
            lifted["rho0"],
        )
        correction = self.residual_head(features).squeeze(-1)
        d_hat = torch.clamp(lifted["rho0"] + correction, 0.0, 1.0)
        predicted_modes = (a_query.float() @ correction.float().unsqueeze(-1)).squeeze(-1)
        correction_modes = measurement_scale.unsqueeze(-1) * predicted_modes
        residual_energy = residual_modes.square().sum(dim=-1) + 1e-8
        data_relative_after = (residual_modes - correction_modes).square().sum(
            dim=-1
        ) / residual_energy
        transport_delta_modes = (
            a_query.float() @ (lifted["rho0"] - lifted["fem_interp"]).float().unsqueeze(-1)
        ).squeeze(-1)
        transport_reference = (
            a_query.float() @ lifted["fem_interp"].float().unsqueeze(-1)
        ).squeeze(-1)
        transport_error = transport_delta_modes.square().sum(dim=-1) / (
            transport_reference.square().sum(dim=-1) + 1e-8
        )
        return {
            "d_hat": d_hat,
            "rho0": lifted["rho0"],
            "fem_interp": lifted["fem_interp"],
            "residual": d_hat - lifted["fem_interp"],
            "plain_correction": correction,
            "raw_residual": correction,
            "adjoint_evidence": adjoint,
            "alpha": lifted["alpha"],
            "delta_logits": lifted["delta_logits"],
            "transport_error": transport_error,
            "measurement_scale": measurement_scale,
            "measurement_residual_modes": residual_modes,
            "data_relative_before": torch.ones_like(data_relative_after),
            "data_relative_after": data_relative_after,
            "a_query": a_query,
        }

    def fixed_quadrature_transport_loss(
        self,
        prior_lift: torch.Tensor,
        tet_ids: torch.Tensor,
        correction_band: torch.Tensor,
        quadrature_weight: torch.Tensor,
        role: torch.Tensor | None = None,
        query_src_tag: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.lifter is None:
            return prior_lift.new_zeros(())
        lifted = self.lifter(
            prior_lift,
            tet_ids,
            correction_band,
            role=role,
            query_src_tag=query_src_tag,
        )
        a_quad = lifted["query_green"].transpose(-1, -2) * quadrature_weight.unsqueeze(-2)
        delta_modes = (
            a_quad.float() @ (lifted["rho0"] - lifted["fem_interp"]).float().unsqueeze(-1)
        ).squeeze(-1)
        reference_modes = (a_quad.float() @ lifted["fem_interp"].float().unsqueeze(-1)).squeeze(-1)
        return (
            delta_modes.square().sum(dim=-1) / (reference_modes.square().sum(dim=-1) + 1e-8)
        ).mean()
