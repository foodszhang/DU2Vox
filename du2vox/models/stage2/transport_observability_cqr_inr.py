"""Transport-consistent, observability-resolved CQR Stage 2 model."""

from __future__ import annotations

import torch
import torch.nn as nn

from du2vox.models.stage2.cqr_observability_projector import CQRObservabilityProjector
from du2vox.models.stage2.cqr_residual_inr import PositionalEncoding
from du2vox.models.stage2.transport_cqr_lifter import TransportConsistentCQRLifter


def normalized_measurement_residual(
    measured_modes: torch.Tensor,
    stage1_modes: torch.Tensor,
    mode: str = "least_squares_stage1_scale",
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return explicit relative-task residual modes and the applied scale."""

    measured = measured_modes.float()
    stage1 = stage1_modes.float()
    if mode == "least_squares_stage1_scale":
        scale = torch.sum(measured * stage1, dim=-1) / (
            torch.sum(stage1.square(), dim=-1) + eps
        )
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


class _CorrectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_hidden_layers: int):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(max(0, n_hidden_layers - 1)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class TransportObservabilityCQRINR(nn.Module):
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
        max_logit_delta: float = 2.0,
        mu_relative: float = 1e-3,
        jitter: float = 1e-6,
        measurement_normalization: str = "least_squares_stage1_scale",
    ):
        super().__init__()
        if prior_dim != 15:
            raise ValueError("TransportObservabilityCQRINR requires prior_lift dim 15")
        self.pe = PositionalEncoding(n_freqs=n_freqs, include_input=True)
        self.prior_dim = prior_dim
        self.view_feat_dim = int(view_feat_dim)
        self.use_band_embedding = bool(use_band_embedding)
        self.band_embed_dim = int(band_embed_dim)
        self.num_bands = int(num_bands)
        self.measurement_normalization = str(measurement_normalization)
        self.lifter = TransportConsistentCQRLifter(
            green_node_modes,
            elements,
            hidden_dim=min(hidden_dim, 128),
            max_logit_delta=max_logit_delta,
        )
        self.projector = CQRObservabilityProjector(mu_relative=mu_relative, jitter=jitter)
        if measurement_basis is None:
            measurement_basis = torch.empty((0, green_node_modes.shape[0]))
        if projected_forward_modes is None:
            projected_forward_modes = torch.empty((green_node_modes.shape[0], 0))
        self.register_buffer("measurement_basis", torch.as_tensor(measurement_basis, dtype=torch.float32))
        self.register_buffer(
            "projected_forward_modes",
            torch.as_tensor(projected_forward_modes, dtype=torch.float32),
        )
        input_dim = self.pe.out_dim + prior_dim + self.view_feat_dim + 2
        if self.use_band_embedding:
            self.band_embedding = nn.Embedding(self.num_bands, self.band_embed_dim)
            input_dim += self.band_embed_dim
        self.observable_head = _CorrectionHead(input_dim, hidden_dim, n_hidden_layers)
        self.ambiguous_head = _CorrectionHead(input_dim, hidden_dim, n_hidden_layers)

    def set_phase(self, phase: str, freeze_lifter_after_phase_a: bool = False) -> None:
        valid_phases = {
            "lifter",
            "observable",
            "full",
            "observable_pretrain",
            "ambiguous_pretrain",
            "joint",
        }
        if phase not in valid_phases:
            raise ValueError(f"Unknown training phase: {phase}")
        train_lifter = phase == "lifter" or not freeze_lifter_after_phase_a
        for parameter in self.lifter.parameters():
            parameter.requires_grad_(train_lifter)
        for parameter in self.observable_head.parameters():
            parameter.requires_grad_(
                phase in {"observable", "full", "observable_pretrain", "joint"}
            )
        for parameter in self.ambiguous_head.parameters():
            parameter.requires_grad_(
                phase in {"full", "ambiguous_pretrain", "joint"}
            )
        self.training_phase = phase

    def build_query_operator(
        self,
        query_green: torch.Tensor,
        candidate_cell_weight: torch.Tensor | None,
        n_valid_candidate_pool: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, n_query = query_green.shape[:2]
        if candidate_cell_weight is None:
            weight = torch.ones((batch, n_query), device=query_green.device, dtype=query_green.dtype)
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rank = self.lifter.green_node_modes.shape[0]
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

    def _head_features(
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
        return torch.cat(features, dim=-1)

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
            measurement_b, coarse_d, coords.shape[0], coords.device
        )
        adjoint = self.projector.adjoint_evidence(residual_modes, a_query)
        features = self._head_features(
            coords, prior_lift, correction_band, view_feat, adjoint, lifted["rho0"]
        )
        raw_observable = self.observable_head(features)
        raw_ambiguous = self.ambiguous_head(features)
        observable = self.projector.project_observable(raw_observable, a_query)
        ambiguous = self.projector.project_ambiguous(raw_ambiguous, a_query)
        d_hat = torch.clamp(lifted["rho0"] + observable + ambiguous, 0.0, 1.0)

        observable_modes = (a_query.float() @ observable.float().unsqueeze(-1)).squeeze(-1)
        ambiguous_modes = (a_query.float() @ ambiguous.float().unsqueeze(-1)).squeeze(-1)
        # least_squares_stage1_scale places the relative-density correction in
        # the same measurement units as the scaled Stage 1 forward prediction.
        correction_modes = measurement_scale.unsqueeze(-1) * (
            observable_modes + ambiguous_modes
        )
        residual_energy = residual_modes.square().sum(dim=-1) + 1e-8
        data_relative_after = (
            (residual_modes - correction_modes).square().sum(dim=-1) / residual_energy
        )
        observable_reduction = 1.0 - data_relative_after
        ambiguous_leakage = ambiguous_modes.square().sum(dim=-1) / (
            ambiguous.float().square().sum(dim=-1) + 1e-8
        )
        transport_delta_modes = (
            a_query.float()
            @ (lifted["rho0"] - lifted["fem_interp"]).float().unsqueeze(-1)
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
            "observable_correction": observable,
            "ambiguous_correction": ambiguous,
            "raw_observable": raw_observable,
            "raw_ambiguous": raw_ambiguous,
            "adjoint_evidence": adjoint,
            "alpha": lifted["alpha"],
            "delta_logits": lifted["delta_logits"],
            "green_similarity": lifted["green_similarity"],
            "green_norm_ratio": lifted["green_norm_ratio"],
            "transport_error": transport_error,
            "observable_residual_reduction": observable_reduction,
            "ambiguous_measurement_leakage": ambiguous_leakage,
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
        lifted = self.lifter(
            prior_lift,
            tet_ids,
            correction_band,
            role=role,
            query_src_tag=query_src_tag,
        )
        a_quad = lifted["query_green"].transpose(-1, -2) * quadrature_weight.unsqueeze(-2)
        delta_modes = (
            a_quad.float()
            @ (lifted["rho0"] - lifted["fem_interp"]).float().unsqueeze(-1)
        ).squeeze(-1)
        reference_modes = (
            a_quad.float() @ lifted["fem_interp"].float().unsqueeze(-1)
        ).squeeze(-1)
        return (
            delta_modes.square().sum(dim=-1)
            / (reference_modes.square().sum(dim=-1) + 1e-8)
        ).mean()
