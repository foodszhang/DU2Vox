"""Residual-aware iterative correction of a Stage-1 FEM inverse state.

The model stays entirely in the 19,990-node FEM space.  Its only voxel operation is
the fixed canonical P1 interpolation used for supervision and evaluation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from du2vox.bridge.canonical_cross_discretization import canonical_p1_torch
from du2vox.models.stage2.error_structured_bridge import FEMContextualResidualBlock


class FEMMeasurementResidual(nn.Module):
    """Fixed forward/adjoint evidence for the current FEM state."""

    def __init__(self, system_matrix: torch.Tensor, eps: float = 1e-8) -> None:
        super().__init__()
        matrix = torch.as_tensor(system_matrix, dtype=torch.float32)
        if matrix.ndim != 2:
            raise ValueError("system_matrix must have shape [N_measurements, N_nodes]")
        self.register_buffer("system_matrix", matrix, persistent=False)
        self.register_buffer(
            "normal_diagonal",
            matrix.square().sum(dim=0).clamp_min(eps),
            persistent=False,
        )
        self.eps = float(eps)

    def operator_scale(self, state: torch.Tensor, value: torch.Tensor | None) -> torch.Tensor:
        if value is None:
            return torch.ones(state.shape[0], 1, device=state.device, dtype=torch.float32)
        scale = value.float().reshape(-1, 1)
        if scale.shape[0] != state.shape[0]:
            raise ValueError("measurement_operator_scale must have one value per case")
        if not torch.isfinite(scale).all() or not torch.all(scale > 0):
            raise ValueError("measurement_operator_scale must be finite and positive")
        return scale

    def forward(
        self,
        state: torch.Tensor,
        measurement: torch.Tensor,
        measurement_operator_scale: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # Keep the large physics products in FP32 even under BF16 autocast.
        with torch.autocast(device_type=state.device.type, enabled=False):
            state_fp32 = state.float()
            measurement_fp32 = measurement.float()
            scale = self.operator_scale(state_fp32, measurement_operator_scale)
            forward_response = (state_fp32 @ self.system_matrix.t()) * scale
            forward_residual = measurement_fp32 - forward_response
            backprojection = (forward_residual @ self.system_matrix) * scale
            scaled_diagonal = self.normal_diagonal * scale.square()
            jacobi = backprojection / scaled_diagonal.clamp_min(self.eps)
            bp_rms = backprojection.square().mean(dim=-1, keepdim=True).sqrt()
            jacobi_rms = jacobi.square().mean(dim=-1, keepdim=True).sqrt()
            normalized_backprojection = backprojection / (bp_rms + self.eps)
            normalized_jacobi = jacobi / (jacobi_rms + self.eps)
            residual_rms = forward_residual.square().mean(dim=-1).sqrt()
        return {
            "forward_residual": forward_residual,
            "backprojection": backprojection,
            "normalized_backprojection": normalized_backprojection,
            "normalized_jacobi": normalized_jacobi,
            "forward_residual_rms": residual_rms,
        }


class ScaleCalibratedFEMMeasurementResidual(FEMMeasurementResidual):
    """Profile out the non-negative per-sample measurement amplitude."""

    def profile_from_forward(
        self, forward_response: torch.Tensor, measurement: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        numerator = (forward_response * measurement).sum(dim=-1, keepdim=True)
        denominator = forward_response.square().sum(dim=-1, keepdim=True)
        # The fluorophore amplitude is physically non-negative. This is the analytic
        # one-dimensional NNLS solution, not a clamp on the reconstructed FEM state.
        amplitude = numerator.clamp_min(0.0) / (denominator + self.eps)
        residual = measurement - amplitude * forward_response
        rms = residual.square().mean(dim=-1).sqrt()
        return {"amplitude": amplitude, "residual": residual, "rms": rms}

    def forward(
        self,
        state: torch.Tensor,
        measurement: torch.Tensor,
        measurement_operator_scale: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        with torch.autocast(device_type=state.device.type, enabled=False):
            state_fp32 = state.float()
            measurement_fp32 = measurement.float()
            scale = self.operator_scale(state_fp32, measurement_operator_scale)
            forward_response = (state_fp32 @ self.system_matrix.t()) * scale
            profiled = self.profile_from_forward(forward_response, measurement_fp32)
            amplitude = profiled["amplitude"]
            residual = profiled["residual"]
            backprojection = amplitude * (residual @ self.system_matrix) * scale
            scaled_diagonal = amplitude.square() * self.normal_diagonal * scale.square() + self.eps
            jacobi = backprojection / scaled_diagonal
            bp_rms = backprojection.square().mean(dim=-1, keepdim=True).sqrt()
            jacobi_rms = jacobi.square().mean(dim=-1, keepdim=True).sqrt()
        return {
            "forward_response": forward_response,
            "amplitude": amplitude.squeeze(-1),
            "forward_residual": residual,
            "backprojection": backprojection,
            "normalized_backprojection": backprojection / (bp_rms + self.eps),
            "normalized_jacobi": jacobi / (jacobi_rms + self.eps),
            "jacobi": jacobi,
            "forward_residual_rms": profiled["rms"],
        }


class DualScaleFEMMeasurementEvidence(ScaleCalibratedFEMMeasurementResidual):
    """Fixed raw and amplitude-profiled evidence from one current FEM state.

    The raw channel exactly follows :class:`FEMMeasurementResidual`: ``y-Ax`` and
    ``A.T(y-Ax)``.  The scale-invariant channel exactly follows the verified V3
    implementation, including its non-negative analytic amplitude and the amplitude
    factor in the gradient with respect to the FEM state.
    """

    def forward(
        self,
        state: torch.Tensor,
        measurement: torch.Tensor,
        measurement_operator_scale: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        with torch.autocast(device_type=state.device.type, enabled=False):
            state_fp32 = state.float()
            measurement_fp32 = measurement.float()
            scale = self.operator_scale(state_fp32, measurement_operator_scale)
            forward_response = (state_fp32 @ self.system_matrix.t()) * scale

            raw_residual = measurement_fp32 - forward_response
            raw_backprojection = (raw_residual @ self.system_matrix) * scale
            raw_diagonal = self.normal_diagonal * scale.square()
            raw_jacobi = raw_backprojection / raw_diagonal.clamp_min(self.eps)

            profiled = self.profile_from_forward(forward_response, measurement_fp32)
            amplitude = profiled["amplitude"]
            si_residual = profiled["residual"]
            si_backprojection = amplitude * (si_residual @ self.system_matrix) * scale
            si_diagonal = amplitude.square() * self.normal_diagonal * scale.square() + self.eps
            si_jacobi = si_backprojection / si_diagonal

            def _normalize(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                # sqrt has an undefined derivative at exactly zero. A profiled
                # amplitude of zero makes the entire SI adjoint exactly zero, so
                # clamp the squared RMS before sqrt to keep online joint backward
                # finite without changing the normalized zero-valued forward.
                norm = value.square().mean(dim=-1, keepdim=True).clamp_min(self.eps**2).sqrt()
                return value / (norm + self.eps), norm

            raw_bp_normed, raw_bp_rms = _normalize(raw_backprojection)
            raw_jacobi_normed, _ = _normalize(raw_jacobi)
            si_bp_normed, si_bp_rms = _normalize(si_backprojection)
            si_jacobi_normed, _ = _normalize(si_jacobi)
            forward_rms = forward_response.square().mean(dim=-1).sqrt()
            measurement_rms = measurement_fp32.square().mean(dim=-1).sqrt()

        return {
            "forward_response": forward_response,
            "forward_response_rms": forward_rms,
            "relative_forward_rms": forward_rms / (measurement_rms + self.eps),
            "amplitude": amplitude.squeeze(-1),
            "raw_residual": raw_residual,
            "raw_residual_rms": raw_residual.square().mean(dim=-1).sqrt(),
            "raw_backprojection": raw_backprojection,
            "raw_backprojection_rms": raw_bp_rms.squeeze(-1),
            "raw_normalized_backprojection": raw_bp_normed,
            "raw_normalized_jacobi": raw_jacobi_normed,
            "si_residual": si_residual,
            "si_residual_rms": profiled["rms"],
            "si_backprojection": si_backprojection,
            "si_backprojection_rms": si_bp_rms.squeeze(-1),
            "si_normalized_backprojection": si_bp_normed,
            "si_normalized_jacobi": si_jacobi_normed,
        }


class DualEvidenceFEMCorrectionCell(nn.Module):
    """Shared mesh-context update fed by raw and scale-invariant physics."""

    def __init__(
        self,
        hidden_dim: int,
        view_feat_dim: int,
        n_context_blocks: int,
        max_iterations: int,
    ) -> None:
        super().__init__()
        if n_context_blocks < 1:
            raise ValueError("At least one contextual block is required")
        # coords(3), initial/current/history(3), local state mean/std(2),
        # raw/SI bp and Jacobi(4), learned mixture(1), local mixture mean/std(2).
        input_dim = 15 + int(view_feat_dim)
        self.evidence_mixer = nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.zeros_(self.evidence_mixer[-1].weight)
        nn.init.constant_(self.evidence_mixer[-1].bias, 1.3862944)  # beta=0.8
        self.node_projection = nn.Linear(input_dim, hidden_dim)
        self.node_normalization = nn.LayerNorm(hidden_dim)
        self.iteration_embedding = nn.Embedding(max_iterations, hidden_dim)
        self.context_blocks = nn.ModuleList(
            FEMContextualResidualBlock(hidden_dim) for _ in range(n_context_blocks)
        )
        # Global mean/max context plus state, two residuals, alpha, and ||Ax||.
        self.global_film = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 6, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )
        self.output_normalization = nn.LayerNorm(hidden_dim)
        self.delta_head = nn.Linear(hidden_dim, 1)
        self.activation = nn.GELU()
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(
        self,
        *,
        initial_state: torch.Tensor,
        current_state: torch.Tensor,
        hidden: torch.Tensor | None,
        node_coords_norm: torch.Tensor,
        knn_indices: torch.Tensor,
        evidence: dict[str, torch.Tensor],
        iteration: int,
        view_features: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state_neighbors = current_state[:, knn_indices]
        state_mean = state_neighbors.mean(dim=2)
        state_std = state_neighbors.std(dim=2, unbiased=False)
        alpha = evidence["amplitude"][:, None].expand_as(current_state)
        mix_input = torch.stack(
            [
                evidence["si_normalized_backprojection"],
                evidence["raw_normalized_backprojection"],
                current_state,
                alpha,
            ],
            dim=-1,
        )
        beta = torch.sigmoid(self.evidence_mixer(mix_input).squeeze(-1))
        mixed = (
            beta * evidence["si_normalized_backprojection"]
            + (1.0 - beta) * evidence["raw_normalized_backprojection"]
        )
        mixed_neighbors = mixed[:, knn_indices]
        pieces = [
            node_coords_norm,
            initial_state.unsqueeze(-1),
            current_state.unsqueeze(-1),
            (current_state - initial_state).unsqueeze(-1),
            state_mean.unsqueeze(-1),
            state_std.unsqueeze(-1),
            evidence["si_normalized_backprojection"].unsqueeze(-1),
            evidence["si_normalized_jacobi"].unsqueeze(-1),
            evidence["raw_normalized_backprojection"].unsqueeze(-1),
            evidence["raw_normalized_jacobi"].unsqueeze(-1),
            mixed.unsqueeze(-1),
            mixed_neighbors.mean(dim=2).unsqueeze(-1),
            mixed_neighbors.std(dim=2, unbiased=False).unsqueeze(-1),
        ]
        if view_features is not None:
            pieces.append(view_features)
        node_input = self.node_normalization(
            self.activation(self.node_projection(torch.cat(pieces, dim=-1)))
        )
        iteration_ids = torch.full(
            (current_state.shape[0],),
            iteration,
            dtype=torch.long,
            device=current_state.device,
        )
        node_input = node_input + self.iteration_embedding(iteration_ids)[:, None, :]
        hidden = node_input if hidden is None else hidden + node_input
        for block in self.context_blocks:
            hidden = block(hidden, knn_indices)
        scalar_context = torch.stack(
            [
                current_state.mean(dim=1),
                current_state.square().mean(dim=1).sqrt(),
                torch.log1p(evidence["si_residual_rms"]),
                torch.log1p(evidence["raw_residual_rms"]),
                torch.log1p(evidence["amplitude"]),
                torch.log1p(evidence["relative_forward_rms"]),
            ],
            dim=-1,
        )
        scale, shift = self.global_film(
            torch.cat([hidden.mean(dim=1), hidden.amax(dim=1), scalar_context], dim=-1)
        ).chunk(2, dim=-1)
        hidden = self.output_normalization(
            hidden * (1.0 + 0.1 * torch.tanh(scale[:, None, :])) + shift[:, None, :]
        )
        delta = self.delta_head(self.activation(hidden)).squeeze(-1)
        dc_direction = beta * torch.tanh(evidence["si_normalized_jacobi"]) + (
            1.0 - beta
        ) * torch.tanh(evidence["raw_normalized_jacobi"])
        return delta, hidden, beta, dc_direction


class ResidualAwareFEMCorrectionCell(nn.Module):
    """Shared contextual update used at every inverse-correction iteration."""

    def __init__(
        self,
        hidden_dim: int,
        view_feat_dim: int,
        n_context_blocks: int,
        max_iterations: int,
    ) -> None:
        super().__init__()
        if n_context_blocks < 1:
            raise ValueError("At least one contextual block is required")
        # coords(3), initial/current/history(3), local state mean/std(2),
        # residual/backprojection/Jacobi and local residual mean/std(5).
        input_dim = 13 + int(view_feat_dim)
        self.node_projection = nn.Linear(input_dim, hidden_dim)
        self.node_normalization = nn.LayerNorm(hidden_dim)
        self.iteration_embedding = nn.Embedding(max_iterations, hidden_dim)
        self.context_blocks = nn.ModuleList(
            FEMContextualResidualBlock(hidden_dim) for _ in range(n_context_blocks)
        )
        # Global mean/max hidden context plus four scalar state/residual statistics.
        self.global_film = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 4, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )
        self.output_normalization = nn.LayerNorm(hidden_dim)
        self.delta_head = nn.Linear(hidden_dim, 1)
        self.activation = nn.GELU()
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(
        self,
        *,
        initial_state: torch.Tensor,
        current_state: torch.Tensor,
        hidden: torch.Tensor | None,
        node_coords_norm: torch.Tensor,
        knn_indices: torch.Tensor,
        normalized_backprojection: torch.Tensor,
        normalized_jacobi: torch.Tensor,
        forward_residual_rms: torch.Tensor,
        iteration: int,
        view_features: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_neighbors = current_state[:, knn_indices]
        state_mean = state_neighbors.mean(dim=2)
        state_std = state_neighbors.std(dim=2, unbiased=False)
        residual_neighbors = normalized_backprojection[:, knn_indices]
        residual_mean = residual_neighbors.mean(dim=2)
        residual_std = residual_neighbors.std(dim=2, unbiased=False)
        pieces = [
            node_coords_norm,
            initial_state.unsqueeze(-1),
            current_state.unsqueeze(-1),
            (current_state - initial_state).unsqueeze(-1),
            state_mean.unsqueeze(-1),
            state_std.unsqueeze(-1),
            normalized_backprojection.unsqueeze(-1),
            normalized_jacobi.unsqueeze(-1),
            residual_mean.unsqueeze(-1),
            residual_std.unsqueeze(-1),
            normalized_backprojection.abs().unsqueeze(-1),
        ]
        if view_features is not None:
            pieces.append(view_features)
        node_input = self.node_normalization(
            self.activation(self.node_projection(torch.cat(pieces, dim=-1)))
        )
        iteration_ids = torch.full(
            (current_state.shape[0],),
            iteration,
            dtype=torch.long,
            device=current_state.device,
        )
        node_input = node_input + self.iteration_embedding(iteration_ids)[:, None, :]
        hidden = node_input if hidden is None else hidden + node_input
        for block in self.context_blocks:
            hidden = block(hidden, knn_indices)

        hidden_mean = hidden.mean(dim=1)
        hidden_max = hidden.amax(dim=1)
        scalar_context = torch.stack(
            [
                current_state.mean(dim=1),
                current_state.square().mean(dim=1).sqrt(),
                normalized_backprojection.abs().mean(dim=1),
                torch.log1p(forward_residual_rms),
            ],
            dim=-1,
        )
        scale, shift = self.global_film(
            torch.cat([hidden_mean, hidden_max, scalar_context], dim=-1)
        ).chunk(2, dim=-1)
        hidden = self.output_normalization(
            hidden * (1.0 + 0.1 * torch.tanh(scale[:, None, :])) + shift[:, None, :]
        )
        delta = self.delta_head(self.activation(hidden)).squeeze(-1)
        return delta, hidden


class IterativeResidualAwareFEMCorrector(nn.Module):
    """Shared-weight FEM correction followed only by analytic P1 transfer."""

    def __init__(
        self,
        *,
        system_matrix: torch.Tensor,
        knn_indices: torch.Tensor,
        hidden_dim: int = 144,
        n_context_blocks: int = 2,
        n_iterations: int = 3,
        view_feat_dim: int = 0,
        view_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if n_iterations < 1:
            raise ValueError("n_iterations must be positive")
        self.register_buffer("knn_indices", knn_indices.long(), persistent=False)
        self.measurement_residual = FEMMeasurementResidual(system_matrix)
        self.cell = ResidualAwareFEMCorrectionCell(
            hidden_dim=hidden_dim,
            view_feat_dim=view_feat_dim,
            n_context_blocks=n_context_blocks,
            max_iterations=n_iterations,
        )
        self.n_iterations = int(n_iterations)
        self.view_encoder = view_encoder

    def correct_nodes(
        self,
        x_h: torch.Tensor,
        measurement_b: torch.Tensor,
        node_coords_norm: torch.Tensor,
        *,
        node_coords_world: torch.Tensor | None = None,
        proj_imgs: torch.Tensor | None = None,
        encoded_views: torch.Tensor | dict[str, torch.Tensor] | None = None,
        measurement_operator_scale: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if node_coords_norm.ndim == 2:
            node_coords_norm = node_coords_norm.unsqueeze(0).expand(x_h.shape[0], -1, -1)
        node_view = None
        if self.view_encoder is not None:
            if node_coords_world is None:
                raise ValueError("node_coords_world is required with multiview evidence")
            if node_coords_world.ndim == 2:
                node_coords_world = node_coords_world.unsqueeze(0).expand(x_h.shape[0], -1, -1)
            if encoded_views is None:
                if proj_imgs is None:
                    raise ValueError("proj_imgs or encoded_views is required")
                encoded_views = self.view_encoder.encode_images(proj_imgs)
            node_view, _ = self.view_encoder.sample_encoded(
                encoded_views, node_coords_world, coords_vox_norm=None
            )

        state = x_h
        hidden = None
        states = []
        corrections = []
        residual_norms = []
        for iteration in range(self.n_iterations):
            evidence = self.measurement_residual(state, measurement_b, measurement_operator_scale)
            delta, hidden = self.cell(
                initial_state=x_h,
                current_state=state,
                hidden=hidden,
                node_coords_norm=node_coords_norm,
                knn_indices=self.knn_indices,
                normalized_backprojection=evidence["normalized_backprojection"],
                normalized_jacobi=evidence["normalized_jacobi"],
                forward_residual_rms=evidence["forward_residual_rms"],
                iteration=iteration,
                view_features=node_view,
            )
            state = state + delta
            corrections.append(delta)
            states.append(state)
            residual_norms.append(evidence["forward_residual_rms"])
        # This post-update value is a diagnostic, not a training objective.  Avoid
        # retaining one additional full A/A.T graph solely for logging.
        with torch.no_grad():
            final_evidence = self.measurement_residual(
                state, measurement_b, measurement_operator_scale
            )
        residual_norms.append(final_evidence["forward_residual_rms"])
        return {
            "stage1_nodes": x_h,
            "step_corrections": torch.stack(corrections, dim=1),
            "step_states": torch.stack(states, dim=1),
            "corrected_nodes": state,
            "measurement_residual_rms": torch.stack(residual_norms, dim=1),
        }

    def forward(
        self,
        x_h: torch.Tensor,
        measurement_b: torch.Tensor,
        node_coords_norm: torch.Tensor,
        query_node_indices: torch.Tensor,
        query_barycentric: torch.Tensor,
        *,
        node_coords_world: torch.Tensor | None = None,
        proj_imgs: torch.Tensor | None = None,
        measurement_operator_scale: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        output = self.correct_nodes(
            x_h,
            measurement_b,
            node_coords_norm,
            node_coords_world=node_coords_world,
            proj_imgs=proj_imgs,
            measurement_operator_scale=measurement_operator_scale,
        )
        output["stage1_fem_voxel"] = canonical_p1_torch(x_h, query_node_indices, query_barycentric)
        output["final_prediction"] = canonical_p1_torch(
            output["corrected_nodes"], query_node_indices, query_barycentric
        )
        return output


class ResidualConsistentIterativeFEMCorrector(IterativeResidualAwareFEMCorrector):
    """Scale-calibrated proximal FEM updates with monotone residual routing."""

    def __init__(
        self,
        *,
        system_matrix: torch.Tensor,
        knn_indices: torch.Tensor,
        hidden_dim: int = 144,
        n_context_blocks: int = 2,
        n_iterations: int = 3,
        view_feat_dim: int = 0,
        view_encoder: nn.Module | None = None,
        max_neural_update: float = 0.5,
        max_dc_update: float = 0.1,
        residual_tolerance: float = 0.0,
        hard_trust_region: bool = True,
    ) -> None:
        super().__init__(
            system_matrix=system_matrix,
            knn_indices=knn_indices,
            hidden_dim=hidden_dim,
            n_context_blocks=n_context_blocks,
            n_iterations=n_iterations,
            view_feat_dim=view_feat_dim,
            view_encoder=view_encoder,
        )
        self.measurement_residual = ScaleCalibratedFEMMeasurementResidual(system_matrix)
        self.max_neural_update = float(max_neural_update)
        self.max_dc_update = float(max_dc_update)
        self.residual_tolerance = float(residual_tolerance)
        self.hard_trust_region = bool(hard_trust_region)
        # 2*sigmoid(-4.595) ~= 0.02. Together with max_dc_update this preserves
        # the Stage-1 identity at initialization while keeping DC trainable.
        self.dc_step_logits = nn.Parameter(torch.full((n_iterations,), -4.59512))
        self.register_buffer(
            "line_search_scales",
            torch.tensor([1.0, 0.5, 0.25, 0.125, 0.0]),
            persistent=False,
        )

    def _monotone_update(
        self,
        state: torch.Tensor,
        direction: torch.Tensor,
        forward_response: torch.Tensor,
        measurement: torch.Tensor,
        baseline_rms: torch.Tensor,
        measurement_operator_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=state.device.type, enabled=False):
            scale = self.measurement_residual.operator_scale(
                state.float(), measurement_operator_scale
            )
            forward_direction = (
                direction.float() @ self.measurement_residual.system_matrix.t()
            ) * scale
            scales = self.line_search_scales.to(state.device)
            candidates = (
                forward_response[:, None, :] + scales[None, :, None] * forward_direction[:, None, :]
            )
            batch, n_scales, n_measurements = candidates.shape
            profiled = self.measurement_residual.profile_from_forward(
                candidates.reshape(batch * n_scales, n_measurements),
                measurement.float()[:, None, :]
                .expand(-1, n_scales, -1)
                .reshape(batch * n_scales, n_measurements),
            )
            candidate_rms = profiled["rms"].reshape(batch, n_scales)
            threshold = baseline_rms[:, None] * (1.0 + self.residual_tolerance)
            acceptable = candidate_rms <= threshold + self.measurement_residual.eps
            selected_index = acceptable.float().argmax(dim=1)
            selected_scale = scales[selected_index]
            selected_rms = candidate_rms.gather(1, selected_index[:, None]).squeeze(1)
        selected = selected_scale[:, None].to(state.dtype)
        # Forward execution obeys the profiled-residual trust region exactly. For
        # rejected proposals, use a straight-through task gradient so the network
        # can learn to turn an inadmissible proposal into an admissible one instead
        # of receiving zero gradient forever. The detached term is identically zero
        # in the forward pass and therefore cannot alter inference semantics.
        accepted_direction = selected * direction
        rejected_proposal_gradient = (1.0 - selected) * (direction - direction.detach())
        next_state = state + accepted_direction + rejected_proposal_gradient
        return next_state, selected_scale, selected_rms

    def correct_nodes(
        self,
        x_h: torch.Tensor,
        measurement_b: torch.Tensor,
        node_coords_norm: torch.Tensor,
        *,
        node_coords_world: torch.Tensor | None = None,
        proj_imgs: torch.Tensor | None = None,
        encoded_views: torch.Tensor | dict[str, torch.Tensor] | None = None,
        measurement_operator_scale: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if node_coords_norm.ndim == 2:
            node_coords_norm = node_coords_norm.unsqueeze(0).expand(x_h.shape[0], -1, -1)
        node_view = None
        if self.view_encoder is not None:
            if node_coords_world is None:
                raise ValueError("node_coords_world is required with multiview evidence")
            if node_coords_world.ndim == 2:
                node_coords_world = node_coords_world.unsqueeze(0).expand(x_h.shape[0], -1, -1)
            if encoded_views is None:
                if proj_imgs is None:
                    raise ValueError("proj_imgs or encoded_views is required")
                encoded_views = self.view_encoder.encode_images(proj_imgs)
            node_view, _ = self.view_encoder.sample_encoded(
                encoded_views, node_coords_world, coords_vox_norm=None
            )

        state = x_h
        hidden = None
        states = []
        corrections = []
        residual_norms = []
        accepted_scales = []
        amplitudes = []
        for iteration in range(self.n_iterations):
            evidence = self.measurement_residual(state, measurement_b, measurement_operator_scale)
            raw_delta, hidden = self.cell(
                initial_state=x_h,
                current_state=state,
                hidden=hidden,
                node_coords_norm=node_coords_norm,
                knn_indices=self.knn_indices,
                normalized_backprojection=evidence["normalized_backprojection"],
                normalized_jacobi=evidence["normalized_jacobi"],
                forward_residual_rms=evidence["forward_residual_rms"],
                iteration=iteration,
                view_features=node_view,
            )
            neural_delta = self.max_neural_update * torch.tanh(raw_delta)
            dc_step = self.max_dc_update * (2.0 * torch.sigmoid(self.dc_step_logits[iteration]))
            # Raw Jacobi values inherit the arbitrary scaling/conditioning of A
            # and can be enormous. The normalized, bounded direction is a learned
            # trust-region DC proposal rather than an unconstrained solver step.
            dc_direction = torch.tanh(evidence["normalized_jacobi"]).to(neural_delta.dtype)
            direction = neural_delta + dc_step * dc_direction
            if self.hard_trust_region:
                next_state, accepted, next_rms = self._monotone_update(
                    state,
                    direction,
                    evidence["forward_response"],
                    measurement_b,
                    evidence["forward_residual_rms"],
                    measurement_operator_scale,
                )
            else:
                # The support target and normalized measurement are not guaranteed
                # to share an exact forward-model optimum. V3 therefore applies the
                # bounded proximal proposal and supervises profiled DC softly.
                next_state = state + direction
                next_evidence = self.measurement_residual(
                    next_state, measurement_b, measurement_operator_scale
                )
                next_rms = next_evidence["forward_residual_rms"]
                accepted = torch.ones_like(next_rms)
            corrections.append(next_state - state)
            state = next_state
            states.append(state)
            residual_norms.append(evidence["forward_residual_rms"])
            accepted_scales.append(accepted)
            amplitudes.append(evidence["amplitude"])
        residual_norms.append(next_rms)
        result = {
            "stage1_nodes": x_h,
            "step_corrections": torch.stack(corrections, dim=1),
            "step_states": torch.stack(states, dim=1),
            "corrected_nodes": state,
            "measurement_residual_rms": torch.stack(residual_norms, dim=1),
            "profiled_amplitude": torch.stack(amplitudes, dim=1),
        }
        if self.hard_trust_region:
            result["accepted_step_scale"] = torch.stack(accepted_scales, dim=1)
        return result


class UnifiedDualEvidenceFEMCorrector(IterativeResidualAwareFEMCorrector):
    """V4 single-model correction with raw and scale-invariant evidence.

    All trainable operations remain in FEM nodal space.  The physics operator is
    fixed, its two channels are recomputed from the current state at every shared-
    weight iteration, and voxel output is produced only by canonical analytic P1.
    """

    def __init__(
        self,
        *,
        system_matrix: torch.Tensor,
        knn_indices: torch.Tensor,
        hidden_dim: int = 144,
        n_context_blocks: int = 2,
        n_iterations: int = 3,
        view_feat_dim: int = 0,
        view_encoder: nn.Module | None = None,
        max_neural_update: float = 0.25,
        max_dc_update: float = 0.1,
        nodal_volume_weights: torch.Tensor | None = None,
        volume_center_neural_update: bool = False,
    ) -> None:
        super().__init__(
            system_matrix=system_matrix,
            knn_indices=knn_indices,
            hidden_dim=hidden_dim,
            n_context_blocks=n_context_blocks,
            n_iterations=n_iterations,
            view_feat_dim=view_feat_dim,
            view_encoder=view_encoder,
        )
        self.measurement_residual = DualScaleFEMMeasurementEvidence(system_matrix)
        self.cell = DualEvidenceFEMCorrectionCell(
            hidden_dim=hidden_dim,
            view_feat_dim=view_feat_dim,
            n_context_blocks=n_context_blocks,
            max_iterations=n_iterations,
        )
        self.max_neural_update = float(max_neural_update)
        self.max_dc_update = float(max_dc_update)
        self.volume_center_neural_update = bool(volume_center_neural_update)
        if self.volume_center_neural_update and nodal_volume_weights is None:
            raise ValueError("nodal_volume_weights are required to volume-center updates")
        if nodal_volume_weights is None:
            nodal_volume_weights = torch.empty(0, dtype=torch.float32)
        nodal_volume_weights = torch.as_tensor(nodal_volume_weights, dtype=torch.float32)
        if nodal_volume_weights.numel() not in {0, system_matrix.shape[1]}:
            raise ValueError("nodal_volume_weights must have one value per FEM node")
        if nodal_volume_weights.numel() and (
            not torch.isfinite(nodal_volume_weights).all()
            or not torch.all(nodal_volume_weights > 0)
        ):
            raise ValueError("nodal_volume_weights must be finite and positive")
        self.register_buffer("nodal_volume_weights", nodal_volume_weights, persistent=False)
        self.dc_step_logits = nn.Parameter(torch.full((n_iterations,), -4.59512))

    def correct_nodes(
        self,
        x_h: torch.Tensor,
        measurement_b: torch.Tensor,
        node_coords_norm: torch.Tensor,
        *,
        node_coords_world: torch.Tensor | None = None,
        proj_imgs: torch.Tensor | None = None,
        encoded_views: torch.Tensor | dict[str, torch.Tensor] | None = None,
        return_terminal_hidden: bool = False,
        return_hidden_trajectory: bool = False,
        measurement_operator_scale: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if node_coords_norm.ndim == 2:
            node_coords_norm = node_coords_norm.unsqueeze(0).expand(x_h.shape[0], -1, -1)
        node_view = None
        if self.view_encoder is not None:
            if node_coords_world is None:
                raise ValueError("node_coords_world is required with multiview evidence")
            if node_coords_world.ndim == 2:
                node_coords_world = node_coords_world.unsqueeze(0).expand(x_h.shape[0], -1, -1)
            if encoded_views is None:
                if proj_imgs is None:
                    raise ValueError("proj_imgs or encoded_views is required")
                encoded_views = self.view_encoder.encode_images(proj_imgs)
            node_view, _ = self.view_encoder.sample_encoded(
                encoded_views, node_coords_world, coords_vox_norm=None
            )

        state = x_h
        hidden = None
        diagnostic_keys = (
            "amplitude",
            "raw_residual_rms",
            "si_residual_rms",
            "raw_backprojection_rms",
            "si_backprojection_rms",
            "relative_forward_rms",
        )
        diagnostics: dict[str, list[torch.Tensor]] = {key: [] for key in diagnostic_keys}
        states: list[torch.Tensor] = []
        corrections: list[torch.Tensor] = []
        evidence_cosines: list[torch.Tensor] = []
        update_norms: list[torch.Tensor] = []
        state_norms: list[torch.Tensor] = []
        dc_norms: list[torch.Tensor] = []
        fusion_means: list[torch.Tensor] = []
        hidden_states: list[torch.Tensor] = []
        for iteration in range(self.n_iterations):
            evidence = self.measurement_residual(state, measurement_b, measurement_operator_scale)
            raw_delta, hidden, beta, dc_direction = self.cell(
                initial_state=x_h,
                current_state=state,
                hidden=hidden,
                node_coords_norm=node_coords_norm,
                knn_indices=self.knn_indices,
                evidence=evidence,
                iteration=iteration,
                view_features=node_view,
            )
            if return_hidden_trajectory:
                hidden_states.append(hidden)
            neural_delta = self.max_neural_update * torch.tanh(raw_delta)
            if self.volume_center_neural_update:
                volume_weights = self.nodal_volume_weights.to(neural_delta.dtype)
                volume_mean = (neural_delta * volume_weights).sum(dim=-1, keepdim=True) / (
                    volume_weights.sum() + 1e-12
                )
                # The neural route redistributes fluorescence within the FEM
                # approximation space; global mass remains the responsibility of
                # the explicit physics/DC route. This removes only the constant P1
                # mode and leaves every spatially varying neural degree of freedom.
                neural_delta = neural_delta - volume_mean
            dc_step = self.max_dc_update * (2.0 * torch.sigmoid(self.dc_step_logits[iteration]))
            dc_delta = dc_step * dc_direction.to(neural_delta.dtype)
            update = neural_delta + dc_delta
            state = state + update
            states.append(state)
            corrections.append(update)
            for key in diagnostic_keys:
                diagnostics[key].append(evidence[key])
            evidence_cosines.append(
                torch.nn.functional.cosine_similarity(
                    evidence["raw_backprojection"],
                    evidence["si_backprojection"],
                    dim=-1,
                    eps=1e-8,
                )
            )
            update_norms.append(update.square().mean(dim=-1).sqrt())
            state_norms.append(state.square().mean(dim=-1).sqrt())
            dc_norms.append(dc_delta.square().mean(dim=-1).sqrt())
            fusion_means.append(beta.mean(dim=-1))

        final_evidence = self.measurement_residual(state, measurement_b, measurement_operator_scale)
        result = {
            "stage1_nodes": x_h,
            "step_corrections": torch.stack(corrections, dim=1),
            "step_states": torch.stack(states, dim=1),
            "corrected_nodes": state,
            # Keep the established loss/evaluator key bound to V3's SI residual.
            "measurement_residual_rms": torch.stack(
                diagnostics["si_residual_rms"] + [final_evidence["si_residual_rms"]],
                dim=1,
            ),
            "raw_residual_rms": torch.stack(
                diagnostics["raw_residual_rms"] + [final_evidence["raw_residual_rms"]],
                dim=1,
            ),
            "si_residual_rms": torch.stack(
                diagnostics["si_residual_rms"] + [final_evidence["si_residual_rms"]],
                dim=1,
            ),
            "profiled_amplitude": torch.stack(diagnostics["amplitude"], dim=1),
            "raw_adjoint_rms": torch.stack(diagnostics["raw_backprojection_rms"], dim=1),
            "si_adjoint_rms": torch.stack(diagnostics["si_backprojection_rms"], dim=1),
            "relative_forward_rms": torch.stack(diagnostics["relative_forward_rms"], dim=1),
            "raw_si_adjoint_cosine": torch.stack(evidence_cosines, dim=1),
            "update_rms": torch.stack(update_norms, dim=1),
            "state_rms": torch.stack(state_norms, dim=1),
            "dc_rms": torch.stack(dc_norms, dim=1),
            "si_fusion_weight": torch.stack(fusion_means, dim=1),
        }
        # Opt-in only: preserving the default key set avoids changing any frozen
        # V4 caller or checkpoint contract.  This is the cell output after the
        # third/final shared iteration, before the scalar update head is applied.
        if return_terminal_hidden:
            result["terminal_hidden"] = hidden
        if return_hidden_trajectory:
            result["hidden_trajectory"] = torch.stack(hidden_states, dim=1)
        return result


class FrozenConvexFEMCorrectorEnsemble(nn.Module):
    """Val-selected convex FEM-state ensemble with one final analytic P1 transfer."""

    def __init__(
        self,
        first: IterativeResidualAwareFEMCorrector,
        second: IterativeResidualAwareFEMCorrector,
        *,
        second_weight: float,
    ) -> None:
        super().__init__()
        if not 0.0 <= second_weight <= 1.0:
            raise ValueError("second_weight must be in [0, 1]")
        self.first = first
        self.second = second
        self.second_weight = float(second_weight)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def correct_nodes(self, *args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        first = self.first.correct_nodes(*args, **kwargs)
        second = self.second.correct_nodes(*args, **kwargs)
        corrected = (1.0 - self.second_weight) * first[
            "corrected_nodes"
        ] + self.second_weight * second["corrected_nodes"]
        return {
            "stage1_nodes": first["stage1_nodes"],
            "first_corrected_nodes": first["corrected_nodes"],
            "second_corrected_nodes": second["corrected_nodes"],
            "corrected_nodes": corrected,
        }

    def forward(
        self,
        x_h: torch.Tensor,
        measurement_b: torch.Tensor,
        node_coords_norm: torch.Tensor,
        query_node_indices: torch.Tensor,
        query_barycentric: torch.Tensor,
        *,
        node_coords_world: torch.Tensor | None = None,
        proj_imgs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        output = self.correct_nodes(
            x_h,
            measurement_b,
            node_coords_norm,
            node_coords_world=node_coords_world,
            proj_imgs=proj_imgs,
        )
        output["final_prediction"] = canonical_p1_torch(
            output["corrected_nodes"], query_node_indices, query_barycentric
        )
        return output
