"""Error-Structured Cross-Discretization Bridging (ESCB).

The computation is sequential:
    FEM inverse correction -> fixed P1 transfer -> voxel representation completion.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from du2vox.bridge.canonical_cross_discretization import canonical_p1_torch
from du2vox.models.stage2.residual_inr import PositionalEncoding


class FEMContextualResidualBlock(nn.Module):
    """One fixed-topology kNN context update with a gated residual connection."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.context_projection = nn.Linear(hidden_dim * 3, hidden_dim)
        self.residual_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()

    def forward(
        self, hidden: torch.Tensor, knn_indices: torch.Tensor
    ) -> torch.Tensor:
        neighbor_mean = hidden[:, knn_indices].mean(dim=2)
        context = torch.cat(
            [hidden, neighbor_mean, hidden - neighbor_mean], dim=-1
        )
        update = self.activation(self.context_projection(context))
        gate = torch.sigmoid(
            self.residual_gate(torch.cat([hidden, neighbor_mean], dim=-1))
        )
        return self.normalization(hidden + gate * update)


class FEMInverseCorrectionNet(nn.Module):
    """Lightweight multi-hop FEM contextual corrector in nodal space."""

    def __init__(
        self,
        knn_indices: torch.Tensor,
        hidden_dim: int = 144,
        n_hidden_layers: int = 3,
        view_feat_dim: int = 0,
    ) -> None:
        super().__init__()
        self.register_buffer("knn_indices", knn_indices.long())
        input_dim = 3 + 3 + int(view_feat_dim)
        if n_hidden_layers < 1:
            raise ValueError("FEM contextual corrector needs at least one block")
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_normalization = nn.LayerNorm(hidden_dim)
        self.context_blocks = nn.ModuleList(
            FEMContextualResidualBlock(hidden_dim)
            for _ in range(n_hidden_layers)
        )
        self.output = nn.Linear(hidden_dim, 1)
        self.activation = nn.GELU()
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        x_h: torch.Tensor,
        node_coords_norm: torch.Tensor,
        view_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if node_coords_norm.ndim == 2:
            node_coords_norm = node_coords_norm.unsqueeze(0).expand(x_h.shape[0], -1, -1)
        neighbor = x_h[:, self.knn_indices]
        neighbor_mean = neighbor.mean(dim=-1)
        neighbor_std = neighbor.std(dim=-1, unbiased=False)
        features = [
            node_coords_norm,
            x_h.unsqueeze(-1),
            neighbor_mean.unsqueeze(-1),
            neighbor_std.unsqueeze(-1),
        ]
        if view_features is not None:
            features.append(view_features)
        hidden = self.input_normalization(
            self.activation(self.input_projection(torch.cat(features, dim=-1)))
        )
        for block in self.context_blocks:
            hidden = block(hidden, self.knn_indices)
        return self.output(hidden).squeeze(-1)


class VoxelRepresentationCompletionNet(nn.Module):
    """Query MLP for only rho_gt - I_h Pi_h rho_gt."""

    def __init__(
        self,
        n_freqs: int = 10,
        hidden_dim: int = 256,
        n_hidden_layers: int = 4,
        view_feat_dim: int = 0,
    ) -> None:
        super().__init__()
        self.pe = PositionalEncoding(n_freqs=n_freqs, include_input=True)
        # original/corrected nodal values (4+4), barycentric (4), P1 values (2)
        context_dim = 14
        input_dim = self.pe.out_dim + context_dim + int(view_feat_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.hidden = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)
        )
        self.output = nn.Linear(hidden_dim, 1)
        self.activation = nn.GELU()
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        coords_norm: torch.Tensor,
        original_nodes: torch.Tensor,
        corrected_nodes: torch.Tensor,
        barycentric: torch.Tensor,
        original_p1: torch.Tensor,
        corrected_p1: torch.Tensor,
        view_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, n_query = coords_norm.shape[:2]
        pieces = [
            self.pe(coords_norm.reshape(-1, 3)),
            original_nodes.reshape(-1, 4),
            corrected_nodes.reshape(-1, 4),
            barycentric.reshape(-1, 4),
            original_p1.reshape(-1, 1),
            corrected_p1.reshape(-1, 1),
        ]
        if view_features is not None:
            pieces.append(view_features.reshape(batch * n_query, -1))
        hidden = self.activation(self.input_proj(torch.cat(pieces, dim=-1)))
        for layer in self.hidden:
            hidden = self.activation(layer(hidden))
        return self.output(hidden).reshape(batch, n_query)


def fixed_representation_target(
    gt_values: torch.Tensor, projected_gt_values: torch.Tensor
) -> torch.Tensor:
    """The fixed scientific target; no inverse prediction is accepted here."""

    return gt_values - projected_gt_values


class ErrorStructuredCrossDiscretizationBridge(nn.Module):
    """Sequential ESCB model with an optional shared multiview encoder."""

    def __init__(
        self,
        inverse_net: FEMInverseCorrectionNet,
        representation_net: VoxelRepresentationCompletionNet,
        view_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.inverse_net = inverse_net
        self.representation_net = representation_net
        self.view_encoder = view_encoder
        self.training_phase = "C"

    def set_phase(self, phase: str) -> None:
        phase = phase.upper()
        if phase not in {"A", "B", "C"}:
            raise ValueError(f"Unknown ESCB phase: {phase}")
        for parameter in self.inverse_net.parameters():
            parameter.requires_grad_(phase in {"A", "C"})
        for parameter in self.representation_net.parameters():
            parameter.requires_grad_(phase in {"B", "C"})
        # Shared view features are trained with inverse pretraining, held fixed in B,
        # and jointly fine-tuned in C.  Thus x_h+ cannot drift during Phase B.
        if self.view_encoder is not None:
            for parameter in self.view_encoder.parameters():
                parameter.requires_grad_(phase in {"A", "C"})
        self.training_phase = phase

    @staticmethod
    def _gather_nodes(values: torch.Tensor, node_indices: torch.Tensor) -> torch.Tensor:
        if node_indices.ndim == 2:
            node_indices = node_indices.unsqueeze(0).expand(values.shape[0], -1, -1)
        return torch.gather(
            values.unsqueeze(1).expand(-1, node_indices.shape[1], -1),
            2,
            node_indices.long(),
        )

    def _view_features(
        self,
        proj_imgs: torch.Tensor | None,
        node_coords_world: torch.Tensor | None,
        query_coords_world: torch.Tensor | None,
        encoded_view_maps: torch.Tensor | dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.view_encoder is None:
            return None, None
        if node_coords_world is None or query_coords_world is None:
            raise ValueError("Multiview ESCB requires images and world coordinates")
        if proj_imgs is None and encoded_view_maps is None:
            raise ValueError("Multiview ESCB requires images or encoded view maps")
        if node_coords_world.ndim == 2:
            node_coords_world = node_coords_world.unsqueeze(0).expand(
                query_coords_world.shape[0], -1, -1
            )
        joined = torch.cat([node_coords_world, query_coords_world], dim=1)
        if encoded_view_maps is None:
            features, _ = self.view_encoder(proj_imgs, joined, coords_vox_norm=None)
        else:
            features, _ = self.view_encoder.sample_encoded(
                encoded_view_maps, joined, coords_vox_norm=None
            )
        n_nodes = node_coords_world.shape[1]
        return features[:, :n_nodes], features[:, n_nodes:]

    def forward(
        self,
        x_h: torch.Tensor,
        node_coords_norm: torch.Tensor,
        query_coords_norm: torch.Tensor,
        query_node_indices: torch.Tensor,
        query_barycentric: torch.Tensor,
        *,
        proj_imgs: torch.Tensor | None = None,
        node_coords_world: torch.Tensor | None = None,
        query_coords_world: torch.Tensor | None = None,
        encoded_view_maps: torch.Tensor | dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        node_view, query_view = self._view_features(
            proj_imgs,
            node_coords_world,
            query_coords_world,
            encoded_view_maps,
        )
        delta_x_h = self.inverse_net(x_h, node_coords_norm, node_view)
        corrected_x_h = x_h + delta_x_h
        original_p1 = canonical_p1_torch(
            x_h, query_node_indices, query_barycentric
        )
        corrected_p1 = canonical_p1_torch(
            corrected_x_h, query_node_indices, query_barycentric
        )
        original_nodes = self._gather_nodes(x_h, query_node_indices)
        corrected_nodes = self._gather_nodes(corrected_x_h, query_node_indices)
        representation = self.representation_net(
            query_coords_norm,
            original_nodes,
            corrected_nodes,
            query_barycentric,
            original_p1,
            corrected_p1,
            query_view,
        )
        final = corrected_p1 + representation
        return {
            "stage1_fem": x_h,
            "inverse_correction": delta_x_h,
            "corrected_fem": corrected_x_h,
            "original_fem_voxel": original_p1,
            "corrected_fem_voxel": corrected_p1,
            "representation_prediction": representation,
            "final_prediction": final,
        }


class PlainTargetFirstVoxelResidual(nn.Module):
    """Strong control: I_h x_h + R, supervised by rho_gt - I_h x_h."""

    def __init__(
        self,
        residual_net: VoxelRepresentationCompletionNet,
        view_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.residual_net = residual_net
        self.view_encoder = view_encoder

    def forward(
        self,
        x_h: torch.Tensor,
        query_coords_norm: torch.Tensor,
        query_node_indices: torch.Tensor,
        query_barycentric: torch.Tensor,
        *,
        proj_imgs: torch.Tensor | None = None,
        query_coords_world: torch.Tensor | None = None,
        encoded_view_maps: torch.Tensor | dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        query_view = None
        if self.view_encoder is not None:
            if query_coords_world is None:
                raise ValueError("Multiview baseline requires images and world coordinates")
            if encoded_view_maps is None:
                if proj_imgs is None:
                    raise ValueError("Multiview baseline requires images")
                query_view, _ = self.view_encoder(
                    proj_imgs, query_coords_world, coords_vox_norm=None
                )
            else:
                query_view, _ = self.view_encoder.sample_encoded(
                    encoded_view_maps, query_coords_world, coords_vox_norm=None
                )
        original_p1 = canonical_p1_torch(
            x_h, query_node_indices, query_barycentric
        )
        original_nodes = ErrorStructuredCrossDiscretizationBridge._gather_nodes(
            x_h, query_node_indices
        )
        residual = self.residual_net(
            query_coords_norm,
            original_nodes,
            original_nodes,
            query_barycentric,
            original_p1,
            original_p1,
            query_view,
        )
        final = original_p1 + residual
        return {
            "stage1_fem": x_h,
            "original_fem_voxel": original_p1,
            "plain_residual_prediction": residual,
            "final_prediction": final,
        }
