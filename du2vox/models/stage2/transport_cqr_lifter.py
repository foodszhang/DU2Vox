"""Convex, Green-relation-aware FEM-to-query field lifting."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional


class TransportConsistentCQRLifter(nn.Module):
    def __init__(
        self,
        green_node_modes: torch.Tensor,
        elements: torch.Tensor,
        hidden_dim: int = 64,
        max_logit_delta: float = 2.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.register_buffer("green_node_modes", torch.as_tensor(green_node_modes, dtype=torch.float32))
        self.register_buffer("elements", torch.as_tensor(elements, dtype=torch.long))
        self.max_logit_delta = float(max_logit_delta)
        self.eps = float(eps)
        # Per vertex: lambda, d_v, P1, Green norm/similarity/distance/log-ratio,
        # RGL residual, band, role one-hot, query-source one-hot.
        feature_dim = 10 + 5 + 4
        self.score_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.score_net[-1].weight)
        nn.init.zeros_(self.score_net[-1].bias)

    def vertex_indices(self, tet_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = (tet_ids >= 0) & (tet_ids < len(self.elements))
        safe = tet_ids.clamp(0, max(len(self.elements) - 1, 0))
        vertices = self.elements[safe]
        return vertices, valid

    def query_green_fingerprint(
        self, prior_lift: torch.Tensor, tet_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vertices, valid = self.vertex_indices(tet_ids)
        node_green = self.green_node_modes.transpose(0, 1)[vertices]
        barycentric = prior_lift[..., 4:8]
        query_green = torch.sum(barycentric.unsqueeze(-1) * node_green, dim=-2)
        query_green = query_green * valid.unsqueeze(-1)
        return query_green, node_green, valid

    def forward(
        self,
        prior_lift: torch.Tensor,
        tet_ids: torch.Tensor,
        correction_band: torch.Tensor,
        role: torch.Tensor | None = None,
        query_src_tag: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if prior_lift.shape[-1] != 15:
            raise ValueError(f"Transport lifter requires prior_lift dim 15, got {prior_lift.shape}")
        role = correction_band if role is None else role
        query_src_tag = torch.zeros_like(role) if query_src_tag is None else query_src_tag
        node_values = prior_lift[..., :4]
        barycentric = prior_lift[..., 4:8]
        fem_interp = torch.sum(node_values * barycentric, dim=-1)
        query_green, node_green, valid = self.query_green_fingerprint(prior_lift, tet_ids)
        node_norm = torch.linalg.vector_norm(node_green.float(), dim=-1).to(prior_lift.dtype)
        query_norm = torch.linalg.vector_norm(query_green.float(), dim=-1).to(prior_lift.dtype)
        dot = torch.sum(node_green * query_green.unsqueeze(-2), dim=-1)
        cosine = dot / (node_norm * query_norm.unsqueeze(-1) + self.eps)
        distance = torch.linalg.vector_norm(
            node_green - query_green.unsqueeze(-2), dim=-1
        ) / (node_norm + query_norm.unsqueeze(-1) + self.eps)
        log_norm_ratio = torch.log((node_norm + self.eps) / (query_norm.unsqueeze(-1) + self.eps))
        residual = prior_lift[..., 13:14].expand_as(node_values)
        band_scalar = correction_band.to(prior_lift.dtype).unsqueeze(-1).expand_as(node_values) / 4.0
        role_one_hot = functional.one_hot(role.long().clamp(0, 4), num_classes=5).to(prior_lift.dtype)
        source_one_hot = functional.one_hot(query_src_tag.long().clamp(0, 3), num_classes=4).to(
            prior_lift.dtype
        )
        shared = torch.cat([role_one_hot, source_one_hot], dim=-1).unsqueeze(-2)
        shared = shared.expand(*node_values.shape, shared.shape[-1])
        features = torch.cat(
            [
                barycentric.unsqueeze(-1),
                node_values.unsqueeze(-1),
                fem_interp.unsqueeze(-1).unsqueeze(-1).expand(*node_values.shape, 1),
                node_norm.unsqueeze(-1),
                query_norm.unsqueeze(-1).unsqueeze(-1).expand(*node_values.shape, 1),
                cosine.unsqueeze(-1),
                distance.unsqueeze(-1),
                log_norm_ratio.unsqueeze(-1),
                residual.unsqueeze(-1),
                band_scalar.unsqueeze(-1),
                shared,
            ],
            dim=-1,
        )
        delta_logits = self.max_logit_delta * torch.tanh(self.score_net(features).squeeze(-1))
        logits = torch.log(barycentric.clamp_min(self.eps)) + delta_logits
        alpha = torch.softmax(logits, dim=-1)
        alpha = torch.where(valid.unsqueeze(-1), alpha, barycentric)
        rho0 = torch.sum(alpha * node_values, dim=-1)
        return {
            "rho0": rho0,
            "fem_interp": fem_interp,
            "alpha": alpha,
            "delta_logits": delta_logits,
            "green_similarity": cosine,
            "green_norm_ratio": torch.exp(log_norm_ratio),
            "query_green": query_green,
        }
