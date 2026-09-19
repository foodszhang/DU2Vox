"""Correction-state cross-space transfer followed by an exact hard-Q decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from du2vox.models.stage2.complement_voxel_detail import (
    ExactVoxelComplement,
    ResidualMLPBlock,
)


class CorrectionStateTransfer(nn.Module):
    """Query-local CST model.

    Shapes use ``N`` canonical voxel queries, ``L`` nodal latent channels, and
    ``D`` transfer channels. ``state`` is ``[N, S]``; ``h0`` and ``hc`` are
    ``[N, L]``. The unconstrained output is projected globally by fixed ``Q``.
    """

    def __init__(
        self,
        *,
        coordinate_dim: int,
        state_dim: int,
        latent_dim: int,
        transfer_dim: int,
        modulation_groups: int,
        decoder_width: int,
        decoder_depth: int,
        complement: ExactVoxelComplement,
        innovation: str = "delta",
        one_ring_context_dim: int = 0,
        direct_view_dim: int = 0,
    ) -> None:
        super().__init__()
        if innovation not in {"none", "delta", "hc_delta"}:
            raise ValueError("innovation must be none, delta, or hc_delta")
        if transfer_dim % modulation_groups:
            raise ValueError("transfer_dim must be divisible by modulation_groups")
        self.innovation = innovation
        self.mode = "hard"
        self.transfer_dim = int(transfer_dim)
        self.modulation_groups = int(modulation_groups)
        self.complement = complement
        self.one_ring_context_dim = int(one_ring_context_dim)
        self.direct_view_dim = int(direct_view_dim)

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, transfer_dim), nn.GELU(), nn.LayerNorm(transfer_dim)
        )
        self.persistent_adapter = nn.Linear(latent_dim, transfer_dim)
        innovation_input = latent_dim * (2 if innovation == "hc_delta" else 1)
        self.innovation_adapter = (
            nn.Linear(innovation_input, transfer_dim)
            if innovation != "none"
            else None
        )
        self.modulation = nn.Linear(transfer_dim, modulation_groups)
        # sigmoid(-2.1972) ~= 0.1: bounded and initially conservative.
        self.alpha_logit = nn.Parameter(torch.tensor(-2.1972246))
        if self.one_ring_context_dim:
            element_input = latent_dim * 2 + 2
            self.element_projection = nn.Sequential(
                nn.Linear(element_input, self.one_ring_context_dim),
                nn.GELU(),
                nn.LayerNorm(self.one_ring_context_dim),
            )
            self.neighbor_difference = nn.Sequential(
                nn.Linear(self.one_ring_context_dim, self.one_ring_context_dim),
                nn.GELU(),
                nn.Linear(self.one_ring_context_dim, self.one_ring_context_dim),
            )
            self.element_norm = nn.LayerNorm(self.one_ring_context_dim)

        decoder_input = (
            coordinate_dim
            + state_dim
            + transfer_dim
            + self.one_ring_context_dim
            + self.direct_view_dim
        )
        self.decoder_input = nn.Linear(decoder_input, decoder_width)
        self.decoder_blocks = nn.ModuleList(
            ResidualMLPBlock(decoder_width) for _ in range(decoder_depth)
        )
        self.decoder_norm = nn.LayerNorm(decoder_width)
        self.decoder_output = nn.Linear(decoder_width, 1)
        self.activation = nn.GELU()
        nn.init.zeros_(self.decoder_output.weight)
        nn.init.zeros_(self.decoder_output.bias)

    def transferred_context(
        self, state: torch.Tensor, h0: torch.Tensor, hc: torch.Tensor
    ) -> torch.Tensor:
        explicit = self.state_encoder(state)
        persistent = self.persistent_adapter(hc)
        if self.innovation == "none":
            return persistent
        delta = hc - h0
        innovation_source = (
            torch.cat([hc, delta], dim=-1)
            if self.innovation == "hc_delta"
            else delta
        )
        innovation = self.innovation_adapter(innovation_source)
        group_gate = torch.tanh(self.modulation(explicit))
        channels_per_group = self.transfer_dim // self.modulation_groups
        gate = group_gate.repeat_interleave(channels_per_group, dim=-1)
        alpha = torch.sigmoid(self.alpha_logit)
        return persistent + innovation * (1.0 + alpha * gate)

    def proposal(
        self,
        coordinate: torch.Tensor,
        state: torch.Tensor,
        h0: torch.Tensor,
        hc: torch.Tensor,
        one_ring: torch.Tensor | None = None,
        direct_views: torch.Tensor | None = None,
    ) -> torch.Tensor:
        transferred = self.transferred_context(state, h0, hc)
        if self.one_ring_context_dim:
            if one_ring is None or one_ring.shape[-1] != self.one_ring_context_dim:
                raise ValueError("Configured CST model requires one-ring context")
        elif one_ring is not None:
            raise ValueError("One-ring context supplied to a context-free CST model")
        if self.direct_view_dim:
            if direct_views is None or direct_views.shape[-1] != self.direct_view_dim:
                raise ValueError("Configured CST model requires direct-view features")
        elif direct_views is not None:
            raise ValueError("Direct-view features supplied to a view-free CST model")
        pieces = [coordinate, state, transferred]
        if one_ring is not None:
            pieces.append(one_ring)
        if direct_views is not None:
            pieces.append(direct_views)
        hidden = self.activation(
            self.decoder_input(torch.cat(pieces, dim=-1))
        )
        for block in self.decoder_blocks:
            hidden = block(hidden)
        return self.decoder_output(self.decoder_norm(hidden)).squeeze(-1)

    def proposal_checkpointed(self, *features: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return self.proposal(*features)
        return checkpoint(self.proposal, *features, use_reentrant=False)

    def constrain(self, proposal: torch.Tensor) -> torch.Tensor:
        return self.complement(proposal)

    def build_one_ring_context(
        self,
        x0: torch.Tensor,
        xc: torch.Tensor,
        h0: torch.Tensor,
        hc: torch.Tensor,
        elements: torch.Tensor,
        face_neighbors: torch.Tensor,
    ) -> torch.Tensor | None:
        """Return one residual face-adjacency message per tetrahedron."""

        if not self.one_ring_context_dim:
            return None
        delta_x = xc - x0
        delta_h = hc - h0
        local = torch.cat(
            [
                xc[elements].mean(dim=1, keepdim=True),
                delta_x[elements].mean(dim=1, keepdim=True),
                hc[elements].mean(dim=1),
                delta_h[elements].mean(dim=1),
            ],
            dim=-1,
        )
        base = self.element_projection(local)
        neighbors = base[face_neighbors]
        message = self.neighbor_difference(neighbors - base[:, None]).mean(dim=1)
        return self.element_norm(base + message)

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logit)
