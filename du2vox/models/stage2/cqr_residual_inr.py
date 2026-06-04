"""
CQR-compatible Stage 2 Residual INR.

This module keeps the original Stage 2 residual design but supports an extended
prior vector. The first 8 prior dimensions are fixed:
    [d_v0, d_v1, d_v2, d_v3, b0, b1, b2, b3]
where b0..b3 are barycentric coordinates.

Any extra prior dimensions are treated as coverage-aware CQR metadata.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, n_freqs: int = 10, include_input: bool = True):
        super().__init__()
        self.n_freqs = n_freqs
        self.include_input = include_input
        freqs = 2.0 ** torch.linspace(0, n_freqs - 1, n_freqs)
        self.register_buffer("freqs", freqs)

    @property
    def out_dim(self) -> int:
        d = self.n_freqs * 2 * 3
        if self.include_input:
            d += 3
        return d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = []
        if self.include_input:
            encoded.append(x)
        for freq in self.freqs:
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))
        return torch.cat(encoded, dim=-1)


class LocalProlongationAdapter(nn.Module):
    """Deprecated experimental adapter kept for old checkpoints/configs."""

    def __init__(self, prior_dim: int, out_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(prior_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.GELU(),
        )

    def forward(self, prior: torch.Tensor) -> torch.Tensor:
        return self.net(prior)


class LocalLiftingAdapter(nn.Module):
    """Deprecated experimental adapter kept for old checkpoints/configs."""

    def __init__(self, prior_dim: int, out_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(prior_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.GELU(),
        )

    def forward(self, prior: torch.Tensor) -> torch.Tensor:
        return self.net(prior)


class CQRResidualINR(nn.Module):
    def __init__(
        self,
        n_freqs: int = 10,
        hidden_dim: int = 256,
        n_hidden_layers: int = 4,
        prior_dim: int = 13,
        skip_connection: bool = True,
        view_feat_dim: int = 0,
        residual_scale: float = 0.1,
        support_head: bool = False,
        use_prolongation_adapter: bool = False,
        prolongation_feat_dim: int = 32,
        use_lifting_adapter: bool = False,
        lifting_feat_dim: int = 32,
        use_band_embedding: bool = False,
        band_embed_dim: int = 8,
        num_bands: int = 5,
        use_residual_gate: bool = False,
        residual_gate_init_bias: float = -2.0,
        residual_gate_source: str = "prior_view",
    ):
        super().__init__()
        if prior_dim < 8:
            raise ValueError("prior_dim must be >= 8")

        self.pe = PositionalEncoding(n_freqs=n_freqs, include_input=True)
        self.prior_dim = prior_dim
        self.view_feat_dim = view_feat_dim
        self.residual_scale = float(residual_scale)
        self.support_head = bool(support_head)
        self.use_prolongation_adapter = bool(use_prolongation_adapter)
        self.prolongation_feat_dim = int(prolongation_feat_dim)
        self.use_lifting_adapter = bool(use_lifting_adapter)
        self.lifting_feat_dim = int(lifting_feat_dim)
        self.use_band_embedding = bool(use_band_embedding)
        self.band_embed_dim = int(band_embed_dim)
        self.num_bands = int(num_bands)
        self.use_residual_gate = bool(use_residual_gate)
        self.residual_gate_bias = float(residual_gate_init_bias)
        self.residual_gate_source = residual_gate_source
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.skip_connection = skip_connection

        in_dim = self.pe.out_dim + prior_dim + view_feat_dim
        if self.use_prolongation_adapter:
            self.prolongation_adapter = LocalProlongationAdapter(
                prior_dim=prior_dim,
                out_dim=self.prolongation_feat_dim,
            )
            in_dim += self.prolongation_feat_dim
        if self.use_lifting_adapter:
            self.lifting_adapter = LocalLiftingAdapter(
                prior_dim=prior_dim,
                out_dim=self.lifting_feat_dim,
            )
            in_dim += self.lifting_feat_dim
        if self.use_band_embedding:
            self.band_embedding = nn.Embedding(self.num_bands, self.band_embed_dim)
            in_dim += self.band_embed_dim
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        mid = n_hidden_layers // 2
        self.hidden_layers = nn.ModuleList()
        for i in range(n_hidden_layers):
            if skip_connection and i == mid:
                self.hidden_layers.append(nn.Linear(hidden_dim * 2, hidden_dim))
            else:
                self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.skip_proj = nn.Linear(in_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        if self.use_residual_gate:
            self.gate_out = nn.Linear(hidden_dim, 1)
            nn.init.zeros_(self.gate_out.weight)
            nn.init.zeros_(self.gate_out.bias)
        if self.support_head:
            self.support_out = nn.Linear(hidden_dim, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(
        self,
        coords: torch.Tensor,
        prior: torch.Tensor,
        view_feat: torch.Tensor | None = None,
        correction_band: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]:
        B, N = coords.shape[:2]
        flat_coords = coords.reshape(B * N, 3)
        flat_prior = prior.reshape(B * N, self.prior_dim)

        pe_q = self.pe(flat_coords)
        features = [pe_q, flat_prior]
        if self.use_prolongation_adapter:
            features.append(self.prolongation_adapter(flat_prior))
        if self.use_lifting_adapter:
            features.append(self.lifting_adapter(flat_prior))
        if view_feat is not None:
            flat_view = view_feat.reshape(B * N, -1)
            features.append(flat_view)
        if self.use_band_embedding:
            if correction_band is None:
                raise ValueError("correction_band is required when use_band_embedding=True")
            flat_band = correction_band.reshape(B * N).long().clamp(0, self.num_bands - 1)
            features.append(self.band_embedding(flat_band))
        x_in = torch.cat(features, dim=-1)

        x = self.act(self.input_proj(x_in))
        mid = self.n_hidden_layers // 2
        for i, layer in enumerate(self.hidden_layers):
            if self.skip_connection and i == mid:
                proj = self.act(self.skip_proj(x_in))
                x = self.act(layer(torch.cat([x, proj], dim=-1)))
            else:
                x = self.act(layer(x))

        raw_residual = self.out(x).squeeze(-1) * self.residual_scale
        if self.use_residual_gate:
            residual_gate = torch.sigmoid(self.gate_out(x).squeeze(-1) - self.residual_gate_bias)
            residual = residual_gate * raw_residual
        else:
            residual_gate = torch.ones_like(raw_residual)
            residual = raw_residual
        fem_interp = (flat_prior[:, :4] * flat_prior[:, 4:8]).sum(dim=-1)
        d_hat = fem_interp + residual
        if self.support_head:
            support_logit = self.support_out(x).squeeze(-1)
            support_prob = torch.sigmoid(support_logit)
            return {
                "d_hat": d_hat.view(B, N),
                "fem_interp": fem_interp.view(B, N),
                "residual": residual.view(B, N),
                "raw_residual": raw_residual.view(B, N),
                "residual_gate": residual_gate.view(B, N),
                "support_logit": support_logit.view(B, N),
                "support_prob": support_prob.view(B, N),
            }
        if self.use_residual_gate:
            return {
                "d_hat": d_hat.view(B, N),
                "fem_interp": fem_interp.view(B, N),
                "residual": residual.view(B, N),
                "raw_residual": raw_residual.view(B, N),
                "residual_gate": residual_gate.view(B, N),
            }
        return d_hat.view(B, N), fem_interp.view(B, N), residual.view(B, N)
