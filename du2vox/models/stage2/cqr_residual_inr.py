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
    ):
        super().__init__()
        if prior_dim < 8:
            raise ValueError("prior_dim must be >= 8")

        self.pe = PositionalEncoding(n_freqs=n_freqs, include_input=True)
        self.prior_dim = prior_dim
        self.view_feat_dim = view_feat_dim
        self.residual_scale = float(residual_scale)
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.skip_connection = skip_connection

        in_dim = self.pe.out_dim + prior_dim + view_feat_dim
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
        self.act = nn.ReLU(inplace=True)

    def forward(
        self,
        coords: torch.Tensor,
        prior: torch.Tensor,
        view_feat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N = coords.shape[:2]
        flat_coords = coords.reshape(B * N, 3)
        flat_prior = prior.reshape(B * N, self.prior_dim)

        pe_q = self.pe(flat_coords)
        if view_feat is not None:
            flat_view = view_feat.reshape(B * N, -1)
            x_in = torch.cat([pe_q, flat_prior, flat_view], dim=-1)
        else:
            x_in = torch.cat([pe_q, flat_prior], dim=-1)

        x = self.act(self.input_proj(x_in))
        mid = self.n_hidden_layers // 2
        for i, layer in enumerate(self.hidden_layers):
            if self.skip_connection and i == mid:
                proj = self.act(self.skip_proj(x_in))
                x = self.act(layer(torch.cat([x, proj], dim=-1)))
            else:
                x = self.act(layer(x))

        residual = self.out(x).squeeze(-1) * self.residual_scale
        fem_interp = (flat_prior[:, :4] * flat_prior[:, 4:8]).sum(dim=-1)
        d_hat = fem_interp + residual
        return d_hat.view(B, N), fem_interp.view(B, N), residual.view(B, N)
