"""
Stage 2 Residual INR.

Architecture (n_hidden_layers=4):
    Input:   PE(q_norm) [pe_dim] + prior_8d [8] = in_dim
             q_norm = 2 * (q - bbox_min) / (bbox_max - bbox_min) - 1  → [-1, 1]
    proj:    Linear(in_dim → hidden)
    h[0..n-1]:  n_hidden_layers × Linear(hidden → hidden) + ReLU
       middle layer (i = n_hidden_layers // 2): concat([h_out, proj(x_in)]) → hidden
    out:     Linear(hidden → 1), zero-init

Zero-init output → step-0 residual = 0 (identity).
"""

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


class ResidualINR(nn.Module):
    def __init__(
        self,
        n_freqs: int = 10,
        hidden_dim: int = 256,
        n_hidden_layers: int = 4,
        prior_dim: int = 8,
        skip_connection: bool = True,
        view_feat_dim: int = 0,  # 0 = DE-only mode (no view features)
    ):
        super().__init__()
        self.pe = PositionalEncoding(n_freqs=n_freqs, include_input=True)
        in_dim = self.pe.out_dim + prior_dim + view_feat_dim  # PE(q_norm) + prior_8d + view_feat

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # Build hidden layers: n_hidden_layers of Linear(hidden, hidden)
        # The middle layer (mid) uses a wider Linear if skip_connection is enabled
        mid = n_hidden_layers // 2
        self.hidden_layers = nn.ModuleList()
        for i in range(n_hidden_layers):
            if skip_connection and i == mid:
                # Skip layer: concat([h, proj_in]) → hidden_dim + hidden_dim
                self.hidden_layers.append(nn.Linear(hidden_dim * 2, hidden_dim))
            else:
                self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Skip projection: x_in → hidden_dim (then ReLU)
        self.skip_proj = nn.Linear(in_dim, hidden_dim)

        # Output: zero-init → residual=0 at init
        self.out = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.skip_connection = skip_connection
        self.view_feat_dim = view_feat_dim
        self.act = nn.ReLU(inplace=True)

    def forward(
        self,
        coords: torch.Tensor,
        prior_8d: torch.Tensor,
        view_feat: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        coords:     [B, N, 3] — coordinates, pre-normalized to [-1, 1]
                    (Stage2DatasetPrecomputed stores grid_coords_norm in .npz)
        prior_8d:   [B, N, 8]
        view_feat:  [B, N, view_feat_dim] or None (DE-only mode)
        Returns: (d_hat, fem_interp, residual)  each [B, N]
        """
        B, N = coords.shape[:2]
        flat_coords = coords.reshape(B * N, 3)

        pe_q = self.pe(flat_coords)                         # [B*N, pe_dim]
        flat_prior = prior_8d.reshape(B * N, 8)             # [B*N, 8]

        if view_feat is not None:
            flat_view = view_feat.reshape(B * N, -1)        # [B*N, view_feat_dim]
            x_in = torch.cat([pe_q, flat_prior, flat_view], dim=-1)
        else:
            x_in = torch.cat([pe_q, flat_prior], dim=-1)    # [B*N, pe_dim + 8]

        # Input projection
        x = self.act(self.input_proj(x_in))                 # [B*N, hidden]

        # Hidden layers with skip connection at middle layer
        mid = self.n_hidden_layers // 2
        for i, layer in enumerate(self.hidden_layers):
            if self.skip_connection and i == mid:
                proj_in = self.act(self.skip_proj(x_in))   # [B*N, hidden]
                x = self.act(layer(torch.cat([x, proj_in], dim=-1)))  # [B*N, hidden]
            else:
                x = self.act(layer(x))

        # Residual
        residual = self.out(x).squeeze(-1)                  # [B*N]

        # FEM interpolation: Σ λi * d_vi
        fem_interp = (flat_prior[:, :4] * flat_prior[:, 4:8]).sum(dim=-1)  # [B*N]
        d_hat = fem_interp + residual

        return d_hat.view(B, N), fem_interp.view(B, N), residual.view(B, N)
