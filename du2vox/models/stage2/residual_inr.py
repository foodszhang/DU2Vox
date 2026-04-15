"""
Stage 2 Residual INR.

Architecture (n_hidden_layers=4):
    Input:   PE(q) [pe_dim] + prior_8d [8] = in_dim
    proj:   Linear(in_dim → hidden)
    h0:     Linear(hidden → hidden) + ReLU
    h1:     Linear(hidden → hidden) + ReLU   ← skip here
    skip:   concat([h1_out, proj(x_in)]) → Linear(hidden+proj→hidden) + ReLU
    h2:     Linear(hidden → hidden) + ReLU
    h3:     Linear(hidden → hidden) + ReLU
    out:    Linear(hidden → 1), zero-init

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
    ):
        super().__init__()
        self.pe = PositionalEncoding(n_freqs=n_freqs, include_input=True)
        in_dim = self.pe.out_dim + prior_dim  # PE(q) + prior_8d

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # Build hidden layers explicitly
        self.h0 = nn.Linear(hidden_dim, hidden_dim)
        self.h1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.ReLU(inplace=True)

        if skip_connection:
            # Middle layer: concat hidden + projected input → hidden
            # skip_proj: x_in (in_dim=59) → in_dim (59), so cat(64,59)=123
            self.h1 = nn.Linear(hidden_dim + in_dim, hidden_dim)
            self.skip_proj = nn.Linear(in_dim, in_dim)

        self.h2 = nn.Linear(hidden_dim, hidden_dim)
        self.h3 = nn.Linear(hidden_dim, hidden_dim)

        # Output: zero-init → residual=0 at init
        self.out = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.skip_connection = skip_connection

    def forward(
        self,
        coords: torch.Tensor,
        prior_8d: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        coords:   [B, N, 3]
        prior_8d: [B, N, 8]
        Returns: (d_hat, fem_interp, residual)  each [B, N]
        """
        B, N = coords.shape[:2]
        flat_coords = coords.reshape(B * N, 3)
        pe_q = self.pe(flat_coords)                    # [B*N, pe_dim]
        flat_prior = prior_8d.reshape(B * N, 8)        # [B*N, 8]
        x_in = torch.cat([pe_q, flat_prior], dim=-1) # [B*N, in_dim]

        # Input projection
        x = torch.relu(self.input_proj(x_in))         # [B*N, hidden]

        # h0
        x = self.act(self.h0(x))                      # [B*N, hidden]

        # h1 (middle layer — skip)
        if self.skip_connection:
            proj_in = torch.relu(self.skip_proj(x_in)) # [B*N, hidden]
            x = self.act(self.h1(torch.cat([x, proj_in], dim=-1)))  # [B*N, hidden]
        else:
            x = self.act(self.h1(x))

        # h2, h3
        x = self.act(self.h2(x))
        x = self.act(self.h3(x))

        # Residual
        residual = self.out(x).squeeze(-1)             # [B*N]

        # FEM interpolation: Σ λi * d_vi
        fem_interp = (flat_prior[:, :4] * flat_prior[:, 4:8]).sum(dim=-1)  # [B*N]
        d_hat = fem_interp + residual

        return d_hat.view(B, N), fem_interp.view(B, N), residual.view(B, N)
