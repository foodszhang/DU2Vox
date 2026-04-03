"""
Graph Convolutional Block (GCNReluBlock).

Standard spectral GCN: out = LeakyReLU(L @ X @ W)
"""

import torch
import torch.nn as nn


class GCNReluBlock(nn.Module):
    """Standard spectral GCN block.

    L @ X @ W + bias, then LeakyReLU activation.
    L can be a dense tensor or sparse tensor.
    """

    def __init__(self, L: torch.Tensor, in_dim: int, out_dim: int):
        super().__init__()
        self.L = L  # [N, N] - can be dense or sparse
        self.weight = nn.Parameter(
            nn.init.kaiming_normal_(
                torch.empty(in_dim, out_dim, dtype=torch.float32),
                mode="fan_out",
            )
        )
        self.bias = nn.Parameter(torch.zeros(out_dim, dtype=torch.float32))
        self.leak_relu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, in_dim]
        returns: [B, N, out_dim]
        """
        # L @ x @ W
        x_w = torch.matmul(x, self.weight)  # [B, N, out_dim]
        if self.L.is_sparse:
            out = torch.sparse.mm(self.L, x_w)  # [B, N, out_dim]
        else:
            out = torch.matmul(self.L, x_w)  # [B, N, out_dim]
        out = out + self.bias
        return self.leak_relu(out)
