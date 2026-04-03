"""
Core building blocks for MS-GDUN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNBlock(nn.Module):
    """Standard spectral GCN: out = LeakyReLU(L @ X @ W + b)."""

    def __init__(self, L: torch.Tensor, in_dim: int, out_dim: int):
        super().__init__()
        self.L = L
        self.weight = nn.Parameter(
            nn.init.kaiming_normal_(
                torch.empty(in_dim, out_dim, dtype=torch.float32),
                mode="fan_out",
            )
        )
        self.bias = nn.Parameter(torch.zeros(out_dim, dtype=torch.float32))
        self.act = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_w = torch.matmul(x, self.weight)
        if self.L.is_sparse:
            out = torch.sparse.mm(self.L, x_w)
        else:
            out = torch.matmul(self.L, x_w)
        out = out + self.bias
        return self.act(out)


class InputBlock(nn.Module):
    """Input module: concat(x, L^TL x, A^TA x - A^T b).

    x:  [B, N, 1]
    b:  [B, S, 1]
    A:  [S, N]
    L:  [N, N]

    Returns: [B, N, 3]
    """

    def __init__(self, L: torch.Tensor, A: torch.Tensor):
        super().__init__()
        self.register_buffer("L", L)
        self.register_buffer("A", A)

    def forward(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        LTLx = torch.matmul(torch.matmul(self.L.t(), self.L), x)
        ATAx = torch.matmul(torch.matmul(self.A.t(), self.A), x)
        ATb = torch.matmul(self.A.t(), b)
        return torch.cat([x, LTLx, ATAx - ATb], dim=-1)


class AdaptiveThreshold(nn.Module):
    """Node-wise adaptive sparse threshold λ_i."""

    def __init__(self, feat_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_dim // 2, 1),
            nn.Softplus(),
        )

    def forward(self, u: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """u: [B, N, 1], feat: [B, N, C]"""
        lam = self.net(feat)  # [B, N, 1]
        return torch.sign(u) * F.softplus(torch.abs(u) - lam)


class UpdateBlock(nn.Module):
    """Wraps AdaptiveThreshold with u = x - grad computation."""

    def __init__(self, feat_dim: int):
        super().__init__()
        self.adaptive_thresh = AdaptiveThreshold(feat_dim)

    def forward(self, x: torch.Tensor, grad: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        u = x - grad
        return self.adaptive_thresh(u, feat)


class SparseUpdate(nn.Module):
    """Sparse update: sign(u) * softplus(|u| - theta).

    Simple version with fixed theta, or adaptive via AdaptiveThreshold.
    """

    def __init__(self, theta: float = 0.0):
        super().__init__()
        self.theta = theta

    def forward(
        self,
        x: torch.Tensor,
        grad: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        u = x - alpha * grad
        return torch.sign(u) * F.softplus(torch.abs(u) - self.theta)
