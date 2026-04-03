"""
Multi-Scale Graph Cross-Attention (MSGC).

KNNGraphCrossAttention: kNN-based graph attention per node.
MultiScaleKNNGraphAttention: Scale-adaptive fusion of L0/L1/L2 branches using L3 as query.
"""

import torch
import torch.nn as nn


class ScaleGate(nn.Module):
    """Learn adaptive weights for multi-scale attention outputs.

    alpha_0 + alpha_1 + alpha_2 = 1 (per node, softmax).
    """

    def __init__(self, feat_dim: int, num_scales: int = 3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_dim // 2, num_scales),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        feat: (B, N, C)
        return: (B, N, num_scales)
        """
        alpha = self.gate(feat)
        return torch.softmax(alpha, dim=-1)


class SensitivityWeighting(nn.Module):
    """Physics-aware weighting using system matrix sensitivity.

    w_i = ||A[:, i]||_2  (node sensitivity)
    """

    def __init__(self, sens_w: torch.Tensor):
        super().__init__()
        self.register_buffer("w", sens_w)

    def forward(self, K: torch.Tensor, V: torch.Tensor):
        """K, V: (B, N, C)"""
        return self.w[None, :, None] * K, self.w[None, :, None] * V


class KNNGraphCrossAttention(nn.Module):
    """Memory-efficient kNN-based graph cross attention.

    Complexity: O(N * k) per forward pass.
    Q = from query branch, K,V = from key/value branch.
    Attention is computed only within kNN neighborhood.
    """

    def __init__(
        self,
        knn_idx: torch.Tensor,
        sens_w: torch.Tensor,
        feat_dim: int,
    ):
        super().__init__()
        self.knn_idx = knn_idx  # (N, k)
        self.Wq = nn.Linear(feat_dim, feat_dim)
        self.Wk = nn.Linear(feat_dim, feat_dim)
        self.Wv = nn.Linear(feat_dim, feat_dim)
        self.sens_weight = SensitivityWeighting(sens_w)
        self.norm = nn.LayerNorm(feat_dim)

    def forward(
        self,
        q_feat: torch.Tensor,
        kv_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        q_feat: (B, N, C)  - query (from L3 branch)
        kv_feat: (B, N, C) - key and value (from L0/L1/L2 branch)

        returns: (B, N, C)
        """
        B, N, C = q_feat.shape
        k = self.knn_idx.shape[1]

        Q = self.Wq(q_feat)
        K = self.Wk(kv_feat)
        V = self.Wv(kv_feat)

        # Sensitivity weighting on K, V
        K, V = self.sens_weight(K, V)

        # Gather kNN keys & values: (B, N, k, C)
        knn_idx = self.knn_idx[None, :, :, None].expand(B, -1, -1, C)
        K_knn = torch.gather(
            K.unsqueeze(2).expand(-1, -1, k, -1),
            dim=1,
            index=knn_idx,
        )
        V_knn = torch.gather(
            V.unsqueeze(2).expand(-1, -1, k, -1),
            dim=1,
            index=knn_idx,
        )

        # Attention scores: (B, N, k)
        attn = (Q.unsqueeze(2) * K_knn).sum(dim=-1) / (C ** 0.5)
        attn = torch.softmax(attn, dim=-1)

        # Weighted sum: (B, N, C)
        out = torch.sum(attn.unsqueeze(-1) * V_knn, dim=2)
        return self.norm(out + Q)


class MultiScaleKNNGraphAttention(nn.Module):
    """Multi-scale + scale-adaptive kNN attention.

    Queries: x_l0, x_l1, x_l2
    Keys/Values: x_l3
    Uses scale-adaptive gate to fuse attention outputs.
    """

    def __init__(
        self,
        feat_dim: int,
        knn_idx: torch.Tensor,
        sens_w: torch.Tensor,
    ):
        super().__init__()
        # Three attention branches: (L0,Q), (L1,Q), (L2,Q) all use L3 as K,V
        self.attn_blocks = nn.ModuleList([
            KNNGraphCrossAttention(knn_idx, sens_w, feat_dim)
            for _ in range(3)
        ])
        self.scale_gate = ScaleGate(feat_dim)
        self.norm = nn.LayerNorm(feat_dim)

    def forward(
        self,
        x_l0: torch.Tensor,
        x_l1: torch.Tensor,
        x_l2: torch.Tensor,
        x_l3: torch.Tensor,
    ) -> torch.Tensor:
        """
        x_l0, x_l1, x_l2: (B, N, C) - query branches
        x_l3: (B, N, C) - key/value branch

        returns: (B, N, C)
        """
        O0 = self.attn_blocks[0](x_l3, x_l0)  # Q=x_l3, KV=x_l0
        O1 = self.attn_blocks[1](x_l3, x_l1)  # Q=x_l3, KV=x_l1
        O2 = self.attn_blocks[2](x_l3, x_l2)  # Q=x_l3, KV=x_l2

        Os = [O0, O1, O2]

        # Scale-adaptive fusion
        alpha = self.scale_gate(x_l3)  # (B, N, 3)
        out = sum(alpha[..., i:i + 1] * Os[i] for i in range(3))

        return self.norm(out + x_l3)
