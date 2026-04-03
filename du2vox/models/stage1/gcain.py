"""
MS-GDUN (GCAIN) full model implementation.

Port of the reference GCAIN architecture from MS_GDUN_for_MICCAI2026/model/MSGDUN.py.
6-stage unrolled network with multi-scale graph attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from du2vox.models.stage1.blocks import GCNBlock, InputBlock, AdaptiveThreshold, UpdateBlock
from du2vox.models.stage1.msgc import MultiScaleKNNGraphAttention


class GCNMultiScal(nn.Module):
    """Multi-scale GCN: applies separate GCN branches for each Laplacian scale.

    Outputs x_L0, x_L1, x_L2, x_L3 (not concatenated, returned separately).
    Each branch: input_dim -> output_dim -> output_dim (two GCN layers).
    """

    def __init__(
        self,
        L0: torch.Tensor,
        L1: torch.Tensor,
        L2: torch.Tensor,
        L3: torch.Tensor,
        in_dim: int,
        out_dim: int,
    ):
        super().__init__()
        # L0 branch
        self.gc00 = GCNBlock(L0, in_dim, out_dim)
        self.gc01 = GCNBlock(L0, out_dim, out_dim)
        # L1 branch
        self.gc10 = GCNBlock(L1, in_dim, out_dim)
        self.gc11 = GCNBlock(L1, out_dim, out_dim)
        # L2 branch
        self.gc20 = GCNBlock(L2, in_dim, out_dim)
        self.gc21 = GCNBlock(L2, out_dim, out_dim)
        # L3 branch
        self.gc30 = GCNBlock(L3, in_dim, out_dim)
        self.gc31 = GCNBlock(L3, out_dim, out_dim)

    def forward(self, x: torch.Tensor):
        x0 = self.gc01(self.gc00(x))
        x1 = self.gc11(self.gc10(x))
        x2 = self.gc21(self.gc20(x))
        x3 = self.gc31(self.gc30(x))
        return x0, x1, x2, x3


class GradientProjection(nn.Module):
    """Project multi-scale attention features to scalar gradient."""

    def __init__(self, feat_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_dim // 2, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (B, N, C) -> (B, N, 1)"""
        return self.net(feat)


class BasicBlock(nn.Module):
    """Single unrolled stage of GCAIN.

    InputBlock -> GCN sequence -> MultiScale GCN -> CrossAttention -> GradientProjection -> AdaptiveThreshold
    """

    def __init__(
        self,
        L: torch.Tensor,
        A: torch.Tensor,
        L0: torch.Tensor,
        L1: torch.Tensor,
        L2: torch.Tensor,
        L3: torch.Tensor,
        knn_idx: torch.Tensor,
        sens_w: torch.Tensor,
        feat_dim: int = 6,
    ):
        super().__init__()

        self.input_block = InputBlock(L, A)

        # GCN sequence: 3 -> 8 -> 16 -> 8 -> feat_dim
        self.gcn_seq = nn.ModuleList([
            GCNBlock(L, 3, 8),
            GCNBlock(L, 8, 16),
            GCNBlock(L, 16, 8),
            GCNBlock(L, 8, feat_dim),
        ])

        # Multi-scale GCN branches
        self.ms_gcn = GCNMultiScal(
            L0=L0, L1=L1, L2=L2, L3=L3,
            in_dim=feat_dim,
            out_dim=feat_dim,
        )

        # Cross attention
        self.attention = MultiScaleKNNGraphAttention(
            feat_dim=feat_dim,
            knn_idx=knn_idx,
            sens_w=sens_w,
        )

        # Gradient projection
        self.grad_proj = GradientProjection(feat_dim)

        # Adaptive threshold sparse update
        self.update = UpdateBlock(feat_dim)

        self.prev_feat = None

    def forward(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Input projection
        feat = self.input_block(x, b)  # [B, N, 3]

        # GCN sequence
        for gcn in self.gcn_seq:
            feat = gcn(feat)  # [B, N, feat_dim]

        self.prev_feat = feat.detach()

        # Multi-scale GCN
        x0, x1, x2, x3 = self.ms_gcn(feat)

        # Cross attention
        feat_attn = self.attention(x_l0=x0, x_l1=x1, x_l2=x2, x_l3=x3)

        # Gradient
        grad = self.grad_proj(feat_attn)  # [B, N, 1]

        # Sparse update with adaptive threshold
        out = self.update(x, grad, feat)
        return out

    def reset_memory(self):
        self.prev_feat = None


class GCAIN_full(nn.Module):
    """GCAIN: 6-stage unrolled network (cold start from zeros).

    Forward: X0=zeros -> BasicBlock_1 -> ... -> BasicBlock_6 -> X6
    """

    def __init__(
        self,
        L: torch.Tensor,
        A: torch.Tensor,
        L0: torch.Tensor,
        L1: torch.Tensor,
        L2: torch.Tensor,
        L3: torch.Tensor,
        knn_idx: torch.Tensor,
        sens_w: torch.Tensor,
        num_layer: int = 6,
        feat_dim: int = 6,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            BasicBlock(
                L=L, A=A,
                L0=L0, L1=L1, L2=L2, L3=L3,
                knn_idx=knn_idx,
                sens_w=sens_w,
                feat_dim=feat_dim,
            )
            for _ in range(num_layer)
        ])

    def forward(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """x: [B, N, 1] (initial guess, typically zeros).
        b: [B, S, 1] (surface measurements).
        """
        for block in self.blocks:
            x = block(x, b)
        return x
