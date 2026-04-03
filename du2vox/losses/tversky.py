"""
Loss functions for FMT reconstruction.

Follows the reference Critic.py from MS_GDUN_for_MICCAI2026.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_coeff(pred: torch.Tensor, label: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Soft Dice coefficient (no binarization).

    pred, label: (B, N, 1) or (B, N)
    Returns: scalar tensor
    """
    if pred.dim() == 3:
        pred = pred.squeeze(-1)
        label = label.squeeze(-1)
    # Clamp to [0, inf) to handle negative predictions (e.g., from AdaptiveThreshold)
    pred = torch.clamp(pred, min=0.0)
    intersection = (pred * label).sum()
    return (2 * intersection + eps) / (pred.sum() + label.sum() + eps)


def binary_dice_coeff(pred: torch.Tensor, label: torch.Tensor, threshold: float = 0.5, eps: float = 1e-8) -> torch.Tensor:
    """Binary Dice coefficient with thresholding.

    pred, label: (B, N, 1)
    threshold: binarization threshold for both pred and label
    Returns: scalar tensor
    """
    if pred.dim() == 3:
        pred = pred.squeeze(-1)
        label = label.squeeze(-1)
    pred_bin = (pred > threshold).float()
    label_bin = (label > threshold).float()
    intersection = (pred_bin * label_bin).sum()
    return (2 * intersection + eps) / (pred_bin.sum() + label_bin.sum() + eps)


def location_error(
    pred: torch.Tensor,
    label: torch.Tensor,
    nodes: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Location error: Euclidean distance between centroids.

    centroid = sum(val * coord) / sum(val)

    pred, label: (B, N, 1)
    nodes: (N, 3)
    Returns: scalar mean over batch
    """
    # Mass
    pred_mass = pred.squeeze(-1).sum(dim=1, keepdim=True) + eps  # [B, 1]
    label_mass = label.squeeze(-1).sum(dim=1, keepdim=True) + eps

    # Weighted centroid — keep (B,1) shape for broadcasting with (B,3) result
    pred_centroid = torch.matmul(pred.squeeze(-1), nodes) / pred_mass  # [B, 3]
    label_centroid = torch.matmul(label.squeeze(-1), nodes) / label_mass  # [B, 3]

    return torch.norm(pred_centroid - label_centroid, dim=-1).mean()


class TverskyLoss(nn.Module):
    """Asymmetric Tversky Loss: α controls FP penalty, β controls FN penalty.

    α=0.3, β=0.7 → FN penalty is 2.3× FP penalty → model predicts larger ROI.
    α=β=0.5 → reduces to standard Dice.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha  # FP weight (low = tolerate false positives)
        self.beta = beta    # FN weight (high = penalize missed tumors)
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)

        # Clamp for numerical stability
        pred = torch.clamp(pred, min=0.0, max=1.0)

        TP = (pred * target).sum(dim=1)
        FP = (pred * (1 - target)).sum(dim=1)
        FN = ((1 - pred) * target).sum(dim=1)

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1 - tversky.mean()


def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor, fg_thresh: float = 0.05) -> torch.Tensor:
    """Foreground-weighted MSE: 70% weight for tumor, 30% for background."""
    weight = torch.ones_like(target)
    fg_mask = (target > fg_thresh)
    n_fg = fg_mask.sum().clamp(min=1)
    n_bg = (~fg_mask).sum().clamp(min=1)
    n_total = target.numel()
    weight[fg_mask] = 0.7 * n_total / n_fg
    weight[~fg_mask] = 0.3 * n_total / n_bg
    return (weight * (pred - target) ** 2).mean()


def criterion(
    pred: torch.Tensor,
    label: torch.Tensor,
    nodes: torch.Tensor,
    weight_tversky: float = 0.7,
    weight_mse: float = 0.3,
) -> torch.Tensor:
    """Combined loss: 0.7 * Tversky(0.1,0.9) + 0.3 * weighted_MSE.

    - Tversky(0.1,0.9): 9× penalty on missed tumors → very aggressive ROI expansion
    - Weighted MSE: pushes tumor center values toward 1.0

    pred, label: (B, N, 1), values in [0, 1] after model output clamp
    nodes: (N, 3) — not used but kept for compatibility
    """
    tversky = TverskyLoss(alpha=0.1, beta=0.9)
    mse = weighted_mse_loss(pred, label)
    return weight_tversky * tversky(pred, label) + weight_mse * mse
