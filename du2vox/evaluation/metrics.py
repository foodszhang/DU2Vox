"""
Evaluation metrics for FMT reconstruction.
"""

import torch
from du2vox.losses.tversky import dice_coeff, binary_dice_coeff, location_error


def evaluate_batch(
    pred: torch.Tensor,
    label: torch.Tensor,
    nodes: torch.Tensor,
    pred_thresh: float = 0.3,
    label_thresh: float = 0.05,
) -> dict:
    """Compute all metrics for a batch.

    pred, label: (B, N, 1), values in [0, 1]
    nodes: (N, 3)
    pred_thresh: threshold for binarizing predictions
    label_thresh: threshold for binarizing ground truth
    """
    pred_squeeze = pred.squeeze(-1) if pred.dim() == 3 else pred
    label_squeeze = label.squeeze(-1) if label.dim() == 3 else label

    # Soft Dice
    dice = dice_coeff(pred, label).item()

    # Binary Dice @ 0.5 (original)
    dice_bin_05 = binary_dice_coeff(pred, label, threshold=0.5).item()

    # Binary Dice @ 0.3
    dice_bin_03 = binary_dice_coeff(pred, label, threshold=0.3).item()

    # Binary Dice @ 0.1
    dice_bin_01 = binary_dice_coeff(pred, label, threshold=0.1).item()

    # Binary Dice @ 0.6
    dice_bin_06 = binary_dice_coeff(pred, label, threshold=0.6).item()

    # Recall/Precision @ 0.3
    pred_bin_03 = (pred_squeeze > 0.3).float()
    label_bin_03 = (label_squeeze > 0.05).float()
    TP_03 = (pred_bin_03 * label_bin_03).sum(dim=1)
    FN_03 = ((1 - pred_bin_03) * label_bin_03).sum(dim=1)
    FP_03 = (pred_bin_03 * (1 - label_bin_03)).sum(dim=1)
    recall_03 = (TP_03 / (TP_03 + FN_03 + 1e-8)).mean().item()
    precision_03 = (TP_03 / (TP_03 + FP_03 + 1e-8)).mean().item()

    # Recall/Precision @ 0.1
    pred_bin_01 = (pred_squeeze > 0.1).float()
    label_bin_01 = (label_squeeze > 0.05).float()
    TP_01 = (pred_bin_01 * label_bin_01).sum(dim=1)
    FN_01 = ((1 - pred_bin_01) * label_bin_01).sum(dim=1)
    FP_01 = (pred_bin_01 * (1 - label_bin_01)).sum(dim=1)
    recall_01 = (TP_01 / (TP_01 + FN_01 + 1e-8)).mean().item()
    precision_01 = (TP_01 / (TP_01 + FP_01 + 1e-8)).mean().item()

    # Recall/Precision @ 0.6
    pred_bin_06 = (pred_squeeze > 0.6).float()
    label_bin_06 = (label_squeeze > 0.05).float()
    TP_06 = (pred_bin_06 * label_bin_06).sum(dim=1)
    FN_06 = ((1 - pred_bin_06) * label_bin_06).sum(dim=1)
    FP_06 = (pred_bin_06 * (1 - label_bin_06)).sum(dim=1)
    recall_06 = (TP_06 / (TP_06 + FN_06 + 1e-8)).mean().item()
    precision_06 = (TP_06 / (TP_06 + FP_06 + 1e-8)).mean().item()

    # Location error
    le = location_error(pred, label, nodes).item()

    # MSE
    mse = torch.nn.functional.mse_loss(pred, label).item()

    # Pred stats
    pred_clamped = torch.clamp(pred_squeeze, min=0.0, max=1.0)
    pred_max = pred_clamped.max().item()
    pred_mean = pred_clamped.mean().item()
    pred_std = pred_clamped.std().item()
    pred_frac_03 = (pred_clamped > 0.3).float().mean().item()
    pred_frac_01 = (pred_clamped > 0.1).float().mean().item()

    return {
        "dice": dice,
        "dice_bin_0.5": dice_bin_05,
        "dice_bin_0.3": dice_bin_03,
        "dice_bin_0.6": dice_bin_06,
        "dice_bin_0.1": dice_bin_01,
        "recall_0.3": recall_03,
        "precision_0.3": precision_03,
        "recall_0.1": recall_01,
        "precision_0.1": precision_01,
        "recall_0.6": recall_06,
        "precision_0.6": precision_06,
        "location_error": le,
        "mse": mse,
        "pred_max": pred_max,
        "pred_mean": pred_mean,
        "pred_std": pred_std,
        "pred_frac_0.3": pred_frac_03,
        "pred_frac_0.1": pred_frac_01,
    }


def summarize_metrics(metric_dicts: list[dict]) -> dict:
    """Average metrics over a list of batch results."""
    keys = metric_dicts[0].keys()
    return {k: sum(d[k] for d in metric_dicts) / len(metric_dicts) for k in keys}
