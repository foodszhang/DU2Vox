"""Frozen amplitude-preserving metrics for continuous FMT fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class SSIM3DProtocol:
    """Definition of the masked true-3D SSIM operator."""

    data_range: float = 2.0
    window_size: int = 11
    gaussian_sigma_vox: float = 1.5
    k1: float = 0.01
    k2: float = 0.03

    @property
    def truncate(self) -> float:
        return ((self.window_size - 1) / 2) / self.gaussian_sigma_vox


def _finite_pair(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    gt = np.asarray(target, dtype=np.float64).reshape(-1)
    if pred.shape != gt.shape:
        raise ValueError(f"prediction shape {pred.shape} != target shape {gt.shape}")
    if not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise ValueError("continuous-field metrics require finite prediction and target")
    return pred, gt


def pearson(prediction: np.ndarray, target: np.ndarray) -> float:
    pred, gt = _finite_pair(prediction, target)
    pred = pred - pred.mean()
    gt = gt - gt.mean()
    denominator = float(np.linalg.norm(pred) * np.linalg.norm(gt))
    return float(np.dot(pred, gt) / denominator) if denominator > 0.0 else float("nan")


def concordance_correlation(prediction: np.ndarray, target: np.ndarray) -> float:
    pred, gt = _finite_pair(prediction, target)
    mean_pred, mean_gt = float(pred.mean()), float(gt.mean())
    var_pred, var_gt = float(pred.var()), float(gt.var())
    covariance = float(np.mean((pred - mean_pred) * (gt - mean_gt)))
    denominator = var_pred + var_gt + (mean_pred - mean_gt) ** 2
    return float(2.0 * covariance / denominator) if denominator > 0.0 else float("nan")


def continuous_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    data_range: float = 2.0,
) -> dict[str, float]:
    """Score raw fields on the complete supplied domain without normalization."""

    pred, gt = _finite_pair(prediction, target)
    error = pred - gt
    mse = float(np.mean(error**2))
    gt_norm = float(np.linalg.norm(gt))
    gt_mass = float(gt.sum())
    prediction_mass = float(pred.sum())
    positive_mass = float(np.clip(pred, 0.0, None).sum())
    negative_mass = float(-np.minimum(pred, 0.0).sum())
    return {
        "psnr_db": float("inf")
        if mse == 0.0
        else float(10.0 * np.log10(float(data_range) ** 2 / mse)),
        "pearson": pearson(pred, gt),
        "ccc": concordance_correlation(pred, gt),
        "relative_l2": float(np.linalg.norm(error) / max(gt_norm, 1e-30)),
        "mse": mse,
        "integrated_ratio_signed": prediction_mass / max(gt_mass, 1e-30),
        "integrated_relative_error": abs(prediction_mass - gt_mass)
        / max(abs(gt_mass), 1e-30),
        "positive_mass_ratio": positive_mass / max(gt_mass, 1e-30),
        "negative_mass_ratio": negative_mass / max(gt_mass, 1e-30),
        "peak_relative_error_global": abs(float(pred.max()) - float(gt.max()))
        / max(float(gt.max()), 1e-30),
    }


def masked_ssim3d(
    prediction_valid: np.ndarray,
    target_valid: np.ndarray,
    valid_flat_indices: np.ndarray,
    grid_shape: tuple[int, int, int],
    *,
    protocol: SSIM3DProtocol = SSIM3DProtocol(),
) -> float:
    """True 3-D Gaussian SSIM averaged over the canonical valid domain.

    Local moments are normalized by the Gaussian-weighted valid mask. This avoids
    treating voxels outside the tetrahedral domain as a large shared zero background.
    Neither input is clamped or independently normalized.
    """

    pred, gt = _finite_pair(prediction_valid, target_valid)
    valid = np.asarray(valid_flat_indices, dtype=np.int64)
    if len(valid) != len(pred):
        raise ValueError("valid_flat_indices length does not match the fields")
    mask = np.zeros(int(np.prod(grid_shape)), dtype=np.float64)
    pred_volume = np.zeros_like(mask)
    gt_volume = np.zeros_like(mask)
    mask[valid] = 1.0
    pred_volume[valid] = pred
    gt_volume[valid] = gt
    mask = mask.reshape(grid_shape)
    pred_volume = pred_volume.reshape(grid_shape)
    gt_volume = gt_volume.reshape(grid_shape)
    kwargs = {
        "sigma": protocol.gaussian_sigma_vox,
        "truncate": protocol.truncate,
        "mode": "constant",
        "cval": 0.0,
    }
    weight = gaussian_filter(mask, **kwargs)
    denominator = np.maximum(weight, 1e-12)
    mean_pred = gaussian_filter(pred_volume * mask, **kwargs) / denominator
    mean_gt = gaussian_filter(gt_volume * mask, **kwargs) / denominator
    second_pred = gaussian_filter(pred_volume**2 * mask, **kwargs) / denominator
    second_gt = gaussian_filter(gt_volume**2 * mask, **kwargs) / denominator
    cross = gaussian_filter(pred_volume * gt_volume * mask, **kwargs) / denominator
    var_pred = np.maximum(second_pred - mean_pred**2, 0.0)
    var_gt = np.maximum(second_gt - mean_gt**2, 0.0)
    covariance = cross - mean_pred * mean_gt
    c1 = (protocol.k1 * protocol.data_range) ** 2
    c2 = (protocol.k2 * protocol.data_range) ** 2
    ssim_map = (
        (2.0 * mean_pred * mean_gt + c1) * (2.0 * covariance + c2)
    ) / (
        (mean_pred**2 + mean_gt**2 + c1)
        * (var_pred + var_gt + c2)
    )
    return float(np.mean(ssim_map.ravel()[valid]))


def _focus_mahalanobis(coords_world: np.ndarray, focus: dict) -> np.ndarray:
    center = np.asarray(focus["center"], dtype=np.float64)
    params = focus["params"]
    radius = float(params.get("radius", focus.get("radius", 1.0)))
    axes = []
    for name in ("rx", "ry", "rz"):
        value = params.get(name, focus.get(name))
        axes.append(radius if value is None else float(value))
    sigmas = np.asarray(axes, dtype=np.float64)
    rotation_value = params.get("rotation_matrix", focus.get("rotation_matrix"))
    rotation = (
        np.eye(3, dtype=np.float64)
        if rotation_value is None
        else np.asarray(rotation_value, dtype=np.float64)
    )
    local = (np.asarray(coords_world, dtype=np.float64) - center) @ rotation
    return np.sqrt(np.sum((local / sigmas) ** 2, axis=1))


def all_gt_source_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    coords_world: np.ndarray,
    foci: list[dict],
    *,
    peak_roi_mahalanobis: float = 2.5,
    support_mahalanobis: float = 3.5,
    detection_fraction: float = 0.5,
    log_floor: float = 2e-6,
) -> dict[str, float]:
    """Amplitude/localization scores in fixed GT-defined ROIs for every source.

    No predicted-component matching or detection censoring is used. Integration
    regions are disjoint: overlapping GT supports are assigned to the source with
    the smallest normalized Mahalanobis distance.
    """

    pred, gt = _finite_pair(prediction, target)
    coords = np.asarray(coords_world, dtype=np.float64)
    if coords.shape != (len(pred), 3):
        raise ValueError("coords_world must have shape [n_valid_voxels, 3]")
    if not foci:
        raise ValueError("At least one GT focus is required")
    distances = np.stack([_focus_mahalanobis(coords, focus) for focus in foci])
    nearest = np.argmin(distances, axis=0)
    local_peaks: list[float] = []
    target_peaks: list[float] = []
    localization: list[float] = []
    integrated_errors: list[float] = []
    detected: list[float] = []
    for index, focus in enumerate(foci):
        peak_roi = distances[index] <= peak_roi_mahalanobis
        support = (distances[index] <= support_mahalanobis) & (nearest == index)
        if not np.any(peak_roi) or not np.any(support):
            raise RuntimeError(f"GT source {index} has no canonical valid voxels")
        peak_indices = np.flatnonzero(peak_roi)
        peak_flat = int(peak_indices[np.argmax(pred[peak_roi])])
        local_peak = float(pred[peak_flat])
        local_peaks.append(local_peak)
        target_peak = float(np.max(gt[peak_roi]))
        target_peaks.append(target_peak)
        localization.append(
            float(np.linalg.norm(coords[peak_flat] - np.asarray(focus["center"])))
        )
        predicted_mass = float(pred[support].sum())
        target_mass = float(gt[support].sum())
        integrated_errors.append(
            abs(predicted_mass - target_mass) / max(abs(target_mass), 1e-30)
        )
        detected.append(float(local_peak >= detection_fraction * target_peak))

    peak_array = np.asarray(local_peaks)
    target_peak_array = np.asarray(target_peaks)
    peak_relative_errors = np.abs(peak_array - target_peak_array) / np.maximum(
        target_peak_array, 1e-30
    )
    contrast_errors = []
    for first in range(len(foci)):
        for second in range(first + 1, len(foci)):
            predicted_log_ratio = np.log(
                max(peak_array[first], log_floor) / max(peak_array[second], log_floor)
            )
            target_log_ratio = np.log(
                max(target_peak_array[first], log_floor)
                / max(target_peak_array[second], log_floor)
            )
            contrast_errors.append(abs(float(predicted_log_ratio - target_log_ratio)))
    return {
        "all_source_peak_relative_error": float(np.mean(peak_relative_errors)),
        "all_source_localization_error_mm": float(np.mean(localization)),
        "all_source_integrated_relative_error": float(np.mean(integrated_errors)),
        "all_source_detection_recall": float(np.mean(detected)),
        "all_source_contrast_log_error": (
            float(np.mean(contrast_errors)) if contrast_errors else float("nan")
        ),
    }


def source_resolved_morphology_metrics(
    prediction: np.ndarray,
    coords_world: np.ndarray,
    valid_flat_indices: np.ndarray,
    grid_shape: tuple[int, int, int],
    spacing_mm: float,
    offset_world_mm: np.ndarray,
    foci: list[dict],
    *,
    peak_roi_mahalanobis: float = 2.5,
    cylinder_radius_mm: float = 0.35,
) -> dict[str, float]:
    """Secondary morphology scores with a separate 50% rule for every GT source."""

    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    coords = np.asarray(coords_world, dtype=np.float64)
    distances = np.stack([_focus_mahalanobis(coords, focus) for focus in foci])
    half_radius = np.sqrt(2.0 * np.log(2.0))
    gt_half = np.any(distances <= half_radius, axis=0)
    pred_half = np.zeros(len(pred), dtype=bool)
    fwhm_errors = []
    for index, focus in enumerate(foci):
        roi = distances[index] <= peak_roi_mahalanobis
        local_peak = float(np.max(pred[roi]))
        if local_peak > 0.0:
            pred_half |= roi & (pred >= 0.5 * local_peak)

        params = focus["params"]
        radius = float(params.get("radius", focus.get("radius", 1.0)))
        sigmas = np.asarray(
            [
                radius if params.get(name, focus.get(name)) is None
                else float(params.get(name, focus.get(name)))
                for name in ("rx", "ry", "rz")
            ]
        )
        major = int(np.argmax(sigmas))
        rotation_value = params.get("rotation_matrix", focus.get("rotation_matrix"))
        rotation = (
            np.eye(3, dtype=np.float64)
            if rotation_value is None
            else np.asarray(rotation_value, dtype=np.float64)
        )
        axis = rotation[:, major]
        relative = coords - np.asarray(focus["center"], dtype=np.float64)
        axial = relative @ axis
        perpendicular = np.linalg.norm(relative - axial[:, None] * axis, axis=1)
        cylinder = (
            (perpendicular <= cylinder_radius_mm)
            & (np.abs(axial) <= 3.5 * sigmas[major])
        )
        bin_index = np.rint(axial[cylinder] / spacing_mm).astype(np.int64)
        unique_bins = np.unique(bin_index)
        profile = np.asarray(
            [pred[cylinder][bin_index == value].mean() for value in unique_bins]
        )
        if profile.size == 0 or float(profile.max()) <= 0.0:
            estimated_fwhm = 0.0
        else:
            above = unique_bins[profile >= 0.5 * float(profile.max())]
            estimated_fwhm = float((above.max() - above.min() + 1) * spacing_mm)
        true_fwhm = float(2.354820045 * sigmas[major])
        fwhm_errors.append(abs(estimated_fwhm / true_fwhm - 1.0))

    intersection = int(np.count_nonzero(gt_half & pred_half))
    denominator = int(np.count_nonzero(gt_half) + np.count_nonzero(pred_half))
    dice = 2.0 * intersection / denominator if denominator else 1.0

    gt_volume = np.zeros(int(np.prod(grid_shape)), dtype=bool)
    pred_volume = np.zeros_like(gt_volume)
    valid = np.asarray(valid_flat_indices, dtype=np.int64)
    gt_volume[valid] = gt_half
    pred_volume[valid] = pred_half
    gt_volume = gt_volume.reshape(grid_shape)
    pred_volume = pred_volume.reshape(grid_shape)
    gt_surface = gt_volume & ~binary_erosion(gt_volume)
    pred_surface = pred_volume & ~binary_erosion(pred_volume)
    if not np.any(pred_surface) or not np.any(gt_surface):
        hd95 = float(np.linalg.norm(np.asarray(grid_shape) * spacing_mm))
    else:
        gt_coords = (
            np.asarray(offset_world_mm)
            + (np.argwhere(gt_surface).astype(np.float64) + 0.5) * spacing_mm
        )
        pred_coords = (
            np.asarray(offset_world_mm)
            + (np.argwhere(pred_surface).astype(np.float64) + 0.5) * spacing_mm
        )
        distances_both = np.concatenate(
            [
                cKDTree(gt_coords).query(pred_coords, k=1)[0],
                cKDTree(pred_coords).query(gt_coords, k=1)[0],
            ]
        )
        hd95 = float(np.quantile(distances_both, 0.95))
    return {
        "dice_50pct_per_source_isosurface": float(dice),
        "hd95_50pct_per_source_isosurface_mm": hd95,
        "principal_axis_fwhm_relative_error": float(np.mean(fwhm_errors)),
    }
