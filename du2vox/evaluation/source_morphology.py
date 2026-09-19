"""Source-resolved iso-intensity morphology metrics for continuous FMT fields.

Complements ``du2vox.evaluation.continuous_field``: that module scores raw
amplitudes over the whole canonical domain, while this one scores the *shape* of
each known GT source separately.

Every source defines its own intensity scale from its local peak, so a weak
source cannot be erased by a global threshold. For source ``i`` in its GT-defined
Mahalanobis ROI:

* GT supports are ``gt >= f * max(gt in ROI_i)`` for ``f`` in {0.5, 0.2};
* predicted supports use the *matched* local peak ``max(pred in ROI_i)``, so the
  comparison is peak-relative on both sides.

Nothing is clamped or independently normalized.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

# Reused deliberately: the Mahalanobis convention must match the frozen
# continuous-field metrics and the GT generator's truncation contract.
from du2vox.evaluation.continuous_field import _focus_mahalanobis


@dataclass(frozen=True)
class SourceMorphologyProtocol:
    """Fixed definitions for source-resolved morphology scoring."""

    roi_mahalanobis: float = 3.5
    tight_fraction: float = 0.5
    loose_fraction: float = 0.2
    detection_fraction: float = 0.5
    weak_source_ratio: float = 1.5


def _dice(first: np.ndarray, second: np.ndarray) -> float:
    first_count = int(np.count_nonzero(first))
    second_count = int(np.count_nonzero(second))
    if first_count + second_count == 0:
        return 1.0
    if first_count == 0 or second_count == 0:
        return 0.0
    return 2.0 * int(np.count_nonzero(first & second)) / (first_count + second_count)


def _volume_ratio(predicted: np.ndarray, target: np.ndarray) -> float:
    target_count = int(np.count_nonzero(target))
    if target_count == 0:
        return float("nan")
    return float(np.count_nonzero(predicted)) / target_count


def _surface_coords(
    flat_mask: np.ndarray,
    valid_flat_indices: np.ndarray,
    grid_shape: tuple[int, int, int],
    spacing_mm: float,
    offset_world_mm: np.ndarray,
) -> np.ndarray:
    volume = np.zeros(int(np.prod(grid_shape)), dtype=bool)
    volume[np.asarray(valid_flat_indices, dtype=np.int64)] = flat_mask
    volume = volume.reshape(grid_shape)
    if not volume.any():
        return np.zeros((0, 3), dtype=np.float64)
    surface = volume & ~binary_erosion(volume)
    return np.asarray(offset_world_mm, dtype=np.float64) + (
        np.argwhere(surface).astype(np.float64) + 0.5
    ) * float(spacing_mm)


def _symmetric_surface_distances(
    first: np.ndarray, second: np.ndarray, fallback: float
) -> np.ndarray:
    if len(first) == 0 or len(second) == 0:
        return np.asarray([fallback], dtype=np.float64)
    forward = cKDTree(second).query(first, k=1)[0]
    backward = cKDTree(first).query(second, k=1)[0]
    return np.concatenate([forward, backward])


def _source_intensity(focus: dict) -> float:
    params = focus.get("params", focus)
    value = params.get("intensity", focus.get("intensity"))
    return float(value) if value is not None else 1.0


def source_morphology_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    coords_world: np.ndarray,
    foci: list[dict],
    valid_flat_indices: np.ndarray,
    grid_shape: tuple[int, int, int],
    spacing_mm: float,
    offset_world_mm: np.ndarray,
    *,
    protocol: SourceMorphologyProtocol = SourceMorphologyProtocol(),
    include_masks: bool = False,
) -> dict:
    """Per-source iso-intensity morphology scores with no detection censoring.

    Overlapping GT supports are assigned to the source with the smallest
    normalized Mahalanobis distance, so per-source regions stay disjoint.

    With ``include_masks`` the per-source rows carry ``_``-prefixed boolean
    support masks for rendering; they are never part of the aggregate metrics.
    """

    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    gt = np.asarray(target, dtype=np.float64).reshape(-1)
    if pred.shape != gt.shape:
        raise ValueError(f"prediction shape {pred.shape} != target shape {gt.shape}")
    coords = np.asarray(coords_world, dtype=np.float64)
    if coords.shape != (len(pred), 3):
        raise ValueError("coords_world must have shape [n_valid_voxels, 3]")
    if not foci:
        raise ValueError("At least one GT focus is required")

    distances = np.stack([_focus_mahalanobis(coords, focus) for focus in foci])
    nearest = np.argmin(distances, axis=0)
    diagonal = float(np.linalg.norm(np.asarray(grid_shape, dtype=np.float64) * spacing_mm))
    intensities = [_source_intensity(focus) for focus in foci]
    max_intensity = max(intensities) if intensities else 0.0

    rows: list[dict] = []
    for index, focus in enumerate(foci):
        roi = (distances[index] <= protocol.roi_mahalanobis) & (nearest == index)
        if not np.any(roi):
            raise RuntimeError(f"GT source {index} has no canonical valid voxels")
        gt_peak = float(gt[roi].max())
        pred_peak = float(pred[roi].max())
        center = np.asarray(focus["center"], dtype=np.float64)

        gt_tight = roi & (gt >= protocol.tight_fraction * gt_peak)
        gt_loose = roi & (gt >= protocol.loose_fraction * gt_peak)
        if pred_peak > 0.0:
            pred_tight = roi & (pred >= protocol.tight_fraction * pred_peak)
            pred_loose = roi & (pred >= protocol.loose_fraction * pred_peak)
        else:
            pred_tight = np.zeros_like(roi)
            pred_loose = np.zeros_like(roi)

        if np.any(pred_tight):
            cle = float(np.linalg.norm(coords[pred_tight].mean(axis=0) - center))
        else:
            cle = diagonal

        peak_flat = int(np.flatnonzero(roi)[int(np.argmax(pred[roi]))])
        peak_localization = float(np.linalg.norm(coords[peak_flat] - center))

        tight_distances = _symmetric_surface_distances(
            _surface_coords(gt_tight, valid_flat_indices, grid_shape, spacing_mm, offset_world_mm),
            _surface_coords(
                pred_tight, valid_flat_indices, grid_shape, spacing_mm, offset_world_mm
            ),
            diagonal,
        )
        loose_distances = _symmetric_surface_distances(
            _surface_coords(gt_loose, valid_flat_indices, grid_shape, spacing_mm, offset_world_mm),
            _surface_coords(
                pred_loose, valid_flat_indices, grid_shape, spacing_mm, offset_world_mm
            ),
            diagonal,
        )

        detected = bool(pred_peak >= protocol.detection_fraction * gt_peak)
        rows.append(
            {
                "source_index": index,
                "intensity": intensities[index],
                "is_weak_source": bool(
                    max_intensity > 0.0
                    and intensities[index] * protocol.weak_source_ratio < max_intensity
                ),
                "gt_peak": gt_peak,
                "pred_peak": pred_peak,
                "detected": detected,
                "dice_50": _dice(gt_tight, pred_tight),
                "dice_20": _dice(gt_loose, pred_loose),
                "cle_50_mm": cle,
                "peak_localization_error_mm": peak_localization,
                "assd_50_mm": float(np.mean(tight_distances)),
                "assd_20_mm": float(np.mean(loose_distances)),
                "hd95_50_mm": float(np.quantile(tight_distances, 0.95)),
                "hd95_20_mm": float(np.quantile(loose_distances, 0.95)),
                "volume_ratio_50": _volume_ratio(pred_tight, gt_tight),
                "volume_ratio_20": _volume_ratio(pred_loose, gt_loose),
                **(
                    {
                        "_roi_mask": roi,
                        "_gt_50_mask": gt_tight,
                        "_gt_20_mask": gt_loose,
                        "_pred_50_mask": pred_tight,
                        "_pred_20_mask": pred_loose,
                    }
                    if include_masks
                    else {}
                ),
            }
        )

    return {
        "n_sources": len(rows),
        "per_source": rows,
        "summary": aggregate_source_rows(rows),
        "summary_weak": aggregate_source_rows([row for row in rows if row["is_weak_source"]]),
    }


def _mean(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def aggregate_source_rows(rows: list[dict]) -> dict:
    """Mean of every per-source metric over the supplied rows (never censored)."""

    if not rows:
        return {"n_sources": 0}
    keys = sorted(
        key
        for key in rows[0]
        if key not in {"source_index", "intensity", "is_weak_source", "detected"}
        and not key.startswith("_")
    )
    return {
        "n_sources": len(rows),
        **{key: _mean([float(row[key]) for row in rows]) for key in keys},
        "detection_recall": _mean([float(row["detected"]) for row in rows]),
    }
