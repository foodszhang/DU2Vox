from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_measurement_proposal(
    sample_dir: str | Path,
    proposal_subdir: str = "proposal",
    proposal_filename: str = "meas_backproj_heatmap.npy",
    proposal_meta_filename: str = "meas_backproj_meta.json",
) -> tuple[np.ndarray, dict[str, Any], tuple[int, int, int]]:
    proposal_dir = Path(sample_dir) / proposal_subdir
    heatmap_path = proposal_dir / proposal_filename
    meta_path = proposal_dir / proposal_meta_filename

    if not heatmap_path.exists():
        raise FileNotFoundError(f"measurement proposal heatmap not found: {heatmap_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"measurement proposal metadata not found: {meta_path}")

    heatmap = np.load(heatmap_path).astype(np.float32)
    with open(meta_path) as f:
        meta = json.load(f)

    grid_size = tuple(int(v) for v in meta.get("grid_size", heatmap.shape))
    if len(grid_size) != 3:
        raise ValueError(f"measurement proposal grid_size must be length 3, got {grid_size}")
    if tuple(heatmap.shape) != grid_size:
        heatmap = heatmap.reshape(grid_size)

    return heatmap, meta, grid_size


def sample_measurement_proposal_points(
    heatmap: np.ndarray,
    meta: dict[str, Any],
    n_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_points <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    scores = np.nan_to_num(heatmap.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    scores = np.clip(scores.reshape(-1), 0.0, None)
    total = float(scores.sum())
    if total <= 0.0:
        raise ValueError("measurement proposal heatmap has zero positive mass")
    prob = scores / total

    cells = rng.choice(prob.size, size=n_points, replace=True, p=prob)
    grid_size = tuple(int(v) for v in meta.get("grid_size", heatmap.shape))
    ijk = np.stack(np.unravel_index(cells, grid_size), axis=1).astype(np.float32)
    jitter = rng.random((n_points, 3), dtype=np.float32)

    if "bbox_min_mm" in meta and "bbox_max_mm" in meta:
        lo = np.asarray(meta["bbox_min_mm"], dtype=np.float32)
        hi = np.asarray(meta["bbox_max_mm"], dtype=np.float32)
        cell = (hi - lo) / np.asarray(grid_size, dtype=np.float32)
        return (lo + (ijk + jitter) * cell).astype(np.float32)

    if "trunk_size_mm" in meta:
        lo = np.asarray(meta.get("origin_mm", [0.0, 0.0, 0.0]), dtype=np.float32)
        hi = lo + np.asarray(meta["trunk_size_mm"], dtype=np.float32)
        cell = (hi - lo) / np.asarray(grid_size, dtype=np.float32)
        return (lo + (ijk + jitter) * cell).astype(np.float32)

    cell_size = np.asarray(meta.get("cell_size_mm", 1.0), dtype=np.float32)
    if cell_size.ndim == 0:
        cell_size = np.full(3, float(cell_size), dtype=np.float32)
    origin = np.asarray(
        meta.get("origin_mm", meta.get("bbox_min_mm", [0.0, 0.0, 0.0])),
        dtype=np.float32,
    )
    return (origin + (ijk + jitter) * cell_size).astype(np.float32)
