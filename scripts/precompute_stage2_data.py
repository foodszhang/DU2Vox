#!/usr/bin/env python3
"""
Offline precomputation for Stage 2 training.

Creates precomputed/*.npz files with regular grids in ROI bbox:
  - grid_coords:  [G, 3] float32 — query point coordinates (mm)
  - prior_8d:    [G, 8] float32 — 8D prior features
  - gt_values:   [G]    float32 — FEM-interpolated GT values
  - valid_mask:  [G]    bool    — True if point inside ROI tet
  - grid_shape:  [3]    int     — (nx, ny, nz) grid dimensions
  - bbox_min:    [3]    float32 — ROI bbox min (padded)
  - bbox_max:    [3]    float32 — ROI bbox max (padded)

Usage:
  python scripts/precompute_stage2_data.py \
    --config configs/stage2/uniform_1000_v2.yaml \
    --split train --grid_spacing 0.2 \
    --output_dir precomputed/train/

  python scripts/precompute_stage2_data.py \
    --config configs/stage2/uniform_1000_v2.yaml \
    --split val --grid_spacing 0.2 \
    --output_dir precomputed/val/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from du2vox.bridge.fem_bridging import FEMBridge


def load_split(path: str) -> list[str]:
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def build_grid(bbox_min: np.ndarray, bbox_max: np.ndarray, spacing: float) -> tuple[np.ndarray, list[int]]:
    """Build regular grid in bbox. Returns coords [G,3] and shape (nx,ny,nz)."""
    lo = bbox_min.astype(np.float32)
    hi = bbox_max.astype(np.float32)

    nx = max(1, int(np.ceil((hi[0] - lo[0]) / spacing)) + 1)
    ny = max(1, int(np.ceil((hi[1] - lo[1]) / spacing)) + 1)
    nz = max(1, int(np.ceil((hi[2] - lo[2]) / spacing)) + 1)

    xs = np.linspace(lo[0], hi[0], nx, dtype=np.float32)
    ys = np.linspace(lo[1], hi[1], ny, dtype=np.float32)
    zs = np.linspace(lo[2], hi[2], nz, dtype=np.float32)

    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
    coords = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)
    return coords, [nx, ny, nz]


def precompute_sample(
    sid: str,
    bridge_dir: Path,
    shared_dir: Path,
    samples_dir: Path,
    grid_spacing: float,
    roi_padding_mm: float,
    n_candidates: int = 16,
) -> dict:
    """Precompute grid data for one sample."""
    # Load bridge metadata
    bd = bridge_dir / sid
    coarse_d = np.load(bd / "coarse_d.npy").astype(np.float32)
    roi_tet_indices = np.load(bd / "roi_tet_indices.npy")
    roi_info = json.loads((bd / "roi_info.json").read_text())

    # Load mesh
    mesh = np.load(shared_dir / "mesh.npz")
    nodes = mesh["nodes"].astype(np.float64)
    elements = mesh["elements"]

    # Build FEMBridge
    bridge = FEMBridge(
        nodes.astype(np.float64),
        elements,
        roi_tet_indices=roi_tet_indices,
        n_candidates=n_candidates,
    )

    # Padded ROI bbox
    bbox = roi_info["roi_bbox_mm"]
    lo = np.array(bbox["min"], dtype=np.float32) - roi_padding_mm
    hi = np.array(bbox["max"], dtype=np.float32) + roi_padding_mm

    # Build regular grid
    grid_coords, grid_shape = build_grid(lo, hi, grid_spacing)
    n_points = len(grid_coords)

    # Prior features for all grid points
    prior_8d, valid = bridge.get_prior_features(
        grid_coords.astype(np.float64), coarse_d, K=n_candidates
    )
    prior_8d = prior_8d.astype(np.float32)
    valid = valid.astype(bool)

    # GT values: load gt_nodes, interpolate at grid points
    gt_nodes = np.load(samples_dir / sid / "gt_nodes.npy").astype(np.float32)
    gt_prior, gt_valid = bridge.get_prior_features(
        grid_coords.astype(np.float64), gt_nodes, K=n_candidates
    )
    gt_values = (gt_prior[:, :4] * gt_prior[:, 4:8]).sum(axis=-1).astype(np.float32)
    gt_values[~gt_valid] = 0.0

    del bridge

    return {
        "grid_coords": grid_coords,
        "prior_8d": prior_8d,
        "gt_values": gt_values,
        "valid_mask": valid,
        "grid_shape": np.array(grid_shape, dtype=np.int32),
        "bbox_min": lo,
        "bbox_max": hi,
    }


def main():
    parser = argparse.ArgumentParser(description="Precompute Stage 2 grid data")
    parser.add_argument("--config", required=True, help="Stage 2 config YAML")
    parser.add_argument("--split", required=True, choices=["train", "val"], help="Split to precompute")
    parser.add_argument("--grid_spacing", type=float, default=0.2, help="Grid spacing in mm (default: 0.2)")
    parser.add_argument("--output_dir", required=True, help="Output directory for .npz files")
    parser.add_argument("--roi_padding_mm", type=float, default=None, help="Override ROI padding (default: from config)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples (for testing)")
    parser.add_argument("--n_candidates", type=int, default=16, help="KDTree candidates per query (default: 16)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    split_key = f"{args.split}_split"
    split_file = cfg["data"][split_key]
    sample_ids = load_split(split_file)

    if args.max_samples:
        sample_ids = sample_ids[: args.max_samples]

    bridge_dir = Path(cfg["data"].get(f"{args.split}_bridge_dir", cfg["data"].get("bridge_dir", "")))
    shared_dir = Path(cfg["data"]["shared_dir"])
    samples_dir = Path(cfg["data"]["samples_dir"])
    roi_padding = args.roi_padding_mm if args.roi_padding_mm is not None else cfg["data"]["roi_padding_mm"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Precompute] Split={args.split}, n_samples={len(sample_ids)}, "
          f"spacing={args.grid_spacing}mm, padding={roi_padding}mm")
    print(f"[Precompute] Bridge={bridge_dir}, Shared={shared_dir}, Samples={samples_dir}")
    print(f"[Precompute] Output={output_dir}")

    total_time = 0.0
    for i, sid in enumerate(sample_ids):
        t0 = time.perf_counter()

        out_path = output_dir / f"{sid}.npz"
        if out_path.exists():
            print(f"  {sid} already exists, skipping")
            continue

        try:
            data = precompute_sample(
                sid=sid,
                bridge_dir=bridge_dir,
                shared_dir=shared_dir,
                samples_dir=samples_dir,
                grid_spacing=args.grid_spacing,
                roi_padding_mm=roi_padding,
                n_candidates=args.n_candidates,
            )
            np.savez(out_path, **data)
            elapsed = time.perf_counter() - t0
            total_time += elapsed

            # Progress info every 50 samples
            if (i + 1) % 50 == 0:
                avg_time = total_time / (i + 1)
                remaining = len(sample_ids) - (i + 1)
                eta = avg_time * remaining
                print(f"  [{i+1}/{len(sample_ids)}] avg={avg_time:.1f}s/sample, ETA={eta/60:.1f}min")

        except Exception as e:
            print(f"  ERROR {sid}: {e}")
            raise

    print(f"\n[Precompute] Done. Total time: {total_time:.1f}s, "
          f"avg: {total_time/max(len(sample_ids),1):.1f}s/sample")


if __name__ == "__main__":
    main()
