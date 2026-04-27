#!/usr/bin/env python3
"""
Offline precomputation for Stage 2 training.

Creates precomputed/*.npz files with regular grids in ROI bbox:
  - grid_coords:  [G, 3] float32 — query point coordinates (mm)
  - prior_8d:    [G, 8] float32 — 8D prior features
  - gt_values:   [G]    float32 — trilinear(gt_voxels) GT values
  - valid_mask:  [G]    bool    — True if point inside ROI tet
  - grid_shape:  [3]    int     — (nx, ny, nz) grid dimensions
  - bbox_min:    [3]    float32 — ROI bbox min (padded, world frame)
  - bbox_max:    [3]    float32 — ROI bbox max (padded, world frame)

Per-sample work (cannot share):
  - ROI bbox + grid construction
  - prior_8d via FEMBridge (coarse_d varies per sample)
  - gt_values via map_coordinates (gt_voxels varies per sample)

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
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).parent.parent))

from du2vox.bridge.fem_bridging import FEMBridge
from du2vox.utils.frame import FrameManifest


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
    samples_dir: Path,
    nodes: np.ndarray,
    elements: np.ndarray,
    grid_spacing: float,
    roi_padding_mm: float,
    n_candidates: int,
    frame_manifest: FrameManifest,
) -> dict:
    """Precompute grid data for one sample."""
    t_sample = time.perf_counter()

    # ── Load bridge metadata ────────────────────────────────────────────────
    bd = bridge_dir / sid
    coarse_d = np.load(bd / "coarse_d.npy").astype(np.float32)
    roi_tet_indices = np.load(bd / "roi_tet_indices.npy")
    roi_info = json.loads((bd / "roi_info.json").read_text())

    # ── Build FEMBridge (cached per sample — not shared) ───────────────────
    t_bridge = time.perf_counter()
    bridge = FEMBridge(
        nodes,
        elements,
        roi_tet_indices=roi_tet_indices,
        n_candidates=n_candidates,
    )

    # ── ROI bbox + grid ─────────────────────────────────────────────────────
    bbox = roi_info["roi_bbox_mm"]
    lo = np.array(bbox["min"], dtype=np.float32) - roi_padding_mm
    hi = np.array(bbox["max"], dtype=np.float32) + roi_padding_mm

    # Clamp to MCX volume
    lo = np.maximum(lo, frame_manifest.mcx_bbox_min.astype(np.float32))
    hi = np.minimum(hi, frame_manifest.mcx_bbox_max.astype(np.float32))

    grid_coords, grid_shape = build_grid(lo, hi, grid_spacing)
    n_points = len(grid_coords)

    grid_coords_norm = (
        2.0 * (grid_coords - lo) / (hi - lo + 1e-8) - 1.0
    ).astype(np.float32)

    # ── Prior features ───────────────────────────────────────────────────────
    t_prior = time.perf_counter()
    prior_8d, valid = bridge.get_prior_features(
        grid_coords.astype(np.float64), coarse_d, K=n_candidates
    )
    prior_8d = prior_8d.astype(np.float32)
    valid = valid.astype(bool)

    # ── GT trilinear lookup ────────────────────────────────────────────────
    t_gt = time.perf_counter()
    gt_voxels = np.load(samples_dir / sid / "gt_voxels.npy").astype(np.float32)
    idx_float = frame_manifest.world_to_gt_index(grid_coords)  # [G, 3]

    gt_values = map_coordinates(
        gt_voxels,
        idx_float.T,
        order=1, mode="constant", cval=0.0, prefilter=False,
    ).astype(np.float32)

    shape_arr = np.array(gt_voxels.shape)
    outside = np.any((idx_float < 0) | (idx_float > shape_arr - 1), axis=1)
    valid_gt = ~outside
    valid = valid & valid_gt
    gt_values[~valid_gt] = 0.0

    del bridge

    t_total = time.perf_counter() - t_sample
    print(
        f"  [{sid}] grid={n_points}, "
        f"valid={valid.sum()}/{n_points} ({100*valid.mean():.0f}%), "
        f"t={t_total:.1f}s "
        f"(bridge={(time.perf_counter()-t_bridge)*1000:.0f}ms "
        f"prior={(time.perf_counter()-t_prior)*1000:.0f}ms "
        f"gt={time.perf_counter()-t_gt:.1f}s)"
    )

    return {
        "grid_coords": grid_coords,
        "grid_coords_norm": grid_coords_norm,
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
        print(f"[Precompute] Limited to {args.max_samples} samples (test mode)")

    bridge_dir = Path(cfg["data"].get(f"{args.split}_bridge_dir", cfg["data"].get("bridge_dir", "")))
    shared_dir = Path(cfg["data"]["shared_dir"])
    samples_dir = Path(cfg["data"]["samples_dir"])
    roi_padding = args.roi_padding_mm if args.roi_padding_mm is not None else cfg["data"]["roi_padding_mm"]

    # ── Load shared assets ONCE ─────────────────────────────────────────────
    print(f"[Precompute] Loading mesh from {shared_dir}...")
    nodes, elements = FrameManifest.load_mesh_nodes(shared_dir)
    print(f"  → {len(nodes)} nodes, {len(elements)} tets")

    frame_manifest = FrameManifest.load(shared_dir)
    print(f"[Precompute] Frame: {frame_manifest.world_frame}")
    print(f"  MCX bbox: min={frame_manifest.mcx_bbox_min}, max={frame_manifest.mcx_bbox_max}")
    print(f"  GT offset: {frame_manifest.gt_offset_world_mm}, spacing={frame_manifest.gt_spacing_mm}mm")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(sample_ids)
    print(f"[Precompute] Split={args.split}, {n_total} samples, spacing={args.grid_spacing}mm, padding={roi_padding}mm")
    print(f"[Precompute] Output={output_dir}")
    print(f"[Precompute] Bridge={bridge_dir}, samples_dir={samples_dir}")
    print("-" * 80)

    # ── Precompute per sample ───────────────────────────────────────────────
    total_time = 0.0
    done = 0
    errors = 0

    for i, sid in enumerate(sample_ids):
        out_path = output_dir / f"{sid}.npz"
        if out_path.exists():
            print(f"  [{i+1}/{n_total}] {sid}: already exists, skipping")
            continue

        t0 = time.perf_counter()
        try:
            data = precompute_sample(
                sid=sid,
                bridge_dir=bridge_dir,
                samples_dir=samples_dir,
                nodes=nodes,
                elements=elements,
                grid_spacing=args.grid_spacing,
                roi_padding_mm=roi_padding,
                n_candidates=args.n_candidates,
                frame_manifest=frame_manifest,
            )
            np.savez_compressed(out_path, **data)
            elapsed = time.perf_counter() - t0
            total_time += elapsed
            done += 1

        except Exception as e:
            errors += 1
            print(f"  [ERROR] {sid}: {e}")
            raise

    print("-" * 80)
    print(f"[Precompute] Done. processed={done}, errors={errors}, "
          f"total={total_time:.1f}s, avg={total_time/max(done,1):.2f}s/sample")


if __name__ == "__main__":
    main()