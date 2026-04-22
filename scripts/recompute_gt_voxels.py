#!/usr/bin/env python3
"""
Recompute gt_values in existing precomputed .npz files using gt_voxels lookup.
grid_coords are already in ATLAS coordinates (from the old bridge output).
Only replaces gt_values — all other fields preserved.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from du2vox.bridge.fem_bridging import FEMBridge

VOZEL_SPACING = 0.2
ROI_SIZE_MM = 30.0


def load_split(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def recompute_gt_values(
    sid: str,
    precomputed_dir: Path,
    samples_dir: Path,
    nodes: np.ndarray,
    grid_spacing: float,
    roi_padding_mm: float,
    n_candidates: int = 16,
) -> dict:
    """Recompute gt_values only. Returns dict with {gt_values, valid_mask}."""
    npz_path = precomputed_dir / f"{sid}.npz"
    if not npz_path.exists():
        return None

    data = dict(np.load(npz_path))
    grid_coords = data["grid_coords"]  # already in ATLAS coordinates
    prior_8d = data["prior_8d"]
    valid = data["valid_mask"].astype(bool)

    # gt_voxels lookup (grid_coords are already ATLAS, no trunk_origin offset needed)
    gt_voxels = np.load(samples_dir / sid / "gt_voxels.npy")
    mesh_center = nodes.mean(axis=0).astype(np.float32)
    voxel_offset = mesh_center - np.float32(ROI_SIZE_MM / 2.0)
    voxel_shape = np.array(gt_voxels.shape, dtype=np.int64)

    # atlas mm → voxel index
    atlas_coords = grid_coords.astype(np.float32)
    idx = np.round(
        (atlas_coords - voxel_offset - np.float32(VOZEL_SPACING / 2.0))
        / np.float32(VOZEL_SPACING)
    ).astype(np.int64)

    # dynamic bounds check
    in_box = ((idx >= 0) & (idx < voxel_shape[None, :])).all(axis=-1)
    idx_clipped = np.clip(idx, np.zeros(3, dtype=np.int64), voxel_shape - 1)

    gt_values = gt_voxels[idx_clipped[:, 0], idx_clipped[:, 1], idx_clipped[:, 2]].astype(np.float32)
    gt_values[~in_box] = 0.0
    gt_values[~valid] = 0.0

    return {
        "gt_values": gt_values,
        "valid_mask": valid,
    }


def main():
    parser = argparse.ArgumentParser(description="Recompute gt_values using gt_voxels")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val"])
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    split_key = f"{args.split}_split"
    sample_ids = load_split(cfg["data"][split_key])
    if args.max_samples:
        sample_ids = sample_ids[: args.max_samples]

    precomputed_dir = Path(cfg["data"][f"precomputed_{args.split}_dir"])
    samples_dir = Path(cfg["data"]["samples_dir"])
    shared_dir = Path(cfg["data"]["shared_dir"])
    grid_spacing = cfg["data"].get("grid_spacing", 0.2)
    roi_padding_mm = cfg["data"].get("roi_padding_mm", 1.0)

    # Load mesh once
    mesh = np.load(shared_dir / "mesh.npz")
    nodes = mesh["nodes"].astype(np.float64)

    print(f"[Recompute] Split={args.split}, n_samples={len(sample_ids)}")
    print(f"[Recompute] Precomputed dir={precomputed_dir}")
    print(f"[Recompute] Samples dir={samples_dir}")

    updated = 0
    errors = 0
    nonzero_counts = []

    for i, sid in enumerate(sample_ids):
        result = recompute_gt_values(
            sid=sid,
            precomputed_dir=precomputed_dir,
            samples_dir=samples_dir,
            nodes=nodes,
            grid_spacing=grid_spacing,
            roi_padding_mm=roi_padding_mm,
        )
        if result is None:
            print(f"  SKIP {sid}: not found")
            errors += 1
            continue

        # Load existing npz, update gt_values, save back
        npz_path = precomputed_dir / f"{sid}.npz"
        data = dict(np.load(npz_path))
        data["gt_values"] = result["gt_values"]
        # valid_mask unchanged

        # Save in-place
        np.savez(npz_path, **data)

        nonzero = (result["gt_values"] != 0).sum()
        nonzero_counts.append(nonzero)
        updated += 1

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(sample_ids)}: nonzero mean={np.mean(nonzero_counts):.0f}")

    print(f"\nDone: {updated} updated, {errors} skipped")
    if nonzero_counts:
        print(f"gt_voxels nonzero per sample: mean={np.mean(nonzero_counts):.0f}, min={np.min(nonzero_counts)}, max={np.max(nonzero_counts)}")


if __name__ == "__main__":
    main()
