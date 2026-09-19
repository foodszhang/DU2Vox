#!/usr/bin/env python3
"""Audit Pi_h targets against an explicit continuous-GT scale contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.bridge.canonical_cross_discretization import (
    CanonicalCrossDiscretization,
)
from du2vox.utils.gt_io import load_canonical_gt, load_normalization_scale


def load_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--operator-cache", type=Path, required=True)
    parser.add_argument("--projection-targets-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--normalization-scale-filename", default="gt_scale.npy")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    metadata_path = args.projection_targets_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "gt_mode": "continuous",
        "normalize_gt": "per_sample_peak",
        "normalization_scale_filename": args.normalization_scale_filename,
    }
    metadata_errors = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    canonical = CanonicalCrossDiscretization(
        args.operator_cache,
        shared_dir=args.shared_dir,
        factorize=True,
        allow_stale_frame_manifest=True,
    )
    ids = [
        sid
        for split in args.splits
        for sid in load_ids(args.dataset_root / "splits" / f"{split}.txt")
    ]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Requested projection-target splits overlap")
    if args.max_samples is not None:
        ids = ids[: args.max_samples]

    rows = []
    sample_scales = metadata.get("sample_scale", {})
    for sid in ids:
        sample_dir = args.dataset_root / "samples" / sid
        scale = load_normalization_scale(sample_dir, args.normalization_scale_filename)
        gt, used_scale = load_canonical_gt(
            sample_dir,
            canonical.operator.valid_flat_indices,
            gt_mode="continuous",
            normalize="per_sample_peak",
            normalization_scale=scale,
        )
        expected_coefficients = canonical.project_coefficients(gt.astype(np.float64))
        target_path = args.projection_targets_dir / f"{sid}.npy"
        saved = np.load(target_path).astype(np.float64)
        shape_exact = saved.shape == expected_coefficients.shape
        if shape_exact:
            difference = saved - expected_coefficients
            relative_l2 = float(
                np.linalg.norm(difference) / max(np.linalg.norm(expected_coefficients), 1e-30)
            )
            max_abs_error = float(np.max(np.abs(difference)))
        else:
            relative_l2 = float("inf")
            max_abs_error = float("inf")
        recorded_scale = sample_scales.get(sid)
        rows.append(
            {
                "sample_id": sid,
                "scale": scale,
                "used_scale_exact": used_scale == scale,
                "metadata_scale_exact": (
                    recorded_scale is not None and float(recorded_scale) == scale
                ),
                "max_abs_error": max_abs_error,
                "relative_l2": relative_l2,
                "finite": bool(np.isfinite(saved).all()),
                "shape_exact": shape_exact,
            }
        )
        if len(rows) % 25 == 0:
            print(f"[{len(rows)}/{len(ids)}]", flush=True)

    tolerance = 5e-6
    passed = (
        not metadata_errors
        and len(rows) == len(ids)
        and all(
            row["used_scale_exact"]
            and row["metadata_scale_exact"]
            and row["finite"]
            and row["shape_exact"]
            and row["relative_l2"] <= tolerance
            for row in rows
        )
    )
    result = {
        "status": "passed" if passed else "failed",
        "contract": expected,
        "metadata_errors": metadata_errors,
        "splits": args.splits,
        "n_samples": len(rows),
        "relative_l2_tolerance": tolerance,
        "max_relative_l2": max((row["relative_l2"] for row in rows), default=None),
        "max_abs_error": max((row["max_abs_error"] for row in rows), default=None),
        "per_sample": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not passed:
        raise RuntimeError("Projection-target contract audit failed")


if __name__ == "__main__":
    main()
