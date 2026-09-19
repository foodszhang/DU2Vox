#!/usr/bin/env python3
"""Select a convex two-corrector ensemble using validation predictions only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.utils.confirmation import require_confirmation_permission


def ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--v1-predictions-dir", type=Path, required=True)
    parser.add_argument("--v3-predictions-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weight-steps", type=int, default=21)
    parser.add_argument("--fixed-v3-weight", type=float)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--allow-confirmation-eval", action="store_true")
    parser.add_argument(
        "--confirmation-manifest",
        type=Path,
        default=Path("data/confirmation_manifest.json"),
    )
    args = parser.parse_args()
    sealed = require_confirmation_permission(
        samples_dir=args.samples_dir,
        allow_confirmation_eval=args.allow_confirmation_eval,
        manifest_path=args.confirmation_manifest,
    )
    if sealed is not None:
        if args.fixed_v3_weight is None:
            raise PermissionError("Weight sweeps are forbidden on confirmation data")
        if abs(args.fixed_v3_weight - 0.55) > 1e-12:
            raise PermissionError("Confirmation ensemble weight is frozen at 0.55")
        if abs(args.threshold - 0.5) > 1e-12:
            raise PermissionError("Confirmation threshold is frozen at 0.5")

    weights = (
        np.asarray([args.fixed_v3_weight], dtype=np.float64)
        if args.fixed_v3_weight is not None
        else np.linspace(0.0, 1.0, args.weight_steps)
    )
    metric_sums = np.zeros((len(weights), 3), dtype=np.float64)
    sample_ids = ids(args.split)
    for sample_id in sample_ids:
        v1_archive = np.load(args.v1_predictions_dir / f"{sample_id}.npz")
        v3_archive = np.load(args.v3_predictions_dir / f"{sample_id}.npz")
        valid = np.asarray(v1_archive["valid_flat_indices"], dtype=np.int64)
        if not np.array_equal(valid, v3_archive["valid_flat_indices"]):
            raise ValueError(f"Valid-domain mismatch for {sample_id}")
        v1 = np.asarray(v1_archive["final_prediction"], dtype=np.float32)
        v3 = np.asarray(v3_archive["final_prediction"], dtype=np.float32)
        gt_volume = np.load(args.samples_dir / sample_id / "gt_voxels.npy", mmap_mode="r")
        target = np.asarray(gt_volume).ravel()[valid] > 0.05
        target_count = float(target.sum())
        for index, weight in enumerate(weights):
            prediction = ((1.0 - weight) * v1 + weight * v3) >= args.threshold
            tp = float(np.logical_and(prediction, target).sum())
            predicted_count = float(prediction.sum())
            fp = predicted_count - tp
            fn = target_count - tp
            metric_sums[index] += (
                2.0 * tp / max(2.0 * tp + fp + fn, 1.0),
                tp / max(predicted_count, 1.0),
                tp / max(target_count, 1.0),
            )
    means = metric_sums / max(len(sample_ids), 1)
    curve = [
        {
            "v3_weight": float(weight),
            "v1_weight": float(1.0 - weight),
            "dice": float(means[index, 0]),
            "precision": float(means[index, 1]),
            "recall": float(means[index, 2]),
        }
        for index, weight in enumerate(weights)
    ]
    best = max(curve, key=lambda row: (row["dice"], row["v3_weight"]))
    result = {
        "mode": "fixed_evaluation" if args.fixed_v3_weight is not None else "selection",
        "split": str(args.split),
        "threshold": args.threshold,
        "n_samples": len(sample_ids),
        "best": best,
        "curve": curve,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
