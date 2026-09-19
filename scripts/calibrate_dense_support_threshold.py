#!/usr/bin/env python3
"""Select one global support threshold on validation and apply it to test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_pair(
    sample_id: str, samples_dir: Path, predictions_dir: Path
) -> tuple[np.ndarray, np.ndarray]:
    archive = np.load(predictions_dir / f"{sample_id}.npz")
    prediction = np.asarray(archive["final_prediction"], dtype=np.float64).ravel()
    valid = np.asarray(archive["valid_flat_indices"], dtype=np.int64)
    gt_volume = np.load(samples_dir / sample_id / "gt_voxels.npy", mmap_mode="r")
    target = (np.asarray(gt_volume).ravel()[valid] > 0.05).astype(np.bool_)
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target mismatch for {sample_id}")
    return prediction, target


def threshold_curve(
    sample_ids: list[str],
    samples_dir: Path,
    predictions_dir: Path,
    thresholds: np.ndarray,
) -> list[dict[str, float]]:
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if np.any(np.diff(thresholds) <= 0):
        raise ValueError("Thresholds must be strictly increasing")
    bins = np.concatenate(([-np.inf], thresholds, [np.inf]))
    sums = np.zeros((len(thresholds), 3), dtype=np.float64)
    for sample_id in sample_ids:
        values, target = load_pair(sample_id, samples_dir, predictions_dir)
        all_histogram = np.histogram(values, bins=bins)[0]
        positive_histogram = np.histogram(values[target], bins=bins)[0]
        predicted = np.cumsum(all_histogram[::-1])[::-1][1:].astype(np.float64)
        true_positive = np.cumsum(positive_histogram[::-1])[::-1][1:].astype(
            np.float64
        )
        false_positive = predicted - true_positive
        false_negative = float(target.sum()) - true_positive
        sums[:, 0] += 2.0 * true_positive / np.maximum(
            2.0 * true_positive + false_positive + false_negative, 1.0
        )
        sums[:, 1] += true_positive / np.maximum(predicted, 1.0)
        sums[:, 2] += true_positive / np.maximum(
            true_positive + false_negative, 1.0
        )
    means = sums / max(len(sample_ids), 1)
    return [
        {
            "threshold": float(threshold),
            "dice": float(means[index, 0]),
            "precision": float(means[index, 1]),
            "recall": float(means[index, 2]),
        }
        for index, threshold in enumerate(thresholds)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--val-split", type=Path, required=True)
    parser.add_argument("--test-split", type=Path, required=True)
    parser.add_argument("--val-predictions-dir", type=Path, required=True)
    parser.add_argument("--test-predictions-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold-min", type=float, default=0.25)
    parser.add_argument("--threshold-max", type=float, default=0.75)
    parser.add_argument("--threshold-steps", type=int, default=101)
    args = parser.parse_args()

    val_ids = load_ids(args.val_split)
    thresholds = np.linspace(
        args.threshold_min, args.threshold_max, args.threshold_steps
    )
    curve = threshold_curve(
        val_ids, args.samples_dir, args.val_predictions_dir, thresholds
    )
    # This selection sees validation only. Resolve ties toward 0.5 to avoid an
    # arbitrary operating-point shift when Dice is numerically flat.
    best = max(curve, key=lambda row: (row["dice"], -abs(row["threshold"] - 0.5)))
    fixed = next(row for row in curve if abs(row["threshold"] - 0.5) < 1e-12)
    test_curve = threshold_curve(
        load_ids(args.test_split),
        args.samples_dir,
        args.test_predictions_dir,
        np.asarray(sorted({0.5, best["threshold"]})),
    )
    test_by_threshold = {row["threshold"]: row for row in test_curve}
    result = {
        "selection_split": "val",
        "selected_threshold": best["threshold"],
        "val_selected": {key: best[key] for key in ("dice", "precision", "recall")},
        "val_fixed_0_5": {
            key: fixed[key] for key in ("dice", "precision", "recall")
        },
        "test_selected": {
            key: test_by_threshold[best["threshold"]][key]
            for key in ("dice", "precision", "recall")
        },
        "test_fixed_0_5": {
            key: test_by_threshold[0.5][key]
            for key in ("dice", "precision", "recall")
        },
        "threshold_curve": curve,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "threshold_curve"}, indent=2))


if __name__ == "__main__":
    main()
