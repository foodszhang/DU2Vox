#!/usr/bin/env python3
"""Independent raw-artifact audit of the historical D1-Q experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.utils.gt_io import load_gt_volume


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return {
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "n": int(finite.size),
    }


def load_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "/home/foods/pro/FMT-SimGen/data/"
            "fmt_simgen_v2_quantitative_gaussian_3k_20k"
        ),
    )
    parser.add_argument(
        "--exp-a-predictions", type=Path, default=Path("output/d1q_expA_predictions_contgt")
    )
    parser.add_argument(
        "--exp-b-predictions", type=Path, default=Path("output/d1q_expB_predictions")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("diagnosis/d1q_independent_forensic_audit.json")
    )
    args = parser.parse_args()

    splits = {
        name: load_ids(args.dataset_root / "splits" / f"{name}.txt")
        for name in ("train", "val", "test")
    }
    split_sets = {name: set(ids) for name, ids in splits.items()}
    split_audit = {
        "counts": {name: len(ids) for name, ids in splits.items()},
        "unique_counts": {name: len(set(ids)) for name, ids in splits.items()},
        "overlap": {
            "train_val": sorted(split_sets["train"] & split_sets["val"]),
            "train_test": sorted(split_sets["train"] & split_sets["test"]),
            "val_test": sorted(split_sets["val"] & split_sets["test"]),
        },
        "sha256": {
            name: sha256(args.dataset_root / "splits" / f"{name}.txt")
            for name in splits
        },
    }

    all_ids = splits["train"] + splits["val"] + splits["test"]
    missing: dict[str, list[str]] = {}
    required = ("measurement_b.npy", "gt_nodes.npy", "gt_voxels.npz", "tumor_params.json", "proj.npz")
    for filename in required:
        absent = [sid for sid in all_ids if not (args.dataset_root / "samples" / sid / filename).exists()]
        missing[filename] = absent

    shapes: Counter[str] = Counter()
    k_counts: Counter[int] = Counter()
    radii: list[float] = []
    amplitudes: list[float] = []
    separations: list[float] = []
    orientation_fields = 0
    for sid in all_ids:
        tumor = json.loads((args.dataset_root / "samples" / sid / "tumor_params.json").read_text())
        foci = tumor["foci"]
        k_counts[len(foci)] += 1
        centers = []
        for focus in foci:
            shapes[focus["shape"]] += 1
            params = focus.get("params", {})
            radii.append(float(params.get("radius", focus.get("radius"))))
            amplitudes.append(float(params.get("intensity", 1.0)))
            if any(key in params or key in focus for key in ("rotation", "orientation", "quaternion")):
                orientation_fields += 1
            centers.append(np.asarray(focus["center"], dtype=np.float64))
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                separations.append(float(np.linalg.norm(centers[i] - centers[j])))

    metric_values: dict[str, dict[str, list[float]]] = {
        "exp_a": {},
        "exp_a_normalized": {},
        "exp_b_normalized": {},
    }

    def append(group: str, key: str, value: float) -> None:
        metric_values[group].setdefault(key, []).append(float(value))

    raw_gt_peaks: list[float] = []
    for sid in splits["test"]:
        for group, pred_root in (
            ("exp_a", args.exp_a_predictions),
            ("exp_a_normalized", args.exp_a_predictions),
            ("exp_b_normalized", args.exp_b_predictions),
        ):
            pred_path = pred_root / f"{sid}.npz"
            if not pred_path.exists():
                continue
            with np.load(pred_path) as archive:
                valid = np.asarray(archive["valid_flat_indices"], dtype=np.int64)
                pred = np.asarray(archive["step3_fem"], dtype=np.float64)
            raw_gt = np.asarray(load_gt_volume(args.dataset_root / "samples" / sid)).ravel()[valid]
            raw_gt = raw_gt.astype(np.float64)
            peak = float(raw_gt.max())
            if group == "exp_a":
                gt = raw_gt
                raw_gt_peaks.append(peak)
            else:
                gt = raw_gt / max(peak, 1e-30)
            signal = gt > 0.01 * float(gt.max())
            background = ~signal
            gt_mass = max(float(gt.sum()), 1e-30)
            positive = np.clip(pred, 0.0, None)
            append(group, "signal_mass_ratio_signed", pred[signal].sum() / gt[signal].sum())
            append(group, "full_mass_ratio_signed", pred.sum() / gt_mass)
            append(group, "full_mass_ratio_nonnegative", positive.sum() / gt_mass)
            append(group, "signal_positive_mass_ratio", positive[signal].sum() / gt_mass)
            append(group, "background_positive_mass_ratio", positive[background].sum() / gt_mass)
            append(group, "background_signed_mass_ratio", pred[background].sum() / gt_mass)
            append(group, "negative_mass_magnitude_ratio", -np.minimum(pred, 0.0).sum() / gt_mass)
            append(group, "global_peak_ratio", pred.max() / max(float(gt.max()), 1e-30))
            append(group, "positive_background_fraction", np.mean(positive[background] > 0.0))
            append(group, "signal_voxel_fraction", np.mean(signal))
            append(group, "mse_full_valid", np.mean((pred - gt) ** 2))
            append(group, "relative_l2_full_valid", np.linalg.norm(pred - gt) / np.linalg.norm(gt))

    metrics = {
        group: {key: summary(values) for key, values in sorted(items.items())}
        for group, items in metric_values.items()
    }
    result = {
        "dataset_root": str(args.dataset_root.resolve()),
        "split_audit": split_audit,
        "missing_assets": {key: value for key, value in missing.items() if value},
        "asset_counts": {
            filename: len(all_ids) - len(missing[filename]) for filename in required
        },
        "source_distribution": {
            "num_foci": {str(key): value for key, value in sorted(k_counts.items())},
            "shapes": dict(sorted(shapes.items())),
            "sigma_mm": summary(radii),
            "amplitude": summary(amplitudes),
            "pair_separation_mm": summary(separations),
            "orientation_field_count": orientation_fields,
            "composition_observed_in_code": "pointwise maximum, not additive mixture",
        },
        "raw_gt_peak_test300": summary(raw_gt_peaks),
        "raw_prediction_reanalysis": metrics,
        "notes": {
            "exp_a_scale": "raw continuous GT versus binary-trained prediction",
            "exp_a_normalized_scale": "per-sample-peak normalized GT versus unchanged binary-trained prediction (the reported 5.95 convention)",
            "exp_b_scale": "per-sample-peak normalized GT; physical rescaling requires unavailable GT peak",
            "signal_definition": "GT > 1% of that case's peak",
            "mass_denominator": "sum GT on the complete canonical valid FEM domain",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
