#!/usr/bin/env python3
"""Frozen raw-amplitude evaluation for aligned continuous-field predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.bridge.canonical_cross_discretization import (  # noqa: E402
    CanonicalCrossDiscretization,
)
from du2vox.evaluation.continuous_field import (  # noqa: E402
    SSIM3DProtocol,
    all_gt_source_metrics,
    continuous_metrics,
    masked_ssim3d,
    source_resolved_morphology_metrics,
)
from du2vox.utils.frame import FrameManifest  # noqa: E402
from du2vox.utils.gt_io import (  # noqa: E402
    load_canonical_gt,
    load_normalization_scale,
)
from du2vox.utils.confirmation import (  # noqa: E402
    validate_validation_dataset_receipt,
)


def parse_prediction(value: str) -> tuple[str, Path, str]:
    if "=" not in value or ":" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=DIRECTORY:NPZ_KEY")
    name, location = value.split("=", 1)
    directory, key = location.rsplit(":", 1)
    return name, Path(directory), key


def load_ids(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if line]


def require_freeze(path: Path | None) -> dict:
    if path is None:
        raise RuntimeError("Development-test scoring requires --freeze-receipt")
    receipt = json.loads(path.read_text())
    if receipt.get("status") != "frozen_on_val300":
        raise RuntimeError("Invalid validation-freeze status")
    if receipt.get("sealed_confirmation_accessed") is not False:
        raise RuntimeError("Validation receipt does not certify unopened confirmation data")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--operator-cache", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--prediction", action="append", type=parse_prediction, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path)
    parser.add_argument("--dataset-receipt", type=Path)
    parser.add_argument("--normalize-gt", choices=["none", "per_sample_peak"], default="none")
    parser.add_argument(
        "--normalization-scale-filename",
        help="Sample-local scalar divisor, e.g. gt_scale.npy.",
    )
    parser.add_argument("--data-range", type=float, default=2.0)
    args = parser.parse_args()

    receipt = require_freeze(args.freeze_receipt) if args.split == "test" else None
    if receipt is not None:
        if args.dataset_receipt is None:
            raise RuntimeError("Development-test scoring requires --dataset-receipt")
        validate_validation_dataset_receipt(receipt, args.dataset_receipt)
    ids = load_ids(args.dataset_root / "splits" / f"{args.split}.txt")
    if len(ids) != 300:
        raise RuntimeError(f"Frozen {args.split} split must have 300 cases, got {len(ids)}")
    methods = {name: (directory, key) for name, directory, key in args.prediction}
    if len(methods) != len(args.prediction):
        raise RuntimeError("Prediction method names must be unique")
    canonical = CanonicalCrossDiscretization(
        args.operator_cache,
        shared_dir=args.shared_dir,
        factorize=False,
        allow_stale_frame_manifest=True,
    )
    frame = FrameManifest.load(args.shared_dir)
    protocol = SSIM3DProtocol(
        data_range=args.data_range,
        window_size=11,
        gaussian_sigma_vox=1.5,
        k1=0.01,
        k2=0.03,
    )
    rows = []
    for case_index, sid in enumerate(ids, start=1):
        sample_dir = args.dataset_root / "samples" / sid
        normalization_scale = (
            load_normalization_scale(sample_dir, args.normalization_scale_filename)
            if args.normalization_scale_filename
            else None
        )
        gt, _ = load_canonical_gt(
            sample_dir,
            canonical.operator.valid_flat_indices,
            gt_mode="continuous",
            normalize=args.normalize_gt,
            normalization_scale=normalization_scale,
        )
        tumor = json.loads((args.dataset_root / "samples" / sid / "tumor_params.json").read_text())
        case = {"sample_id": sid, "methods": {}}
        for name, (directory, key) in methods.items():
            prediction_path = directory / f"{sid}.npz"
            if not prediction_path.exists():
                raise FileNotFoundError(prediction_path)
            with np.load(prediction_path) as archive:
                prediction = np.asarray(archive[key], dtype=np.float64).reshape(-1)
            if len(prediction) != len(gt):
                raise RuntimeError(
                    f"{name}/{sid}: {len(prediction)} values != {len(gt)} valid voxels"
                )
            metrics = continuous_metrics(prediction, gt, data_range=protocol.data_range)
            metrics["ssim3d"] = masked_ssim3d(
                prediction,
                gt,
                canonical.operator.valid_flat_indices,
                canonical.operator.grid_shape,
                protocol=protocol,
            )
            metrics.update(
                all_gt_source_metrics(
                    prediction, gt, canonical.operator.coords_world, tumor["foci"]
                )
            )
            metrics.update(
                source_resolved_morphology_metrics(
                    prediction,
                    canonical.operator.coords_world,
                    canonical.operator.valid_flat_indices,
                    canonical.operator.grid_shape,
                    canonical.operator.spacing_mm,
                    frame.gt_offset_world_mm,
                    tumor["foci"],
                )
            )
            case["methods"][name] = metrics
        rows.append(case)
        print(f"[{case_index}/300] {sid}", flush=True)
    metric_names = sorted(next(iter(rows[0]["methods"].values())))
    summary = {
        name: {
            metric: float(np.nanmean([row["methods"][name][metric] for row in rows]))
            for metric in metric_names
        }
        for name in methods
    }
    result = {
        "protocol": {
            "domain": "all 1,677,645 canonical valid voxel centers",
            "amplitude": "raw frozen physical/simulation scale; no independent normalization",
            "ssim3d": {
                "data_range": protocol.data_range,
                "window_size": protocol.window_size,
                "gaussian_sigma_vox": protocol.gaussian_sigma_vox,
                "k1": protocol.k1,
                "k2": protocol.k2,
                "masked_local_moments": True,
                "slice_wise": False,
            },
            "source_metrics": "all GT sources; fixed GT ROIs; no detection censoring",
        },
        "split": args.split,
        "normalize_gt": args.normalize_gt,
        "normalization_scale_filename": args.normalization_scale_filename,
        "data_range": args.data_range,
        "n_samples": len(rows),
        "methods": list(methods),
        "summary": summary,
        "freeze_receipt": str(args.freeze_receipt.resolve()) if receipt else None,
        "dataset_receipt": (str(args.dataset_receipt.resolve()) if args.dataset_receipt else None),
        "per_sample": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")


if __name__ == "__main__":
    main()
