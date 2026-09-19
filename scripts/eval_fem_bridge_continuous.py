#!/usr/bin/env python3
"""Evaluate saved FEM bridge states on the continuous canonical voxel domain."""

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
from du2vox.evaluation.continuous_field import (
    SSIM3DProtocol,
    all_gt_source_metrics,
    continuous_metrics,
    masked_ssim3d,
)
from du2vox.utils.gt_io import load_canonical_gt, load_normalization_scale


def load_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--operator-cache", type=Path, required=True)
    parser.add_argument("--bridge-dir", type=Path)
    parser.add_argument(
        "--coefficient-source",
        choices=["bridge", "gt_nodes"],
        default="bridge",
        help="Evaluate saved coarse_d or the normalized nodal-GT approximation oracle.",
    )
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--normalization-scale-filename", default="gt_scale.npy")
    parser.add_argument("--data-range", type=float, default=1.0)
    parser.add_argument("--skip-ssim", action="store_true")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    if args.coefficient_source == "bridge" and args.bridge_dir is None:
        parser.error("--bridge-dir is required when --coefficient-source=bridge")

    ids = load_ids(args.dataset_root / "splits" / f"{args.split}.txt")
    if args.max_samples is not None:
        ids = ids[: args.max_samples]
    canonical = CanonicalCrossDiscretization(
        args.operator_cache,
        shared_dir=args.shared_dir,
        factorize=False,
        allow_stale_frame_manifest=True,
    )
    protocol = SSIM3DProtocol(data_range=args.data_range)
    rows = []
    for index, sid in enumerate(ids, start=1):
        sample_dir = args.dataset_root / "samples" / sid
        scale = load_normalization_scale(sample_dir, args.normalization_scale_filename)
        gt, _ = load_canonical_gt(
            sample_dir,
            canonical.operator.valid_flat_indices,
            gt_mode="continuous",
            normalize="per_sample_peak",
            normalization_scale=scale,
        )
        if args.coefficient_source == "bridge":
            coefficients = np.load(args.bridge_dir / sid / "coarse_d.npy").astype(np.float64)
        else:
            coefficients = np.load(sample_dir / "gt_nodes.npy").astype(np.float64) / scale
        prediction = np.asarray(canonical.p @ coefficients).reshape(-1)
        metrics = continuous_metrics(prediction, gt, data_range=args.data_range)
        tumor = json.loads((sample_dir / "tumor_params.json").read_text())
        foci = tumor["foci"]
        metrics.update(
            all_gt_source_metrics(
                prediction,
                gt,
                canonical.operator.coords_world,
                foci,
            )
        )
        if not args.skip_ssim:
            metrics["ssim3d"] = masked_ssim3d(
                prediction,
                gt,
                canonical.operator.valid_flat_indices,
                canonical.operator.grid_shape,
                protocol=protocol,
            )
        intensities = [float(focus["params"]["intensity"]) for focus in foci]
        intensity_ratio = max(intensities) / min(intensities)
        rows.append(
            {
                "sample_id": sid,
                "num_foci": len(foci),
                "source_intensity_ratio": intensity_ratio,
                **metrics,
            }
        )
        if index % 25 == 0 or index == len(ids):
            print(f"[{index}/{len(ids)}]", flush=True)

    keys = [
        key for key in rows[0] if key not in {"sample_id", "num_foci", "source_intensity_ratio"}
    ]

    def summarize(group: list[dict]) -> dict:
        result = {}
        for key in keys:
            values = np.asarray([row[key] for row in group], dtype=np.float64)
            finite = values[np.isfinite(values)]
            result[key] = {
                "mean": float(np.mean(finite)) if len(finite) else float("nan"),
                "median": float(np.median(finite)) if len(finite) else float("nan"),
            }
        return result

    intensity_groups = {
        "single_source": [row for row in rows if row["num_foci"] == 1],
        "multi_ratio_le_1p5": [
            row for row in rows if row["num_foci"] > 1 and row["source_intensity_ratio"] <= 1.5
        ],
        "multi_ratio_gt_1p5": [
            row for row in rows if row["num_foci"] > 1 and row["source_intensity_ratio"] > 1.5
        ],
    }
    result = {
        "split": args.split,
        "n_samples": len(rows),
        "domain": "full canonical valid FEM domain",
        "coefficient_source": args.coefficient_source,
        "normalization_scale_filename": args.normalization_scale_filename,
        "prediction_independently_normalized": False,
        "ssim3d_computed": not args.skip_ssim,
        "source_metrics": "all GT sources in fixed GT-defined ROIs; no detection censoring",
        "summary": summarize(rows),
        "by_num_foci": {
            str(num_foci): {
                "n_samples": len(group),
                "summary": summarize(group),
            }
            for num_foci in (1, 2, 3)
            if (group := [row for row in rows if row["num_foci"] == num_foci])
        },
        "by_source_intensity_ratio": {
            name: {
                "n_samples": len(group),
                "summary": summarize(group),
            }
            for name, group in intensity_groups.items()
            if group
        },
        "per_sample": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")


if __name__ == "__main__":
    main()
