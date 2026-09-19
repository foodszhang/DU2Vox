#!/usr/bin/env python3
"""Dense common-domain metrics for a frozen cached prediction ensemble."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.evaluation.error_structured import _reconstruction_metrics
from experiments.cross_discretization_decomposition.run_analysis import focus_metadata
from scripts.train_iterative_fem_corrector import build_dataset, load_ids
from du2vox.utils.confirmation import require_confirmation_permission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--v1-predictions-dir", type=Path, required=True)
    parser.add_argument("--v3-predictions-dir", type=Path, required=True)
    parser.add_argument("--v3-weight", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-predictions-dir", type=Path)
    parser.add_argument("--allow-confirmation-eval", action="store_true")
    parser.add_argument(
        "--confirmation-manifest",
        type=Path,
        default=Path("data/confirmation_manifest.json"),
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    data = cfg["data"]
    require_confirmation_permission(
        samples_dir=data["samples_dir"],
        allow_confirmation_eval=args.allow_confirmation_eval,
        manifest_path=args.confirmation_manifest,
    )
    os.environ["DU2VOX_SHARED_DIR"] = str(data["shared_dir"])
    if data.get("allow_stale_frame_manifest", False):
        os.environ["DU2VOX_ALLOW_STALE_FRAME_MANIFEST"] = "1"
    if data.get("frame_manifest_sha256"):
        os.environ["DU2VOX_FRAME_MANIFEST_SHA256"] = str(
            data["frame_manifest_sha256"]
        )
    sample_ids = load_ids(data[f"{args.split}_split"])
    dataset = build_dataset(
        cfg,
        args.split,
        sample_ids,
        int(cfg["training"].get("n_query_points", 8192)),
    )
    if args.save_predictions_dir is not None:
        args.save_predictions_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        v1_archive = np.load(args.v1_predictions_dir / f"{sample_id}.npz")
        v3_archive = np.load(args.v3_predictions_dir / f"{sample_id}.npz")
        valid = np.asarray(v1_archive["valid_flat_indices"], dtype=np.int64)
        if not np.array_equal(valid, dataset.valid_flat_indices):
            raise ValueError(f"Canonical valid-domain mismatch for {sample_id}")
        v1 = np.asarray(v1_archive["final_prediction"], dtype=np.float64)
        v3 = np.asarray(v3_archive["final_prediction"], dtype=np.float64)
        prediction = (1.0 - args.v3_weight) * v1 + args.v3_weight * v3
        gt_volume = np.load(dataset.samples_dir / sample_id / "gt_voxels.npy", mmap_mode="r")
        gt = (np.asarray(gt_volume).ravel()[valid] > 0.05).astype(np.float32)
        tumor = focus_metadata(dataset.samples_dir / sample_id)["tumor_params"]
        metrics = _reconstruction_metrics(prediction, gt, dataset, tumor)
        rows.append({"sample_id": sample_id, **metrics})
        if args.save_predictions_dir is not None:
            np.savez_compressed(
                args.save_predictions_dir / f"{sample_id}.npz",
                final_prediction=prediction.astype(np.float32),
                valid_flat_indices=valid,
                grid_shape=np.asarray(dataset.grid_shape),
            )
    numeric_keys = [key for key in rows[0] if key != "sample_id"]
    summary = {
        key: float(np.nanmean([float(row[key]) for row in rows]))
        for key in numeric_keys
    }
    result = {
        "model": "frozen_v1_v3_convex_ensemble",
        "split": args.split,
        "v1_weight": 1.0 - args.v3_weight,
        "v3_weight": args.v3_weight,
        "threshold": 0.5,
        "n_samples": len(rows),
        "summary": summary,
        "per_sample": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "per_sample"}, indent=2))


if __name__ == "__main__":
    main()
