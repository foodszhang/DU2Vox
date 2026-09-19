#!/usr/bin/env python3
"""Evaluate a frozen Stage-1 FEM state on the certified canonical P1 domain."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.bridge.canonical_cross_discretization import CanonicalCrossDiscretization
from experiments.cross_discretization_decomposition.decomposition import (
    binary_metrics,
    component_metrics,
    surface_metrics,
)
from experiments.cross_discretization_decomposition.run_analysis import (
    focus_metadata,
    weak_recall,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--bridge-dir", type=Path, required=True)
    parser.add_argument("--operator-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    samples_dir = Path(cfg["data"]["samples_dir"])
    shared_dir = Path(cfg["data"]["shared_dir"])
    canonical = CanonicalCrossDiscretization(
        args.operator_cache, shared_dir=shared_dir, factorize=False
    )
    ids = load_ids(args.split_file)
    missing = [
        sid for sid in ids if not (args.bridge_dir / sid / "coarse_d.npy").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} coarse states; first missing ID: {missing[0]}"
        )

    operator = canonical.operator
    rows: list[dict[str, float | str]] = []
    for index, sid in enumerate(ids, start=1):
        coefficients = np.load(args.bridge_dir / sid / "coarse_d.npy").astype(
            np.float64
        )
        if coefficients.shape != (operator.p.shape[1],):
            raise ValueError(f"{sid}: unexpected FEM shape {coefficients.shape}")
        prediction = canonical.prolong(coefficients)
        gt_volume = np.load(samples_dir / sid / "gt_voxels.npy", mmap_mode="r")
        gt = (
            np.asarray(gt_volume).ravel()[operator.valid_flat_indices] > 0.05
        ).astype(np.float64)
        metrics = binary_metrics(prediction, gt, threshold=0.5)
        metrics.update(
            surface_metrics(
                prediction,
                gt,
                operator.valid_flat_indices,
                operator.grid_shape,
                operator.spacing_mm,
                threshold=0.5,
            )
        )
        metrics.update(
            component_metrics(
                prediction,
                gt,
                operator.valid_flat_indices,
                operator.grid_shape,
                operator.spacing_mm,
            )
        )
        tumor = focus_metadata(samples_dir / sid)["tumor_params"]
        metrics["weak_recall"] = weak_recall(
            prediction, gt, operator.coords_world, tumor
        )
        rows.append({"sample_id": sid, **metrics})
        if index % 25 == 0 or index == len(ids):
            print(f"[canonical {index}/{len(ids)}] {sid} dice={metrics['dice']:.4f}")

    numeric_keys = [key for key in rows[0] if key != "sample_id"]
    summary = {
        key: float(np.nanmean([float(row[key]) for row in rows]))
        for key in numeric_keys
    }
    result = {
        "split": args.split,
        "n_samples": len(rows),
        "selection_domain": "val300" if args.split == "val" else "frozen_on_val300",
        "threshold": 0.5,
        "gt_semantics": "gt_voxels > 0.05",
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "bridge_dir": str(args.bridge_dir.resolve()),
        "operator_cache": str(args.operator_cache.resolve()),
        "operator_hashes": canonical.cache_hashes,
        "summary": summary,
        "per_sample": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    print(f"[canonical] mean Dice={summary['dice']:.6f}")
    print(f"[canonical] wrote {args.output}")


if __name__ == "__main__":
    main()
