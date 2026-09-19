#!/usr/bin/env python3
"""Execute the complete frozen confirmation comparison as one audited transaction."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.bridge.canonical_cross_discretization import CanonicalCrossDiscretization
from du2vox.evaluation.error_structured import _reconstruction_metrics
from du2vox.evaluation.iterative_fem import (
    evaluate_iterative_fem_dense,
    save_iterative_fem_result,
)
from du2vox.utils.confirmation import (
    require_confirmation_permission,
    sha256_file,
)
from experiments.cross_discretization_decomposition.run_analysis import focus_metadata
from scripts.train_iterative_fem_corrector import build_dataset, build_model


def _model_type(variant: str) -> str:
    return {
        "residual_consistent_v2": "residual_consistent_iterative_fem_v2",
        "scale_calibrated_proximal_v3": "scale_calibrated_proximal_fem_v3",
        "unified_dual_evidence_v4": "unified_dual_evidence_fem_v4",
    }.get(variant, "iterative_residual_aware_fem")


def _confirmation_config(
    config_path: Path,
    *,
    samples_dir: Path,
    split: Path,
    bridge_dir: Path,
    projection_targets_dir: Path,
) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text())
    data = cfg["data"]
    data["samples_dir"] = str(samples_dir)
    data["test_split"] = str(split)
    data["test_bridge_dir"] = str(bridge_dir)
    data["projection_targets_dir"] = str(projection_targets_dir)
    return cfg


def _evaluate_model(
    *,
    config_path: Path,
    checkpoint_path: Path,
    samples_dir: Path,
    split: Path,
    bridge_dir: Path,
    projection_targets_dir: Path,
    output_path: Path,
    predictions_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    cfg = _confirmation_config(
        config_path,
        samples_dir=samples_dir,
        split=split,
        bridge_dir=bridge_dir,
        projection_targets_dir=projection_targets_dir,
    )
    data = cfg["data"]
    os.environ["DU2VOX_SHARED_DIR"] = str(data["shared_dir"])
    if data.get("allow_stale_frame_manifest", False):
        os.environ["DU2VOX_ALLOW_STALE_FRAME_MANIFEST"] = "1"
    if data.get("frame_manifest_sha256"):
        os.environ["DU2VOX_FRAME_MANIFEST_SHA256"] = str(
            data["frame_manifest_sha256"]
        )
    ids = [line.strip() for line in split.read_text().splitlines() if line.strip()]
    dataset = build_dataset(
        cfg, "test", ids, int(cfg["training"].get("n_query_points", 8192))
    )
    canonical = CanonicalCrossDiscretization(
        data["operator_cache"], shared_dir=data["shared_dir"], factorize=False
    )
    model = build_model(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    expected = _model_type(cfg["model"].get("variant", "v1"))
    if checkpoint.get("model_type") != expected:
        raise RuntimeError(f"Wrong checkpoint type for {config_path}: expected {expected}")
    model.load_state_dict(checkpoint["model"])
    result = evaluate_iterative_fem_dense(
        model,
        dataset,
        canonical,
        device,
        max_samples=len(ids),
        save_predictions_dir=predictions_dir,
    )
    result.update(
        {
            "model": expected,
            "split": "sealed_confirmation",
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_dense_val_dice": checkpoint.get("dense_val_dice"),
            "threshold": 0.5,
        }
    )
    save_iterative_fem_result(result, output_path)
    return result


def _evaluate_ensemble(
    *,
    config_path: Path,
    samples_dir: Path,
    split: Path,
    bridge_dir: Path,
    projection_targets_dir: Path,
    v1_predictions_dir: Path,
    v3_predictions_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    cfg = _confirmation_config(
        config_path,
        samples_dir=samples_dir,
        split=split,
        bridge_dir=bridge_dir,
        projection_targets_dir=projection_targets_dir,
    )
    ids = [line.strip() for line in split.read_text().splitlines() if line.strip()]
    dataset = build_dataset(
        cfg, "test", ids, int(cfg["training"].get("n_query_points", 8192))
    )
    rows: list[dict[str, Any]] = []
    for sid in ids:
        v1 = np.load(v1_predictions_dir / f"{sid}.npz")
        v3 = np.load(v3_predictions_dir / f"{sid}.npz")
        valid = np.asarray(v1["valid_flat_indices"], dtype=np.int64)
        if not np.array_equal(valid, dataset.valid_flat_indices):
            raise RuntimeError(f"Canonical domain mismatch for {sid}")
        prediction = 0.45 * np.asarray(
            v1["final_prediction"], dtype=np.float64
        ) + 0.55 * np.asarray(v3["final_prediction"], dtype=np.float64)
        gt_volume = np.load(samples_dir / sid / "gt_voxels.npy", mmap_mode="r")
        gt = (np.asarray(gt_volume).ravel()[valid] > 0.05).astype(np.float32)
        tumor = focus_metadata(samples_dir / sid)["tumor_params"]
        rows.append(
            {"sample_id": sid, **_reconstruction_metrics(prediction, gt, dataset, tumor)}
        )
    numeric_keys = [key for key in rows[0] if key != "sample_id"]
    result = {
        "model": "frozen_v1_v3_convex_ensemble",
        "split": "sealed_confirmation",
        "v1_weight": 0.45,
        "v3_weight": 0.55,
        "threshold": 0.5,
        "n_samples": len(rows),
        "summary": {
            key: float(np.nanmean([float(row[key]) for row in rows]))
            for key in numeric_keys
        },
        "per_sample": rows,
    }
    output_path.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frozen-protocol", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--bridge-dir", type=Path, required=True)
    parser.add_argument("--projection-targets-dir", type=Path, required=True)
    parser.add_argument("--stage1-config", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--v1-config", type=Path, required=True)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--v3-config", type=Path, required=True)
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--v4-config", type=Path, required=True)
    parser.add_argument("--v4-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--allow-confirmation-eval", action="store_true")
    args = parser.parse_args()
    manifest = require_confirmation_permission(
        samples_dir=json.loads(args.manifest.read_text())["samples_dir"],
        allow_confirmation_eval=args.allow_confirmation_eval,
        manifest_path=args.manifest,
    )
    if manifest is None:
        raise RuntimeError("The supplied dataset is not the sealed confirmation cohort")
    if not args.frozen_protocol.is_file():
        raise FileNotFoundError("Frozen protocol must exist before confirmation access")
    if args.receipt.exists():
        raise RuntimeError(
            "A confirmation evaluation receipt already exists; reruns are forbidden"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "status": "started",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha256_file(args.manifest),
        "frozen_protocol_sha256": sha256_file(args.frozen_protocol),
        "threshold": 0.5,
        "ensemble_weights": {"v1": 0.45, "v3": 0.55},
        "checkpoints": {
            "stage1": sha256_file(args.stage1_checkpoint),
            "v1": sha256_file(args.v1_checkpoint),
            "v3": sha256_file(args.v3_checkpoint),
            "v4": sha256_file(args.v4_checkpoint),
        },
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    samples_dir = Path(manifest["samples_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: dict[str, dict[str, Any]] = {}
    try:
        if args.bridge_dir.exists() or args.projection_targets_dir.exists():
            raise RuntimeError(
                "Confirmation derived predictions already exist before the one-shot suite"
            )
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("bridge_stage1_to_stage2.py")),
                "--config",
                str(args.stage1_config),
                "--checkpoint",
                str(args.stage1_checkpoint),
                "--split_file",
                str(args.split),
                "--output_dir",
                str(args.bridge_dir),
                "--samples_dir",
                str(samples_dir),
                "--device",
                str(device),
                "--batch_size",
                "32",
                "--skip_roi",
            ],
            check=True,
        )
        canonical_for_targets = CanonicalCrossDiscretization(
            yaml.safe_load(args.v4_config.read_text())["data"]["operator_cache"],
            shared_dir=yaml.safe_load(args.v4_config.read_text())["data"]["shared_dir"],
            factorize=True,
        )
        args.projection_targets_dir.mkdir(parents=True, exist_ok=False)
        ids = [line.strip() for line in args.split.read_text().splitlines() if line.strip()]
        for sid in ids:
            gt_volume = np.load(samples_dir / sid / "gt_voxels.npy", mmap_mode="r")
            gt = (
                np.asarray(gt_volume).ravel()[
                    canonical_for_targets.operator.valid_flat_indices
                ]
                > 0.05
            ).astype(np.float64)
            np.save(
                args.projection_targets_dir / f"{sid}.npy",
                canonical_for_targets.project_coefficients(gt).astype(np.float32),
            )
        for name, config, checkpoint in (
            ("v1", args.v1_config, args.v1_checkpoint),
            ("v3", args.v3_config, args.v3_checkpoint),
            ("v4", args.v4_config, args.v4_checkpoint),
        ):
            results[name] = _evaluate_model(
                config_path=config,
                checkpoint_path=checkpoint,
                samples_dir=samples_dir,
                split=args.split,
                bridge_dir=args.bridge_dir,
                projection_targets_dir=args.projection_targets_dir,
                output_path=args.output_dir / f"{name}.json",
                predictions_dir=args.output_dir / f"{name}_predictions",
                device=device,
            )
        results["ensemble"] = _evaluate_ensemble(
            config_path=args.v1_config,
            samples_dir=samples_dir,
            split=args.split,
            bridge_dir=args.bridge_dir,
            projection_targets_dir=args.projection_targets_dir,
            v1_predictions_dir=args.output_dir / "v1_predictions",
            v3_predictions_dir=args.output_dir / "v3_predictions",
            output_path=args.output_dir / "ensemble.json",
        )
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("summarize_sealed_confirmation.py")),
                "--samples-dir",
                str(samples_dir),
                "--manifest",
                str(args.manifest),
                "--v1",
                str(args.output_dir / "v1.json"),
                "--v3",
                str(args.output_dir / "v3.json"),
                "--v4",
                str(args.output_dir / "v4.json"),
                "--ensemble",
                str(args.output_dir / "ensemble.json"),
                "--output",
                str(args.output_dir / "statistics.json"),
                "--allow-confirmation-eval",
            ],
            check=True,
        )
    except BaseException as error:
        receipt.update(
            {
                "status": "failed_after_confirmation_access",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": repr(error),
            }
        )
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        raise
    receipt.update(
        {
            "status": "completed",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "result_files": {
                name: sha256_file(args.output_dir / f"{name}.json")
                for name in ("v1", "v3", "v4", "ensemble")
            },
            "statistics_sha256": sha256_file(args.output_dir / "statistics.json"),
        }
    )
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
