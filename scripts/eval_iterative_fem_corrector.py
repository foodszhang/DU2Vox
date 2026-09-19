#!/usr/bin/env python3
"""Evaluate a frozen iterative FEM corrector on the dense canonical domain."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.bridge.canonical_cross_discretization import CanonicalCrossDiscretization
from du2vox.evaluation.iterative_fem import (
    evaluate_iterative_fem_dense,
    save_iterative_fem_result,
)
from scripts.train_iterative_fem_corrector import (
    build_dataset,
    build_model,
    load_ids,
)
from du2vox.utils.confirmation import (
    require_confirmation_permission,
    validate_validation_dataset_receipt,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_freeze_receipt(
    path: Path, checkpoint: Path, config: Path, dataset_receipt: Path
) -> None:
    receipt = json.loads(path.read_text())
    if receipt.get("status") != "frozen_on_val300":
        raise RuntimeError("Development-test evaluation requires a frozen val300 receipt")
    authorized = {
        (item["checkpoint_sha256"], item["config_sha256"])
        for item in receipt.get("candidates", [])
    }
    if (sha256(checkpoint), sha256(config)) not in authorized:
        raise RuntimeError("Checkpoint/config pair is not frozen in the val receipt")
    if receipt.get("sealed_confirmation_accessed") is not False:
        raise RuntimeError("Invalid receipt: sealed confirmation access is not false")
    validate_validation_dataset_receipt(receipt, dataset_receipt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-predictions-dir", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--diagnostic-max-dc-update",
        type=float,
        help="Validation-only counterfactual override; does not modify the checkpoint.",
    )
    parser.add_argument(
        "--diagnostic-max-neural-update",
        type=float,
        help="Validation-only counterfactual override; does not modify the checkpoint.",
    )
    parser.add_argument(
        "--diagnostic-volume-center-neural-update",
        action="store_true",
        help="Validation-only counterfactual removal of neural-update constant mode.",
    )
    parser.add_argument("--freeze-receipt", type=Path)
    parser.add_argument("--allow-confirmation-eval", action="store_true")
    parser.add_argument(
        "--confirmation-manifest",
        type=Path,
        default=Path("data/confirmation_manifest.json"),
    )
    args = parser.parse_args()
    if args.split == "test" and (
        args.diagnostic_max_dc_update is not None
        or args.diagnostic_max_neural_update is not None
        or args.diagnostic_volume_center_neural_update
    ):
        raise RuntimeError("Diagnostic update overrides are restricted to validation")
    cfg = yaml.safe_load(args.config.read_text())
    data = cfg["data"]
    require_confirmation_permission(
        samples_dir=data["samples_dir"],
        allow_confirmation_eval=args.allow_confirmation_eval,
        manifest_path=args.confirmation_manifest,
    )
    if args.split == "test":
        if args.freeze_receipt is None:
            raise RuntimeError("Development-test evaluation requires --freeze-receipt")
        dataset_receipt = data.get("dataset_receipt")
        if not dataset_receipt:
            raise RuntimeError("Config must declare data.dataset_receipt")
        validate_freeze_receipt(
            args.freeze_receipt,
            args.checkpoint,
            args.config,
            Path(dataset_receipt),
        )
    os.environ["DU2VOX_SHARED_DIR"] = str(data["shared_dir"])
    if data.get("allow_stale_frame_manifest", False):
        os.environ["DU2VOX_ALLOW_STALE_FRAME_MANIFEST"] = "1"
    if data.get("frame_manifest_sha256"):
        os.environ["DU2VOX_FRAME_MANIFEST_SHA256"] = str(
            data["frame_manifest_sha256"]
        )
    ids = load_ids(data[f"{args.split}_split"])
    if args.max_samples is not None:
        ids = ids[: args.max_samples]
    dataset = build_dataset(
        cfg, args.split, ids, int(cfg["training"].get("n_query_points", 8192))
    )
    canonical = CanonicalCrossDiscretization(
        data["operator_cache"], shared_dir=data["shared_dir"], factorize=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.diagnostic_volume_center_neural_update:
        cfg["model"]["volume_center_neural_update"] = True
    model = build_model(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    variant = cfg["model"].get("variant", "v1")
    configured_model_type = {
        "residual_consistent_v2": "residual_consistent_iterative_fem_v2",
        "scale_calibrated_proximal_v3": "scale_calibrated_proximal_fem_v3",
        "unified_dual_evidence_v4": "unified_dual_evidence_fem_v4",
    }.get(variant, "iterative_residual_aware_fem")
    if checkpoint.get("model_type") != configured_model_type:
        raise RuntimeError("Checkpoint has the wrong model type")
    model.load_state_dict(checkpoint["model"])
    diagnostic_overrides = {}
    if args.diagnostic_max_dc_update is not None:
        if args.diagnostic_max_dc_update < 0.0:
            raise ValueError("--diagnostic-max-dc-update must be non-negative")
        model.max_dc_update = float(args.diagnostic_max_dc_update)
        diagnostic_overrides["max_dc_update"] = model.max_dc_update
    if args.diagnostic_max_neural_update is not None:
        if args.diagnostic_max_neural_update < 0.0:
            raise ValueError("--diagnostic-max-neural-update must be non-negative")
        model.max_neural_update = float(args.diagnostic_max_neural_update)
        diagnostic_overrides["max_neural_update"] = model.max_neural_update
    if args.diagnostic_volume_center_neural_update:
        diagnostic_overrides["volume_center_neural_update"] = True
    result = evaluate_iterative_fem_dense(
        model,
        dataset,
        canonical,
        device,
        max_samples=len(ids),
        save_predictions_dir=args.save_predictions_dir,
    )
    result.update(
        {
            "model": configured_model_type,
            "split": args.split,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_dense_val_dice": checkpoint.get("dense_val_dice"),
            "checkpoint_dense_val_metric": checkpoint.get(
                "dense_val_metric", checkpoint.get("dense_val_dice")
            ),
            "checkpoint_selection_metric": checkpoint.get(
                "selection_metric", "legacy_step3_dice"
            ),
            "diagnostic_overrides": diagnostic_overrides,
        }
    )
    save_iterative_fem_result(result, args.output)


if __name__ == "__main__":
    main()
