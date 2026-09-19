#!/usr/bin/env python3
"""Dense evaluation and detail diagnostics for frozen voxel-detail checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.ndimage import binary_erosion, distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.evaluation.error_structured import (
    _reconstruction_metrics,
    cosine,
    relative_errors,
)
from du2vox.data.error_structured_dataset import ErrorStructuredFixedDomainDataset
from experiments.cross_discretization_decomposition.run_analysis import focus_metadata
from scripts.train_complement_voxel_detail import FullDomainEngine, load_ids, sha256_file
from du2vox.models.stage2.complement_voxel_detail import normalize_detail_mode
from du2vox.utils.gt_io import load_gt_volume
from du2vox.utils.confirmation import validate_validation_dataset_receipt


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted(
        key
        for key, value in rows[0].items()
        if key != "sample_id" and isinstance(value, (int, float, np.floating))
    )
    return {key: float(np.nanmean([float(row[key]) for row in rows])) for key in keys}


def paired_bootstrap(values: list[float], *, seed: int, draws: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(array), size=(draws, len(array)))
    means = array[sampled].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "paired_bootstrap_95_ci": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "bootstrap_draws": draws,
        "seed": seed,
    }


def validate_freeze_receipt(
    path: Path, checkpoint: Path, config: Path, dataset_receipt: Path
) -> dict[str, Any]:
    receipt = json.loads(path.read_text())
    if receipt.get("status") != "frozen_on_val300":
        raise RuntimeError("Development-test evaluation requires a frozen val300 receipt")
    actual = sha256_file(checkpoint)
    config_sha = sha256_file(config)
    authorized = {
        (item["checkpoint_sha256"], item["config_sha256"]) for item in receipt.get("candidates", [])
    }
    if (actual, config_sha) not in authorized:
        raise RuntimeError("Checkpoint/config pair is not frozen in the val receipt")
    if receipt.get("sealed_confirmation_accessed") is not False:
        raise RuntimeError("Invalid receipt: sealed confirmation access is not false")
    validate_validation_dataset_receipt(receipt, dataset_receipt)
    return receipt


def build_metric_dataset(
    cfg: dict[str, Any], split: str, ids: list[str]
) -> ErrorStructuredFixedDomainDataset:
    data = cfg["data"]
    if cfg.get("coarse_source") is not None:
        bridge_dir = cfg["coarse_source"][f"{split}_bridge_dir"]
    else:
        backbone_cfg = yaml.safe_load(Path(cfg["frozen_backbone"]["config"]).read_text())
        bridge_dir = backbone_cfg["data"][f"{split}_bridge_dir"]
    return ErrorStructuredFixedDomainDataset(
        sample_ids=ids,
        samples_dir=data["samples_dir"],
        bridge_dir=bridge_dir,
        projection_targets_dir=data["projection_targets_dir"],
        operator_cache=data["operator_cache"],
        shared_dir=data["shared_dir"],
        n_query_points=1,
        seed=int(cfg["training"]["seed"]),
        multiview=False,
        gt_mode=data.get("gt_mode", "binary"),
        normalize_gt=data.get("normalize_gt", "none"),
        normalization_scale_filename=data.get("normalization_scale_filename"),
        binary_threshold=float(data.get("binary_threshold", 0.05)),
        metric_data_range=float(data.get("data_range", 2.0)),
    )


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--save-predictions-dir", type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    environment_data = cfg["data"]
    if "frozen_backbone" in cfg:
        backbone_cfg = yaml.safe_load(Path(cfg["frozen_backbone"]["config"]).read_text())
        environment_data = backbone_cfg["data"]
    os.environ["DU2VOX_SHARED_DIR"] = str(environment_data["shared_dir"])
    if environment_data.get("allow_stale_frame_manifest", False):
        os.environ["DU2VOX_ALLOW_STALE_FRAME_MANIFEST"] = "1"
    if environment_data.get("frame_manifest_sha256"):
        os.environ["DU2VOX_FRAME_MANIFEST_SHA256"] = str(environment_data["frame_manifest_sha256"])
    receipt = None
    if args.split == "test":
        if args.freeze_receipt is None:
            raise RuntimeError("Development-test evaluation requires --freeze-receipt")
        dataset_receipt = cfg["data"].get("dataset_receipt")
        if not dataset_receipt:
            raise RuntimeError("Config must declare data.dataset_receipt")
        receipt = validate_freeze_receipt(
            args.freeze_receipt,
            args.checkpoint,
            args.config,
            Path(dataset_receipt),
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = FullDomainEngine(cfg, device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    mode = normalize_detail_mode(checkpoint.get("mode", checkpoint.get("constrained", True)))
    model = engine.build_model(mode)
    model.load_state_dict(checkpoint["model"])
    if "view_encoder" in checkpoint:
        engine.encoder.load_state_dict(checkpoint["view_encoder"])
    elif cfg.get("coarse_source") is not None:
        raise RuntimeError("Stage1 hard-Q checkpoint is missing its view encoder")
    model.eval()
    engine.encoder.eval()
    ids = load_ids(cfg["data"][f"{args.split}_split"])
    if args.max_samples is not None:
        ids = ids[: args.max_samples]
    cache_audit = engine.audit_coarse_cache(args.split, ids)
    dataset = build_metric_dataset(cfg, args.split, ids)
    coarse_prefix = "stage1" if cfg.get("coarse_source") is not None else "v4"
    rows: list[dict[str, Any]] = []
    if args.save_predictions_dir is not None:
        args.save_predictions_dir.mkdir(parents=True, exist_ok=True)
    for index, sid in enumerate(ids):
        _, coarse, detail, final = engine.predict(model, sid, args.split)
        gt, target = engine._targets(sid)
        coarse_np = coarse.squeeze(0).float().cpu().numpy().astype(np.float64)
        detail_np = detail.squeeze(0).float().cpu().numpy().astype(np.float64)
        final_np = final.squeeze(0).float().cpu().numpy().astype(np.float64)
        gt_np = gt.cpu().numpy().astype(np.float64)
        target_np = target.cpu().numpy().astype(np.float64)
        tumor = focus_metadata(Path(cfg["data"]["samples_dir"]) / sid)["tumor_params"]
        row: dict[str, Any] = {"sample_id": sid}
        for prefix, prediction in ((coarse_prefix, coarse_np), ("final", final_np)):
            metrics = _reconstruction_metrics(prediction, gt_np, dataset, tumor)
            row.update({f"{prefix}_{key}": value for key, value in metrics.items()})
            _, relative_l2 = relative_errors(prediction, gt_np)
            row[f"{prefix}_relative_l2"] = relative_l2
        projected = engine.complement.project_numpy(detail_np)
        leakage = float(np.dot(projected, projected) / max(np.dot(detail_np, detail_np), 1e-30))
        coarse_coefficients = np.load(engine.coarse_path(sid, args.split)).reshape(-1)
        final_coeff = engine.complement.coefficients_numpy(final_np)
        preservation = final_coeff - coarse_coefficients.astype(np.float64)
        rel_l1, rel_l2 = relative_errors(detail_np, target_np)
        gt_norm = float(np.linalg.norm(target_np))
        pred_norm = float(np.linalg.norm(detail_np))
        energy = np.square(detail_np)
        gt_energy = np.square(target_np)
        row.update(
            {
                "delta_dice": row["final_dice"] - row[f"{coarse_prefix}_dice"],
                "coarse_leakage": leakage,
                "coarse_preservation_relative_l2": float(
                    np.linalg.norm(preservation) / max(np.linalg.norm(coarse_coefficients), 1e-30)
                ),
                "coarse_preservation_rms": float(np.sqrt(np.mean(preservation**2))),
                "detail_cosine": cosine(detail_np, target_np),
                "detail_relative_l1": rel_l1,
                "detail_relative_l2": rel_l2,
                "detail_energy_ratio": float(energy.sum() / max(gt_energy.sum(), 1e-30)),
                "predicted_detail_norm": pred_norm,
                "gt_detail_norm": gt_norm,
                "predicted_positive_fraction": float(np.mean(detail_np > 0)),
                "predicted_negative_fraction": float(np.mean(detail_np < 0)),
                "predicted_positive_energy_fraction": float(
                    energy[detail_np > 0].sum() / max(energy.sum(), 1e-30)
                ),
                "gt_positive_fraction": float(np.mean(target_np > 0)),
                "gt_negative_fraction": float(np.mean(target_np < 0)),
            }
        )
        if args.save_predictions_dir is not None:
            np.savez_compressed(
                args.save_predictions_dir / f"{sid}.npz",
                coarse_prediction=coarse_np.astype(np.float32),
                detail=detail_np.astype(np.float32),
                final_prediction=final_np.astype(np.float32),
                valid_flat_indices=engine.canonical.operator.valid_flat_indices,
                grid_shape=np.asarray(engine.canonical.operator.grid_shape),
            )
        gt_volume = load_gt_volume(Path(cfg["data"]["samples_dir"]) / sid)
        gt_binary = np.asarray(gt_volume) > 0.05
        boundary = gt_binary & ~binary_erosion(gt_binary)
        distance = distance_transform_edt(~boundary, sampling=engine.canonical.operator.spacing_mm)
        valid_distance = distance.ravel()[engine.canonical.operator.valid_flat_indices]
        total_energy = max(float(energy.sum()), 1e-30)
        boundary_02 = float(energy[valid_distance <= 0.2].sum() / total_energy)
        boundary_06 = float(energy[valid_distance <= 0.6].sum() / total_energy)
        row["boundary_energy_le_0.2mm"] = boundary_02
        row["boundary_energy_le_0.6mm"] = boundary_06
        row["boundary_energy_outside_0.6mm"] = 1.0 - boundary_06
        rows.append(row)
        if (index + 1) % 25 == 0:
            print(f"[{mode} {args.split} {index + 1}/{len(ids)}]", flush=True)
    result = {
        "n_samples": len(rows),
        "mode": mode,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_dense_val_dice": checkpoint["dense_val_dice"],
        "training_lr": float(checkpoint["config"]["training"]["lr"]),
        "training_epochs": int(checkpoint["config"]["training"]["epochs"]),
        "use_views": bool(checkpoint["config"]["model"].get("use_views", True)),
        "initial_decoder_sha256": checkpoint.get("initial_decoder_sha256"),
        "initial_encoder_sha256": checkpoint.get("initial_encoder_sha256"),
        "training_order_sha256": checkpoint.get("training_order_sha256"),
        "parameter_counts": checkpoint.get("parameter_counts"),
        "split": args.split,
        "coarse_source": coarse_prefix,
        "coarse_cache_audit": cache_audit,
        "freeze_receipt": str(args.freeze_receipt.resolve()) if receipt else None,
        "freeze_receipt_sha256": (sha256_file(args.freeze_receipt) if receipt else None),
        "summary": summarize(rows),
        "paired_delta_dice": paired_bootstrap(
            [float(row["delta_dice"]) for row in rows],
            seed=int(cfg["training"]["seed"]),
            draws=args.bootstrap_draws,
        ),
        "per_sample": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")


if __name__ == "__main__":
    main()
