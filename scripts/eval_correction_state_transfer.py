#!/usr/bin/env python3
"""Evaluate a validation-selected CST checkpoint on the canonical dense domain."""

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

from du2vox.evaluation.error_structured import (  # noqa: E402
    _reconstruction_metrics,
    cosine,
    relative_errors,
)
from experiments.cross_discretization_decomposition.run_analysis import (  # noqa: E402
    focus_metadata,
)
from scripts.train_correction_state_transfer import CSTEngine, load_config  # noqa: E402
from scripts.train_iterative_fem_corrector import build_dataset, load_ids  # noqa: E402


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        key
        for key, value in rows[0].items()
        if key != "sample_id" and isinstance(value, (int, float, np.floating))
    ]
    return {
        key: float(np.nanmean([float(row[key]) for row in rows]))
        for key in sorted(keys)
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-freeze", type=Path)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    backbone_cfg = yaml.safe_load(Path(cfg["frozen_backbone"]["config"]).read_text())
    os.environ["DU2VOX_SHARED_DIR"] = str(backbone_cfg["data"]["shared_dir"])
    os.environ["DU2VOX_ALLOW_STALE_FRAME_MANIFEST"] = "1"
    os.environ["DU2VOX_FRAME_MANIFEST_SHA256"] = str(
        backbone_cfg["data"]["frame_manifest_sha256"]
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not checkpoint.get("hard_q", False):
        raise RuntimeError("CST evaluation requires an exact hard-Q checkpoint")
    if args.split == "test":
        if args.validation_freeze is None:
            raise RuntimeError("Development-test requires a validation freeze receipt")
        freeze = json.loads(args.validation_freeze.read_text())
        if not freeze.get("frozen_on_validation", False):
            raise RuntimeError("Invalid CST validation freeze receipt")
        if Path(freeze["selected_checkpoint"]).resolve() != args.checkpoint.resolve():
            raise RuntimeError("Development-test checkpoint differs from frozen selection")
        if file_config := freeze.get("selected_config"):
            if Path(file_config).resolve() != args.config.resolve():
                raise RuntimeError("Development-test config differs from frozen selection")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = CSTEngine(cfg, device)
    model = engine.build_model()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    ids = load_ids(cfg["data"][f"{args.split}_split"])
    if args.max_samples is None and len(ids) != 300:
        raise RuntimeError("Formal CST evaluation requires all 300 cases")
    if args.max_samples is not None:
        ids = ids[: args.max_samples]
    dataset = build_dataset(backbone_cfg, args.split, ids, 1)
    rows: list[dict[str, Any]] = []
    for index, sid in enumerate(ids):
        _, coarse, detail, final = engine.predict(model, sid, args.split)
        gt, target = engine._targets(sid)
        arrays = [
            tensor.squeeze(0).float().cpu().numpy().astype(np.float64)
            for tensor in (coarse, detail, final)
        ]
        coarse_np, detail_np, final_np = arrays
        gt_np = gt.cpu().numpy().astype(np.float64)
        target_np = target.cpu().numpy().astype(np.float64)
        tumor = focus_metadata(Path(cfg["data"]["samples_dir"]) / sid)["tumor_params"]
        row: dict[str, Any] = {"sample_id": sid}
        for prefix, prediction in (("coarse", coarse_np), ("final", final_np)):
            metrics = _reconstruction_metrics(prediction, gt_np, dataset, tumor)
            row.update({f"{prefix}_{key}": value for key, value in metrics.items()})
        projected = engine.complement.project_numpy(detail_np)
        projected_final = engine.complement.coefficients_numpy(final_np)
        expected_state = np.load(
            Path(cfg["data"]["v4_states_root"]) / args.split / f"{sid}.npy"
        ).astype(np.float64)
        rel_l1, rel_l2 = relative_errors(detail_np, target_np)
        detail_energy = np.square(detail_np)
        gt_volume = np.load(
            Path(cfg["data"]["samples_dir"]) / sid / "gt_voxels.npy", mmap_mode="r"
        )
        gt_binary = np.asarray(gt_volume) > 0.05
        boundary = gt_binary & ~binary_erosion(gt_binary)
        distances = distance_transform_edt(
            ~boundary, sampling=engine.canonical.operator.spacing_mm
        ).ravel()[engine.canonical.operator.valid_flat_indices]
        energy = max(float(detail_energy.sum()), 1e-30)
        row.update(
            {
                "stage2_marginal_dice": row["final_dice"] - row["coarse_dice"],
                "final_relative_l2": float(
                    np.linalg.norm(final_np - gt_np) / max(np.linalg.norm(gt_np), 1e-30)
                ),
                "detail_cosine": cosine(detail_np, target_np),
                "detail_relative_l1": rel_l1,
                "detail_relative_l2": rel_l2,
                "detail_energy_recovery": float(
                    np.linalg.norm(detail_np) / max(np.linalg.norm(target_np), 1e-30)
                ),
                "boundary_energy_le_0.2mm": float(
                    detail_energy[distances <= 0.2].sum() / energy
                ),
                "boundary_energy_le_0.6mm": float(
                    detail_energy[distances <= 0.6].sum() / energy
                ),
                "coarse_leakage": float(
                    np.dot(projected, projected)
                    / max(np.dot(detail_np, detail_np), 1e-30)
                ),
                "coarse_preservation_relative_l2": float(
                    np.linalg.norm(projected_final - expected_state)
                    / max(np.linalg.norm(expected_state), 1e-30)
                ),
            }
        )
        rows.append(row)
        if (index + 1) % 25 == 0:
            print(f"[CST {args.split} {index + 1}/{len(ids)}]", flush=True)
    payload = {
        "method": "CST",
        "split": "development_test" if args.split == "test" else "validation",
        "n_samples": len(rows),
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_validation_final_dice": checkpoint["dense_val_dice"],
        "parameter_count": checkpoint["parameter_count"],
        "threshold": 0.5,
        "hard_q": True,
        "frozen_v4": True,
        "confirmation_data_used": False,
        "validation_freeze_receipt": (
            str(args.validation_freeze.resolve()) if args.validation_freeze else None
        ),
        "summary": summarize(rows),
        "per_sample": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")


if __name__ == "__main__":
    main()
