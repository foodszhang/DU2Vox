#!/usr/bin/env python3
"""Evaluate FEM-only and final outputs of an online hierarchical V4 checkpoint."""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.evaluation.error_structured import _reconstruction_metrics, cosine
from experiments.cross_discretization_decomposition.run_analysis import focus_metadata
from scripts.train_iterative_fem_corrector import build_dataset, load_ids
from scripts.train_online_hierarchical_v4 import OnlineHierarchicalEngine


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted(
        key for key, value in rows[0].items()
        if key != "sample_id" and isinstance(value, (int, float, np.floating))
    )
    return {key: float(np.nanmean([float(row[key]) for row in rows])) for key in keys}


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    backbone_cfg = yaml.safe_load(Path(cfg["frozen_backbone"]["config"]).read_text())
    data = backbone_cfg["data"]
    os.environ["DU2VOX_SHARED_DIR"] = str(data["shared_dir"])
    if data.get("allow_stale_frame_manifest", False):
        os.environ["DU2VOX_ALLOW_STALE_FRAME_MANIFEST"] = "1"
    os.environ["DU2VOX_FRAME_MANIFEST_SHA256"] = str(data["frame_manifest_sha256"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    arm = checkpoint.get("arm", "frozen_online")
    engine = OnlineHierarchicalEngine(cfg, device, arm)
    detail_model = engine.build_model("hard")
    detail_state = (
        checkpoint["detail_model"] if "detail_model" in checkpoint else checkpoint["model"]
    )
    detail_model.load_state_dict(detail_state)
    if "v4_model" in checkpoint:
        engine.v4.load_state_dict(checkpoint["v4_model"])
    detail_model.eval()
    engine.v4.eval()
    ids = load_ids(cfg["data"][f"{args.split}_split"])
    if args.max_samples is not None:
        ids = ids[: args.max_samples]
    dataset = build_dataset(backbone_cfg, args.split, ids, 1)
    rows: list[dict[str, Any]] = []
    for index, sid in enumerate(ids):
        _, coarse, detail, final, state, v4_output = engine.predict_online(
            detail_model, sid, args.split
        )
        gt, target = engine._targets(sid)
        coarse_np = coarse.squeeze(0).float().cpu().numpy().astype(np.float64)
        detail_np = detail.squeeze(0).float().cpu().numpy().astype(np.float64)
        final_np = final.squeeze(0).float().cpu().numpy().astype(np.float64)
        state_np = state.float().cpu().numpy().astype(np.float64)
        gt_np = gt.cpu().numpy().astype(np.float64)
        target_np = target.cpu().numpy().astype(np.float64)
        frozen_state = np.load(
            Path(cfg["data"]["v4_states_root"]) / args.split / f"{sid}.npy"
        ).astype(np.float64)
        tumor = focus_metadata(Path(cfg["data"]["samples_dir"]) / sid)["tumor_params"]
        row: dict[str, Any] = {"sample_id": sid}
        for prefix, prediction in (("fem", coarse_np), ("final", final_np)):
            metrics = _reconstruction_metrics(prediction, gt_np, dataset, tumor)
            row.update({f"{prefix}_{key}": value for key, value in metrics.items()})
        projected_detail = engine.complement.project_numpy(detail_np)
        projected_final = engine.complement.coefficients_numpy(final_np)
        state_delta = state_np - frozen_state
        residuals = v4_output["measurement_residual_rms"].squeeze(0).float().cpu().numpy()
        row.update(
            {
                "delta_final_vs_fem_dice": row["final_dice"] - row["fem_dice"],
                "fem_state_delta_relative_l1": float(
                    np.abs(state_delta).sum() / max(np.abs(frozen_state).sum(), 1e-30)
                ),
                "fem_state_delta_relative_l2": float(
                    np.linalg.norm(state_delta) / max(np.linalg.norm(frozen_state), 1e-30)
                ),
                "measurement_residual_ratio": float(residuals[-1] / max(residuals[0], 1e-30)),
                "detail_coarse_leakage": float(
                    np.dot(projected_detail, projected_detail)
                    / max(np.dot(detail_np, detail_np), 1e-30)
                ),
                "final_coarse_preservation_relative_l2": float(
                    np.linalg.norm(projected_final - state_np)
                    / max(np.linalg.norm(state_np), 1e-30)
                ),
                "detail_cosine": cosine(detail_np, target_np),
                "detail_norm": float(np.linalg.norm(detail_np)),
            }
        )
        rows.append(row)
        if (index + 1) % 25 == 0:
            print(f"[{arm} {args.split} {index + 1}/{len(ids)}]", flush=True)
    result = {
        "n_samples": len(rows),
        "arm": arm,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_dense_val_dice": checkpoint["dense_val_dice"],
        "split": args.split,
        "summary": summarize(rows),
        "per_sample": rows,
        "confirmation_data_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")


if __name__ == "__main__":
    main()
