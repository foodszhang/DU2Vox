#!/usr/bin/env python3
"""
Stage 2 evaluation: load checkpoint, run on val split, report metrics.

Usage:
    python scripts/eval_stage2.py \\
        --config configs/stage2/uniform_1000_v2.yaml \\
        --checkpoint checkpoints/stage2/stage1_uniform_v2_visible/best.pth
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from du2vox.models.stage2.residual_inr import ResidualINR
from du2vox.models.stage2.stage2_dataset import Stage2Dataset


def load_split(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def build_loader(cfg, sample_ids, batch_size=8):
    ds = Stage2Dataset(
        bridge_dir=cfg["data"]["bridge_dir"],
        shared_dir=cfg["data"]["shared_dir"],
        samples_dir=cfg["data"]["samples_dir"],
        sample_ids=sample_ids,
        n_query_points=cfg["data"]["n_query_points"],
        roi_padding_mm=cfg["data"]["roi_padding_mm"],
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def evaluate(model, loader, device="cuda"):
    model.eval()
    all_d_hat = []
    all_gt = []
    all_valid = []

    with torch.no_grad():
        for batch in loader:
            coords  = batch["coords"].to(device)
            prior   = batch["prior_8d"].to(device)
            gt      = batch["gt"]
            valid   = batch["valid"]

            d_hat, _, _ = model(coords, prior)
            d_hat = d_hat.cpu().numpy()
            valid_np = valid.numpy()
            gt_np = gt.numpy()

            all_d_hat.append(d_hat)
            all_gt.append(gt_np)
            all_valid.append(valid_np)

    d_hat_all = np.concatenate([d.flatten() for d in all_d_hat])
    gt_all    = np.concatenate([g.flatten() for g in all_gt])
    valid_all = np.concatenate([v.flatten() for v in all_valid])

    valid_mask = valid_all > 0
    if valid_mask.sum() == 0:
        return {"mse": float("nan"), "mae": float("nan"), "n_valid": 0}

    mse = float(np.mean((d_hat_all[valid_mask] - gt_all[valid_mask]) ** 2))
    mae = float(np.mean(np.abs(d_hat_all[valid_mask] - gt_all[valid_mask])))

    # Also report per-sample metrics
    d_hat_cat = np.concatenate(all_d_hat)
    gt_cat    = np.concatenate(all_gt)
    valid_cat = np.concatenate(all_valid)

    per_sample_mse = {}
    offset = 0
    for i, batch in enumerate(loader):
        B = batch["coords"].shape[0]
        N = batch["coords"].shape[1]
        for b in range(B):
            v = valid_cat[offset:offset+N]
            if v.sum() > 0:
                mse_s = np.mean((d_hat_cat[offset:offset+N][v>0] - gt_cat[offset:offset+N][v>0])**2)
                per_sample_mse[batch["sample_id"][b]] = float(mse_s)
            offset += N

    return {
        "mse": mse,
        "mae": mae,
        "n_valid": int(valid_mask.sum()),
        "per_sample_mse": per_sample_mse,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    split_file = cfg["data"][f"{args.split}_split"]
    sample_ids = load_split(split_file)
    if args.max_samples:
        sample_ids = sample_ids[: args.max_samples]

    print(f"[Eval Stage 2] Split={args.split}, n_samples={len(sample_ids)}")

    model = ResidualINR(
        n_freqs=cfg["model"]["n_freqs"],
        hidden_dim=cfg["model"]["hidden_dim"],
        n_hidden_layers=cfg["model"]["n_hidden_layers"],
        prior_dim=cfg["model"]["prior_dim"],
        skip_connection=cfg["model"]["skip_connection"],
    ).cuda()

    state = torch.load(args.checkpoint, map_location="cuda")
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {args.checkpoint}")

    loader = build_loader(cfg, sample_ids, batch_size=cfg["training"]["batch_size"])
    metrics = evaluate(model, loader)

    print(f"\n=== Stage 2 Evaluation ===")
    print(f"MSE:  {metrics['mse']:.6f}")
    print(f"MAE:  {metrics['mae']:.6f}")
    print(f"Valid query points: {metrics['n_valid']}")

    # Per-sample MSE summary
    if metrics["per_sample_mse"]:
        mses = list(metrics["per_sample_mse"].values())
        print(f"\nPer-sample MSE: mean={np.mean(mses):.6f}, "
              f"min={np.min(mses):.6f}, max={np.max(mses):.6f}")

    return metrics


if __name__ == "__main__":
    main()
