#!/usr/bin/env python3
"""
Stage 2 evaluation: full-grid metrics with FEM baseline comparison.

Metrics reported:
  - Stage 2 Dice@0.5 (per-sample averaged)
  - FEM Dice@0.5 (per-sample averaged)
  - ΔDice = Stage2 - FEM
  - Stage 2 MSE
  - FEM MSE
  - Per-sample metrics saved to JSON

Usage:
    python scripts/eval_stage2.py \
        --config configs/stage2/uniform_1000_v2.yaml \
        --checkpoint checkpoints/stage2/baseline_de_only/best.pth
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from du2vox.models.stage2.residual_inr import ResidualINR
from du2vox.models.stage2.stage2_dataset import Stage2DatasetPrecomputed


def load_split(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def compute_dice(pred, target, threshold=0.5):
    pred_bin = (pred >= threshold).astype(float)
    target_bin = (target >= threshold).astype(float)
    intersection = (pred_bin * target_bin).sum()
    return (2 * intersection / (pred_bin.sum() + target_bin.sum() + 1e-8))


def evaluate_full_grid(
    model,
    precomputed_dir: Path,
    sample_ids: list,
    batch_points: int = 8192,
    device: str = "cuda",
    view_feat_dim: int = 0,
) -> dict:
    """
    Full-grid evaluation: load complete precomputed grids, run model on all points.

    Returns per-sample and aggregated metrics with FEM baseline comparison.
    """
    model.eval()
    per_sample = []

    for sid in sample_ids:
        path = precomputed_dir / f"{sid}.npz"
        if not path.exists():
            print(f"  WARNING: {sid} not found, skipping")
            continue

        data = dict(np.load(path))
        # Legacy fallback: if grid_coords_norm not present, normalize raw coords using bbox metadata
        if "grid_coords_norm" in data:
            coords_norm = data["grid_coords_norm"]   # [G, 3] — already normalized [-1,1]
        else:
            raw = data["grid_coords"]
            bbox_min = data["bbox_min"]
            bbox_max = data["bbox_max"]
            coords_norm = (2.0 * (raw - bbox_min) / (bbox_max - bbox_min + 1e-8) - 1.0).astype(np.float32)
        coords_raw  = data["grid_coords"]         # [G, 3] — raw mm, for FEM eval
        prior_all   = data["prior_8d"]            # [G, 8]
        gt_all      = data["gt_values"]          # [G]
        valid_all   = data["valid_mask"]          # [G]

        v_mask = valid_all > 0
        if v_mask.sum() == 0:
            continue

        coords_v = coords_norm[v_mask]   # normalized [-1,1] — for model input
        prior_v  = prior_all[v_mask]
        gt_v     = gt_all[v_mask]

        # Batch forward through model
        d_hat_full = []
        for start in range(0, len(coords_v), batch_points):
            end = min(start + batch_points, len(coords_v))
            c = torch.from_numpy(coords_v[start:end]).unsqueeze(0).float().to(device)
            p = torch.from_numpy(prior_v[start:end]).unsqueeze(0).float().to(device)
            B_b, N_b = c.shape[:2]
            # Provide zero view features when model was trained with view_feat_dim > 0
            zero_view_b = (torch.zeros(B_b, N_b, view_feat_dim).to(device)
                           if view_feat_dim > 0 else None)
            with torch.no_grad():
                d_hat_b, fem_b, _ = model(c, p, zero_view_b)
            d_hat_full.append(d_hat_b.squeeze(0).cpu().float().numpy())

        d_hat_v = np.concatenate(d_hat_full)

        # FEM baseline interpolation
        fem_v = (prior_v[:, :4] * prior_v[:, 4:8]).sum(axis=-1)

        # Dice metrics
        stage2_dice_05 = compute_dice(d_hat_v, gt_v, 0.5)
        fem_dice_05 = compute_dice(fem_v, gt_v, 0.5)

        stage2_dice_01 = compute_dice(d_hat_v, gt_v, 0.1)
        fem_dice_01 = compute_dice(fem_v, gt_v, 0.1)

        stage2_mse = float(np.mean((d_hat_v - gt_v) ** 2))
        fem_mse = float(np.mean((fem_v - gt_v) ** 2))

        per_sample.append({
            "sample_id": sid,
            "stage2_dice_05": float(stage2_dice_05),
            "fem_dice_05": float(fem_dice_05),
            "delta_dice_05": float(stage2_dice_05 - fem_dice_05),
            "stage2_dice_01": float(stage2_dice_01),
            "fem_dice_01": float(fem_dice_01),
            "stage2_mse": stage2_mse,
            "fem_mse": fem_mse,
            "n_valid": int(v_mask.sum()),
        })

    if not per_sample:
        return {"error": "no samples evaluated"}

    # Aggregate
    keys = ["stage2_dice_05", "fem_dice_05", "delta_dice_05",
            "stage2_dice_01", "fem_dice_01", "stage2_mse", "fem_mse"]
    summary = {k: float(np.mean([s[k] for s in per_sample])) for k in keys}
    summary["per_sample"] = {s["sample_id"]: s for s in per_sample}
    summary["n_evaluated"] = len(per_sample)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--batch_points", type=int, default=8192)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    split_file = cfg["data"][f"{args.split}_split"]
    sample_ids = load_split(split_file)
    if args.max_samples:
        sample_ids = sample_ids[: args.max_samples]

    precomputed_dir = cfg["data"].get(f"precomputed_{args.split}_dir")
    if not precomputed_dir or not Path(precomputed_dir).exists():
        print(f"ERROR: precomputed_{args.split}_dir not set or not found in config.")
        print(f"  Set it in config to point to precomputed grid data.")
        sys.exit(1)

    print(f"[Eval Stage 2] Split={args.split}, n_samples={len(sample_ids)}")
    print(f"[Eval Stage 2] Precomputed dir={precomputed_dir}")

    view_feat_dim = cfg["model"].get("view_feat_dim", 0)
    model = ResidualINR(
        n_freqs=cfg["model"]["n_freqs"],
        hidden_dim=cfg["model"]["hidden_dim"],
        n_hidden_layers=cfg["model"]["n_hidden_layers"],
        prior_dim=cfg["model"]["prior_dim"],
        skip_connection=cfg["model"]["skip_connection"],
        view_feat_dim=view_feat_dim,
    ).cuda()

    state = torch.load(args.checkpoint, map_location="cuda")
    # Handle wrapped multiview checkpoint format: {"residual_inr": ..., "view_encoder": ...}
    if "residual_inr" in state:
        state = state["residual_inr"]
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {args.checkpoint}, view_feat_dim={view_feat_dim}")

    metrics = evaluate_full_grid(
        model,
        Path(precomputed_dir),
        sample_ids,
        batch_points=args.batch_points,
        view_feat_dim=view_feat_dim,
    )

    print(f"\n=== Stage 2 Evaluation (Full Grid, per-sample averaged) ===")
    print(f"N evaluated:  {metrics['n_evaluated']}")
    print(f"Stage2 Dice@0.5: {metrics['stage2_dice_05']:.4f}")
    print(f"FEM Dice@0.5:     {metrics['fem_dice_05']:.4f}")
    print(f"ΔDice@0.5:       {metrics['delta_dice_05']:+.4f}")
    print(f"Stage2 Dice@0.1: {metrics['stage2_dice_01']:.4f}")
    print(f"FEM Dice@0.1:     {metrics['fem_dice_01']:.4f}")
    print(f"Stage2 MSE:    {metrics['stage2_mse']:.6f}")
    print(f"FEM MSE:       {metrics['fem_mse']:.6f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nPer-sample results saved to {out_path}")

    return metrics


if __name__ == "__main__":
    main()
