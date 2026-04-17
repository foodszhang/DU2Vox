#!/usr/bin/env python3
"""
Stage 2 Residual INR training entry point.

Supports two dataset modes:
- precomputed:  Stage2DatasetPrecomputed (fast, fork-safe, num_workers>0)
- on-demand:    Stage2Dataset (slow, FEMBridge not fork-safe, num_workers=0)
- multiview:    Stage2DatasetPrecomputedMultiview (DE + MCX projections)

Usage:
    python scripts/train_stage2.py --config configs/stage2/uniform_1000_v2.yaml
    python scripts/train_stage2.py --config configs/stage2/uniform_1000_v2.yaml --experiment_name baseline_de_only
    python scripts/train_stage2.py --config configs/stage2/full_multiview.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from du2vox.models.stage2.residual_inr import ResidualINR
from du2vox.models.stage2.stage2_dataset import (
    Stage2Dataset,
    Stage2DatasetPrecomputed,
    Stage2DatasetPrecomputedMultiview,
)


def load_split(split_file: str):
    with open(split_file) as f:
        return [l.strip() for l in f if l.strip()]


def build_dataloader(
    cfg: dict,
    sample_ids: list,
    shuffle: bool = False,
    precomputed_dir: str | None = None,
    bridge_dir: str | None = None,
) -> DataLoader:
    """Build a DataLoader. Selects dataset type based on config."""
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"].get("num_workers", 4)
    n_query = cfg["data"]["n_query_points"]

    if precomputed_dir and Path(precomputed_dir).exists():
        # Check if multiview mode is enabled
        if cfg["model"].get("view_encoder", False):
            dataset = Stage2DatasetPrecomputedMultiview(
                precomputed_dir=precomputed_dir,
                samples_dir=cfg["data"]["samples_dir"],
                sample_ids=sample_ids,
                n_query_points=n_query,
            )
        else:
            dataset = Stage2DatasetPrecomputed(
                precomputed_dir=precomputed_dir,
                sample_ids=sample_ids,
                n_query_points=n_query,
            )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None,
        )
    else:
        # Fallback to on-demand dataset
        dataset = Stage2Dataset(
            bridge_dir=bridge_dir or cfg["data"]["bridge_dir"],
            shared_dir=cfg["data"]["shared_dir"],
            samples_dir=cfg["data"]["samples_dir"],
            sample_ids=sample_ids,
            n_query_points=n_query,
            roi_padding_mm=cfg["data"]["roi_padding_mm"],
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # FEM bridge is not fork-safe
            pin_memory=True,
        )


def train_step(
    model: nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    grad_clip_norm: float = 1.0,
    view_encoder: Optional[nn.Module] = None,
    dice_weight: float = 0.0,
) -> dict:
    """
    Train one batch. Supports both DE-only and multiview modes.

    model: ResidualINR (DE-only) or ResidualINR (multiview, receives view_feat externally)
    view_encoder: ViewEncoderModule or None (DE-only mode)
    """
    coords   = batch["coords"].cuda()      # [B, N, 3] — normalized [-1,1] for INR
    prior    = batch["prior_8d"].cuda()   # [B, N, 8]
    gt       = batch["gt"].cuda()          # [B, N]
    valid    = batch["valid"].cuda()       # [B, N]

    is_multiview = "proj_imgs" in batch and "coords_world" in batch

    if is_multiview:
        coords_world = batch["coords_world"].cuda()  # [B, N, 3] — world mm for projection
        proj_imgs = batch["proj_imgs"].cuda()        # [B, 7, 1, 256, 256]
        # Phase 3: voxel-space coords for projection (preserves aspect ratio)
        coords_vox = batch.get("coords_mcx_vox_norm")
        coords_vox = coords_vox.cuda() if coords_vox is not None else None
        view_feat, visibility = view_encoder(
            proj_imgs, coords_world, coords_vox_norm=coords_vox
        )  # [B, N, view_feat_dim], [B, N, 7]
        d_hat, fem_interp, residual = model(coords, prior, view_feat)
    else:
        d_hat, fem_interp, residual = model(coords, prior)

    # Loss only on valid ROI points
    valid_mask = valid.flatten()
    if valid_mask.sum() > 0:
        pred_flat = d_hat.flatten()[valid_mask]
        gt_flat = gt.flatten()[valid_mask]

        # Final prediction = FEM baseline + residual; clamp to [0, 1] for Dice
        final_pred = torch.clamp(fem_interp.flatten()[valid_mask] + pred_flat, 0.0, 1.0)

        mse_loss = nn.functional.mse_loss(pred_flat, gt_flat)

        # Phase 4: MSE + Dice mixed loss (Dice on final clamped prediction)
        if dice_weight > 0:
            pred_bin = (final_pred >= 0.5).float()
            gt_bin = (gt_flat >= 0.5).float()
            intersection = (pred_bin * gt_bin).sum()
            dice_val = 2 * intersection / (pred_bin.sum() + gt_bin.sum() + 1e-8)
            dice_loss = 1.0 - dice_val  # scalar, higher is worse
            loss = (1 - dice_weight) * mse_loss + dice_weight * dice_loss
        else:
            loss = mse_loss
    else:
        loss = torch.tensor(0.0, device=coords.device)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    if view_encoder is not None:
        torch.nn.utils.clip_grad_norm_(view_encoder.parameters(), grad_clip_norm)
    optimizer.step()

    with torch.no_grad():
        if valid_mask.sum() > 0:
            fem_loss = nn.functional.mse_loss(
                fem_interp.flatten()[valid_mask], gt.flatten()[valid_mask]
            )
            residual_norm = residual.flatten()[valid_mask].abs().mean()
        else:
            fem_loss = torch.tensor(0.0)
            residual_norm = torch.tensor(0.0)

    return {
        "loss": loss.item(),
        "fem_baseline_loss": fem_loss.item(),
        "residual_norm": residual_norm.item(),
        "valid_count": int(valid_mask.sum()),
    }


def compute_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute Dice coefficient at a threshold."""
    pred_bin = (pred >= threshold).float()
    target_bin = (target >= threshold).float()
    intersection = (pred_bin * target_bin).sum()
    return (2 * intersection / (pred_bin.sum() + target_bin.sum() + 1e-8)).item()


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    view_encoder: Optional[nn.Module] = None,
) -> dict:
    """Validate with per-sample Dice averaging and FEM baseline comparison."""
    model.eval()
    per_sample_metrics = []
    total_loss = 0.0
    n_valid_points = 0

    with torch.no_grad():
        for batch in val_loader:
            coords  = batch["coords"].cuda()
            prior   = batch["prior_8d"].cuda()
            gt      = batch["gt"]
            valid   = batch["valid"]
            sids    = batch["sample_id"]

            B, N = coords.shape[:2]

            is_multiview = "proj_imgs" in batch and "coords_world" in batch
            if is_multiview:
                coords_world = batch["coords_world"].cuda()
                proj_imgs = batch["proj_imgs"].cuda()
                coords_vox = batch.get("coords_mcx_vox_norm")
                coords_vox = coords_vox.cuda() if coords_vox is not None else None
                view_feat, _ = view_encoder(
                    proj_imgs, coords_world, coords_vox_norm=coords_vox
                )
                d_hat, fem_interp, _ = model(coords, prior, view_feat)
            else:
                d_hat, fem_interp, _ = model(coords, prior)
            d_hat = d_hat.cpu()
            fem_interp = fem_interp.cpu()
            gt_np = gt.numpy()
            valid_np = valid.numpy()

            # Accumulate val_loss in the same loop
            v_all = valid_np.flatten()
            g_all = gt_np.flatten()
            p_all = d_hat.numpy().flatten()
            if v_all.sum() > 0:
                total_loss += nn.functional.mse_loss(
                    torch.from_numpy(p_all[v_all > 0]),
                    torch.from_numpy(g_all[v_all > 0])
                ).item() * int(v_all.sum())
                n_valid_points += int(v_all.sum())

            for b in range(B):
                v = valid_np[b]
                g = gt_np[b]
                p = d_hat[b].numpy()
                f = fem_interp[b].numpy()

                v_mask = v > 0
                if v_mask.sum() == 0:
                    continue

                stage2_d = compute_dice(
                    torch.from_numpy(p[v_mask]),
                    torch.from_numpy(g[v_mask]),
                    0.5
                )
                fem_d = compute_dice(
                    torch.from_numpy(f[v_mask]),
                    torch.from_numpy(g[v_mask]),
                    0.5
                )

                stage2_mse = float(np.mean((p[v_mask] - g[v_mask]) ** 2))
                fem_mse = float(np.mean((f[v_mask] - g[v_mask]) ** 2))

                per_sample_metrics.append({
                    "sample_id": sids[b],
                    "stage2_dice_05": stage2_d,
                    "fem_dice_05": fem_d,
                    "delta_dice_05": stage2_d - fem_d,
                    "stage2_mse": stage2_mse,
                    "fem_mse": fem_mse,
                })

    if not per_sample_metrics:
        return {
            "val_loss": 0.0, "val_valid": 0,
            "stage2_dice_05": 0.0, "fem_dice_05": 0.0, "delta_dice_05": 0.0,
            "stage2_mse": 0.0, "fem_mse": 0.0,
            "per_sample": {},
        }

    keys = ["stage2_dice_05", "fem_dice_05", "delta_dice_05", "stage2_mse", "fem_mse"]
    summary = {k: float(np.mean([m[k] for m in per_sample_metrics])) for k in keys}
    summary["per_sample"] = {m["sample_id"]: {k: m[k] for k in keys} for m in per_sample_metrics}
    summary["val_loss"] = total_loss / max(n_valid_points, 1)
    summary["val_valid"] = n_valid_points

    return summary


def main():
    parser = argparse.ArgumentParser(description="Stage 2 Residual INR training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/stage2")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    exp_name = args.experiment_name or cfg["experiment"]["name"]
    max_epochs = args.max_epochs or cfg["training"]["max_epochs"]

    # Load splits
    train_ids = load_split(cfg["data"]["train_split"])
    val_ids   = load_split(cfg["data"]["val_split"])

    if args.max_samples:
        train_ids = train_ids[: args.max_samples]
        val_ids   = val_ids[: max(1, args.max_samples // 4)]

    # Precomputed mode
    precomputed_train = cfg["data"].get("precomputed_train_dir")
    precomputed_val = cfg["data"].get("precomputed_val_dir")

    if precomputed_train:
        print(f"[Stage2] Mode: precomputed (train={precomputed_train}, val={precomputed_val})")
    else:
        print(f"[Stage2] Mode: on-demand (bridge_dir fallback)")
    print(f"[Stage2] Training: {len(train_ids)} samples, Val: {len(val_ids)} samples")

    # Build model
    view_encoder_cfg = cfg["model"].get("view_encoder", False)

    if view_encoder_cfg:
        # Multiview mode: ViewEncoderModule + ResidualINR
        from du2vox.models.stage2.view_encoder import ViewEncoderModule

        view_encoder = ViewEncoderModule(
            view_feat_dim=cfg["model"]["view_feat_dim"],
            fusion_method=cfg["model"].get("fusion_method", "attn"),
            encoder_out_channels=cfg["model"].get("encoder_out_channels", 32),
            encoder_base_channels=cfg["model"].get("encoder_base_channels", 32),
        ).cuda()

        model = ResidualINR(
            n_freqs=cfg["model"]["n_freqs"],
            hidden_dim=cfg["model"]["hidden_dim"],
            n_hidden_layers=cfg["model"]["n_hidden_layers"],
            prior_dim=cfg["model"]["prior_dim"],
            skip_connection=cfg["model"]["skip_connection"],
            view_feat_dim=cfg["model"]["view_feat_dim"],
        ).cuda()

        # Joint optimizer with separate LR for view encoder
        lr_scale = cfg["model"].get("view_encoder_lr_scale", 1.0)
        ve_lr = cfg["training"]["lr"] * lr_scale
        optimizer = torch.optim.AdamW(
            [
                {"params": model.parameters(), "lr": cfg["training"]["lr"]},
                {"params": view_encoder.parameters(), "lr": ve_lr},
            ],
            weight_decay=cfg["training"]["weight_decay"],
        )
        print(f"[Stage2] Multiview mode: view_feat_dim={cfg['model']['view_feat_dim']}, "
              f"fusion={cfg['model'].get('fusion_method', 'mean')}, ve_lr={ve_lr:.0e}")
    else:
        # DE-only mode
        view_encoder = None
        model = ResidualINR(
            n_freqs=cfg["model"]["n_freqs"],
            hidden_dim=cfg["model"]["hidden_dim"],
            n_hidden_layers=cfg["model"]["n_hidden_layers"],
            prior_dim=cfg["model"]["prior_dim"],
            skip_connection=cfg["model"]["skip_connection"],
        ).cuda()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["training"]["lr"],
            weight_decay=cfg["training"]["weight_decay"],
        )

    warmup_epochs = cfg["training"].get("warmup_epochs", 5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["scheduler"].get("T_max", max_epochs),
        eta_min=cfg["training"]["scheduler"].get("eta_min", 1e-6),
    )

    train_loader = build_dataloader(
        cfg, train_ids, shuffle=True,
        precomputed_dir=precomputed_train,
        bridge_dir=cfg["data"].get("train_bridge_dir", cfg["data"].get("bridge_dir", "")),
    )
    val_loader = build_dataloader(
        cfg, val_ids, shuffle=False,
        precomputed_dir=precomputed_val,
        bridge_dir=cfg["data"].get("val_bridge_dir", cfg["data"].get("bridge_dir", "")),
    )

    # Training loop
    log_path = Path("logs") / exp_name
    log_path.mkdir(parents=True, exist_ok=True)

    best_dice = 0.0
    best_val_loss = float("inf")
    patience  = cfg["training"].get("early_stopping_patience", 20)
    patience_counter = 0
    train_log = []

    print(f"\n{'Epoch':>5}  {'Loss':>10}  {'ValLoss':>10}  {'S2Dice':>8}  {'FemDice':>8}  {'ΔDice':>8}  {'ResNorm':>8}  {'FemMSE':>10}  {'Valid':>7}  {'Time':>6}")
    print("-" * 110)

    for epoch in range(1, max_epochs + 1):
        t0 = time.perf_counter()
        model.train()
        epoch_loss = 0.0
        epoch_fem = 0.0
        epoch_res = 0.0
        epoch_valid = 0
        n_steps = 0

        # Warmup: linear lr ramp
        if epoch <= warmup_epochs:
            lr_scale = epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = cfg["training"]["lr"] * lr_scale
        else:
            scheduler.step()

        for batch in train_loader:
            metrics = train_step(
                model, batch, optimizer,
                grad_clip_norm=cfg["training"].get("grad_clip_norm", 1.0),
                view_encoder=view_encoder,
                dice_weight=cfg["training"].get("dice_weight", 0.0),
            )
            epoch_loss += metrics["loss"]
            epoch_fem  += metrics["fem_baseline_loss"]
            epoch_res  += metrics["residual_norm"]
            epoch_valid += metrics["valid_count"]
            n_steps += 1

        val_metrics = validate(model, val_loader, view_encoder=view_encoder)
        elapsed = time.perf_counter() - t0

        avg_loss = epoch_loss / max(n_steps, 1)
        avg_fem  = epoch_fem  / max(n_steps, 1)
        avg_res  = epoch_res  / max(n_steps, 1)

        entry = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_loss": val_metrics["val_loss"],
            "stage2_dice_05": val_metrics["stage2_dice_05"],
            "fem_dice_05": val_metrics["fem_dice_05"],
            "delta_dice_05": val_metrics["delta_dice_05"],
            "stage2_mse": val_metrics["stage2_mse"],
            "fem_mse": val_metrics["fem_mse"],
            "residual_norm": avg_res,
            "valid_count": epoch_valid,
            "elapsed_s": elapsed,
            "lr": optimizer.param_groups[0]["lr"],
        }
        train_log.append(entry)

        # Log every epoch
        print(
            f"{epoch:>5}  {avg_loss:>10.6f}  {val_metrics['val_loss']:>10.6f}  "
            f"{val_metrics['stage2_dice_05']:>8.4f}  {val_metrics['fem_dice_05']:>8.4f}  "
            f"{val_metrics['delta_dice_05']:>8.4f}  "
            f"{avg_res:>8.4f}  {val_metrics['fem_mse']:>10.6f}  "
            f"{epoch_valid:>7}  {elapsed:>5.1f}s"
        )

        # Save best by stage2_dice_05
        if val_metrics["stage2_dice_05"] > best_dice:
            best_dice = val_metrics["stage2_dice_05"]
            best_val_loss = val_metrics["val_loss"]
            ckpt_path = Path(args.checkpoint_dir) / exp_name / "best.pth"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            if view_encoder is not None:
                torch.save({
                    "residual_inr": model.state_dict(),
                    "view_encoder": view_encoder.state_dict(),
                }, ckpt_path)
            else:
                torch.save(model.state_dict(), ckpt_path)
            patience_counter = 0
            print(f"  -> Best model saved (stage2_dice_05={best_dice:.4f}, val_loss={best_val_loss:.6f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience and epoch > warmup_epochs:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Save train log
    with open(log_path / "train_log.json", "w") as f:
        json.dump(train_log, f, indent=2)

    print(f"\nTraining complete. Best stage2_dice_05: {best_dice:.4f}, Best val_loss: {best_val_loss:.6f}")
    print(f"Checkpoints: {Path(args.checkpoint_dir) / exp_name / 'best.pth'}")


if __name__ == "__main__":
    main()
