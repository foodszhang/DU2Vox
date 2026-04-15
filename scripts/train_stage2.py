#!/usr/bin/env python3
"""
Stage 2 Residual INR training entry point.

Usage:
    python scripts/train_stage2.py --config configs/stage2/uniform_1000_v2.yaml

Accepts --max_samples N and --max_epochs M for quick smoke tests.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from du2vox.models.stage2.residual_inr import ResidualINR
from du2vox.models.stage2.stage2_dataset import Stage2Dataset


def load_split(split_file: str):
    with open(split_file) as f:
        return [l.strip() for l in f if l.strip()]


def build_dataloader(cfg: dict, sample_ids: list, shuffle: bool = False,
                     bridge_dir: str | None = None):
    dataset = Stage2Dataset(
        bridge_dir=bridge_dir or cfg["data"]["bridge_dir"],
        shared_dir=cfg["data"]["shared_dir"],
        samples_dir=cfg["data"]["samples_dir"],
        sample_ids=sample_ids,
        n_query_points=cfg["data"]["n_query_points"],
        roi_padding_mm=cfg["data"]["roi_padding_mm"],
    )
    return DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=shuffle,
        num_workers=0,  # FEM bridge is not fork-safe
        pin_memory=True,
    )


def train_step(model: ResidualINR, batch: dict, optimizer: torch.optim.Optimizer) -> dict:
    coords   = batch["coords"].cuda()      # [B, N, 3]
    prior    = batch["prior_8d"].cuda()   # [B, N, 8]
    gt       = batch["gt"].cuda()         # [B, N]
    valid    = batch["valid"].cuda()      # [B, N]

    d_hat, fem_interp, residual = model(coords, prior)

    # Loss only on valid ROI points
    valid_mask = valid.flatten()
    if valid_mask.sum() > 0:
        loss = nn.functional.mse_loss(d_hat.flatten()[valid_mask], gt.flatten()[valid_mask])
    else:
        loss = torch.tensor(0.0, device=coords.device)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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


def validate(model: ResidualINR, val_loader: DataLoader) -> dict:
    model.eval()
    total_loss = 0.0
    total_valid = 0
    n_batches = 0
    all_preds = []
    all_gt = []

    with torch.no_grad():
        for batch in val_loader:
            coords  = batch["coords"].cuda()
            prior   = batch["prior_8d"].cuda()
            gt      = batch["gt"].cuda()
            valid   = batch["valid"].cuda()

            d_hat, _, _ = model(coords, prior)

            valid_mask = valid.flatten()
            if valid_mask.sum() > 0:
                loss = nn.functional.mse_loss(
                    d_hat.flatten()[valid_mask], gt.flatten()[valid_mask]
                )
                total_loss += loss.item()
                total_valid += int(valid_mask.sum())
                n_batches += 1

                all_preds.append(d_hat.flatten()[valid_mask].cpu())
                all_gt.append(gt.flatten()[valid_mask].cpu())

    all_preds = torch.cat(all_preds)
    all_gt = torch.cat(all_gt)

    dice_05 = compute_dice(all_preds, all_gt, 0.5)
    dice_01 = compute_dice(all_preds, all_gt, 0.1)
    dice_08 = compute_dice(all_preds, all_gt, 0.8)

    return {
        "val_loss": total_loss / max(n_batches, 1),
        "val_valid": total_valid,
        "dice_05": dice_05,
        "dice_01": dice_01,
        "dice_08": dice_08,
    }


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

    print(f"[Stage2] Training: {len(train_ids)} samples, Val: {len(val_ids)} samples")

    # Build model
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

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["scheduler"].get("T_max", max_epochs),
        eta_min=cfg["training"]["scheduler"].get("eta_min", 1e-6),
    )

    train_bridge = cfg["data"].get("train_bridge_dir", cfg["data"].get("bridge_dir", ""))
    val_bridge = cfg["data"].get("val_bridge_dir", cfg["data"].get("bridge_dir", ""))
    train_loader = build_dataloader(cfg, train_ids, shuffle=True, bridge_dir=train_bridge)
    val_loader = build_dataloader(cfg, val_ids, shuffle=False, bridge_dir=val_bridge)

    # Training loop
    log_path = Path("logs") / exp_name
    log_path.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_dice = 0.0
    patience  = cfg["training"].get("early_stopping_patience", 20)
    patience_counter = 0
    train_log = []

    print(f"\n{'Epoch':>5}  {'Loss':>10}  {'ValLoss':>10}  {'Dice05':>8}  {'Dice01':>8}  {'Dice08':>8}  {'ResNorm':>8}  {'FemBase':>10}  {'Valid':>7}  {'Time':>6}")
    print("-" * 100)

    for epoch in range(1, max_epochs + 1):
        t0 = time.perf_counter()
        model.train()
        epoch_loss = 0.0
        epoch_fem = 0.0
        epoch_res = 0.0
        epoch_valid = 0
        n_steps = 0

        for batch in train_loader:
            metrics = train_step(model, batch, optimizer)
            epoch_loss += metrics["loss"]
            epoch_fem  += metrics["fem_baseline_loss"]
            epoch_res  += metrics["residual_norm"]
            epoch_valid += metrics["valid_count"]
            n_steps += 1

        scheduler.step()

        val_metrics = validate(model, val_loader)
        elapsed = time.perf_counter() - t0

        avg_loss = epoch_loss / max(n_steps, 1)
        avg_fem  = epoch_fem  / max(n_steps, 1)
        avg_res  = epoch_res  / max(n_steps, 1)

        entry = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_loss": val_metrics["val_loss"],
            "dice_05": val_metrics["dice_05"],
            "dice_01": val_metrics["dice_01"],
            "dice_08": val_metrics["dice_08"],
            "fem_baseline_loss": avg_fem,
            "residual_norm": avg_res,
            "valid_count": epoch_valid,
            "elapsed_s": elapsed,
        }
        train_log.append(entry)

        # Log every epoch
        print(
            f"{epoch:>5}  {avg_loss:>10.6f}  {val_metrics['val_loss']:>10.6f}  "
            f"{val_metrics['dice_05']:>8.4f}  {val_metrics['dice_01']:>8.4f}  {val_metrics['dice_08']:>8.4f}  "
            f"{avg_res:>8.4f}  {avg_fem:>10.6f}  {epoch_valid:>7}  {elapsed:>5.1f}s"
        )

        # Save best by dice_05
        if val_metrics["dice_05"] > best_dice:
            best_dice = val_metrics["dice_05"]
            best_val = val_metrics["val_loss"]
            ckpt_path = Path(args.checkpoint_dir) / exp_name / "best.pth"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            patience_counter = 0
            print(f"  -> Best model saved (dice_05={best_dice:.4f}, val_loss={best_val:.6f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Save train log
    with open(log_path / "train_log.json", "w") as f:
        json.dump(train_log, f, indent=2)

    print(f"\nTraining complete. Best val_loss: {best_val:.6f}, Best dice_05: {best_dice:.4f}")
    print(f"Checkpoints: {Path(args.checkpoint_dir) / exp_name / 'best.pth'}")


if __name__ == "__main__":
    main()
