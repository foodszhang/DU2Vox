#!/usr/bin/env python3
"""
MS-GDUN training script for FMT-SimGen.

Usage:
    python scripts/train_stage1.py --config configs/stage1/gcain_full.yaml
    python scripts/train_stage1.py --config configs/stage1/gcain_full.yaml --resume checkpoints/latest.pth
"""

import argparse
import os
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from du2vox.models.stage1.gcain import GCAIN_full
from du2vox.data.dataset import FMTSimGenDataset
from du2vox.losses.tversky import criterion
from du2vox.evaluation.metrics import evaluate_batch, summarize_metrics


def train():
    parser = argparse.ArgumentParser(description="MS-GDUN training for FMT-SimGen")
    parser.add_argument(
        "--config", type=str, default="configs/stage1/gcain_full.yaml",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    paths_cfg = cfg["paths"]

    shared_dir = Path(paths_cfg["shared_dir"])
    samples_dir = Path(paths_cfg["samples_dir"])
    splits_dir = Path(paths_cfg["splits_dir"])
    checkpoint_dir = Path(paths_cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    train_set = FMTSimGenDataset(
        shared_dir=shared_dir,
        samples_dir=samples_dir,
        split_file=splits_dir / "train.txt",
    )
    val_set = FMTSimGenDataset(
        shared_dir=shared_dir,
        samples_dir=samples_dir,
        split_file=splits_dir / "val.txt",
    )

    train_loader = DataLoader(train_set, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=train_cfg["batch_size"], shuffle=False)

    # Use actual dimensions from loaded mesh (not config)
    n_nodes = train_set.nodes.shape[0]
    n_surface = train_set.A.shape[0]
    print(f"Train: {len(train_set)} samples, Val: {len(val_set)} samples")
    print(f"Shared assets: {n_nodes} nodes, {n_surface} surface nodes")

    # Verify config matches
    cfg_n_nodes = model_cfg.get("n_nodes", n_nodes)
    if cfg_n_nodes != n_nodes:
        print(f"WARNING: config n_nodes={cfg_n_nodes} != actual {n_nodes}, using actual")
    cfg_n_surface = model_cfg.get("n_surface", n_surface)
    if cfg_n_surface != n_surface:
        print(f"WARNING: config n_surface={cfg_n_surface} != actual {n_surface}, using actual")

    # Move shared assets to GPU
    A = train_set.A.cuda()           # [S, N]
    L = train_set.L.cuda()           # [N, N]
    L0 = train_set.L0.cuda()
    L1 = train_set.L1.cuda()
    L2 = train_set.L2.cuda()
    L3 = train_set.L3.cuda()
    knn_idx = train_set.knn_idx.cuda()
    sens_w = train_set.sens_w.cuda()
    nodes = train_set.nodes.cuda()

    print(f"A GPU memory: {A.element_size() * A.nelement() / 1e6:.1f} MB")
    print(f"L (dense) GPU memory: {L.element_size() * L.nelement() / 1e6:.1f} MB")

    # Model
    model = GCAIN_full(
        L=L,
        A=A,
        L0=L0,
        L1=L1,
        L2=L2,
        L3=L3,
        knn_idx=knn_idx,
        sens_w=sens_w,
        num_layer=model_cfg["num_layer"],
        feat_dim=model_cfg["feat_dim"],
    ).cuda()

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=train_cfg["scheduler_patience"],
        factor=0.5,
    )

    # Resume from checkpoint
    start_epoch = 1
    best_val_loss = float("inf")
    best_dice = 0.0
    epochs_without_improvement = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cuda")
        # Backward compatibility: old checkpoints are plain state_dict (OrderedDict)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if ckpt.get("scheduler_state_dict") and scheduler:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            best_dice = ckpt.get("best_dice", 0.0)
            epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)
        else:
            # Plain state_dict (old format)
            model.load_state_dict(ckpt)
            start_epoch = 1
            print("  (old checkpoint format: starting from epoch 1)")
        print(f"Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}, best_dice={best_dice:.4f}")
    else:
        print("Starting training from scratch")

    for epoch in range(start_epoch, train_cfg["max_epochs"] + 1):
        # Training
        model.train()
        train_losses = []
        for batch in train_loader:
            b = batch["b"].cuda()    # [B, S, 1]
            gt = batch["gt"].cuda() # [B, N, 1]

            X0 = torch.zeros(b.size(0), n_nodes, 1, device="cuda")
            pred = model(X0, b)  # [B, N, 1]
            pred = torch.clamp(pred, min=0.0, max=1.0)

            loss = criterion(pred, gt, nodes)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        train_loss_mean = sum(train_losses) / len(train_losses)

        # Validation
        model.eval()
        val_metrics = []
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                b = batch["b"].cuda()
                gt = batch["gt"].cuda()
                X0 = torch.zeros(b.size(0), n_nodes, 1, device="cuda")
                pred = model(X0, b)
                pred = torch.clamp(pred, min=0.0, max=1.0)
                loss = criterion(pred, gt, nodes)
                val_losses.append(loss.item())
                val_metrics.append(evaluate_batch(pred, gt, nodes))

        val_loss_mean = sum(val_losses) / len(val_losses)
        val_summary = summarize_metrics(val_metrics)
        current_dice = val_summary.get("dice_bin_0.3", 0.0)

        scheduler.step(val_loss_mean)

        # Early stopping
        early_stop = train_cfg.get("early_stopping", {})
        patience = early_stop.get("patience", 0)
        if patience > 0:
            monitor = early_stop.get("monitor", "val_dice_03")
            current_val = current_dice if "dice" in monitor else val_loss_mean
            if current_val < best_dice if "dice" in monitor else current_val > best_val_loss:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {patience} epochs)")
                    break
            else:
                epochs_without_improvement = 0

        # Every 10 epochs or last epoch: full metrics
        if epoch % 10 == 0 or epoch == train_cfg["max_epochs"]:
            print(
                f"Epoch {epoch:03d}/{train_cfg['max_epochs']} | "
                f"Train: {train_loss_mean:.4f} | "
                f"Val: {val_loss_mean:.4f} | "
                f"Dice: {val_summary['dice']:.4f} | "
                f"Dice_bin@0.3: {current_dice:.4f} | "
                f"Recall@0.3: {val_summary.get('recall_0.3', 0.0):.4f} | "
                f"Prec@0.3: {val_summary.get('precision_0.3', 0.0):.4f} | "
                f"Dice_bin@0.1: {val_summary.get('dice_bin_0.1', 0.0):.4f} | "
                f"Recall@0.1: {val_summary.get('recall_0.1', 0.0):.4f} | "
                f"Prec@0.1: {val_summary.get('precision_0.1', 0.0):.4f} | "
                f"frac>0.1: {val_summary.get('pred_frac_0.1', 0.0):.4f}"
            )
        else:
            print(
                f"Epoch {epoch:03d}/{train_cfg['max_epochs']} | "
                f"Train: {train_loss_mean:.4f} | "
                f"Val: {val_loss_mean:.4f} | "
                f"Dice: {val_summary['dice']:.4f} | "
                f"Dice_bin@0.3: {current_dice:.4f} | "
                f"Recall@0.1: {val_summary.get('recall_0.1', 0.0):.4f}"
            )

        # Save latest checkpoint every epoch (overwrite)
        latest_path = checkpoint_dir / "latest.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "best_dice": best_dice,
            "epochs_without_improvement": epochs_without_improvement,
        }, latest_path)

        # Save best checkpoint when Dice improves
        if current_dice > best_dice:
            best_dice = current_dice
            best_val_loss = val_loss_mean
            ckpt_path = checkpoint_dir / "best.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "best_dice": best_dice,
                "epochs_without_improvement": epochs_without_improvement,
            }, ckpt_path)
            print(f"  -> Saved best model (Dice={current_dice:.4f}) to {ckpt_path}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}, Best Dice_bin@0.3: {best_dice:.4f}")


if __name__ == "__main__":
    train()
