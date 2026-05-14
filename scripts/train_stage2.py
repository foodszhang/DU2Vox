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
from du2vox.models.stage2.cqr_residual_inr import CQRResidualINR
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
    deterministic: bool = False,
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
                shared_dir=cfg["data"].get("shared_dir"),
                deterministic=deterministic,
            )
        else:
            dataset = Stage2DatasetPrecomputed(
                precomputed_dir=precomputed_dir,
                sample_ids=sample_ids,
                n_query_points=n_query,
                deterministic=deterministic,
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
    loss_type: str = "gisc",
) -> dict:
    """
    Train one batch. Supports both DE-only and multiview modes.

    model: ResidualINR (DE-only) or ResidualINR (multiview, receives view_feat externally)
    view_encoder: ViewEncoderModule or None (DE-only mode)

    loss_type: "gisc" (weighted BCE + sparse + Focal Tversky) or "soft_dice" (pure soft Dice).
    """
    coords = batch["coords"].cuda()  # [B, N, 3] — normalized [-1,1] for INR
    prior = batch.get("prior_ext", batch["prior_8d"]).cuda()
    gt = batch["gt"].cuda()  # [B, N]
    valid = batch["valid"].cuda()  # [B, N]

    is_multiview = "proj_imgs" in batch and "coords_world" in batch

    if is_multiview:
        coords_world = batch["coords_world"].cuda()  # [B, N, 3] — world mm for projection
        proj_imgs = batch["proj_imgs"].cuda()  # [B, 7, 1, 256, 256]
        # Phase 3: voxel-space coords for projection (preserves aspect ratio)
        coords_vox = batch.get("coords_mcx_vox_norm")
        coords_vox = coords_vox.cuda() if coords_vox is not None else None
        view_feat, visibility = view_encoder(
            proj_imgs, coords_world, coords_vox_norm=coords_vox
        )  # [B, N, view_feat_dim], [B, N, 7]
        # B4: apply mcx_valid mask to zero out view features for points outside MCX volume
        if "mcx_valid" in batch:
            mcx_valid = batch["mcx_valid"].cuda()  # [B, N]
            view_feat = view_feat * mcx_valid.unsqueeze(-1).float()
        d_hat, fem_interp, residual = model(coords, prior, view_feat)
    else:
        d_hat, fem_interp, residual = model(coords, prior)

    # Loss only on valid ROI points
    valid_mask = valid.flatten()
    if valid_mask.sum() > 0:
        pred_flat = d_hat.flatten()[valid_mask]
        gt_flat = gt.flatten()[valid_mask]

        # d_hat already includes fem_interp + residual, no need to add again
        final_pred = torch.clamp(pred_flat, 0.0, 1.0)

        p = final_pred.clamp(1e-6, 1 - 1e-6)
        eps = 1e-6

        if loss_type == "soft_dice":
            # Pure soft Dice loss: 1 - 2*TP/(P+G)
            g = gt_flat.clamp(eps, 1 - eps)
            TP = (p * g).sum()
            dice_loss = 1 - 2 * TP / (p.sum() + g.sum() + eps)
            loss = dice_loss
            loss_components = {
                "dice": dice_loss.detach(),
                "bce": torch.tensor(0.0),
                "sparse": torch.tensor(0.0),
                "focal_tv": torch.tensor(0.0),
            }
        else:
            # loss_type: "focal" | "mse" | "asym_tversky" | "focal_v3"
            g = (gt_flat >= 0.5).float()
            p_t = p.clamp(eps, 1 - eps)

            if loss_type == "mse":
                # Pure MSE on residual targets — baseline to test residual learning
                target = torch.clamp(gt_flat, 0.0, 1.0)
                mse_loss = ((p_t - target) ** 2).mean()
                loss = mse_loss
                loss_components = {
                    "dice": mse_loss.detach(),
                    "focal": torch.tensor(0.0),
                    "bce": torch.tensor(0.0),
                    "sparse": torch.tensor(0.0),
                    "focal_tv": torch.tensor(0.0),
                }

            elif loss_type == "mse_support":
                # CQR-friendly hybrid loss:
                #   MSE keeps relative-intensity regression,
                #   BCE protects the 0.5 support boundary,
                #   residual L2 prevents aggressive correction away from FEM prior.
                target = torch.clamp(gt_flat, 0.0, 1.0)
                mse_loss = ((p_t - target) ** 2).mean()

                g_bin = (gt_flat >= 0.5).float()
                p_prob = p_t.clamp(eps, 1 - eps)

                pos = g_bin.sum()
                neg = (1.0 - g_bin).sum()
                pos_weight = (neg / (pos + eps)).clamp(1.0, 20.0)

                point_weight = torch.where(
                    g_bin > 0.5,
                    pos_weight,
                    torch.ones_like(g_bin),
                )

                bce_loss = torch.nn.functional.binary_cross_entropy(
                    p_prob,
                    g_bin,
                    weight=point_weight,
                    reduction="mean",
                )

                res_l2_loss = (residual.flatten()[valid_mask] ** 2).mean()

                loss = mse_loss + 0.05 * bce_loss + 0.01 * res_l2_loss

                loss_components = {
                    "dice": mse_loss.detach(),
                    "focal": bce_loss.detach(),
                    "bce": bce_loss.detach(),
                    "sparse": torch.tensor(0.0),
                    "focal_tv": res_l2_loss.detach(),
                }

            elif loss_type == "asym_tversky":
                # Asymmetric Tversky: alpha=0.3 (FN penalty) < beta=0.7 (FP penalty)
                # This penalizes missing tumor more than false positives
                alpha, beta = 0.3, 0.7
                tp = (p_t * g).sum()
                fn = ((1 - p_t) * g).sum()
                fp = (p_t * (1 - g)).sum()
                tversky = 1 - tp / (tp + alpha * fn + beta * fp + eps)
                # Combine with light MSE for smooth gradients
                mse_term = ((p_t - gt_flat) ** 2).mean() * 0.1
                loss = tversky + mse_term
                loss_components = {
                    "dice": tversky.detach(),
                    "focal": mse_term.detach(),
                    "bce": torch.tensor(0.0),
                    "sparse": torch.tensor(0.0),
                    "focal_tv": torch.tensor(0.0),
                }
            elif loss_type == "focal_v3":
                # Focal loss v3: gamma=1.5 + lighter sparse + residual L2 reg
                pt = torch.where(g > 0.5, p_t, 1 - p_t)
                focal_weight = (1 - pt) ** 1.5
                bce_raw = -torch.log(pt.clamp(eps, 1 - eps))
                focal_loss = (focal_weight * bce_raw).mean()
                sparse_loss = 0.002 * (p_t * (1 - g)).mean()
                res_l2_loss = 0.01 * (residual.flatten()[valid_mask] ** 2).mean()
                loss = focal_loss + sparse_loss + res_l2_loss
                loss_components = {
                    "dice": torch.tensor(0.0),
                    "focal": focal_loss.detach(),
                    "bce": torch.tensor(0.0),
                    "sparse": sparse_loss.detach(),
                    "focal_tv": res_l2_loss.detach(),
                }
            else:
                # Original focal loss (loss_type == "focal"): gamma=2.0
                pt = torch.where(g > 0.5, p_t, 1 - p_t)
                focal_weight = (1 - pt) ** 2.0
                bce_raw = -torch.log(pt.clamp(eps, 1 - eps))
                focal_loss = (focal_weight * bce_raw).mean()
                sparse_loss = 0.01 * (p_t * (1 - g)).mean()
                loss = focal_loss + sparse_loss
                loss_components = {
                    "dice": torch.tensor(0.0),
                    "focal": focal_loss.detach(),
                    "bce": torch.tensor(0.0),
                    "sparse": sparse_loss.detach(),
                    "focal_tv": torch.tensor(0.0),
                }
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
        "bce": loss_components.get("bce", torch.tensor(0.0)).item()
        if valid_mask.sum() > 0
        else 0.0,
        "dice": loss_components.get("dice", torch.tensor(0.0)).item()
        if valid_mask.sum() > 0
        else 0.0,
        "sparse": loss_components.get("sparse", torch.tensor(0.0)).item()
        if valid_mask.sum() > 0
        else 0.0,
        "focal_tv": loss_components.get("focal_tv", torch.tensor(0.0)).item()
        if valid_mask.sum() > 0
        else 0.0,
        "focal": loss_components.get("focal", torch.tensor(0.0)).item()
        if valid_mask.sum() > 0
        else 0.0,
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
            coords = batch["coords"].cuda()
            prior = batch.get("prior_ext", batch["prior_8d"]).cuda()
            gt = batch["gt"]
            valid = batch["valid"]
            sids = batch["sample_id"]

            B, N = coords.shape[:2]

            is_multiview = "proj_imgs" in batch and "coords_world" in batch
            if is_multiview:
                coords_world = batch["coords_world"].cuda()
                proj_imgs = batch["proj_imgs"].cuda()
                coords_vox = batch.get("coords_mcx_vox_norm")
                coords_vox = coords_vox.cuda() if coords_vox is not None else None
                view_feat, _ = view_encoder(proj_imgs, coords_world, coords_vox_norm=coords_vox)
                # B4: apply mcx_valid mask
                if "mcx_valid" in batch:
                    mcx_valid = batch["mcx_valid"].cuda()
                    view_feat = view_feat * mcx_valid.unsqueeze(-1).float()
                d_hat, fem_interp, _ = model(coords, prior, view_feat)
            else:
                d_hat, fem_interp, _ = model(coords, prior)
            d_hat = d_hat.cpu()
            fem_interp = fem_interp.cpu()
            gt_np = gt.numpy()
            valid_np = valid.numpy()

            # Accumulate val_loss in the same loop (clamp to [0,1] per B2)
            p_all = np.clip(d_hat.numpy(), 0.0, 1.0).flatten()
            v_all = valid_np.flatten()
            g_all = gt_np.flatten()
            if v_all.sum() > 0:
                total_loss += nn.functional.mse_loss(
                    torch.from_numpy(p_all[v_all > 0]), torch.from_numpy(g_all[v_all > 0])
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
                    torch.from_numpy(p[v_mask]), torch.from_numpy(g[v_mask]), 0.5
                )
                fem_d = compute_dice(torch.from_numpy(f[v_mask]), torch.from_numpy(g[v_mask]), 0.5)

                p_clipped = np.clip(p, 0.0, 1.0)
                stage2_mse = float(np.mean((p_clipped[v_mask] - g[v_mask]) ** 2))
                fem_mse = float(np.mean((f[v_mask] - g[v_mask]) ** 2))

                per_sample_metrics.append(
                    {
                        "sample_id": sids[b],
                        "stage2_dice_05": stage2_d,
                        "fem_dice_05": fem_d,
                        "delta_dice_05": stage2_d - fem_d,
                        "stage2_mse": stage2_mse,
                        "fem_mse": fem_mse,
                    }
                )

    if not per_sample_metrics:
        return {
            "val_loss": 0.0,
            "val_valid": 0,
            "stage2_dice_05": 0.0,
            "fem_dice_05": 0.0,
            "delta_dice_05": 0.0,
            "stage2_mse": 0.0,
            "fem_mse": 0.0,
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
    val_ids = load_split(cfg["data"]["val_split"])

    if args.max_samples:
        train_ids = train_ids[: args.max_samples]
        val_ids = val_ids[: max(1, args.max_samples // 4)]

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
    prior_dim = int(cfg["model"].get("prior_dim", 8))
    model_type = cfg["model"].get("model_type", "")
    use_cqr_model = (model_type == "cqr_residual_inr") or (prior_dim > 8)
    ModelCls = CQRResidualINR if use_cqr_model else ResidualINR

    if view_encoder_cfg:
        # Multiview mode: ViewEncoderModule + ResidualINR
        from du2vox.models.stage2.view_encoder import ViewEncoderModule

        view_encoder = ViewEncoderModule(
            view_feat_dim=cfg["model"]["view_feat_dim"],
            fusion_method=cfg["model"].get("fusion_method", "attn"),
            encoder_out_channels=cfg["model"].get("encoder_out_channels", 32),
            encoder_base_channels=cfg["model"].get("encoder_base_channels", 32),
        ).cuda()

        model = ModelCls(
            n_freqs=cfg["model"]["n_freqs"],
            hidden_dim=cfg["model"]["hidden_dim"],
            n_hidden_layers=cfg["model"]["n_hidden_layers"],
            prior_dim=prior_dim,
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
        print(
            f"[Stage2] Multiview mode: view_feat_dim={cfg['model']['view_feat_dim']}, "
            f"fusion={cfg['model'].get('fusion_method', 'mean')}, ve_lr={ve_lr:.0e}"
        )
    else:
        # DE-only mode
        view_encoder = None
        model = ModelCls(
            n_freqs=cfg["model"]["n_freqs"],
            hidden_dim=cfg["model"]["hidden_dim"],
            n_hidden_layers=cfg["model"]["n_hidden_layers"],
            prior_dim=prior_dim,
            skip_connection=cfg["model"]["skip_connection"],
        ).cuda()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["training"]["lr"],
            weight_decay=cfg["training"]["weight_decay"],
        )

    warmup_epochs = cfg["training"].get("warmup_epochs", 5)
    loss_type = cfg.get("loss", {}).get("type", "gisc")  # "gisc" or "soft_dice"
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["scheduler"].get("T_max", max_epochs),
        eta_min=cfg["training"]["scheduler"].get("eta_min", 1e-6),
    )

    train_loader = build_dataloader(
        cfg,
        train_ids,
        shuffle=True,
        precomputed_dir=precomputed_train,
        bridge_dir=cfg["data"].get("train_bridge_dir", cfg["data"].get("bridge_dir", "")),
    )
    val_loader = build_dataloader(
        cfg,
        val_ids,
        shuffle=False,
        precomputed_dir=precomputed_val,
        bridge_dir=cfg["data"].get("val_bridge_dir", cfg["data"].get("bridge_dir", "")),
        deterministic=True,
    )

    # Training loop
    log_path = Path("logs") / exp_name
    log_path.mkdir(parents=True, exist_ok=True)

    best_delta = -float("inf")  # Allow negative — training may degrade, delta tells us how much
    best_ckpt_info = None  # {epoch, stage2_dice_05, fem_dice_05, delta_dice_05}
    best_val_loss = float("inf")
    patience = cfg["training"].get("early_stopping_patience", 20)
    patience_counter = 0
    train_log = []

    print(
        f"\n{'Epoch':>5}  {'Loss':>10}  {'ValLoss':>10}  {'S2Dice':>8}  {'FemDice':>8}  {'ΔDice':>8}  {'ResNorm':>8}  {'FemMSE':>10}  {'Valid':>7}  {'Time':>6}"
    )
    print("-" * 100)

    for epoch in range(1, max_epochs + 1):
        t0 = time.perf_counter()
        model.train()
        epoch_loss = 0.0
        epoch_fem = 0.0
        epoch_res = 0.0
        epoch_bce = 0.0
        epoch_sparse = 0.0
        epoch_focal_tv = 0.0
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
                model,
                batch,
                optimizer,
                grad_clip_norm=cfg["training"].get("grad_clip_norm", 1.0),
                view_encoder=view_encoder,
                loss_type=loss_type,
            )
            epoch_loss += metrics["loss"]
            epoch_fem += metrics["fem_baseline_loss"]
            epoch_res += metrics["residual_norm"]
            epoch_bce += metrics["bce"]
            epoch_sparse += metrics["sparse"]
            epoch_focal_tv += metrics["focal_tv"]
            epoch_valid += metrics["valid_count"]
            n_steps += 1

        val_metrics = validate(model, val_loader, view_encoder=view_encoder)
        elapsed = time.perf_counter() - t0

        avg_loss = epoch_loss / max(n_steps, 1)
        avg_fem = epoch_fem / max(n_steps, 1)
        avg_res = epoch_res / max(n_steps, 1)
        avg_bce = epoch_bce / max(n_steps, 1)
        avg_sparse = epoch_sparse / max(n_steps, 1)
        avg_focal_tv = epoch_focal_tv / max(n_steps, 1)

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
            "bce": avg_bce,
            "sparse": avg_sparse,
            "focal_tv": avg_focal_tv,
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
        print(f"         bce={avg_bce:.4f}  sp={avg_sparse:.4f}  ft={avg_focal_tv:.4f}")

        # Save best by delta_dice_05 (relative improvement over FEM baseline)
        # Noise tolerance: only save if delta improved by > 0.0005
        delta = val_metrics["delta_dice_05"]
        improved = delta > best_delta + 0.0005
        if improved:
            best_delta = delta
            best_ckpt_info = {
                "epoch": epoch,
                "stage2_dice_05": val_metrics["stage2_dice_05"],
                "fem_dice_05": val_metrics["fem_dice_05"],
                "delta_dice_05": delta,
            }
            best_val_loss = val_metrics["val_loss"]
            ckpt_path = Path(args.checkpoint_dir) / exp_name / "best.pth"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            if view_encoder is not None:
                torch.save(
                    {
                        "residual_inr": model.state_dict(),
                        "view_encoder": view_encoder.state_dict(),
                    },
                    ckpt_path,
                )
            else:
                torch.save(model.state_dict(), ckpt_path)
            patience_counter = 0
            print(
                f"  -> Best ckpt saved: ΔDice={delta:+.4f} (S2={val_metrics['stage2_dice_05']:.4f} vs FEM={val_metrics['fem_dice_05']:.4f})"
            )
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience and epoch > warmup_epochs:
            print(
                f"\nEarly stopping at epoch {epoch} (best ΔDice={best_delta:+.4f} at ep={best_ckpt_info['epoch']})"
            )
            break

    # Save train log
    with open(log_path / "train_log.json", "w") as f:
        json.dump(train_log, f, indent=2)

    print(
        f"\nTraining complete. Best ΔDice@0.5: {best_delta:+.4f} (ep={best_ckpt_info['epoch']}, S2={best_ckpt_info['stage2_dice_05']:.4f}, FEM={best_ckpt_info['fem_dice_05']:.4f})"
    )
    print(f"Best val_loss: {best_val_loss:.6f}")
    print(f"Checkpoints: {Path(args.checkpoint_dir) / exp_name / 'best.pth'}")


if __name__ == "__main__":
    main()
