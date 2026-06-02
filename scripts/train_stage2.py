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
import os
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


def build_stage2_model(ModelCls, cfg: dict, prior_dim: int, view_feat_dim: int = 0) -> nn.Module:
    cqr_ratios = cfg.get("cqr", {}).get("ratios", {}) or {}
    default_num_bands = 5 if float(cqr_ratios.get("proposal", 0.0)) > 0.0 else 4
    kwargs = dict(
        n_freqs=cfg["model"]["n_freqs"],
        hidden_dim=cfg["model"]["hidden_dim"],
        n_hidden_layers=cfg["model"]["n_hidden_layers"],
        prior_dim=prior_dim,
        skip_connection=cfg["model"]["skip_connection"],
        view_feat_dim=view_feat_dim,
    )
    if ModelCls is CQRResidualINR:
        kwargs["residual_scale"] = cfg["model"].get("residual_scale", 1.0)
        kwargs["support_head"] = cfg["model"].get("support_head", False)
        kwargs["use_prolongation_adapter"] = cfg["model"].get("use_prolongation_adapter", False)
        kwargs["prolongation_feat_dim"] = cfg["model"].get("prolongation_feat_dim", 32)
        kwargs["use_lifting_adapter"] = cfg["model"].get("use_lifting_adapter", False)
        kwargs["lifting_feat_dim"] = cfg["model"].get("lifting_feat_dim", 32)
        kwargs["use_band_embedding"] = cfg["model"].get("use_band_embedding", False)
        kwargs["band_embed_dim"] = cfg["model"].get("band_embed_dim", 8)
        kwargs["num_bands"] = cfg["model"].get("num_bands", default_num_bands)
    return ModelCls(**kwargs)


def unpack_model_output(output):
    if isinstance(output, dict):
        return output
    d_hat, fem_interp, residual = output
    return {"d_hat": d_hat, "fem_interp": fem_interp, "residual": residual}


def select_stage2_prediction(output: dict, cfg: dict) -> torch.Tensor:
    output_mode = cfg["model"].get("output_mode", "residual")
    if output_mode == "residual":
        return output["d_hat"]
    if output_mode == "support":
        if "support_prob" not in output:
            raise ValueError("output_mode=support requires model.support_head=true")
        return output["support_prob"]
    if output_mode == "hybrid":
        if "support_prob" not in output:
            raise ValueError("output_mode=hybrid requires model.support_head=true")
        alpha = float(cfg["model"].get("hybrid_alpha", 0.5))
        return alpha * output["support_prob"] + (1.0 - alpha) * output["d_hat"].clamp(0.0, 1.0)
    raise ValueError(f"Unknown output_mode: {output_mode}")


def select_prior(batch: dict, prior_source: str, expected_dim: int) -> torch.Tensor:
    if prior_source == "prior_8d":
        prior = batch["prior_8d"]
    elif prior_source == "prior_ext":
        prior = batch["prior_ext"]
    elif prior_source == "prior_lift":
        prior = batch["prior_lift"]
    elif prior_source == "prior_prolong":
        prior = batch["prior_prolong"]
    else:
        raise ValueError(f"Unknown prior_source: {prior_source}")

    if prior.shape[-1] != expected_dim:
        raise ValueError(
            f"prior_source={prior_source} produced dim={prior.shape[-1]}, expected={expected_dim}"
        )
    return prior.cuda()


def load_split(split_file: str):
    with open(split_file) as f:
        return [line.strip() for line in f if line.strip()]


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
        resample_train = bool(cfg["data"].get("resample_queries_each_epoch", False)) and not deterministic
        # Check if multiview mode is enabled
        if cfg["model"].get("view_encoder", False):
            dataset = Stage2DatasetPrecomputedMultiview(
                precomputed_dir=precomputed_dir,
                samples_dir=cfg["data"]["samples_dir"],
                sample_ids=sample_ids,
                n_query_points=n_query,
                shared_dir=cfg["data"].get("shared_dir"),
                deterministic=deterministic,
                resample_queries_each_epoch=resample_train,
                query_epoch_seed_stride=cfg["data"].get("query_epoch_seed_stride", 1000003),
                base_seed=cfg["data"].get("query_base_seed", 0),
                projection_file=cfg["data"].get("projection_file", "proj.npz"),
                fallback_projection_file=cfg["data"].get("fallback_projection_file"),
                projection_norm=cfg["data"].get("projection_norm", "none"),
                projection_eps=cfg["data"].get("projection_eps", 1.0e-8),
                projection_transform=cfg["data"].get("projection_transform", "none"),
            )
        else:
            dataset = Stage2DatasetPrecomputed(
                precomputed_dir=precomputed_dir,
                sample_ids=sample_ids,
                n_query_points=n_query,
                deterministic=deterministic,
                resample_queries_each_epoch=resample_train,
                query_epoch_seed_stride=cfg["data"].get("query_epoch_seed_stride", 1000003),
                base_seed=cfg["data"].get("query_base_seed", 0),
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
    scaler: torch.amp.GradScaler,
    grad_clip_norm: float = 1.0,
    view_encoder: Optional[nn.Module] = None,
    loss_type: str = "gisc",
    loss_cfg: dict | None = None,
    model_cfg: dict | None = None,
    prior_source: str = "prior_ext",
    expected_prior_dim: int = 8,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    accumulation_steps: int = 1,
    step_optimizer: bool = True,
) -> dict:
    """
    Train one batch. Supports both DE-only and multiview modes.

    model: ResidualINR (DE-only) or ResidualINR (multiview, receives view_feat externally)
    view_encoder: ViewEncoderModule or None (DE-only mode)

    loss_type: "gisc" (weighted BCE + sparse + Focal Tversky) or "soft_dice" (pure soft Dice).
    """
    coords = batch["coords"].cuda()  # [B, N, 3] — normalized [-1,1] for INR
    prior = select_prior(batch, prior_source, expected_prior_dim)
    correction_band = batch.get("correction_band")
    correction_band = correction_band.cuda() if correction_band is not None else None
    gt = batch["gt"].cuda()  # [B, N]
    valid = batch["valid"].cuda()  # [B, N]

    is_multiview = "proj_imgs" in batch and "coords_world" in batch
    freeze_view_encoder = bool((model_cfg or {}).get("freeze_view_encoder", False))
    loss_components = {
        "dice": torch.tensor(0.0, device=coords.device),
        "focal": torch.tensor(0.0, device=coords.device),
        "bce": torch.tensor(0.0, device=coords.device),
        "sparse": torch.tensor(0.0, device=coords.device),
        "focal_tv": torch.tensor(0.0, device=coords.device),
    }

    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
        if is_multiview:
            coords_world = batch["coords_world"].cuda()  # [B, N, 3] — world mm for projection
            proj_imgs = batch["proj_imgs"].cuda()  # [B, 7, 1, 256, 256]
            # Phase 3: voxel-space coords for projection (preserves aspect ratio)
            coords_vox = batch.get("coords_mcx_vox_norm")
            coords_vox = coords_vox.cuda() if coords_vox is not None else None
            if freeze_view_encoder:
                with torch.no_grad():
                    view_feat, visibility = view_encoder(
                        proj_imgs, coords_world, coords_vox_norm=coords_vox
                    )
            else:
                view_feat, visibility = view_encoder(
                    proj_imgs, coords_world, coords_vox_norm=coords_vox
                )  # [B, N, view_feat_dim], [B, N, 7]
            # B4: apply mcx_valid mask to zero out view features for points outside MCX volume
            if "mcx_valid" in batch:
                mcx_valid = batch["mcx_valid"].cuda()  # [B, N]
                view_feat = view_feat * mcx_valid.unsqueeze(-1).float()
            output = unpack_model_output(model(coords, prior, view_feat, correction_band=correction_band))
        else:
            output = unpack_model_output(model(coords, prior, correction_band=correction_band))
        d_hat = output["d_hat"]
        fem_interp = output["fem_interp"]
        residual = output["residual"]

        # Loss only on valid ROI points
        valid_mask = valid.flatten()
        if valid_mask.sum() > 0:
            if model_cfg is not None:
                pred_for_dice = select_stage2_prediction(output, {"model": model_cfg})
            else:
                pred_for_dice = d_hat
            pred_flat = pred_for_dice.flatten()[valid_mask]
            gt_flat = gt.flatten()[valid_mask]
            point_weight = torch.ones_like(gt_flat)
            if "query_weight" in batch:
                qw = batch["query_weight"].cuda().flatten()[valid_mask]
                point_weight = point_weight * qw
            if "residual_indicator" in batch:
                ri = batch["residual_indicator"].cuda().flatten()[valid_mask]
                residual_weight = float((loss_cfg or {}).get("residual_indicator_weight", 0.5))
                point_weight = point_weight * (1.0 + residual_weight * ri)
            point_weight = point_weight / (point_weight.mean().detach() + 1e-6)

            # d_hat already includes fem_interp + residual, no need to add again
            final_pred = torch.clamp(pred_flat, 0.0, 1.0)

            p = final_pred.clamp(1e-6, 1 - 1e-6)
            eps = 1e-6

            if loss_type == "hybrid_support":
                if "support_logit" not in output:
                    raise ValueError("loss.type=hybrid_support requires model.support_head=true")
                loss_cfg = loss_cfg or {}
                support_threshold = float(loss_cfg.get("support_threshold", 0.5))
                gt_bin = (gt_flat >= support_threshold).float()
                support_logit = output["support_logit"].flatten()[valid_mask]
                support_prob = output["support_prob"].flatten()[valid_mask]
                mse_loss = nn.functional.mse_loss(
                    d_hat.flatten()[valid_mask].clamp(0.0, 1.0),
                    gt_flat.clamp(0.0, 1.0),
                )
                bce_loss = nn.functional.binary_cross_entropy_with_logits(support_logit, gt_bin)
                dice_loss = 1.0 - (2.0 * (support_prob * gt_bin).sum() + eps) / (
                    support_prob.sum() + gt_bin.sum() + eps
                )
                loss = (
                    float(loss_cfg.get("mse_weight", 0.3)) * mse_loss
                    + float(loss_cfg.get("bce_weight", 0.3)) * bce_loss
                    + float(loss_cfg.get("dice_weight", 0.4)) * dice_loss
                )
                loss_components = {
                    "dice": dice_loss.detach(),
                    "bce": bce_loss.detach(),
                    "sparse": torch.tensor(0.0),
                    "focal_tv": mse_loss.detach(),
                    "focal": torch.tensor(0.0),
                }
            elif loss_type == "soft_dice":
                g = gt_flat.clamp(eps, 1 - eps)
                tp = (p * g).sum()
                dice_loss = 1.0 - 2.0 * tp / (p.sum() + g.sum() + eps)
                loss = dice_loss
                loss_components = {
                    "dice": dice_loss.detach(),
                    "bce": torch.tensor(0.0),
                    "sparse": torch.tensor(0.0),
                    "focal_tv": torch.tensor(0.0),
                }
            else:
                # loss_type: "focal" | "mse" | "asym_tversky" | "focal_v3" | "dice_mse"
                g = (gt_flat >= 0.5).float()
                p_t = p.clamp(eps, 1 - eps)

                if loss_type == "dice_mse":
                    target = torch.clamp(gt_flat, 0.0, 1.0)
                    target_soft = target.clamp(eps, 1.0 - eps)
                    tp = (p_t * target_soft).sum()
                    dice_loss = 1.0 - 2.0 * tp / (p_t.sum() + target_soft.sum() + eps)
                    mse_loss = ((p_t - target) ** 2).mean()
                    res_l2_loss = (residual.flatten()[valid_mask] ** 2).mean()
                    loss_cfg = loss_cfg or {}
                    loss = (
                        float(loss_cfg.get("dice_weight", 0.7)) * dice_loss
                        + float(loss_cfg.get("mse_weight", 0.3)) * mse_loss
                        + float(loss_cfg.get("residual_l2_weight", 0.0)) * res_l2_loss
                    )
                    loss_components = {
                        "dice": dice_loss.detach(),
                        "focal": mse_loss.detach(),
                        "bce": torch.tensor(0.0),
                        "sparse": torch.tensor(0.0),
                        "focal_tv": res_l2_loss.detach(),
                    }

                elif loss_type in {"mse_bce_dice", "sparse_support"}:
                    loss_cfg = loss_cfg or {}
                    support_threshold = float(loss_cfg.get("support_threshold", 0.5))
                    target = gt_flat.clamp(0.0, 1.0)
                    gt_bin = (gt_flat >= support_threshold).float()
                    pred = p_t.clamp(eps, 1 - eps)

                    weight_sum = point_weight.sum() + eps
                    mse_each = (pred - target) ** 2
                    mse_loss = (mse_each * point_weight).sum() / weight_sum
                    bce_weight = point_weight
                    if loss_type == "sparse_support":
                        pos_weight = float(loss_cfg.get("pos_weight", 1.0))
                        bce_weight = bce_weight * torch.where(gt_bin > 0.5, pos_weight, 1.0)
                        bce_weight = bce_weight / (bce_weight.mean().detach() + eps)
                    bce_each = -(gt_bin * torch.log(pred) + (1.0 - gt_bin) * torch.log(1.0 - pred))
                    bce_loss = (bce_each * bce_weight).sum() / (bce_weight.sum() + eps)
                    dice_loss = 1.0 - (2.0 * (pred * gt_bin * point_weight).sum() + eps) / (
                        (pred * point_weight).sum() + (gt_bin * point_weight).sum() + eps
                    )
                    loss = (
                        float(loss_cfg.get("mse_weight", 0.6)) * mse_loss
                        + float(loss_cfg.get("bce_weight", 0.2)) * bce_loss
                        + float(loss_cfg.get("dice_weight", 0.2)) * dice_loss
                    )
                    loss_components = {
                        "dice": dice_loss.detach(),
                        "focal": mse_loss.detach(),
                        "bce": bce_loss.detach(),
                        "sparse": torch.tensor(0.0),
                        "focal_tv": torch.tensor(0.0),
                    }

                elif loss_type == "mse":
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
                    target = torch.clamp(gt_flat, 0.0, 1.0)
                    mse_loss = ((p_t - target) ** 2).mean()
                    g_bin = (gt_flat >= 0.5).float()
                    p_prob = p_t.clamp(eps, 1 - eps)
                    pos = g_bin.sum()
                    neg = (1.0 - g_bin).sum()
                    pos_weight = (neg / (pos + eps)).clamp(1.0, 20.0)
                    point_weight = torch.where(g_bin > 0.5, pos_weight, torch.ones_like(g_bin))
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
                    alpha, beta = 0.3, 0.7
                    tp = (p_t * g).sum()
                    fn = ((1 - p_t) * g).sum()
                    fp = (p_t * (1 - g)).sum()
                    tversky = 1 - tp / (tp + alpha * fn + beta * fp + eps)
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

    scaled_loss = loss / max(int(accumulation_steps), 1)
    scaler.scale(scaled_loss).backward()
    if step_optimizer:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        if view_encoder is not None and not freeze_view_encoder:
            torch.nn.utils.clip_grad_norm_(view_encoder.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

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
    cfg: dict | None = None,
    prior_source: str = "prior_ext",
    expected_prior_dim: int = 8,
) -> dict:
    """Validate with per-sample Dice averaging and FEM baseline comparison."""
    model.eval()
    per_sample_metrics = []
    total_loss = 0.0
    n_valid_points = 0

    with torch.no_grad():
        for batch in val_loader:
            coords = batch["coords"].cuda()
            prior = select_prior(batch, prior_source, expected_prior_dim)
            correction_band = batch.get("correction_band")
            correction_band = correction_band.cuda() if correction_band is not None else None
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
                output = unpack_model_output(model(coords, prior, view_feat, correction_band=correction_band))
            else:
                output = unpack_model_output(model(coords, prior, correction_band=correction_band))
            if cfg is not None:
                d_hat = select_stage2_prediction(output, cfg)
            else:
                d_hat = output["d_hat"]
            fem_interp = output["fem_interp"]
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
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if cfg.get("data", {}).get("shared_dir"):
        os.environ["DU2VOX_SHARED_DIR"] = str(cfg["data"]["shared_dir"])

    exp_name = args.experiment_name or cfg["experiment"]["name"]
    max_epochs = args.max_epochs or cfg["training"]["max_epochs"]

    # Load splits
    train_ids = load_split(cfg["data"]["train_split"])
    val_ids = load_split(cfg["data"]["val_split"])
    test_split = cfg.get("data", {}).get("test_split")
    test_count = len(load_split(test_split)) if test_split else 0

    if args.max_samples:
        train_ids = train_ids[: args.max_samples]
        val_ids = val_ids[: max(1, args.max_samples // 4)]

    # Precomputed mode
    precomputed_train = cfg["data"].get("precomputed_train_dir")
    precomputed_val = cfg["data"].get("precomputed_val_dir")

    # Build model
    view_encoder_cfg = cfg["model"].get("view_encoder", False)
    prior_dim = int(cfg["model"].get("prior_dim", 8))
    prior_source = cfg["model"].get("prior_source", "prior_ext")
    expected_prior_dim = 8 if prior_source == "prior_8d" else prior_dim
    model_type = cfg["model"].get("model_type", "")
    use_cqr_model = (model_type == "cqr_residual_inr") or (prior_dim > 8)
    ModelCls = CQRResidualINR if use_cqr_model else ResidualINR

    if precomputed_train:
        print(f"[Stage2] Mode: precomputed (train={precomputed_train}, val={precomputed_val})")
    else:
        print("[Stage2] Mode: on-demand (bridge_dir fallback)")
    print(f"[Stage2] Data root: {cfg['data'].get('dataset_root', cfg['data'].get('samples_dir', ''))}")
    print(
        f"[Stage2] Splits: train={cfg['data']['train_split']}, "
        f"val={cfg['data']['val_split']}, test={test_split or 'N/A'}"
    )
    print(f"[Stage2] Training: {len(train_ids)} samples, Val: {len(val_ids)} samples, Test: {test_count} samples")
    print(
        f"[Stage2] Model: model_type={model_type or 'residual_inr'}, "
        f"prior_dim={prior_dim}, prior_source={prior_source}, use_cqr_model={use_cqr_model}, "
        f"output_mode={cfg['model'].get('output_mode', 'residual')}"
    )
    if ModelCls is CQRResidualINR:
        print(
            f"[Stage2] CQR prior_source={prior_source}, prior_dim={prior_dim}, "
            f"use_prolongation_adapter={cfg['model'].get('use_prolongation_adapter', False)}, "
            f"prolongation_feat_dim={cfg['model'].get('prolongation_feat_dim', 32)}, "
            f"use_band_embedding={cfg['model'].get('use_band_embedding', False)}, "
            f"band_embed_dim={cfg['model'].get('band_embed_dim', 8)}, "
            f"use_lifting_adapter={cfg['model'].get('use_lifting_adapter', False)}, "
            f"lifting_feat_dim={cfg['model'].get('lifting_feat_dim', 32)}, "
            f"residual_scale={cfg['model'].get('residual_scale', 1.0)}"
        )
    print(f"[Stage2] Loss: {cfg['loss']['type']} weights={cfg.get('loss', {})}")
    print(
        f"[Stage2] Precision: amp={cfg['training'].get('amp', False)}, "
        f"amp_dtype={cfg['training'].get('amp_dtype', 'fp16')}, "
        f"grad_accum_steps={cfg['training'].get('grad_accum_steps', 1)}"
    )
    print(
        f"[Stage2] LR: base_lr={cfg['training']['lr']}, "
        f"view_encoder_lr_scale={cfg['model'].get('view_encoder_lr_scale', 1.0)}"
    )
    print(f"[Stage2] Data: train_precomputed={precomputed_train}, val_precomputed={precomputed_val}")
    print(
        f"[Projection] input_file={cfg['data'].get('projection_file', 'proj.npz')}, "
        f"norm={cfg['data'].get('projection_norm', 'none')}, "
        f"transform={cfg['data'].get('projection_transform', 'none')}"
    )
    print(
        f"[Stage2] Query resampling: resample_queries_each_epoch="
        f"{cfg['data'].get('resample_queries_each_epoch', False)}, "
        f"query_epoch_seed_stride={cfg['data'].get('query_epoch_seed_stride', 1000003)}"
    )
    print(
        f"[Stage2] CQR measurement_proposal="
        f"{cfg.get('cqr', {}).get('measurement_proposal', {}).get('enabled', False)} "
        f"ratio={cfg.get('cqr', {}).get('ratios', {}).get('proposal', 0.0)}"
    )

    if view_encoder_cfg:
        # Multiview mode: ViewEncoderModule + ResidualINR
        from du2vox.models.stage2.view_encoder import ViewEncoderModule

        view_encoder = ViewEncoderModule(
            view_feat_dim=cfg["model"]["view_feat_dim"],
            fusion_method=cfg["model"].get("fusion_method", "attn"),
            encoder_out_channels=cfg["model"].get("encoder_out_channels", 32),
            encoder_base_channels=cfg["model"].get("encoder_base_channels", 32),
            projection_transform=cfg["model"].get("view_projection_transform", "log1p"),
            multiscale_cfg=cfg["model"].get("view_multiscale", {}),
        ).cuda()
        freeze_view_encoder = bool(cfg["model"].get("freeze_view_encoder", False))
        if freeze_view_encoder:
            view_encoder.eval()
            for param in view_encoder.parameters():
                param.requires_grad_(False)

        model = build_stage2_model(
            ModelCls,
            cfg,
            prior_dim=prior_dim,
            view_feat_dim=cfg["model"]["view_feat_dim"],
        ).cuda()

        # Joint optimizer with separate LR for view encoder
        lr_scale = cfg["model"].get("view_encoder_lr_scale", 1.0)
        ve_lr = cfg["training"]["lr"] * lr_scale
        param_groups = [{"params": model.parameters(), "lr": cfg["training"]["lr"]}]
        if not freeze_view_encoder:
            param_groups.append({"params": view_encoder.parameters(), "lr": ve_lr})
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=cfg["training"]["weight_decay"],
        )
        print(
            f"[Stage2] Multiview mode: view_feat_dim={cfg['model']['view_feat_dim']}, "
            f"fusion={cfg['model'].get('fusion_method', 'mean')}, ve_lr={ve_lr:.0e}, "
            f"freeze_view_encoder={freeze_view_encoder}, "
            f"view_multiscale={cfg['model'].get('view_multiscale', {}).get('enabled', False)}"
        )
    else:
        # DE-only mode
        view_encoder = None
        model = build_stage2_model(ModelCls, cfg, prior_dim=prior_dim).cuda()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["training"]["lr"],
            weight_decay=cfg["training"]["weight_decay"],
        )

    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location="cuda")
        if isinstance(ckpt, dict) and "residual_inr" in ckpt:
            model.load_state_dict(ckpt["residual_inr"])
            if view_encoder is not None and "view_encoder" in ckpt:
                view_encoder.load_state_dict(ckpt["view_encoder"])
        else:
            model.load_state_dict(ckpt)
        print(f"[Stage2] Resumed model weights from {args.resume_checkpoint}")

    warmup_epochs = cfg["training"].get("warmup_epochs", 5)
    loss_type = cfg.get("loss", {}).get("type", "gisc")  # "gisc" or "soft_dice"
    use_amp = bool(cfg["training"].get("amp", False))
    amp_dtype_name = cfg["training"].get("amp_dtype", "fp16")
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bf16" else torch.float16
    grad_accum_steps = max(1, int(cfg["training"].get("grad_accum_steps", 1)))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["scheduler"].get("T_max", max_epochs),
        eta_min=cfg["training"]["scheduler"].get("eta_min", 1e-6),
    )
    warmup_base_lrs = [pg["lr"] for pg in optimizer.param_groups]
    for i, lr in enumerate(warmup_base_lrs):
        print(f"[Stage2] param_group[{i}].lr = {lr:.0e}")

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

    if args.resume_checkpoint:
        val_metrics = validate(
            model,
            val_loader,
            view_encoder=view_encoder,
            cfg=cfg,
            prior_source=prior_source,
            expected_prior_dim=expected_prior_dim,
        )
        best_delta = val_metrics["delta_dice_05"]
        best_val_loss = val_metrics["val_loss"]
        best_ckpt_info = {
            "epoch": 0,
            "stage2_dice_05": val_metrics["stage2_dice_05"],
            "fem_dice_05": val_metrics["fem_dice_05"],
            "delta_dice_05": best_delta,
        }
        print(
            f"[Stage2] Resume baseline: ΔDice={best_delta:+.4f} "
            f"(S2={val_metrics['stage2_dice_05']:.4f} vs FEM={val_metrics['fem_dice_05']:.4f})"
        )

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
        optimizer.zero_grad(set_to_none=True)

        # Warmup: linear lr ramp
        if epoch <= warmup_epochs:
            lr_scale = epoch / warmup_epochs
            for pg, base_lr in zip(optimizer.param_groups, warmup_base_lrs):
                pg["lr"] = base_lr * lr_scale
        else:
            scheduler.step()

        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)

        for batch_idx, batch in enumerate(train_loader):
            step_optimizer = ((batch_idx + 1) % grad_accum_steps == 0) or (batch_idx + 1 == len(train_loader))
            metrics = train_step(
                model,
                batch,
                optimizer,
                scaler,
                grad_clip_norm=cfg["training"].get("grad_clip_norm", 1.0),
                view_encoder=view_encoder,
                loss_type=loss_type,
                loss_cfg=cfg.get("loss", {}),
                model_cfg=cfg.get("model", {}),
                prior_source=prior_source,
                expected_prior_dim=expected_prior_dim,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                accumulation_steps=grad_accum_steps,
                step_optimizer=step_optimizer,
            )
            epoch_loss += metrics["loss"]
            epoch_fem += metrics["fem_baseline_loss"]
            epoch_res += metrics["residual_norm"]
            epoch_bce += metrics["bce"]
            epoch_sparse += metrics["sparse"]
            epoch_focal_tv += metrics["focal_tv"]
            epoch_valid += metrics["valid_count"]
            n_steps += 1

        val_metrics = validate(
            model,
            val_loader,
            view_encoder=view_encoder,
            cfg=cfg,
            prior_source=prior_source,
            expected_prior_dim=expected_prior_dim,
        )
        elapsed = time.perf_counter() - t0

        avg_loss = epoch_loss / max(n_steps, 1)
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
