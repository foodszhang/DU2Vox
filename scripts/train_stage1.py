#!/usr/bin/env python3
"""
MS-GDUN training script for FMT-SimGen.
ALL parameters are read from the config file — no hardcoding.

Usage:
    python scripts/train_stage1.py --config configs/stage1/gaussian_1000.yaml
    python scripts/train_stage1.py --config configs/stage1/gaussian_1000.yaml --resume runs/gcain_gaussian_1000/checkpoints/best.pth
"""

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau

from du2vox.models.stage1.gcain import GCAIN_full
from du2vox.data.dataset import FMTSimGenDataset
from du2vox.losses.tversky import criterion, criterion_gaussian, criterion_support
from du2vox.evaluation.metrics import evaluate_batch, summarize_metrics


def compute_loss(pred, gt, nodes, loss_cfg):
    """Select loss function based on loss.type config."""
    loss_type = loss_cfg.get("type", "uniform")
    if loss_type == "gaussian":
        return criterion_gaussian(
            pred, gt, nodes,
            weight_tversky=loss_cfg.get("tversky_weight", 0.5),
            weight_mse=loss_cfg.get("mse_weight", 0.2),
            weight_core=loss_cfg.get("core_weight", 0.3),
            core_threshold=loss_cfg.get("core_threshold", 0.6),
            tversky_alpha=loss_cfg.get("tversky_alpha", 0.1),
            tversky_beta=loss_cfg.get("tversky_beta", 0.9),
        )
    if loss_type == "support":
        return criterion_support(
            pred, gt, nodes,
            weight_tversky=loss_cfg.get("tversky_weight", 0.5),
            weight_bce=loss_cfg.get("bce_weight", 0.3),
            weight_mse=loss_cfg.get("mse_weight", 0.2),
            tversky_alpha=loss_cfg.get("tversky_alpha", 0.3),
            tversky_beta=loss_cfg.get("tversky_beta", 0.7),
        )
    return criterion(
        pred, gt, nodes,
        weight_tversky=loss_cfg.get("tversky_weight", 0.7),
        weight_mse=loss_cfg.get("mse_weight", 0.3),
        tversky_alpha=loss_cfg.get("tversky_alpha", 0.1),
        tversky_beta=loss_cfg.get("tversky_beta", 0.9),
    )


class DualLogger:
    """Simultaneously write to stdout and file, each line forcibly flushed."""

    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.file = open(filepath, "a", encoding="utf-8", buffering=1)

    def write(self, msg):
        self.terminal.write(msg)
        self.terminal.flush()
        self.file.write(msg)
        self.file.flush()

    def flush(self):
        self.terminal.flush()
        self.file.flush()


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_activation_fn(activation: str, leaky_slope: float):
    """Return a callable that applies the configured output activation."""
    if activation == "sigmoid":
        return lambda x: torch.sigmoid(x)
    elif activation == "leaky_relu":
        def fn(x):
            return torch.nn.functional.leaky_relu(x, negative_slope=leaky_slope).clamp(max=1.0)
        return fn
    else:
        return lambda x: x.clamp(min=0.0, max=1.0)


def build_scheduler(sched_cfg: dict, optimizer):
    sched_type = sched_cfg.get("type", "CosineAnnealingWarmRestarts")
    if sched_type == "CosineAnnealingWarmRestarts":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=sched_cfg.get("T_0", 50),
            T_mult=sched_cfg.get("T_mult", 2),
            eta_min=sched_cfg.get("eta_min", 1e-6),
        )
    elif sched_type == "CosineAnnealingLR":
        return CosineAnnealingLR(
            optimizer,
            T_max=sched_cfg.get("T_max", 300),
            eta_min=sched_cfg.get("eta_min", 1e-6),
        )
    elif sched_type == "ReduceLROnPlateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=sched_cfg.get("patience", 10),
            factor=sched_cfg.get("factor", 0.5),
        )
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")


def metric_improved(metric: str, current: float, best: float) -> bool:
    if metric == "val_loss":
        return current < best
    return current > best


def save_checkpoint(path: Path, epoch: int, model, optimizer, scheduler, best_metrics: dict, primary_metric: str) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_metrics": best_metrics,
            "best_val_loss": best_metrics.get("val_loss", float("inf")),
            "best_dice": best_metrics.get("dice_bin_0.5", 0.0),
            "primary_metric": primary_metric,
        },
        path,
    )


def train():
    parser = argparse.ArgumentParser(description="MS-GDUN training for FMT-SimGen")
    parser.add_argument(
        "--config", type=str, required=True,
    )
    parser.add_argument(
        "--resume", type=str, default=None,
    )
    parser.add_argument(
        "--resume_weights_only", action="store_true",
        help="Load only model weights from --resume and reset optimizer/scheduler/best metrics.",
    )
    args = parser.parse_args()

    # ── Load config ──
    cfg = load_config(args.config)

    exp_cfg = cfg["experiment"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    loss_cfg = cfg["loss"]
    log_cfg = cfg.get("logging", {})

    # ── Experiment output directory ──
    run_dir = Path("runs") / exp_cfg["name"]
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Copy config into experiment dir for reproducibility
    shutil.copy2(args.config, run_dir / "config.yaml")

    # ── Log file setup ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = run_dir / f"train_{timestamp}.log"
    sys.stdout = DualLogger(log_path)

    print(f"[LOG] Logging to {log_path}")
    print(f"[LOG] Started at {datetime.now().isoformat()}")
    print(f"[LOG] Config: {args.config}")
    print(f"[LOG] Experiment: {exp_cfg['name']}")
    print(f"[LOG] Output dir: {run_dir}")

    # ── Data paths ──
    shared_dir = Path(data_cfg["shared_dir"])
    samples_dir = Path(data_cfg["samples_dir"])
    splits_dir = Path(data_cfg["splits_dir"])

    # ── Dataset (val reuses train's shared assets to avoid duplication) ──
    train_set = FMTSimGenDataset(
        shared_dir=shared_dir,
        samples_dir=samples_dir,
        split_file=splits_dir / "train.txt",
        normalize_b=data_cfg.get("normalize_b", True),
        normalize_gt=data_cfg.get("normalize_gt", True),
        normalize_gt_mode=data_cfg.get("normalize_gt_mode", "per_sample"),
        binarize_gt=data_cfg.get("binarize_gt", False),
        binarize_threshold=data_cfg.get("binarize_threshold", 0.05),
        use_visible_mask=data_cfg.get("use_visible_mask", False),
    )
    val_set = FMTSimGenDataset(
        shared_dir=None,
        samples_dir=samples_dir,
        split_file=splits_dir / "val.txt",
        normalize_b=data_cfg.get("normalize_b", True),
        normalize_gt=data_cfg.get("normalize_gt", True),
        normalize_gt_mode=data_cfg.get("normalize_gt_mode", "per_sample"),
        binarize_gt=data_cfg.get("binarize_gt", False),
        binarize_threshold=data_cfg.get("binarize_threshold", 0.05),
        shared=train_set,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=train_cfg.get("batch_size", 2),
        shuffle=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=train_cfg.get("batch_size", 2),
        shuffle=False,
    )

    n_nodes = train_set.nodes.shape[0]
    n_surface = train_set.A.shape[0]
    print(f"Train: {len(train_set)} samples, Val: {len(val_set)} samples")
    print(f"Shared assets: {n_nodes} nodes, {n_surface} surface nodes")
    print(f"A shape: {tuple(train_set.A.shape)}, first b shape: {tuple(train_set.b_list[0].shape)}")
    print(f"visible_mask applied: {train_set.visible_mask is not None}")
    assert train_set.b_list[0].shape[0] == train_set.A.shape[0]

    # ── Move shared assets to GPU ──
    A = train_set.A.cuda()
    L = train_set.L.cuda()
    L0 = train_set.L0.cuda()
    L1 = train_set.L1.cuda()
    L2 = train_set.L2.cuda()
    L3 = train_set.L3.cuda()
    knn_idx = train_set.knn_idx.cuda()
    sens_w = train_set.sens_w.cuda()
    nodes = train_set.nodes.cuda()

    print(f"A dense size: {A.element_size() * A.nelement() / 1e6:.1f} MB")
    print(f"L dense size: {L.element_size() * L.nelement() / 1e6:.1f} MB")

    # ── Model ──
    model = GCAIN_full(
        L=L, A=A, LTL=None, ATA=None,
        L0=L0, L1=L1, L2=L2, L3=L3,
        knn_idx=knn_idx,
        sens_w=sens_w,
        num_layer=model_cfg["num_layer"],
        feat_dim=model_cfg["feat_dim"],
    ).cuda()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    # ── Scheduler ──
    scheduler = build_scheduler(train_cfg["scheduler"], optimizer)

    # ── Activation function ──
    apply_activation = build_activation_fn(
        train_cfg.get("activation", "leaky_relu"),
        train_cfg.get("leaky_relu_slope", 0.01),
    )

    # ── Resume from checkpoint ──
    start_epoch = 1
    checkpoint_cfg = cfg.get("checkpoint", {})
    primary_metric = checkpoint_cfg.get("primary_metric", "dice_bin_0.5")
    best_metrics = {
        "dice_bin_0.3": 0.0,
        "dice_bin_0.5": 0.0,
        "dice": 0.0,
        "val_loss": float("inf"),
    }
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cuda")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            if args.resume_weights_only:
                print("  Loaded model weights only; optimizer/scheduler/best metrics reset")
            elif "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if not args.resume_weights_only and "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                print("  Scheduler state restored from checkpoint")
            if not args.resume_weights_only:
                start_epoch = ckpt.get("epoch", 0) + 1
                best_metrics.update(ckpt.get("best_metrics", {}))
                best_metrics["val_loss"] = min(best_metrics["val_loss"], ckpt.get("best_val_loss", float("inf")))
                best_metrics["dice_bin_0.3"] = max(best_metrics["dice_bin_0.3"], ckpt.get("best_dice", 0.0))
        else:
            model.load_state_dict(ckpt)
            start_epoch = 1
            print("  (old checkpoint format: starting from epoch 1)")
        print(f"Resumed at epoch {start_epoch}, best_metrics={best_metrics}")
    else:
        print("Starting training from scratch")

    # ── Training loop config ──
    max_epochs = train_cfg["max_epochs"]
    grad_clip_norm = train_cfg.get("grad_clip_norm", 1.0)
    diag_epochs = log_cfg.get("diag_epochs", 3)
    detail_every = log_cfg.get("detail_every", 10)
    milestone_every = log_cfg.get("milestone_every", 50)

    # CSV header
    print(
        "[CSV] epoch,train_loss,val_loss,dice,dice_03,dice_01,dice_06,"
        "recall_01,prec_03,loc_err,mse,pred_max,pred_mean,pred_std,lr"
    )

    try:
        for epoch in range(start_epoch, max_epochs + 1):
            # ── Training ──
            model.train()
            train_losses = []
            for batch in train_loader:
                b = batch["b"].cuda()
                gt = batch["gt"].cuda()

                X0 = torch.zeros(b.size(0), n_nodes, 1, device="cuda")
                pred = model(X0, b)
                pred = apply_activation(pred)

                loss = compute_loss(pred, gt, nodes, loss_cfg)

                optimizer.zero_grad()
                loss.backward()

                if epoch <= diag_epochs:
                    grad_norms = [
                        (n, p.grad.norm().item())
                        for n, p in model.named_parameters()
                        if p.grad is not None
                    ]
                    total_grad = sum(g for _, g in grad_norms)
                    print(f"  [DIAG] total_grad_norm={total_grad:.6f}, "
                          f"pred: min={pred.min():.4f} max={pred.max():.4f} "
                          f"mean={pred.mean():.4f} std={pred.std():.4f}")
                    if total_grad == 0:
                        print("  [FATAL] gradient all-zero!")
                        break

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()
                train_losses.append(loss.item())

            train_loss_mean = sum(train_losses) / len(train_losses)

            # ── Validation ──
            model.eval()
            val_metrics = []
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    b = batch["b"].cuda()
                    gt = batch["gt"].cuda()
                    X0 = torch.zeros(b.size(0), n_nodes, 1, device="cuda")
                    pred = model(X0, b)
                    pred = apply_activation(pred)
                    loss = compute_loss(pred, gt, nodes, loss_cfg)
                    val_losses.append(loss.item())
                    val_metrics.append(evaluate_batch(pred, gt, nodes))

            val_loss_mean = sum(val_losses) / len(val_losses)
            val_summary = summarize_metrics(val_metrics)
            current_dice03 = val_summary.get("dice_bin_0.3", 0.0)
            current_dice05 = val_summary.get("dice_bin_0.5", 0.0)

            # ── Scheduler step ──
            sched_type = train_cfg["scheduler"].get("type", "CosineAnnealingWarmRestarts")
            if sched_type in {"CosineAnnealingWarmRestarts", "CosineAnnealingLR"}:
                scheduler.step()
            else:
                scheduler.step(val_loss_mean)

            lr_current = optimizer.param_groups[0]["lr"]

            # ── Epoch log (every epoch) ──
            print(
                f"Epoch {epoch:03d}/{max_epochs} | "
                f"lr={lr_current:.1e} | "
                f"Train: {train_loss_mean:.4f} | "
                f"Val: {val_loss_mean:.4f} | "
                f"Dice: {val_summary['dice']:.4f} | "
                f"Dice@0.3: {current_dice03:.4f} | "
                f"Dice@0.5: {current_dice05:.4f} | "
                f"Dice@0.6: {val_summary.get('dice_bin_0.6', 0.0):.4f} | "
                f"Dice@0.1: {val_summary.get('dice_bin_0.1', 0.0):.4f} | "
                f"Rec@0.5: {val_summary.get('recall_pred05_gt05', 0.0):.4f} | "
                f"Prec@0.5: {val_summary.get('precision_pred05_gt05', 0.0):.4f} | "
                f"pred: [{val_summary.get('pred_mean', 0):.3f}±{val_summary.get('pred_std', 0):.3f}] "
                f"max={val_summary.get('pred_max', 0):.3f} "
                f"f>0.1={val_summary.get('pred_frac_0.1', 0):.3f}"
            )

            # ── Detailed val output ──
            if epoch % detail_every == 0 or epoch == max_epochs:
                print("  ── Val Detail ──")
                print(f"  Dice_bin@0.5: {val_summary.get('dice_bin_0.5', 0):.4f}")
                print(f"  Dice_bin@0.3: {val_summary.get('dice_bin_0.3', 0):.4f}")
                print(f"  Dice_bin@0.6: {val_summary.get('dice_bin_0.6', 0):.4f}")
                print(f"  Dice_bin@0.1: {val_summary.get('dice_bin_0.1', 0):.4f}")
                print(f"  Recall@0.3:   {val_summary.get('recall_0.3', 0):.4f}")
                print(f"  Recall@0.5:   {val_summary.get('recall_pred05_gt05', 0):.4f}")
                print(f"  Recall@0.1:   {val_summary.get('recall_0.1', 0):.4f}")
                print(f"  Prec@0.3:     {val_summary.get('precision_0.3', 0):.4f}")
                print(f"  Prec@0.5:     {val_summary.get('precision_pred05_gt05', 0):.4f}")
                print(f"  Prec@0.1:     {val_summary.get('precision_0.1', 0):.4f}")
                print(f"  LocError:     {val_summary.get('location_error', 0):.4f}")
                print(f"  MSE:          {val_summary.get('mse', 0):.6f}")
                print(f"  pred_frac>0.3:{val_summary.get('pred_frac_0.3', 0):.4f}")
                print(f"  pred_frac>0.1:{val_summary.get('pred_frac_0.1', 0):.4f}")
                print("  ───────────────")

            # ── Save latest ──
            latest_path = checkpoint_dir / "latest.pth"
            save_checkpoint(latest_path, epoch, model, optimizer, scheduler, best_metrics, primary_metric)

            # ── Save metric-specific best checkpoints ──
            metric_values = {
                "dice_bin_0.3": current_dice03,
                "dice_bin_0.5": current_dice05,
                "dice": val_summary.get("dice", 0.0),
                "val_loss": val_loss_mean,
            }
            metric_paths = {
                "dice_bin_0.3": "best_dice03.pth",
                "dice_bin_0.5": "best_dice05.pth",
                "dice": "best_softdice.pth",
                "val_loss": "best_val_loss.pth",
            }
            for metric, value in metric_values.items():
                if metric_improved(metric, value, best_metrics[metric]):
                    best_metrics[metric] = value
                    save_checkpoint(
                        checkpoint_dir / metric_paths[metric],
                        epoch,
                        model,
                        optimizer,
                        scheduler,
                        best_metrics,
                        primary_metric,
                    )
                    print(f"  -> Best {metric}: {value:.4f} @ Epoch {epoch}")
                    if metric == primary_metric:
                        save_checkpoint(
                            checkpoint_dir / "best.pth",
                            epoch,
                            model,
                            optimizer,
                            scheduler,
                            best_metrics,
                            primary_metric,
                        )

            # ── Milestone checkpoint ──
            if epoch % milestone_every == 0:
                milestone_path = checkpoint_dir / f"epoch_{epoch:03d}.pth"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_metrics": best_metrics,
                    "best_val_loss": best_metrics["val_loss"],
                    "best_dice": best_metrics["dice_bin_0.5"],
                    "primary_metric": primary_metric,
                }, milestone_path)
                print(f"  -> Milestone checkpoint saved to {milestone_path}")

            # ── CSV summary ──
            print(
                f"[CSV] {epoch},{train_loss_mean:.6f},{val_loss_mean:.6f},"
                f"{val_summary['dice']:.6f},{current_dice03:.6f},"
                f"{val_summary.get('dice_bin_0.1', 0):.6f},"
                f"{val_summary.get('dice_bin_0.6', 0):.6f},"
                f"{val_summary.get('recall_0.1', 0):.6f},"
                f"{val_summary.get('precision_0.3', 0):.6f},"
                f"{val_summary.get('location_error', 0):.6f},"
                f"{val_summary.get('mse', 0):.8f},"
                f"{val_summary.get('pred_max', 0):.6f},"
                f"{val_summary.get('pred_mean', 0):.6f},"
                f"{val_summary.get('pred_std', 0):.6f},"
                f"{lr_current:.8f}"
            )

    except KeyboardInterrupt:
        print(f"\n[LOG] Training interrupted at epoch {epoch}")
    except Exception as e:
        print(f"\n[LOG] Training failed at epoch {epoch}: {e}")
        raise
    finally:
        print(
            f"\n[LOG] Final best: Dice@0.3={best_metrics['dice_bin_0.3']:.4f}, "
            f"Dice@0.5={best_metrics['dice_bin_0.5']:.4f}, "
            f"soft Dice={best_metrics['dice']:.4f}, val_loss={best_metrics['val_loss']:.4f}"
        )
        print(f"[LOG] Primary checkpoint metric: {primary_metric}")
        print(f"[LOG] Ended at {datetime.now().isoformat()}")
        if isinstance(sys.stdout, DualLogger):
            sys.stdout.file.close()
            sys.stdout = sys.stdout.terminal


if __name__ == "__main__":
    train()
