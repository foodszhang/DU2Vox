#!/usr/bin/env python3
"""Train ESCB or the matched plain target-first voxel residual baseline."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.bridge.canonical_cross_discretization import (
    CanonicalCrossDiscretization,
)
from du2vox.data.error_structured_dataset import (
    ErrorStructuredFixedDomainDataset,
)
from du2vox.evaluation.error_structured import evaluate_dense, save_dense_result
from du2vox.models.stage2.error_structured_bridge import (
    ErrorStructuredCrossDiscretizationBridge,
    FEMInverseCorrectionNet,
    PlainTargetFirstVoxelResidual,
    VoxelRepresentationCompletionNet,
    fixed_representation_target,
)


def load_ids(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_view_encoder(model_cfg: dict[str, Any]) -> torch.nn.Module | None:
    if not model_cfg.get("view_encoder", True):
        return None
    from du2vox.models.stage2.view_encoder import ViewEncoderModule

    return ViewEncoderModule(
        view_feat_dim=int(model_cfg.get("view_feat_dim", 32)),
        fusion_method=model_cfg.get("fusion_method", "attn"),
        encoder_out_channels=int(model_cfg.get("encoder_out_channels", 32)),
        encoder_base_channels=int(model_cfg.get("encoder_base_channels", 32)),
        projection_transform=model_cfg.get("view_projection_transform", "none"),
        multiscale_cfg=model_cfg.get("view_multiscale", {}),
    )


def build_model(cfg: dict[str, Any], model_type: str, shared_dir: str | Path) -> torch.nn.Module:
    model_cfg = cfg["model"]
    view_dim = int(model_cfg.get("view_feat_dim", 32)) if model_cfg.get("view_encoder", True) else 0
    representation = VoxelRepresentationCompletionNet(
        n_freqs=int(model_cfg.get("n_freqs", 10)),
        hidden_dim=int(model_cfg.get("representation_hidden_dim", 256)),
        n_hidden_layers=int(model_cfg.get("representation_hidden_layers", 4)),
        view_feat_dim=view_dim,
    )
    view_encoder = build_view_encoder(model_cfg)
    if model_type == "baseline":
        return PlainTargetFirstVoxelResidual(representation, view_encoder)
    if model_type != "escb":
        raise ValueError(f"Unknown model type: {model_type}")
    knn = torch.from_numpy(np.load(Path(shared_dir) / "knn_idx_full.npy"))
    inverse = FEMInverseCorrectionNet(
        knn,
        hidden_dim=int(model_cfg.get("inverse_hidden_dim", 128)),
        n_hidden_layers=int(model_cfg.get("inverse_hidden_layers", 3)),
        view_feat_dim=view_dim,
    )
    return ErrorStructuredCrossDiscretizationBridge(inverse, representation, view_encoder)


def build_dataset(
    cfg: dict[str, Any], split: str, ids: list[str], n_query_points: int
) -> ErrorStructuredFixedDomainDataset:
    data = cfg["data"]
    return ErrorStructuredFixedDomainDataset(
        sample_ids=ids,
        samples_dir=data["samples_dir"],
        bridge_dir=data[f"{split}_bridge_dir"],
        projection_targets_dir=data["projection_targets_dir"],
        operator_cache=data["operator_cache"],
        shared_dir=data["shared_dir"],
        n_query_points=n_query_points,
        seed=int(cfg["training"].get("seed", 20260831)),
        multiview=bool(cfg["model"].get("view_encoder", True)),
        projection_file=data.get("projection_file", "proj.npz"),
        projection_norm=data.get("projection_norm", "per_view_max"),
        projection_transform=data.get("projection_transform", "none"),
        gt_mode=data.get("gt_mode", "binary"),
        normalize_gt=data.get("normalize_gt", "none"),
        normalization_scale_filename=data.get("normalization_scale_filename"),
        binary_threshold=float(data.get("binary_threshold", 0.05)),
        signal_query_fraction=float(data.get("signal_query_fraction", 0.0)),
        metric_data_range=float(data.get("data_range", 2.0)),
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def forward_model(model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    common = {
        "proj_imgs": batch.get("proj_imgs"),
        "query_coords_world": batch["query_coords_world"],
    }
    if isinstance(model, ErrorStructuredCrossDiscretizationBridge):
        return model(
            batch["x_h"],
            batch["node_coords_norm"],
            batch["query_coords_norm"],
            batch["query_node_indices"],
            batch["query_barycentric"],
            node_coords_world=batch["node_coords_world"],
            **common,
        )
    return model(
        batch["x_h"],
        batch["query_coords_norm"],
        batch["query_node_indices"],
        batch["query_barycentric"],
        **common,
    )


def cosine_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = prediction.reshape(prediction.shape[0], -1)
    truth = target.reshape(target.shape[0], -1)
    similarity = F.cosine_similarity(pred, truth, dim=-1, eps=1e-8)
    return 1.0 - similarity.mean()


def compute_loss(
    model_type: str,
    phase: str,
    output: dict[str, torch.Tensor],
    batch: dict[str, Any],
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    if model_type == "baseline":
        target = batch["gt"] - output["original_fem_voxel"]
        target_loss = F.mse_loss(output["plain_residual_prediction"], target)
        final_loss = F.mse_loss(output["final_prediction"], batch["gt"])
        total = float(loss_cfg.get("lambda_plain_target", 1.0)) * target_loss
        total = total + float(loss_cfg.get("lambda_final", 1.0)) * final_loss
        return total, {
            "plain_target_mse": float(target_loss.detach()),
            "final_mse": float(final_loss.detach()),
        }

    inverse_target = batch["pi_gt"] - batch["x_h"]
    inv_mse = F.mse_loss(output["inverse_correction"], inverse_target)
    inv_cos = cosine_loss(output["inverse_correction"], inverse_target)
    # Recompute through the contract helper, rather than accepting any corrected
    # FEM prediction as an argument.
    repr_target = fixed_representation_target(batch["gt"], batch["pi_gt_voxel"])
    if not torch.equal(repr_target, batch["representation_target"]):
        raise RuntimeError("Representation target changed from canonical fixed target")
    repr_mse = F.mse_loss(output["representation_prediction"], repr_target)
    repr_cos = cosine_loss(output["representation_prediction"], repr_target)
    final_mse = F.mse_loss(output["final_prediction"], batch["gt"])
    if phase == "A":
        total = inv_mse + float(loss_cfg.get("cosine_weight", 0.0)) * inv_cos
    elif phase == "B":
        total = repr_mse + float(loss_cfg.get("cosine_weight", 0.0)) * repr_cos
    elif phase == "C":
        total = (
            float(loss_cfg["lambda_inv"]) * inv_mse
            + float(loss_cfg["lambda_repr"]) * repr_mse
            + float(loss_cfg["lambda_final"]) * final_mse
        )
    else:
        raise ValueError(phase)
    return total, {
        "inverse_target_mse": float(inv_mse.detach()),
        "inverse_target_cosine_loss": float(inv_cos.detach()),
        "representation_target_mse": float(repr_mse.detach()),
        "representation_target_cosine_loss": float(repr_cos.detach()),
        "final_mse": float(final_mse.detach()),
    }


def gradient_sum(module: torch.nn.Module) -> float:
    return float(
        sum(
            parameter.grad.detach().abs().sum().item()
            for parameter in module.parameters()
            if parameter.grad is not None
        )
    )


def assert_gradient_contract(
    model: ErrorStructuredCrossDiscretizationBridge, phase: str
) -> dict[str, float]:
    inverse = gradient_sum(model.inverse_net)
    representation = gradient_sum(model.representation_net)
    expected = {
        "A": (True, False),
        "B": (False, True),
        "C": (True, True),
    }[phase]
    if (inverse > 0) != expected[0] or (representation > 0) != expected[1]:
        raise RuntimeError(
            f"Phase {phase} gradient contract failed: inverse={inverse}, "
            f"representation={representation}"
        )
    return {"inverse_grad_l1": inverse, "representation_grad_l1": representation}


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    model_type: str,
    phase: str,
    epoch: int,
    metric: float,
    cfg: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_type": model_type,
            "phase": phase,
            "epoch": epoch,
            "dense_val_dice": metric,
            "config": cfg,
        },
        path,
    )


def train_phase(
    *,
    model: torch.nn.Module,
    model_type: str,
    phase: str,
    epochs: int,
    train_dataset: ErrorStructuredFixedDomainDataset,
    val_dataset: ErrorStructuredFixedDomainDataset,
    canonical: CanonicalCrossDiscretization,
    cfg: dict[str, Any],
    run_dir: Path,
    device: torch.device,
    max_val_samples: int | None,
    resume_checkpoint: dict[str, Any] | None = None,
) -> Path:
    training = cfg["training"]
    loss_cfg = cfg["loss"]
    if isinstance(model, ErrorStructuredCrossDiscretizationBridge):
        model.set_phase(phase)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training["lr"]),
        weight_decay=float(training.get("weight_decay", 1e-5)),
    )
    start_epoch = 1
    if resume_checkpoint is not None:
        if resume_checkpoint.get("phase") != phase:
            raise RuntimeError(f"Cannot resume phase {phase} from {resume_checkpoint.get('phase')}")
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
    loader = DataLoader(
        train_dataset,
        batch_size=int(training.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(training.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    amp_enabled = bool(training.get("amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if training.get("amp_dtype", "bf16") == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    best_metric = -float("inf")
    best_path = (
        run_dir
        / "checkpoints"
        / (
            "best_dense_val_delta_dice.pth"
            if phase in {"C", "baseline"}
            else f"best_phase_{phase.lower()}.pth"
        )
    )
    patience = 0
    gradient_checked = model_type == "baseline"
    for epoch in range(start_epoch, epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        totals: dict[str, float] = {}
        count = 0
        for batch in loader:
            batch = move_batch(batch, device)
            # Clear every module, including parameters frozen in the current phase;
            # otherwise a previous phase's .grad can masquerade as a routing leak.
            model.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype):
                output = forward_model(model, batch)
                loss, details = compute_loss(model_type, phase, output, batch, loss_cfg)
            scaler.scale(loss).backward()
            was_unscaled = False
            if not gradient_checked:
                scaler.unscale_(optimizer)
                was_unscaled = True
                gradients = assert_gradient_contract(model, phase)
                print(f"[gradient-contract] phase={phase} {gradients}", flush=True)
                gradient_checked = True
            if float(training.get("grad_clip_norm", 0.0)) > 0:
                if not was_unscaled:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["grad_clip_norm"])
                )
            scaler.step(optimizer)
            scaler.update()
            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach())
            for key, value in details.items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1
        means = {key: value / max(count, 1) for key, value in totals.items()}
        print(f"[train] phase={phase} epoch={epoch}/{epochs} {means}", flush=True)
        save_checkpoint(
            run_dir / "checkpoints" / f"last_phase_{phase.lower()}.pth",
            model,
            optimizer,
            model_type=model_type,
            phase=phase,
            epoch=epoch,
            metric=float("nan"),
            cfg=cfg,
        )

        eval_interval = int(cfg["validation"].get("dense_interval", 5))
        if epoch % eval_interval != 0 and epoch != epochs:
            continue
        dense = evaluate_dense(
            model,
            val_dataset,
            canonical,
            device,
            batch_points=int(cfg["validation"].get("batch_points", 32768)),
            phase_a_only=phase == "A",
            max_samples=max_val_samples,
        )
        save_dense_result(
            dense,
            run_dir / "dense_val" / f"phase_{phase.lower()}_epoch_{epoch:03d}.json",
        )
        metric_key = "corrected_dice" if phase == "A" else "final_dice"
        metric = float(dense["summary"][metric_key])
        if metric > best_metric:
            best_metric = metric
            patience = 0
            save_checkpoint(
                best_path,
                model,
                optimizer,
                model_type=model_type,
                phase=phase,
                epoch=epoch,
                metric=metric,
                cfg=cfg,
            )
        else:
            patience += 1
        print(
            f"[dense-val] phase={phase} epoch={epoch} {metric_key}={metric:.6f} "
            f"best={best_metric:.6f}",
            flush=True,
        )
        if patience >= int(training.get("early_stopping_patience", 3)):
            break
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=["escb", "baseline"], required=True)
    parser.add_argument("--experiment-name")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    os.environ["DU2VOX_SHARED_DIR"] = str(cfg["data"]["shared_dir"])
    if cfg["data"].get("allow_stale_frame_manifest", False):
        os.environ["DU2VOX_ALLOW_STALE_FRAME_MANIFEST"] = "1"
    if cfg["data"].get("frame_manifest_sha256"):
        os.environ["DU2VOX_FRAME_MANIFEST_SHA256"] = str(cfg["data"]["frame_manifest_sha256"])
    training = cfg["training"]
    seed_all(int(training.get("seed", 20260831)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if float(cfg["loss"]["lambda_inv"]) <= 0 or float(cfg["loss"]["lambda_repr"]) <= 0:
        raise ValueError("ESCB joint lambda_inv and lambda_repr must both be positive")
    train_ids = load_ids(cfg["data"]["train_split"])
    val_ids = load_ids(cfg["data"]["val_split"])
    if args.max_samples is not None:
        train_ids = train_ids[: args.max_samples]
    if args.max_val_samples is not None:
        val_ids = val_ids[: args.max_val_samples]
    if args.smoke:
        train_ids = train_ids[: min(4, len(train_ids))]
        val_ids = val_ids[: min(4, len(val_ids))]
    experiment = args.experiment_name or f"{cfg['experiment']['name']}_{args.model}"
    run_dir = Path(cfg.get("runs_root", "runs")) / experiment
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, run_dir / "config.yaml")

    n_query = int(training["n_query_points"])
    if args.smoke:
        n_query = min(n_query, 1024)
    train_dataset = build_dataset(cfg, "train", train_ids, n_query)
    val_dataset = build_dataset(cfg, "val", val_ids, n_query)
    canonical = CanonicalCrossDiscretization(
        cfg["data"]["operator_cache"],
        shared_dir=cfg["data"]["shared_dir"],
        factorize=True,
    )
    model = build_model(cfg, args.model, cfg["data"]["shared_dir"]).to(device)
    resume_checkpoint = None
    if args.resume is not None:
        resume_checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if resume_checkpoint.get("model_type") != args.model:
            raise RuntimeError("Resume checkpoint model type does not match")
        model.load_state_dict(resume_checkpoint["model"])
    parameter_counts = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    (run_dir / "model_info.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "parameters": parameter_counts,
                "train_samples": len(train_ids),
                "val_samples": len(val_ids),
                "fixed_domain_points": canonical.p.shape[0],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[model] {args.model} {parameter_counts} device={device}", flush=True)

    if args.model == "baseline":
        phases = [("baseline", int(training["baseline_epochs"]))]
    else:
        phases = [
            ("A", int(training["phase_a_epochs"])),
            ("B", int(training["phase_b_epochs"])),
            ("C", int(training["phase_c_epochs"])),
        ]
    if args.smoke:
        phases = [(phase, 1) for phase, _ in phases]
    completed = []
    resume_phase = resume_checkpoint.get("phase") if resume_checkpoint else None
    reached_resume_phase = resume_phase is None
    for phase, epochs in phases:
        if not reached_resume_phase:
            if phase != resume_phase:
                continue
            reached_resume_phase = True
        best = train_phase(
            model=model,
            model_type=args.model,
            phase=phase,
            epochs=epochs,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            canonical=canonical,
            cfg=cfg,
            run_dir=run_dir,
            device=device,
            max_val_samples=len(val_ids),
            resume_checkpoint=(resume_checkpoint if phase == resume_phase else None),
        )
        resume_checkpoint = None
        completed.append({"phase": phase, "checkpoint": str(best)})
    (run_dir / "training_complete.json").write_text(
        json.dumps({"model": args.model, "phases": completed}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
