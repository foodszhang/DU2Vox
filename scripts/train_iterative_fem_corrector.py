#!/usr/bin/env python3
"""Train the residual-aware iterative FEM inverse-state corrector."""

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

from du2vox.bridge.canonical_cross_discretization import CanonicalCrossDiscretization
from du2vox.data.error_structured_dataset import ErrorStructuredFixedDomainDataset
from du2vox.evaluation.fem_volume import lumped_nodal_volumes
from du2vox.evaluation.iterative_fem import (
    evaluate_iterative_fem_dense,
    save_iterative_fem_result,
)
from du2vox.models.stage2.iterative_fem_corrector import (
    IterativeResidualAwareFEMCorrector,
    ResidualConsistentIterativeFEMCorrector,
    UnifiedDualEvidenceFEMCorrector,
)
from du2vox.utils.fem_system import load_fem_system_matrix
from du2vox.utils.frame import FrameManifest


def load_ids(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


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
        load_measurement=True,
        normalize_measurement=bool(data.get("normalize_b", True)),
        measurement_operator_scaling=data.get("measurement_operator_scaling", "none"),
        use_visible_mask=bool(data.get("use_visible_mask", False)),
        # Must match the contract used when precomputing projection_targets_dir.
        gt_mode=data.get("gt_mode", "binary"),
        normalize_gt=data.get("normalize_gt", "none"),
        normalization_scale_filename=data.get("normalization_scale_filename"),
        binary_threshold=float(data.get("binary_threshold", 0.05)),
        # Share of queries drawn from the signal region; see the dataset for why
        # uniform sampling is inadequate for a continuous target.
        signal_query_fraction=float(data.get("signal_query_fraction", 0.0)),
        metric_data_range=float(data.get("data_range", 2.0)),
    )


def build_model(cfg: dict[str, Any]) -> IterativeResidualAwareFEMCorrector:
    data = cfg["data"]
    model_cfg = cfg["model"]
    view_encoder = None
    view_dim = 0
    if model_cfg.get("view_encoder", True):
        # Imported only after main() has installed the config's frame environment.
        from du2vox.models.stage2.view_encoder import ViewEncoderModule

        view_dim = int(model_cfg.get("view_feat_dim", 32))
        view_encoder = ViewEncoderModule(
            view_feat_dim=view_dim,
            fusion_method=model_cfg.get("fusion_method", "attn"),
            encoder_out_channels=int(model_cfg.get("encoder_out_channels", 32)),
            encoder_base_channels=int(model_cfg.get("encoder_base_channels", 32)),
            projection_transform=model_cfg.get("view_projection_transform", "none"),
            multiscale_cfg=model_cfg.get("view_multiscale", {}),
        )
    system_matrix = load_fem_system_matrix(
        data["shared_dir"], use_visible_mask=bool(data.get("use_visible_mask", False))
    )
    knn = torch.from_numpy(np.load(Path(data["shared_dir"]) / "knn_idx_full.npy"))
    variant = model_cfg.get("variant", "v1")
    if variant == "unified_dual_evidence_v4":
        model_class = UnifiedDualEvidenceFEMCorrector
    elif variant in {"residual_consistent_v2", "scale_calibrated_proximal_v3"}:
        model_class = ResidualConsistentIterativeFEMCorrector
    else:
        model_class = IterativeResidualAwareFEMCorrector
    extra: dict[str, float] = {}
    if model_class is ResidualConsistentIterativeFEMCorrector:
        extra = {
            "max_neural_update": float(model_cfg.get("max_neural_update", 0.5)),
            "max_dc_update": float(model_cfg.get("max_dc_update", 0.1)),
            "residual_tolerance": float(model_cfg.get("residual_tolerance", 0.0)),
            "hard_trust_region": variant == "residual_consistent_v2",
        }
    elif model_class is UnifiedDualEvidenceFEMCorrector:
        volume_center = bool(model_cfg.get("volume_center_neural_update", False))
        nodal_volume_weights = None
        if volume_center:
            nodes, elements = FrameManifest.load_mesh_nodes(data["shared_dir"])
            nodal_volume_weights = torch.from_numpy(
                lumped_nodal_volumes(nodes, elements).astype(np.float32)
            )
        extra = {
            "max_neural_update": float(model_cfg.get("max_neural_update", 0.25)),
            "max_dc_update": float(model_cfg.get("max_dc_update", 0.1)),
            "nodal_volume_weights": nodal_volume_weights,
            "volume_center_neural_update": volume_center,
        }
    return model_class(
        system_matrix=system_matrix,
        knn_indices=knn,
        hidden_dim=int(model_cfg.get("hidden_dim", 144)),
        n_context_blocks=int(model_cfg.get("context_blocks", 2)),
        n_iterations=int(model_cfg.get("iterations", 3)),
        view_feat_dim=view_dim,
        view_encoder=view_encoder,
        **extra,
    )


def cosine_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(prediction, target, dim=-1, eps=1e-8).mean()


def masked_case_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean within each case and then across cases, with an explicit stratum."""

    weights = mask.to(values.dtype)
    counts = weights.sum(dim=-1)
    if torch.any(counts <= 0):
        raise RuntimeError("Explicit query-loss stratum is empty for at least one case")
    return ((values * weights).sum(dim=-1) / counts).mean()


def fem_mass_relative_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    nodal_volume_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Relative integrated-field loss under the P1 lumped FEM volume measure."""

    weights = nodal_volume_weights.to(prediction.dtype)
    prediction_mass = (prediction * weights).sum(dim=-1)
    target_mass = (target * weights).sum(dim=-1)
    relative_error = (prediction_mass - target_mass) / target_mass.abs().clamp_min(1e-8)
    return relative_error.square().mean(), prediction_mass / target_mass.clamp_min(1e-8)


def compute_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, Any],
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    states = output["step_states"]
    target_state = batch["pi_gt"]
    inverse_target = target_state - batch["x_h"]
    n_steps = states.shape[1]
    configured_weights = loss_cfg.get("step_weights", None)
    if configured_weights is None:
        weights = torch.arange(1, n_steps + 1, device=states.device, dtype=states.dtype)
    else:
        if len(configured_weights) != n_steps:
            raise ValueError("loss.step_weights must match model.iterations")
        weights = states.new_tensor(configured_weights)
    weights = weights / weights.sum()
    step_mses = torch.stack([F.mse_loss(states[:, step], target_state) for step in range(n_steps)])
    deep_mse = (weights * step_mses).sum()
    final_state_mse = step_mses[-1]
    total_correction = states[:, -1] - batch["x_h"]
    inverse_cosine = cosine_loss(total_correction, inverse_target)
    projected_mse = F.mse_loss(output["final_prediction"], batch["pi_gt_voxel"])
    temperature = float(loss_cfg.get("support_temperature", 0.1))
    # Support is defined RELATIVE TO THE PER-CASE PEAK so that it is exactly the
    # set the Dice@50% metric scores. An absolute intensity cut is only equivalent
    # when every case peaks at 1, which no longer holds under the raw-amplitude
    # continuous contract (LPR peaks run to ~1.4). Dividing both sides by the peak
    # also leaves ``support_temperature`` on the dimensionless intensity scale it
    # was originally defined on.
    support_fraction = float(loss_cfg.get("support_fraction", 0.5))
    gt_peak = batch["gt_peak"].reshape(-1, 1).clamp_min(1e-8)
    support_probability = torch.sigmoid(
        (output["final_prediction"] / gt_peak - support_fraction) / temperature
    )
    # Tversky scores set overlap, so it needs a support mask rather than a soft
    # intensity field. Under the binary contract the peak is exactly 1, so this
    # reduces to ``gt >= 0.5``; under the continuous contract it yields the
    # peak-relative support instead of letting a faint voxel contribute only
    # fractionally to the true positives.
    target_support = (batch["gt"] / gt_peak >= support_fraction).to(support_probability.dtype)
    reduce_dims = tuple(range(1, support_probability.ndim))
    true_positive = (support_probability * target_support).sum(dim=reduce_dims)
    false_positive = (support_probability * (1.0 - target_support)).sum(dim=reduce_dims)
    false_negative = ((1.0 - support_probability) * target_support).sum(dim=reduce_dims)
    alpha = float(loss_cfg.get("tversky_alpha", 0.5))
    beta = float(loss_cfg.get("tversky_beta", 0.5))
    support_loss = (
        1.0
        - (
            (true_positive + 1e-6)
            / (true_positive + alpha * false_positive + beta * false_negative + 1e-6)
        ).mean()
    )
    residual_ratio = output["measurement_residual_rms"][:, -1] / output["measurement_residual_rms"][
        :, 0
    ].detach().clamp_min(1e-8)
    residual_margin = float(loss_cfg.get("residual_target_ratio", 1.0))
    residual_excess = F.relu(residual_ratio - residual_margin).square().mean()

    # ── Continuous intensity losses, on the voxel domain ────────────────────
    # batch["gt"] is the GT target at the sampled queries. Under the historical
    # binary contract it is 0/1 and these terms only refine the support
    # boundary; under the continuous contract it carries the actual source
    # profile and per-focus amplitude, which is what makes the model regress
    # intensity instead of merely detecting support.
    voxel_prediction = output["final_prediction"]
    voxel_target = batch["gt"]
    voxel_mse = F.mse_loss(voxel_prediction, voxel_target)
    strata = batch.get("query_sampling_stratum")
    voxel_global_mse = voxel_mse.new_zeros(())
    voxel_source_mse = voxel_mse.new_zeros(())
    if strata is not None and (
        float(loss_cfg.get("lambda_voxel_global_mse", 0.0)) > 0.0
        or float(loss_cfg.get("lambda_voxel_source_mse", 0.0)) > 0.0
    ):
        squared_error = (voxel_prediction - voxel_target).square()
        if float(loss_cfg.get("lambda_voxel_global_mse", 0.0)) > 0.0:
            voxel_global_mse = masked_case_mean(squared_error, strata == 0)
        if float(loss_cfg.get("lambda_voxel_source_mse", 0.0)) > 0.0:
            voxel_source_mse = masked_case_mean(squared_error, strata == 1)
    # Scale-aware: unlike MSE this does not let a strongly fluorescing sample
    # dominate the gradient, which matters because per-sample yield varies.
    voxel_relative_l2 = torch.linalg.vector_norm(
        voxel_prediction - voxel_target
    ) / torch.linalg.vector_norm(voxel_target).clamp_min(1e-8)
    # Shape agreement independent of overall scale.
    field_cosine = cosine_loss(voxel_prediction, voxel_target)
    # Peak recovery. The source peak is the headline quantitative target, and a
    # mean over the top-k queries is a differentiable stand-in for max.
    peak_fraction = float(loss_cfg.get("peak_fraction", 0.01))
    n_peak = max(1, int(voxel_target.shape[-1] * peak_fraction))
    peak_prediction = voxel_prediction.topk(n_peak, dim=-1).values.mean(dim=-1)
    peak_target = voxel_target.topk(n_peak, dim=-1).values.mean(dim=-1).detach()
    peak_loss = F.mse_loss(peak_prediction, peak_target)
    # Non-negativity prior on the physically-interpretable output only. The
    # nodal reference state inside the approximation space is expected to carry
    # negative coefficients (that is what makes it the L2 optimum), so this term
    # deliberately constrains the P1-lifted voxel field rho_hat rather than the
    # state. Gradients still reach the nodes through the canonical P1 transfer,
    # so CoarseFEM-Voxel separation is preserved. A fluorophore concentration
    # cannot be negative; without this, the data-consistency direction (a signed
    # A^T(y - Ax), which pushes down on most nodes because Stage 1 overshoots)
    # has no counterweight. See diagnosis/lpr_negativity_audit_val300.json.
    voxel_negativity = torch.clamp(voxel_prediction, max=0.0).square().mean()
    negative_mass_excess = voxel_mse.new_zeros(())
    if strata is not None and float(loss_cfg.get("lambda_negative_mass_excess", 0.0)) > 0.0:
        global_mask = (strata == 0).to(voxel_prediction.dtype)
        global_count = global_mask.sum(dim=-1).clamp_min(1.0)
        prediction_negative_mean = (
            F.relu(-voxel_prediction) * global_mask
        ).sum(dim=-1) / global_count
        oracle_negative_mean = (
            F.relu(-batch["pi_gt_voxel"]) * global_mask
        ).sum(dim=-1) / global_count
        volume_weights = batch["nodal_volume_weights"].to(target_state.dtype)
        target_field_mean = (target_state * volume_weights).sum(dim=-1) / volume_weights.sum(
            dim=-1
        ).clamp_min(1e-8)
        prediction_negative_ratio = prediction_negative_mean / target_field_mean.clamp_min(1e-8)
        oracle_negative_ratio = oracle_negative_mean / target_field_mean.clamp_min(1e-8)
        # Sampled-L2 projection itself has small negative side lobes. Penalize
        # only excess negative mass beyond that case-specific approximation-space
        # floor instead of forcing an unattainable zero-negative FEM state.
        negative_mass_excess = F.relu(
            prediction_negative_ratio - oracle_negative_ratio.detach()
        ).square().mean()
    fem_mass_loss = voxel_mse.new_zeros(())
    fem_mass_ratio = voxel_mse.new_ones(())
    if float(loss_cfg.get("lambda_fem_mass", 0.0)) > 0.0:
        fem_mass_loss, per_case_mass_ratio = fem_mass_relative_loss(
            states[:, -1], target_state, batch["nodal_volume_weights"]
        )
        fem_mass_ratio = per_case_mass_ratio.mean()

    total = (
        float(loss_cfg.get("lambda_final_state", 1.0)) * final_state_mse
        + float(loss_cfg.get("lambda_deep_supervision", 0.5)) * deep_mse
        + float(loss_cfg.get("lambda_projected", 0.5)) * projected_mse
        + float(loss_cfg.get("lambda_cosine", 0.05)) * inverse_cosine
        + float(loss_cfg.get("lambda_support", 0.0)) * support_loss
        + float(loss_cfg.get("lambda_data_consistency", 0.0)) * residual_excess
        + float(loss_cfg.get("lambda_voxel_mse", 0.0)) * voxel_mse
        + float(loss_cfg.get("lambda_voxel_relative_l2", 0.0)) * voxel_relative_l2
        + float(loss_cfg.get("lambda_field_cosine", 0.0)) * field_cosine
        + float(loss_cfg.get("lambda_peak", 0.0)) * peak_loss
        + float(loss_cfg.get("lambda_voxel_global_mse", 0.0)) * voxel_global_mse
        + float(loss_cfg.get("lambda_voxel_source_mse", 0.0)) * voxel_source_mse
        + float(loss_cfg.get("lambda_negativity", 0.0)) * voxel_negativity
        + float(loss_cfg.get("lambda_negative_mass_excess", 0.0)) * negative_mass_excess
        + float(loss_cfg.get("lambda_fem_mass", 0.0)) * fem_mass_loss
    )
    details = {
        "final_state_mse": float(final_state_mse.detach()),
        "deep_state_mse": float(deep_mse.detach()),
        "projected_target_mse": float(projected_mse.detach()),
        "inverse_cosine_loss": float(inverse_cosine.detach()),
        "support_tversky_loss": float(support_loss.detach()),
        "profiled_residual_excess": float(residual_excess.detach()),
        "voxel_mse": float(voxel_mse.detach()),
        "voxel_global_mse": float(voxel_global_mse.detach()),
        "voxel_source_mse": float(voxel_source_mse.detach()),
        "voxel_relative_l2": float(voxel_relative_l2.detach()),
        "field_cosine_loss": float(field_cosine.detach()),
        "peak_loss": float(peak_loss.detach()),
        "voxel_negativity": float(voxel_negativity.detach()),
        "negative_mass_excess": float(negative_mass_excess.detach()),
        "fem_mass_relative_squared": float(fem_mass_loss.detach()),
        "fem_mass_ratio": float(fem_mass_ratio.detach()),
        "measurement_residual_ratio": float(
            (
                output["measurement_residual_rms"][:, -1]
                / output["measurement_residual_rms"][:, 0].clamp_min(1e-8)
            )
            .mean()
            .detach()
        ),
    }
    for step, value in enumerate(step_mses, start=1):
        details[f"step{step}_state_mse"] = float(value.detach())
    if "accepted_step_scale" in output:
        details["accepted_step_scale"] = float(output["accepted_step_scale"].mean().detach())
    return total, details


def save_checkpoint(
    path: Path,
    model: IterativeResidualAwareFEMCorrector,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    metric: float,
    cfg: dict[str, Any],
    model_type: str,
) -> None:
    selection_metric = cfg.get("validation", {}).get(
        "selection_metric", f"step{int(cfg['model'].get('iterations', 3))}_dice"
    )
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_type": model_type,
        "epoch": epoch,
        "dense_val_metric": metric,
        "selection_metric": selection_metric,
        "config": cfg,
    }
    if selection_metric.endswith("_dice"):
        payload["dense_val_dice"] = metric
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-name")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    data = cfg["data"]
    training = cfg["training"]
    os.environ["DU2VOX_SHARED_DIR"] = str(data["shared_dir"])
    if data.get("allow_stale_frame_manifest", False):
        os.environ["DU2VOX_ALLOW_STALE_FRAME_MANIFEST"] = "1"
    if data.get("frame_manifest_sha256"):
        os.environ["DU2VOX_FRAME_MANIFEST_SHA256"] = str(data["frame_manifest_sha256"])
    seed_all(int(training.get("seed", 20260831)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ids = load_ids(data["train_split"])
    val_ids = load_ids(data["val_split"])
    if args.max_samples is not None:
        train_ids = train_ids[: args.max_samples]
    if args.max_val_samples is not None:
        val_ids = val_ids[: args.max_val_samples]
    if args.smoke:
        # Explicit smoke bounds are authoritative. The LPR protocol requests a
        # 20/8 smoke; silently shrinking it to 4/4 leaves only one training batch
        # and makes stability diagnostics dominated by a single update.
        if args.max_samples is None:
            train_ids = train_ids[:4]
        if args.max_val_samples is None:
            val_ids = val_ids[:4]
    n_query = int(training.get("n_query_points", 8192))
    if args.smoke:
        n_query = min(n_query, 1024)
    train_dataset = build_dataset(cfg, "train", train_ids, n_query)
    val_dataset = build_dataset(cfg, "val", val_ids, n_query)
    canonical = CanonicalCrossDiscretization(
        data["operator_cache"], shared_dir=data["shared_dir"], factorize=False
    )
    model = build_model(cfg).to(device)
    variant = cfg["model"].get("variant", "v1")
    configured_model_type = {
        "residual_consistent_v2": "residual_consistent_iterative_fem_v2",
        "scale_calibrated_proximal_v3": "scale_calibrated_proximal_fem_v3",
        "unified_dual_evidence_v4": "unified_dual_evidence_fem_v4",
    }.get(variant, "iterative_residual_aware_fem")
    initialization_report: dict[str, Any] | None = None
    initialization_path = training.get("initialization_checkpoint")
    if initialization_path and args.resume is None:
        source = torch.load(initialization_path, map_location=device, weights_only=False)
        source_state = dict(source["model"])
        target_state = model.state_dict()
        if variant == "unified_dual_evidence_v4":
            node_key = "cell.node_projection.weight"
            if node_key in source_state and node_key in target_state:
                source_weight = source_state[node_key]
                target_weight = torch.zeros_like(target_state[node_key])
                target_weight[:, :10] = source_weight[:, :10]
                target_weight[:, 12] = source_weight[:, 12]
                target_weight[:, 13:15] = source_weight[:, 10:12]
                target_weight[:, 15:] = source_weight[:, 13:]
                source_state[node_key] = target_weight
            film_key = "cell.global_film.0.weight"
            if film_key in source_state and film_key in target_state:
                source_weight = source_state[film_key]
                target_weight = torch.zeros_like(target_state[film_key])
                hidden_twice = target_weight.shape[1] - 6
                target_weight[:, : hidden_twice + 2] = source_weight[:, : hidden_twice + 2]
                # V3's log profiled-residual feature initializes V4's SI slot.
                target_weight[:, hidden_twice + 2] = source_weight[:, hidden_twice + 3]
                source_state[film_key] = target_weight
        compatible = {
            key: value
            for key, value in source_state.items()
            if key in target_state and target_state[key].shape == value.shape
        }
        incompatible = sorted(set(source_state) - set(compatible))
        model.load_state_dict(compatible, strict=False)
        initialization_report = {
            "checkpoint": str(Path(initialization_path).resolve()),
            "loaded_tensors": len(compatible),
            "skipped_tensors": incompatible,
        }
        print(f"[initialization] {initialization_report}", flush=True)
    if (
        model.measurement_residual.system_matrix.shape[0]
        != train_dataset[0]["measurement_b"].shape[0]
    ):
        raise RuntimeError("Measurement vector and system matrix conventions differ")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("lr", 2e-4)),
        weight_decay=float(training.get("weight_decay", 1e-5)),
    )
    start_epoch = 1
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("model_type") != configured_model_type:
            raise RuntimeError("Resume checkpoint has the wrong model type")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # The config is authoritative for a resumed stability run. AdamW otherwise
        # silently restores the old checkpoint LR and ignores a deliberate decay.
        resumed_lr = float(training.get("lr", 2e-4))
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = resumed_lr
        start_epoch = int(checkpoint["epoch"]) + 1

    experiment = args.experiment_name or cfg["experiment"]["name"]
    run_dir = Path(cfg.get("runs_root", "runs")) / experiment
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, run_dir / "config.yaml")
    counts = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "view_encoder": sum(parameter.numel() for parameter in model.view_encoder.parameters())
        if model.view_encoder is not None
        else 0,
        "fem_corrector": sum(parameter.numel() for parameter in model.cell.parameters()),
        "learned_transfer": 0,
        "voxel_head": 0,
    }
    (run_dir / "model_info.json").write_text(
        json.dumps(
            {
                "model": configured_model_type,
                "parameters": counts,
                "iterations": model.n_iterations,
                "train_samples": len(train_ids),
                "val_samples": len(val_ids),
                "initialization": initialization_report,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[model] parameters={counts} device={device}", flush=True)

    loader = DataLoader(
        train_dataset,
        batch_size=int(training.get("batch_size", 4)),
        shuffle=True,
        num_workers=int(training.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )
    amp_enabled = bool(training.get("amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if training.get("amp_dtype", "bf16") == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    epochs = int(training.get("epochs", 25))
    if args.max_epochs is not None:
        epochs = min(epochs, args.max_epochs)
    if args.smoke:
        epochs = min(epochs, 2)
    best_path = run_dir / "checkpoints" / "best_dense_val_delta_dice.pth"
    best_metric = -float("inf")
    if best_path.exists() and args.resume is not None:
        previous_best = torch.load(best_path, map_location="cpu", weights_only=False)
        best_metric = float(
            previous_best["dense_val_metric"]
            if "dense_val_metric" in previous_best
            else previous_best["dense_val_dice"]
        )
    patience = 0
    gradient_checked = False
    for epoch in range(start_epoch, epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        totals: dict[str, float] = {}
        batches = 0
        for batch in loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype):
                output = model(
                    batch["x_h"],
                    batch["measurement_b"],
                    batch["node_coords_norm"],
                    batch["query_node_indices"],
                    batch["query_barycentric"],
                    node_coords_world=batch["node_coords_world"],
                    proj_imgs=batch.get("proj_imgs"),
                    measurement_operator_scale=batch.get("measurement_operator_scale"),
                )
                loss, details = compute_loss(output, batch, cfg["loss"])
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch {epoch}; "
                    "the last finite checkpoint was preserved"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if not gradient_checked:
                grad = sum(
                    parameter.grad.detach().abs().sum().item()
                    for parameter in model.cell.parameters()
                    if parameter.grad is not None
                )
                if grad <= 0:
                    raise RuntimeError("FEM corrector received no gradient")
                if any(model.measurement_residual.parameters()):
                    raise RuntimeError("Fixed measurement residual operator has parameters")
                print(f"[gradient-contract] fem_corrector_grad_l1={grad:.6g}")
                gradient_checked = True
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training.get("grad_clip_norm", 1.0))
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"Non-finite gradient norm at epoch {epoch}; optimizer step skipped"
                )
            scaler.step(optimizer)
            scaler.update()
            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach())
            for key, value in details.items():
                totals[key] = totals.get(key, 0.0) + value
            batches += 1
        means = {key: value / max(batches, 1) for key, value in totals.items()}
        print(f"[train] epoch={epoch}/{epochs} {means}", flush=True)
        save_checkpoint(
            run_dir / "checkpoints" / "last.pth",
            model,
            optimizer,
            epoch=epoch,
            metric=float("nan"),
            cfg=cfg,
            model_type=configured_model_type,
        )
        interval = int(cfg["validation"].get("dense_interval", 5))
        if epoch % interval != 0 and epoch != epochs:
            continue
        result = evaluate_iterative_fem_dense(
            model,
            val_dataset,
            canonical,
            device,
            max_samples=len(val_ids),
        )
        save_iterative_fem_result(result, run_dir / "dense_val" / f"epoch_{epoch:03d}.json")
        selection_key = cfg["validation"].get("selection_metric", f"step{model.n_iterations}_dice")
        metric = float(result["summary"][selection_key])
        if metric > best_metric:
            best_metric = metric
            patience = 0
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch=epoch,
                metric=metric,
                cfg=cfg,
                model_type=configured_model_type,
            )
        else:
            patience += 1
        print(
            f"[dense-val] epoch={epoch} {selection_key}={metric:.6f} best={best_metric:.6f}",
            flush=True,
        )
        if patience >= int(training.get("early_stopping_patience", 3)):
            break
    (run_dir / "training_complete.json").write_text(
        json.dumps(
            {"model": configured_model_type, "best_checkpoint": str(best_path)},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
