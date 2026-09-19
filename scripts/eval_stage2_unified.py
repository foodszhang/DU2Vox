#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent.parent))

from du2vox.models.stage2.cqr_residual_inr import CQRResidualINR
from du2vox.models.stage2.plain_cqr_residual_control import PlainCQRResidualControl
from du2vox.models.stage2.residual_inr import ResidualINR
from du2vox.models.stage2.transport_observability_cqr_inr import (
    TransportObservabilityCQRINR,
)
from du2vox.models.stage2.stage2_dataset import load_projection_stack
from du2vox.physics.compressed_green_operator import CompressedGreenOperator
from du2vox.utils.frame import FrameManifest


ROLE_NAMES = {0: "bg", 1: "core", 2: "halo", 3: "sentinel", 4: "proposal"}

METRIC_KEYS = [
    "s2_dice_05",
    "fem_dice_05",
    "delta_dice_05",
    "s2_iou_05",
    "fem_iou_05",
    "s2_precision_05",
    "fem_precision_05",
    "s2_recall_05",
    "fem_recall_05",
    "s2_specificity_05",
    "fem_specificity_05",
    "hd95_05",
    "component_recall_05",
    "fp_component_count_05",
    "source_localization_error_mm",
    "source_separation_success",
    "weak_source_recall_05",
    "measurement_error",
    "correction_forward_error",
    "mse_s2",
    "mse_fem",
    "residual_norm",
    "gt_pos_ratio_05",
    "s2_pos_ratio_05",
    "fem_pos_ratio_05",
    "residual_gate_mean",
    "residual_gate_gt_pos_mean",
    "residual_gate_gt_neg_mean",
    "residual_mean_gt_pos",
    "residual_mean_gt_neg",
    "raw_residual_mean",
    "raw_residual_abs_mean",
    "s2_minus_fem_mean",
    "s2_minus_fem_gt_pos_mean",
    "s2_minus_fem_gt_neg_mean",
    "core_gate_mean",
    "halo_gate_mean",
    "bg_gate_mean",
    "proposal_gate_mean",
]

ROLE_SUBSETS = ["all", "core", "core_halo", "halo", "proposal", "bg", "non_bg"]

BASE_FIELDNAMES = ["sample_id", "role_subset", "num_foci", "n_valid", *METRIC_KEYS]
ROLE_FIELDNAMES = [
    f"{name}_{suffix}"
    for suffix in ["count", "gt_pos_05", "s2_pos_05", "fem_pos_05", "dice_05", "gate_mean"]
    for name in ["bg", "core", "halo", "sentinel", "proposal"]
]


def get_residual_gate_logit_bias(model_cfg: dict[str, Any], warn_prefix: str = "[Eval]") -> float:
    if "residual_gate_logit_bias" in model_cfg:
        return float(model_cfg.get("residual_gate_logit_bias", -2.0))
    if "residual_gate_init_bias" in model_cfg:
        print(
            f"{warn_prefix}[WARN] residual_gate_init_bias is deprecated; "
            "use residual_gate_logit_bias"
        )
        return -float(model_cfg.get("residual_gate_init_bias", 2.0))
    return -2.0


def validate_residual_gate_contract(model_cfg: dict[str, Any]) -> None:
    if str(model_cfg.get("residual_gate_cap", "none")).lower() != "rgl":
        return
    prior_source = model_cfg.get("prior_source", "prior_ext")
    prior_dim = int(model_cfg.get("prior_dim", 8))
    if prior_source != "prior_lift" or prior_dim != 15:
        raise ValueError(
            "residual_gate_cap='rgl' requires model.prior_source='prior_lift' "
            "and model.prior_dim=15"
        )


def load_split(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def binary_metrics(pred: np.ndarray, gt: np.ndarray, thr: float = 0.5) -> dict[str, float]:
    eps = 1e-8
    pred_bin = pred >= thr
    gt_bin = gt >= thr
    tp = float((pred_bin & gt_bin).sum())
    fp = float((pred_bin & ~gt_bin).sum())
    fn = float((~pred_bin & gt_bin).sum())
    tn = float((~pred_bin & ~gt_bin).sum())
    pred_sum = float(pred_bin.sum())
    gt_sum = float(gt_bin.sum())
    union = float((pred_bin | gt_bin).sum())
    return {
        "dice": 2.0 * tp / (pred_sum + gt_sum + eps),
        "iou": tp / (union + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "specificity": tn / (tn + fp + eps),
        "pred_pos_ratio": float(pred_bin.mean()) if len(pred_bin) else 0.0,
        "gt_pos_ratio": float(gt_bin.mean()) if len(gt_bin) else 0.0,
    }


def _component_centers(points: np.ndarray) -> np.ndarray:
    """Return centers of nontrivial proximity components on a CQR point cloud."""

    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)
    if len(points) == 1:
        return points.astype(np.float32, copy=False)
    tree = cKDTree(points)
    nearest = tree.query(points, k=2)[0][:, 1]
    positive = nearest[np.isfinite(nearest) & (nearest > 0)]
    radius = 0.5 if len(positive) == 0 else max(0.3, 1.75 * float(np.median(positive)))
    parent = np.arange(len(points), dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for left, right in tree.query_pairs(radius):
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left
    groups: dict[int, list[int]] = {}
    for index in range(len(points)):
        groups.setdefault(find(index), []).append(index)
    kept = [indices for indices in groups.values() if len(indices) >= 3]
    if not kept:
        kept = [max(groups.values(), key=len)]
    return np.stack([points[indices].mean(axis=0) for indices in kept]).astype(np.float32)


def source_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    coords: np.ndarray,
    tumor_params: dict[str, Any] | None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Component, localization, separation, weak-source, and HD95 metrics."""

    pred_points = coords[pred >= threshold]
    gt_points = coords[gt >= threshold]
    if len(pred_points) and len(gt_points):
        pred_to_gt = cKDTree(gt_points).query(pred_points, k=1)[0]
        gt_to_pred = cKDTree(pred_points).query(gt_points, k=1)[0]
        hd95 = float(max(np.percentile(pred_to_gt, 95), np.percentile(gt_to_pred, 95)))
    else:
        hd95 = float(np.linalg.norm(np.ptp(coords, axis=0))) if len(coords) else 0.0

    pred_centers = _component_centers(pred_points)
    foci = (tumor_params or {}).get("foci", [])
    if not foci:
        true_centers = _component_centers(gt_points)
        radii = np.full(len(true_centers), 1.0, dtype=np.float32)
        intensities = np.ones(len(true_centers), dtype=np.float32)
    else:
        true_centers = np.asarray([focus["center"] for focus in foci], dtype=np.float32)

        def focus_extent(focus: dict[str, Any], axis: str) -> float:
            params = focus.get("params", {}) or {}
            for value in (
                focus.get(axis),
                params.get(axis),
                focus.get("radius"),
                params.get("radius"),
                1.0,
            ):
                if value is not None:
                    return float(value)
            return 1.0

        radii = np.asarray(
            [
                max(
                    focus_extent(focus, "rx"),
                    focus_extent(focus, "ry"),
                    focus_extent(focus, "rz"),
                )
                for focus in foci
            ],
            dtype=np.float32,
        )
        intensities = np.asarray(
            [float((focus.get("params", {}) or {}).get("intensity") or 1.0) for focus in foci],
            dtype=np.float32,
        )

    if len(true_centers) == 0:
        return {
            "hd95_05": hd95,
            "component_recall_05": 1.0,
            "fp_component_count_05": float(len(pred_centers)),
            "source_localization_error_mm": 0.0,
            "source_separation_success": 1.0,
            "weak_source_recall_05": 1.0,
        }
    if len(pred_centers) == 0:
        diag = float(np.linalg.norm(np.ptp(coords, axis=0))) if len(coords) else 0.0
        return {
            "hd95_05": hd95,
            "component_recall_05": 0.0,
            "fp_component_count_05": 0.0,
            "source_localization_error_mm": diag,
            "source_separation_success": 0.0,
            "weak_source_recall_05": 0.0,
        }

    distances = np.linalg.norm(true_centers[:, None, :] - pred_centers[None, :, :], axis=-1)
    nearest_pred = distances.argmin(axis=1)
    nearest_distance = distances[np.arange(len(true_centers)), nearest_pred]
    matched = nearest_distance <= 1.5 * radii
    fp = sum(
        np.all(np.linalg.norm(true_centers - center[None, :], axis=1) > 1.5 * radii)
        for center in pred_centers
    )
    weak_index = int(np.argmin(intensities))
    weak_center = true_centers[weak_index]
    weak_radius = max(float(radii[weak_index]), 1e-6)
    weak_mask = np.linalg.norm(coords - weak_center, axis=1) <= weak_radius
    weak_gt = weak_mask & (gt >= threshold)
    weak_recall = (
        float(((pred >= threshold) & weak_gt).sum() / weak_gt.sum())
        if weak_gt.any()
        else float(matched[weak_index])
    )
    separation = float(matched.all() and len(set(nearest_pred.tolist())) == len(true_centers))
    return {
        "hd95_05": hd95,
        "component_recall_05": float(matched.mean()),
        "fp_component_count_05": float(fp),
        "source_localization_error_mm": float(nearest_distance.mean()),
        "source_separation_success": separation,
        "weak_source_recall_05": weak_recall,
    }


def make_role_mask(role: np.ndarray, subset: str) -> np.ndarray:
    if subset == "all":
        return np.ones_like(role, dtype=bool)
    if subset == "core":
        return role == 1
    if subset == "core_halo":
        return (role == 1) | (role == 2)
    if subset == "halo":
        return role == 2
    if subset == "proposal":
        return role == 4
    if subset == "bg":
        return role == 0
    if subset == "non_bg":
        return role != 0
    raise ValueError(subset)


def mean_dict(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, float]:
    out = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if key in row and row[key] != ""]
        out[key] = float(np.mean(vals)) if vals else 0.0
    return out


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_num_foci(precomputed_dir: Path, samples_dir: Path | None, sample_id: str) -> int:
    npz_path = precomputed_dir / f"{sample_id}.npz"
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as data:
            for key in ["num_foci", "n_foci", "foci_count"]:
                if key in data.files:
                    return int(np.asarray(data[key]).reshape(-1)[0])

    if samples_dir is None:
        return -1

    sample_dir = samples_dir / sample_id
    for name in ["meta.json", "tumor_params.json"]:
        meta = read_json(sample_dir / name)
        if not meta:
            continue
        for key in ["num_foci", "n_foci", "foci_count"]:
            if key in meta:
                return int(meta[key])
        for key in ["foci", "centers", "sources"]:
            if isinstance(meta.get(key), list):
                return len(meta[key])
    return -1


def build_model(
    cfg: dict[str, Any], device: torch.device
) -> tuple[torch.nn.Module, torch.nn.Module | None]:
    prior_dim = int(cfg["model"].get("prior_dim", 8))
    model_type = cfg["model"].get("model_type", "")
    use_transport_model = model_type == "transport_observability_cqr_inr"
    use_plain_control = model_type == "plain_cqr_residual_control"
    use_cqr_model = (
        use_transport_model
        or use_plain_control
        or (model_type == "cqr_residual_inr")
        or (prior_dim > 8)
    )
    if use_transport_model:
        model_cls = TransportObservabilityCQRINR
    elif use_plain_control:
        model_cls = PlainCQRResidualControl
    else:
        model_cls = CQRResidualINR if use_cqr_model else ResidualINR
    cqr_ratios = cfg.get("cqr", {}).get("ratios", {}) or {}
    default_num_bands = 5 if float(cqr_ratios.get("proposal", 0.0)) > 0.0 else 4

    view_encoder = None
    view_feat_dim = 0
    if cfg["model"].get("view_encoder", False):
        from du2vox.models.stage2.view_encoder import ViewEncoderModule

        view_feat_dim = int(cfg["model"]["view_feat_dim"])
        view_encoder = ViewEncoderModule(
            view_feat_dim=view_feat_dim,
            fusion_method=cfg["model"].get("fusion_method", "attn"),
            encoder_out_channels=cfg["model"].get("encoder_out_channels", 32),
            encoder_base_channels=cfg["model"].get("encoder_base_channels", 32),
            projection_transform=cfg["model"].get("view_projection_transform", "none"),
            multiscale_cfg=cfg["model"].get("view_multiscale", {}),
        ).to(device)

    kwargs = dict(
        n_freqs=cfg["model"]["n_freqs"],
        hidden_dim=cfg["model"]["hidden_dim"],
        n_hidden_layers=cfg["model"]["n_hidden_layers"],
        prior_dim=prior_dim,
        skip_connection=cfg["model"]["skip_connection"],
        view_feat_dim=view_feat_dim,
    )
    if model_cls in {TransportObservabilityCQRINR, PlainCQRResidualControl}:
        transport = cfg.get("transport", {}) or {}
        operator = CompressedGreenOperator.load(transport["operator_cache"])
        with np.load(Path(cfg["data"]["shared_dir"]) / "mesh.npz", allow_pickle=False) as mesh:
            elements = mesh["elements"].copy()
        extra_kwargs = {}
        if model_cls is PlainCQRResidualControl:
            extra_kwargs["lifting_mode"] = cfg["model"].get("lifting_mode", "p1")
        model = model_cls(
            green_node_modes=operator.green_node_modes,
            elements=elements,
            measurement_basis=operator.measurement_basis,
            projected_forward_modes=operator.projected_forward_modes,
            n_freqs=cfg["model"]["n_freqs"],
            hidden_dim=cfg["model"]["hidden_dim"],
            n_hidden_layers=cfg["model"]["n_hidden_layers"],
            prior_dim=prior_dim,
            view_feat_dim=view_feat_dim,
            use_band_embedding=cfg["model"].get("use_band_embedding", True),
            band_embed_dim=cfg["model"].get("band_embed_dim", 8),
            num_bands=cfg["model"].get("num_bands", default_num_bands),
            max_logit_delta=transport.get("max_logit_delta", 2.0),
            mu_relative=cfg.get("observability", {}).get("mu_relative", 1e-3),
            jitter=cfg.get("observability", {}).get("jitter", 1e-6),
            measurement_normalization=transport.get(
                "measurement_normalization", "least_squares_stage1_scale"
            ),
            **extra_kwargs,
        ).to(device)
        return model, view_encoder
    if model_cls is CQRResidualINR:
        validate_residual_gate_contract(cfg["model"])
        kwargs["residual_scale"] = cfg["model"].get("residual_scale", 1.0)
        kwargs["support_head"] = cfg["model"].get("support_head", False)
        kwargs["use_prolongation_adapter"] = cfg["model"].get("use_prolongation_adapter", False)
        kwargs["prolongation_feat_dim"] = cfg["model"].get("prolongation_feat_dim", 32)
        kwargs["use_lifting_adapter"] = cfg["model"].get("use_lifting_adapter", False)
        kwargs["lifting_feat_dim"] = cfg["model"].get("lifting_feat_dim", 32)
        kwargs["use_band_embedding"] = cfg["model"].get("use_band_embedding", False)
        kwargs["band_embed_dim"] = cfg["model"].get("band_embed_dim", 8)
        kwargs["num_bands"] = cfg["model"].get("num_bands", default_num_bands)
        kwargs["use_residual_gate"] = cfg["model"].get("use_residual_gate", False)
        kwargs["residual_gate_logit_bias"] = get_residual_gate_logit_bias(cfg["model"])
        kwargs["residual_gate_cap"] = cfg["model"].get("residual_gate_cap", "none")
        kwargs["residual_gate_cap_min"] = cfg["model"].get("residual_gate_cap_min", 0.0)
        kwargs["residual_gate_source"] = cfg["model"].get("residual_gate_source", "prior_view")
    model = model_cls(**kwargs).to(device)
    return model, view_encoder


def unpack_model_output(output):
    if isinstance(output, dict):
        return output
    d_hat, fem_interp, residual = output
    return {"d_hat": d_hat, "fem_interp": fem_interp, "residual": residual}


def select_stage2_prediction(output: dict[str, torch.Tensor], cfg: dict[str, Any]) -> torch.Tensor:
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


def select_prior(data: dict[str, np.ndarray], cfg: dict[str, Any], valid: np.ndarray) -> np.ndarray:
    prior_source = cfg["model"].get("prior_source", "prior_ext")
    prior_dim = int(cfg["model"].get("prior_dim", 8))
    expected_dim = 8 if prior_source == "prior_8d" else prior_dim
    if prior_source == "prior_8d":
        prior = data["prior_8d"]
    elif prior_source == "prior_ext":
        prior = data["prior_ext"]
    elif prior_source == "prior_lift":
        prior = data["prior_lift"]
    elif prior_source == "prior_prolong":
        prior = data["prior_prolong"]
    else:
        raise ValueError(f"Unknown prior_source: {prior_source}")
    if prior.shape[-1] != expected_dim:
        raise ValueError(
            f"prior_source={prior_source} produced dim={prior.shape[-1]}, expected={expected_dim}"
        )
    return prior.astype(np.float32)[valid]


def load_checkpoint(
    checkpoint: Path,
    model: torch.nn.Module,
    view_encoder: torch.nn.Module | None,
    device: torch.device,
) -> None:
    ckpt = torch.load(checkpoint, map_location=device)
    if isinstance(ckpt, dict):
        print(f"[Eval] checkpoint keys={list(ckpt.keys())[:20]}")
    else:
        print(f"[Eval] checkpoint type={type(ckpt).__name__}")

    if isinstance(ckpt, dict) and "residual_inr" in ckpt:
        state_dict = ckpt["residual_inr"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    if (
        isinstance(state_dict, dict)
        and "band_embedding.weight" in state_dict
        and hasattr(model, "band_embedding")
        and state_dict["band_embedding.weight"].shape != model.band_embedding.weight.shape
    ):
        old_weight = state_dict["band_embedding.weight"]
        new_weight = model.band_embedding.weight.detach().clone()
        n = min(old_weight.shape[0], new_weight.shape[0])
        new_weight[:n] = old_weight[:n]
        state_dict = dict(state_dict)
        state_dict["band_embedding.weight"] = new_weight
        print(
            "[Eval][WARN] padded band_embedding.weight "
            f"from {tuple(old_weight.shape)} to {tuple(new_weight.shape)}"
        )
    model.load_state_dict(state_dict)

    if view_encoder is not None:
        if isinstance(ckpt, dict) and "view_encoder" in ckpt:
            view_encoder.load_state_dict(ckpt["view_encoder"])
        elif isinstance(ckpt, dict) and "view_encoder_state_dict" in ckpt:
            view_encoder.load_state_dict(ckpt["view_encoder_state_dict"])
        else:
            print("[WARN] view_encoder enabled but no view_encoder weights found")


def normalize_coords(data: dict[str, np.ndarray]) -> np.ndarray:
    if "grid_coords_norm" in data:
        return data["grid_coords_norm"].astype(np.float32)
    raw = data["grid_coords"].astype(np.float32)
    bbox_min = data["bbox_min"].astype(np.float32)
    bbox_max = data["bbox_max"].astype(np.float32)
    return (2.0 * (raw - bbox_min) / (bbox_max - bbox_min + 1e-8) - 1.0).astype(np.float32)


def load_proj_imgs(samples_dir: Path, sample_id: str, cfg: dict[str, Any]) -> np.ndarray:
    data_cfg = cfg.get("data", {})
    proj_imgs, _ = load_projection_stack(
        samples_dir / sample_id,
        projection_file=data_cfg.get("projection_file", "proj.npz"),
        fallback_projection_file=data_cfg.get("fallback_projection_file"),
        projection_norm=data_cfg.get("projection_norm", "none"),
        projection_eps=data_cfg.get("projection_eps", 1.0e-8),
        projection_transform=data_cfg.get("projection_transform", "none"),
    )
    return proj_imgs[:, None, :, :]


def mcx_valid_mask(frame: FrameManifest | None, coords_world: np.ndarray) -> np.ndarray:
    if frame is None:
        return np.ones(len(coords_world), dtype=bool)
    lo = frame.mcx_bbox_min
    hi = frame.mcx_bbox_max
    return (
        (coords_world[:, 0] >= lo[0])
        & (coords_world[:, 0] <= hi[0])
        & (coords_world[:, 1] >= lo[1])
        & (coords_world[:, 1] <= hi[1])
        & (coords_world[:, 2] >= lo[2])
        & (coords_world[:, 2] <= hi[2])
    )


def load_transport_eval_arrays(
    cfg: dict[str, Any],
    data: dict[str, np.ndarray],
    valid: np.ndarray,
    samples_dir: Path | None,
    sample_id: str,
) -> dict[str, Any]:
    sidecar_root = Path(cfg["transport"]["sidecar_root"])
    sidecar_path = next(
        (
            sidecar_root / split / f"{sample_id}.npz"
            for split in ("train", "val", "test")
            if (sidecar_root / split / f"{sample_id}.npz").exists()
        ),
        None,
    )
    if sidecar_path is None:
        raise FileNotFoundError(f"No physics sidecar found for {sample_id}")
    with np.load(sidecar_path, allow_pickle=False) as sidecar:
        output = {
            "candidate_cell_weight": sidecar["candidate_cell_weight"][valid],
            "n_valid_candidate_pool": int(sidecar["n_valid_candidate_pool"]),
        }
    output["tet_ids"] = data["tet_ids"][valid]
    output["role"] = data["role"][valid]
    output["query_src_tag"] = data["query_src_tag"][valid]
    split_name = sidecar_path.parent.name
    bridge_dir = Path(cfg["data"][f"{split_name}_bridge_dir"])
    output["coarse_d"] = np.load(bridge_dir / sample_id / "coarse_d.npy").reshape(-1)
    if samples_dir is None:
        raise ValueError("samples_dir is required for transport evaluation")
    output["measurement_b"] = np.load(samples_dir / sample_id / "measurement_b.npy").reshape(-1)
    return output


def run_model_on_sample(
    model: torch.nn.Module,
    view_encoder: torch.nn.Module | None,
    cfg: dict[str, Any],
    data: dict[str, np.ndarray],
    samples_dir: Path | None,
    frame: FrameManifest | None,
    sample_id: str,
    batch_points: int,
    role_subset: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    valid = data["valid_mask"].astype(bool)
    if role_subset != "all":
        if "role" not in data:
            raise ValueError("--role_subset requires role field in batch / npz")
        valid = valid & make_role_mask(data["role"], role_subset)

    coords_norm = normalize_coords(data)[valid]
    coords_world = data["grid_coords"].astype(np.float32)[valid]
    prior = select_prior(data, cfg, valid)
    if "correction_band" in data:
        correction_band = data["correction_band"].astype(np.int64)[valid]
    elif "role" in data:
        correction_band = data["role"].astype(np.int64)[valid]
    else:
        correction_band = np.zeros(len(coords_norm), dtype=np.int64)

    if len(coords_norm) == 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            valid,
            {
                "residual_gate": np.zeros((0,), dtype=np.float32),
                "raw_residual": np.zeros((0,), dtype=np.float32),
            },
        )

    d_hat_chunks = []
    fem_chunks = []
    residual_chunks = []
    gate_chunks = []
    raw_residual_chunks = []
    physics_summary: dict[str, float] = {}
    mechanism_chunks: dict[str, list[np.ndarray]] = {}
    proj_imgs = None
    mcx_valid = None
    if view_encoder is not None:
        if samples_dir is None:
            raise ValueError("samples_dir is required for multiview evaluation")
        proj_imgs = (
            torch.from_numpy(load_proj_imgs(samples_dir, sample_id, cfg)).unsqueeze(0).to(device)
        )
        mcx_valid = mcx_valid_mask(frame, coords_world)

    transport_model = isinstance(model, (TransportObservabilityCQRINR, PlainCQRResidualControl))
    transport_arrays = None
    if transport_model:
        transport_arrays = load_transport_eval_arrays(cfg, data, valid, samples_dir, sample_id)
        # The low-rank projection couples all evaluated queries and must not be
        # recomputed independently on arbitrary inference chunks.
        batch_points = len(coords_norm)

    for start in range(0, len(coords_norm), batch_points):
        end = min(start + batch_points, len(coords_norm))
        coords_b = torch.from_numpy(coords_norm[start:end]).unsqueeze(0).to(device)
        prior_b = torch.from_numpy(prior[start:end]).unsqueeze(0).to(device)
        band_b = torch.from_numpy(correction_band[start:end]).unsqueeze(0).to(device)
        if transport_model:
            assert transport_arrays is not None
            transport_kwargs = {
                "tet_ids": torch.from_numpy(transport_arrays["tet_ids"][start:end])
                .unsqueeze(0)
                .to(device),
                "role": torch.from_numpy(transport_arrays["role"][start:end])
                .unsqueeze(0)
                .to(device),
                "query_src_tag": torch.from_numpy(transport_arrays["query_src_tag"][start:end])
                .unsqueeze(0)
                .to(device),
                "candidate_cell_weight": torch.from_numpy(
                    transport_arrays["candidate_cell_weight"][start:end]
                )
                .unsqueeze(0)
                .to(device),
                "n_valid_candidate_pool": torch.tensor(
                    [transport_arrays["n_valid_candidate_pool"]], device=device
                ),
                "coarse_d": torch.from_numpy(transport_arrays["coarse_d"]).unsqueeze(0).to(device),
                "measurement_b": torch.from_numpy(transport_arrays["measurement_b"])
                .unsqueeze(0)
                .to(device),
            }
            if view_encoder is None:
                output = unpack_model_output(
                    model(coords_b, prior_b, correction_band=band_b, **transport_kwargs)
                )
            else:
                world_b = torch.from_numpy(coords_world[start:end]).unsqueeze(0).to(device)
                view_feat, _ = view_encoder(proj_imgs, world_b, coords_vox_norm=None)
                valid_b = torch.from_numpy(mcx_valid[start:end]).unsqueeze(0).to(device)
                view_feat = view_feat * valid_b.unsqueeze(-1).float()
                output = unpack_model_output(
                    model(
                        coords_b,
                        prior_b,
                        view_feat,
                        correction_band=band_b,
                        **transport_kwargs,
                    )
                )
        elif view_encoder is None:
            output = unpack_model_output(model(coords_b, prior_b, correction_band=band_b))
        else:
            world_b = torch.from_numpy(coords_world[start:end]).unsqueeze(0).to(device)
            view_feat, _ = view_encoder(proj_imgs, world_b, coords_vox_norm=None)
            valid_b = torch.from_numpy(mcx_valid[start:end]).unsqueeze(0).to(device)
            view_feat = view_feat * valid_b.unsqueeze(-1).float()
            output = unpack_model_output(
                model(coords_b, prior_b, view_feat, correction_band=band_b)
            )
        d_hat_b = select_stage2_prediction(output, cfg)
        fem_b = output["fem_interp"]
        residual_b = output["residual"]
        gate_b = output.get("residual_gate", torch.zeros_like(residual_b))
        raw_residual_b = output.get("raw_residual", residual_b)
        if "a_query" in output and "rho0" in output:
            correction_b = output.get(
                "plain_correction",
                output.get("partition_correction", output["d_hat"] - output["rho0"]),
            )
            gt_b = (
                torch.from_numpy(data["gt_values"].astype(np.float32)[valid][start:end])
                .unsqueeze(0)
                .to(device)
            )
            target_correction_b = gt_b - output["rho0"]
            pred_modes = (output["a_query"].float() @ correction_b.float().unsqueeze(-1)).squeeze(
                -1
            )
            target_modes = (
                output["a_query"].float() @ target_correction_b.float().unsqueeze(-1)
            ).squeeze(-1)
            corr_error = torch.sqrt(
                (pred_modes - target_modes).square().sum(dim=-1)
                / (target_modes.square().sum(dim=-1) + 1e-8)
            ).mean()
            physics_summary["measurement_error"] = float(
                torch.sqrt(output["data_relative_after"].clamp_min(0.0)).mean()
            )
            physics_summary["correction_forward_error"] = float(corr_error)
            if "alpha" in output:
                barycentric = prior_b[..., 4:8]
                alpha = output["alpha"]
                physics_summary["alpha_lambda_l1"] = float(torch.abs(alpha - barycentric).mean())
                physics_summary["alpha_entropy"] = float(
                    (-(alpha.clamp_min(1e-8) * alpha.clamp_min(1e-8).log()).sum(dim=-1)).mean()
                )
                mechanism_chunks.setdefault("alpha_lambda_deviation", []).append(
                    torch.abs(alpha - barycentric)
                    .mean(dim=-1)
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .float()
                    .numpy()
                )
            rho_delta = output["rho0"] - output["fem_interp"]
            physics_summary["rho_tc_p1_l1"] = float(rho_delta.abs().mean())
            physics_summary["rho_tc_p1_l2"] = float(torch.sqrt(rho_delta.square().mean()))
            physics_summary["transport_error"] = float(output["transport_error"].mean())
            mechanism_chunks.setdefault("rho_tc_minus_p1", []).append(
                rho_delta.squeeze(0).detach().cpu().float().numpy()
            )
            mechanism_chunks.setdefault("correction", []).append(
                correction_b.squeeze(0).detach().cpu().float().numpy()
            )
            if "observable_correction" in output:
                observable = output["observable_correction"]
                ambiguous = output["ambiguous_correction"]
                combined_energy = (observable + ambiguous).square().sum() + 1e-8
                physics_summary["observable_energy_fraction"] = float(
                    observable.square().sum() / combined_energy
                )
                physics_summary["ambiguous_energy_fraction"] = float(
                    ambiguous.square().sum() / combined_energy
                )
                physics_summary["raw_observable_norm"] = float(
                    torch.sqrt(output["raw_observable"].square().mean())
                )
                physics_summary["projected_observable_norm"] = float(
                    torch.sqrt(observable.square().mean())
                )
                physics_summary["observable_projection_ratio"] = float(
                    output["observable_correction"].abs().mean()
                    / (output["raw_observable"].abs().mean() + 1e-8)
                )
                physics_summary["raw_ambiguous_norm"] = float(
                    torch.sqrt(output["raw_ambiguous"].square().mean())
                )
                physics_summary["projected_ambiguous_norm"] = float(
                    torch.sqrt(ambiguous.square().mean())
                )
                physics_summary["ambiguous_projection_ratio"] = float(
                    ambiguous.abs().mean() / (output["raw_ambiguous"].abs().mean() + 1e-8)
                )
                for key, value in (
                    ("observable_correction", observable),
                    ("ambiguous_correction", ambiguous),
                    ("raw_observable", output["raw_observable"]),
                    ("raw_ambiguous", output["raw_ambiguous"]),
                ):
                    mechanism_chunks.setdefault(key, []).append(
                        value.squeeze(0).detach().cpu().float().numpy()
                    )
        d_hat_chunks.append(d_hat_b.squeeze(0).detach().cpu().float().numpy())
        fem_chunks.append(fem_b.squeeze(0).detach().cpu().float().numpy())
        residual_chunks.append(residual_b.squeeze(0).detach().cpu().float().numpy())
        gate_chunks.append(gate_b.squeeze(0).detach().cpu().float().numpy())
        raw_residual_chunks.append(raw_residual_b.squeeze(0).detach().cpu().float().numpy())

    return (
        np.concatenate(d_hat_chunks),
        np.concatenate(fem_chunks),
        np.concatenate(residual_chunks),
        valid,
        {
            "residual_gate": np.concatenate(gate_chunks),
            "raw_residual": np.concatenate(raw_residual_chunks),
            **physics_summary,
            **{key: np.concatenate(chunks) for key, chunks in mechanism_chunks.items() if chunks},
        },
    )


def evaluate_sample(
    sample_id: str,
    role_subset: str,
    num_foci: int,
    data: dict[str, np.ndarray],
    d_hat: np.ndarray,
    fem: np.ndarray,
    residual: np.ndarray,
    valid: np.ndarray,
    diagnostics: dict[str, np.ndarray] | None = None,
    tumor_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gt = data["gt_values"].astype(np.float32)[valid]
    diagnostics = diagnostics or {}
    residual_gate = diagnostics.get("residual_gate")
    if residual_gate is None or len(residual_gate) != len(gt):
        residual_gate = np.zeros_like(gt, dtype=np.float32)
    raw_residual = diagnostics.get("raw_residual")
    if raw_residual is None or len(raw_residual) != len(gt):
        raw_residual = residual
    gt_pos = gt >= 0.5
    gt_neg = ~gt_pos
    s2_minus_fem = d_hat - fem

    def masked_mean(
        values: np.ndarray, mask: np.ndarray | None = None, abs_value: bool = False
    ) -> float:
        if mask is None:
            selected = values
        else:
            selected = values[mask]
        if len(selected) == 0:
            return 0.0
        if abs_value:
            selected = np.abs(selected)
        return float(np.mean(selected))

    s2_metrics = binary_metrics(d_hat, gt, 0.5)
    fem_metrics = binary_metrics(fem, gt, 0.5)
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "role_subset": role_subset,
        "num_foci": num_foci,
        "n_valid": int(valid.sum()),
        "s2_dice_05": s2_metrics["dice"],
        "fem_dice_05": fem_metrics["dice"],
        "delta_dice_05": s2_metrics["dice"] - fem_metrics["dice"],
        "s2_iou_05": s2_metrics["iou"],
        "fem_iou_05": fem_metrics["iou"],
        "s2_precision_05": s2_metrics["precision"],
        "fem_precision_05": fem_metrics["precision"],
        "s2_recall_05": s2_metrics["recall"],
        "fem_recall_05": fem_metrics["recall"],
        "s2_specificity_05": s2_metrics["specificity"],
        "fem_specificity_05": fem_metrics["specificity"],
        "mse_s2": float(np.mean((d_hat - gt) ** 2)) if len(gt) else 0.0,
        "mse_fem": float(np.mean((fem - gt) ** 2)) if len(gt) else 0.0,
        "residual_norm": float(np.mean(np.abs(residual))) if len(residual) else 0.0,
        "gt_pos_ratio_05": s2_metrics["gt_pos_ratio"],
        "s2_pos_ratio_05": s2_metrics["pred_pos_ratio"],
        "fem_pos_ratio_05": fem_metrics["pred_pos_ratio"],
        "residual_gate_mean": masked_mean(residual_gate),
        "residual_gate_gt_pos_mean": masked_mean(residual_gate, gt_pos),
        "residual_gate_gt_neg_mean": masked_mean(residual_gate, gt_neg),
        "residual_mean_gt_pos": masked_mean(residual, gt_pos),
        "residual_mean_gt_neg": masked_mean(residual, gt_neg),
        "raw_residual_mean": masked_mean(raw_residual),
        "raw_residual_abs_mean": masked_mean(raw_residual, abs_value=True),
        "s2_minus_fem_mean": masked_mean(s2_minus_fem),
        "s2_minus_fem_gt_pos_mean": masked_mean(s2_minus_fem, gt_pos),
        "s2_minus_fem_gt_neg_mean": masked_mean(s2_minus_fem, gt_neg),
        "measurement_error": float(diagnostics.get("measurement_error", 0.0)),
        "correction_forward_error": float(diagnostics.get("correction_forward_error", 0.0)),
    }
    for key in (
        "alpha_lambda_l1",
        "alpha_entropy",
        "rho_tc_p1_l1",
        "rho_tc_p1_l2",
        "transport_error",
        "observable_energy_fraction",
        "ambiguous_energy_fraction",
        "raw_observable_norm",
        "projected_observable_norm",
        "observable_projection_ratio",
        "raw_ambiguous_norm",
        "projected_ambiguous_norm",
        "ambiguous_projection_ratio",
    ):
        if key in diagnostics:
            row[key] = float(diagnostics[key])
    if "role" in data:
        role_values = data["role"][valid]
        for vector_key in ("alpha_lambda_deviation", "rho_tc_minus_p1"):
            if vector_key not in diagnostics:
                continue
            values = np.asarray(diagnostics[vector_key])
            for role_id, role_name in ROLE_NAMES.items():
                mask = role_values == role_id
                if mask.any():
                    row[f"{vector_key}_{role_name}"] = float(np.mean(np.abs(values[mask])))
    row.update(
        source_metrics(
            d_hat,
            gt,
            data["grid_coords"].astype(np.float32)[valid],
            tumor_params,
        )
    )

    if "role" in data:
        role = data["role"][valid]
        for rid, name in ROLE_NAMES.items():
            mask = role == rid
            row[f"{name}_count"] = int(mask.sum())
            if mask.any():
                s2_role = binary_metrics(d_hat[mask], gt[mask], 0.5)
                fem_role = binary_metrics(fem[mask], gt[mask], 0.5)
                row[f"{name}_gt_pos_05"] = s2_role["gt_pos_ratio"]
                row[f"{name}_s2_pos_05"] = s2_role["pred_pos_ratio"]
                row[f"{name}_fem_pos_05"] = fem_role["pred_pos_ratio"]
                row[f"{name}_dice_05"] = s2_role["dice"]
                row[f"{name}_gate_mean"] = masked_mean(residual_gate, mask)
            else:
                row[f"{name}_gt_pos_05"] = ""
                row[f"{name}_s2_pos_05"] = ""
                row[f"{name}_fem_pos_05"] = ""
                row[f"{name}_dice_05"] = ""
                row[f"{name}_gate_mean"] = ""
    return row


def summarize_roles(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out = {}
    for name in ROLE_NAMES.values():
        keys = [
            f"{name}_count",
            f"{name}_gt_pos_05",
            f"{name}_s2_pos_05",
            f"{name}_fem_pos_05",
            f"{name}_dice_05",
            f"{name}_gate_mean",
        ]
        out[name] = mean_dict(rows, keys)
    return out


def summarize_by_foci(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out = {}
    foci_values = sorted({int(row["num_foci"]) for row in rows if int(row["num_foci"]) >= 0})
    for num_foci in foci_values:
        group = [row for row in rows if int(row["num_foci"]) == num_foci]
        out[str(num_foci)] = mean_dict(group, METRIC_KEYS)
        out[str(num_foci)]["n_samples"] = len(group)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=BASE_FIELDNAMES + ROLE_FIELDNAMES, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Stage2 evaluator")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_points", type=int, default=8192)
    parser.add_argument("--role_subset", choices=ROLE_SUBSETS, default="all")
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if cfg.get("data", {}).get("shared_dir"):
        os.environ["DU2VOX_SHARED_DIR"] = str(cfg["data"]["shared_dir"])
    if cfg.get("data", {}).get("allow_stale_frame_manifest", False):
        os.environ["DU2VOX_ALLOW_STALE_FRAME_MANIFEST"] = "1"
    if cfg.get("data", {}).get("frame_manifest_sha256"):
        os.environ["DU2VOX_FRAME_MANIFEST_SHA256"] = str(cfg["data"]["frame_manifest_sha256"])

    split_file = cfg["data"][f"{args.split}_split"]
    sample_ids = load_split(split_file)
    if args.max_samples is not None:
        sample_ids = sample_ids[: args.max_samples]

    precomputed_dir = Path(cfg["data"].get(f"precomputed_{args.split}_dir", ""))
    if not precomputed_dir.exists():
        raise SystemExit(f"precomputed_{args.split}_dir not found: {precomputed_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, view_encoder = build_model(cfg, device)
    load_checkpoint(Path(args.checkpoint), model, view_encoder, device)
    model.eval()
    if view_encoder is not None:
        view_encoder.eval()

    samples_dir = Path(cfg["data"]["samples_dir"]) if cfg["data"].get("samples_dir") else None
    frame = FrameManifest.load(cfg["data"]["shared_dir"]) if cfg["data"].get("shared_dir") else None
    print(
        f"[Eval] split={args.split}, n_samples={len(sample_ids)}, "
        f"mode=all_valid, role_subset={args.role_subset}"
    )
    print(f"[Eval] precomputed_dir={precomputed_dir}")

    rows = []
    with torch.no_grad():
        for sample_id in sample_ids:
            npz_path = precomputed_dir / f"{sample_id}.npz"
            if not npz_path.exists():
                print(f"[WARN] missing npz: {npz_path}")
                continue
            data = dict(np.load(npz_path, allow_pickle=False))
            num_foci = get_num_foci(precomputed_dir, samples_dir, sample_id)
            d_hat, fem, residual, valid, diagnostics = run_model_on_sample(
                model=model,
                view_encoder=view_encoder,
                cfg=cfg,
                data=data,
                samples_dir=samples_dir,
                frame=frame,
                sample_id=sample_id,
                batch_points=args.batch_points,
                role_subset=args.role_subset,
                device=device,
            )
            rows.append(
                evaluate_sample(
                    sample_id,
                    args.role_subset,
                    num_foci,
                    data,
                    d_hat,
                    fem,
                    residual,
                    valid,
                    diagnostics,
                    tumor_params=read_json(samples_dir / sample_id / "tumor_params.json")
                    if samples_dir is not None
                    else None,
                )
            )

    overall = mean_dict(rows, METRIC_KEYS)
    result = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "role_subset": args.role_subset,
        "output_mode": cfg["model"].get("output_mode", "residual"),
        "hybrid_alpha": cfg["model"].get("hybrid_alpha", 0.5),
        "n_samples": len(rows),
        "overall": overall,
        "by_foci": summarize_by_foci(rows),
        "role": summarize_roles(rows),
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    write_csv(Path(args.out_csv), rows)

    print(f"[Eval] wrote JSON: {out_json}")
    print(f"[Eval] wrote CSV: {args.out_csv}")
    print(
        f"[Eval] s2_dice_05={overall['s2_dice_05']:.4f}, "
        f"fem_dice_05={overall['fem_dice_05']:.4f}, "
        f"delta={overall['delta_dice_05']:+.4f}"
    )


if __name__ == "__main__":
    main()
