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

sys.path.insert(0, str(Path(__file__).parent.parent))

from du2vox.models.stage2.cqr_residual_inr import CQRResidualINR
from du2vox.models.stage2.residual_inr import ResidualINR
from du2vox.models.stage2.stage2_dataset import load_projection_stack
from du2vox.utils.frame import FrameManifest


ROLE_NAMES = {0: "bg", 1: "core", 2: "halo", 3: "proposal"}

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
    "mse_s2",
    "mse_fem",
    "residual_norm",
    "gt_pos_ratio_05",
    "s2_pos_ratio_05",
    "fem_pos_ratio_05",
]

ROLE_SUBSETS = ["all", "core", "core_halo", "halo", "proposal", "bg", "non_bg"]

BASE_FIELDNAMES = ["sample_id", "role_subset", "num_foci", "n_valid", *METRIC_KEYS]
ROLE_FIELDNAMES = [
    f"{name}_{suffix}"
    for suffix in ["count", "gt_pos_05", "s2_pos_05", "fem_pos_05", "dice_05"]
    for name in ["bg", "core", "halo", "proposal"]
]


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
    pred_sum = float(pred_bin.sum())
    gt_sum = float(gt_bin.sum())
    union = float((pred_bin | gt_bin).sum())
    return {
        "dice": 2.0 * tp / (pred_sum + gt_sum + eps),
        "iou": tp / (union + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "pred_pos_ratio": float(pred_bin.mean()) if len(pred_bin) else 0.0,
        "gt_pos_ratio": float(gt_bin.mean()) if len(gt_bin) else 0.0,
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
        return role == 3
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


def build_model(cfg: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module | None]:
    prior_dim = int(cfg["model"].get("prior_dim", 8))
    model_type = cfg["model"].get("model_type", "")
    use_cqr_model = (model_type == "cqr_residual_inr") or (prior_dim > 8)
    model_cls = CQRResidualINR if use_cqr_model else ResidualINR

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
            projection_transform=cfg["model"].get("view_projection_transform", "log1p"),
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
    if model_cls is CQRResidualINR:
        kwargs["residual_scale"] = cfg["model"].get("residual_scale", 1.0)
        kwargs["support_head"] = cfg["model"].get("support_head", False)
        kwargs["use_prolongation_adapter"] = cfg["model"].get("use_prolongation_adapter", False)
        kwargs["prolongation_feat_dim"] = cfg["model"].get("prolongation_feat_dim", 32)
        kwargs["use_lifting_adapter"] = cfg["model"].get("use_lifting_adapter", False)
        kwargs["lifting_feat_dim"] = cfg["model"].get("lifting_feat_dim", 32)
        kwargs["use_band_embedding"] = cfg["model"].get("use_band_embedding", False)
        kwargs["band_embed_dim"] = cfg["model"].get("band_embed_dim", 8)
        kwargs["num_bands"] = cfg["model"].get("num_bands", 4)
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
        model.load_state_dict(ckpt["residual_inr"])
    elif isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        )

    d_hat_chunks = []
    fem_chunks = []
    residual_chunks = []
    proj_imgs = None
    mcx_valid = None
    if view_encoder is not None:
        if samples_dir is None:
            raise ValueError("samples_dir is required for multiview evaluation")
        proj_imgs = torch.from_numpy(load_proj_imgs(samples_dir, sample_id, cfg)).unsqueeze(0).to(device)
        mcx_valid = mcx_valid_mask(frame, coords_world)

    for start in range(0, len(coords_norm), batch_points):
        end = min(start + batch_points, len(coords_norm))
        coords_b = torch.from_numpy(coords_norm[start:end]).unsqueeze(0).to(device)
        prior_b = torch.from_numpy(prior[start:end]).unsqueeze(0).to(device)
        band_b = torch.from_numpy(correction_band[start:end]).unsqueeze(0).to(device)
        if view_encoder is None:
            output = unpack_model_output(model(coords_b, prior_b, correction_band=band_b))
        else:
            world_b = torch.from_numpy(coords_world[start:end]).unsqueeze(0).to(device)
            view_feat, _ = view_encoder(proj_imgs, world_b, coords_vox_norm=None)
            valid_b = torch.from_numpy(mcx_valid[start:end]).unsqueeze(0).to(device)
            view_feat = view_feat * valid_b.unsqueeze(-1).float()
            output = unpack_model_output(model(coords_b, prior_b, view_feat, correction_band=band_b))
        d_hat_b = select_stage2_prediction(output, cfg)
        fem_b = output["fem_interp"]
        residual_b = output["residual"]
        d_hat_chunks.append(d_hat_b.squeeze(0).detach().cpu().float().numpy())
        fem_chunks.append(fem_b.squeeze(0).detach().cpu().float().numpy())
        residual_chunks.append(residual_b.squeeze(0).detach().cpu().float().numpy())

    return (
        np.concatenate(d_hat_chunks),
        np.concatenate(fem_chunks),
        np.concatenate(residual_chunks),
        valid,
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
) -> dict[str, Any]:
    gt = data["gt_values"].astype(np.float32)[valid]
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
        "mse_s2": float(np.mean((d_hat - gt) ** 2)) if len(gt) else 0.0,
        "mse_fem": float(np.mean((fem - gt) ** 2)) if len(gt) else 0.0,
        "residual_norm": float(np.mean(np.abs(residual))) if len(residual) else 0.0,
        "gt_pos_ratio_05": s2_metrics["gt_pos_ratio"],
        "s2_pos_ratio_05": s2_metrics["pred_pos_ratio"],
        "fem_pos_ratio_05": fem_metrics["pred_pos_ratio"],
    }

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
            else:
                row[f"{name}_gt_pos_05"] = ""
                row[f"{name}_s2_pos_05"] = ""
                row[f"{name}_fem_pos_05"] = ""
                row[f"{name}_dice_05"] = ""
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
        writer = csv.DictWriter(f, fieldnames=BASE_FIELDNAMES + ROLE_FIELDNAMES, extrasaction="ignore")
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
            d_hat, fem, residual, valid = run_model_on_sample(
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
            rows.append(evaluate_sample(sample_id, args.role_subset, num_foci, data, d_hat, fem, residual, valid))

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
