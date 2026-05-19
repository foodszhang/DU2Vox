#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).parent.parent))

from du2vox.bridge.coverage_field import compute_coverage_field
from du2vox.bridge.fem_bridging import FEMBridge
from du2vox.models.stage2.stage2_dataset import MCX_ANGLES
from du2vox.utils.frame import FrameManifest
from scripts.eval_stage2_unified import build_model, load_checkpoint


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


def load_split(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def make_bbox_grid(bbox: dict[str, list[float]], spacing: float, padding: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = np.asarray(bbox["min"], dtype=np.float32) - float(padding)
    hi = np.asarray(bbox["max"], dtype=np.float32) + float(padding)
    axes = [np.arange(lo[i], hi[i] + spacing * 0.5, spacing, dtype=np.float32) for i in range(3)]
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.stack([m.ravel() for m in mesh], axis=1).astype(np.float32)
    return points, lo.astype(np.float32), hi.astype(np.float32)


def normalize_coords(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return (2.0 * (points - lo) / (hi - lo + 1e-8) - 1.0).astype(np.float32)


def binary_metrics(pred: np.ndarray, gt: np.ndarray, thr: float = 0.5) -> dict[str, float]:
    pred_bin = pred >= thr
    gt_bin = gt >= thr
    tp = float((pred_bin & gt_bin).sum())
    fp = float((pred_bin & ~gt_bin).sum())
    fn = float((~pred_bin & gt_bin).sum())
    union = float((pred_bin | gt_bin).sum())
    return {
        "dice": 2.0 * tp / (float(pred_bin.sum() + gt_bin.sum()) + 1e-8),
        "iou": tp / (union + 1e-8),
        "precision": tp / (tp + fp + 1e-8),
        "recall": tp / (tp + fn + 1e-8),
        "pred_pos_ratio": float(pred_bin.mean()) if len(pred_bin) else 0.0,
        "gt_pos_ratio": float(gt_bin.mean()) if len(gt_bin) else 0.0,
    }


def mean_dict(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, float]:
    out = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if key in row]
        out[key] = float(np.mean(vals)) if vals else 0.0
    return out


def load_proj_imgs(samples_dir: Path, sample_id: str) -> np.ndarray:
    proj_path = samples_dir / sample_id / "proj.npz"
    if not proj_path.exists():
        return np.zeros((7, 1, 256, 256), dtype=np.float32)
    proj_data = np.load(proj_path)
    proj_imgs = np.stack([proj_data[str(angle)].astype(np.float32) for angle in MCX_ANGLES], axis=0)
    return proj_imgs[:, None, :, :]


def mcx_valid_mask(frame: FrameManifest, coords_world: np.ndarray) -> np.ndarray:
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


def sample_gt(frame: FrameManifest, samples_dir: Path, sample_id: str, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gt_voxels = np.load(samples_dir / sample_id / "gt_voxels.npy").astype(np.float32)
    idx = frame.world_to_gt_index(points)
    shape = np.asarray(gt_voxels.shape)
    inside = ~np.any((idx < 0) | (idx > shape - 1), axis=1)
    gt = map_coordinates(gt_voxels, idx.T, order=1, mode="constant", cval=0.0, prefilter=False).astype(np.float32)
    gt[~inside] = 0.0
    return gt, inside


def build_prior_arrays(
    cfg: dict[str, Any],
    nodes: np.ndarray,
    elements: np.ndarray,
    eval_bridge_dir: Path,
    sample_id: str,
    points: np.ndarray,
    n_candidates: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bd = eval_bridge_dir / sample_id
    coarse_d = np.load(bd / "coarse_d.npy").astype(np.float32)
    roi_tets = np.load(bd / "roi_tet_indices.npy").astype(np.int64)
    bridge = FEMBridge(nodes, elements, roi_tets, n_candidates=n_candidates)
    tet_ids, bary = bridge.locate_points_batch(points)
    valid = tet_ids >= 0
    prior_8d = np.zeros((len(points), 8), dtype=np.float32)
    if valid.any():
        node_ids = elements[tet_ids[valid]]
        prior_8d[valid, :4] = coarse_d[node_ids]
        prior_8d[valid, 4:8] = bary[valid].astype(np.float32)

    prior_dim = int(cfg["model"].get("prior_dim", 8))
    if prior_dim <= 8:
        return prior_8d, valid, tet_ids, coarse_d

    field = compute_coverage_field(
        coarse_d,
        elements,
        roi_tet_indices=roi_tets,
        cfg=cfg.get("cqr", {}).get("coverage", {}),
    )
    prior_ext = np.zeros((len(points), prior_dim), dtype=np.float32)
    prior_ext[:, :8] = prior_8d
    if valid.any():
        prior_ext[valid, 8] = field["coverage_score"][tet_ids[valid]]
        prior_ext[valid, 9:13] = field["risk_components"][tet_ids[valid], : prior_dim - 9]
    return prior_ext, valid, tet_ids, coarse_d


def run_model(
    model: torch.nn.Module,
    view_encoder: torch.nn.Module | None,
    coords_norm: np.ndarray,
    coords_world: np.ndarray,
    prior: np.ndarray,
    samples_dir: Path,
    frame: FrameManifest,
    sample_id: str,
    batch_points: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d_hat_chunks = []
    fem_chunks = []
    residual_chunks = []
    proj_imgs = None
    mcx_valid = None
    if view_encoder is not None:
        proj_imgs = torch.from_numpy(load_proj_imgs(samples_dir, sample_id)).unsqueeze(0).to(device)
        mcx_valid = mcx_valid_mask(frame, coords_world)

    for start in range(0, len(coords_norm), batch_points):
        end = min(start + batch_points, len(coords_norm))
        coords_b = torch.from_numpy(coords_norm[start:end]).unsqueeze(0).to(device)
        prior_b = torch.from_numpy(prior[start:end]).unsqueeze(0).to(device)
        if view_encoder is None:
            d_hat_b, fem_b, residual_b = model(coords_b, prior_b)
        else:
            world_b = torch.from_numpy(coords_world[start:end]).unsqueeze(0).to(device)
            view_feat, _ = view_encoder(proj_imgs, world_b, coords_vox_norm=None)
            valid_b = torch.from_numpy(mcx_valid[start:end]).unsqueeze(0).to(device)
            view_feat = view_feat * valid_b.unsqueeze(-1).float()
            d_hat_b, fem_b, residual_b = model(coords_b, prior_b, view_feat)
        d_hat_chunks.append(d_hat_b.squeeze(0).detach().cpu().float().numpy())
        fem_chunks.append(fem_b.squeeze(0).detach().cpu().float().numpy())
        residual_chunks.append(residual_b.squeeze(0).detach().cpu().float().numpy())
    return np.concatenate(d_hat_chunks), np.concatenate(fem_chunks), np.concatenate(residual_chunks)


def evaluate_sample(
    sample_id: str,
    cfg: dict[str, Any],
    model: torch.nn.Module,
    view_encoder: torch.nn.Module | None,
    nodes: np.ndarray,
    elements: np.ndarray,
    frame: FrameManifest,
    samples_dir: Path,
    common_bridge_dir: Path,
    eval_bridge_dir: Path,
    grid_spacing: float,
    padding: float,
    batch_points: int,
    n_candidates: int,
    device: torch.device,
) -> dict[str, Any] | None:
    common_info_path = common_bridge_dir / sample_id / "roi_info.json"
    if not common_info_path.exists():
        print(f"[WARN] missing common roi_info: {common_info_path}")
        return None
    common_info = load_json(common_info_path)
    points, bbox_min, bbox_max = make_bbox_grid(common_info["roi_bbox_mm"], grid_spacing, padding)
    coords_norm_all = normalize_coords(points, bbox_min, bbox_max)
    prior_all, valid, _, _ = build_prior_arrays(cfg, nodes, elements, eval_bridge_dir, sample_id, points, n_candidates)
    gt_all, gt_inside = sample_gt(frame, samples_dir, sample_id, points)
    valid = valid & gt_inside
    if not valid.any():
        return {"sample_id": sample_id, "n_points": len(points), "n_valid": 0, **{key: 0.0 for key in METRIC_KEYS}}

    coords_norm = coords_norm_all[valid]
    coords_world = points[valid]
    prior = prior_all[valid]
    gt = gt_all[valid]
    d_hat, fem, residual = run_model(
        model,
        view_encoder,
        coords_norm,
        coords_world,
        prior,
        samples_dir,
        frame,
        sample_id,
        batch_points,
        device,
    )
    s2 = binary_metrics(d_hat, gt)
    fem_metrics = binary_metrics(fem, gt)
    return {
        "sample_id": sample_id,
        "n_points": int(len(points)),
        "n_valid": int(valid.sum()),
        "s2_dice_05": s2["dice"],
        "fem_dice_05": fem_metrics["dice"],
        "delta_dice_05": s2["dice"] - fem_metrics["dice"],
        "s2_iou_05": s2["iou"],
        "fem_iou_05": fem_metrics["iou"],
        "s2_precision_05": s2["precision"],
        "fem_precision_05": fem_metrics["precision"],
        "s2_recall_05": s2["recall"],
        "fem_recall_05": fem_metrics["recall"],
        "mse_s2": float(np.mean((d_hat - gt) ** 2)),
        "mse_fem": float(np.mean((fem - gt) ** 2)),
        "residual_norm": float(np.mean(np.abs(residual))),
        "gt_pos_ratio_05": s2["gt_pos_ratio"],
        "s2_pos_ratio_05": s2["pred_pos_ratio"],
        "fem_pos_ratio_05": fem_metrics["pred_pos_ratio"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_id", "n_points", "n_valid", *METRIC_KEYS]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage2 common-domain evaluator")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--common_bridge_dir", required=True)
    parser.add_argument("--eval_bridge_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--grid_spacing_mm", type=float, default=1.0)
    parser.add_argument("--padding_mm", type=float, default=1.0)
    parser.add_argument("--batch_points", type=int, default=8192)
    parser.add_argument("--n_candidates", type=int, default=16)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    sample_ids = load_split(cfg["data"][f"{args.split}_split"])
    if args.max_samples is not None:
        sample_ids = sample_ids[: args.max_samples]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, view_encoder = build_model(cfg, device)
    load_checkpoint(Path(args.checkpoint), model, view_encoder, device)
    model.eval()
    if view_encoder is not None:
        view_encoder.eval()

    samples_dir = Path(cfg["data"]["samples_dir"])
    shared_dir = cfg["data"]["shared_dir"]
    nodes, elements = FrameManifest.load_mesh_nodes(shared_dir)
    frame = FrameManifest.load(shared_dir)
    rows = []
    with torch.no_grad():
        for sample_id in sample_ids:
            row = evaluate_sample(
                sample_id=sample_id,
                cfg=cfg,
                model=model,
                view_encoder=view_encoder,
                nodes=nodes,
                elements=elements,
                frame=frame,
                samples_dir=samples_dir,
                common_bridge_dir=Path(args.common_bridge_dir),
                eval_bridge_dir=Path(args.eval_bridge_dir),
                grid_spacing=args.grid_spacing_mm,
                padding=args.padding_mm,
                batch_points=args.batch_points,
                n_candidates=args.n_candidates,
                device=device,
            )
            if row is not None:
                rows.append(row)

    result = {
        "eval_domain": "common_grid",
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "common_bridge_dir": args.common_bridge_dir,
        "eval_bridge_dir": args.eval_bridge_dir,
        "grid_spacing_mm": args.grid_spacing_mm,
        "padding_mm": args.padding_mm,
        "n_samples": len(rows),
        "overall": mean_dict(rows, METRIC_KEYS),
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    write_csv(Path(args.out_csv), rows)
    print(f"[CommonEval] wrote {out_json} and {args.out_csv}")


if __name__ == "__main__":
    main()
