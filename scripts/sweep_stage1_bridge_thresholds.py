#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.ndimage import map_coordinates
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))

from du2vox.bridge.fem_bridging import FEMBridge
from du2vox.bridge.roi_derivation import derive_roi
from du2vox.data.dataset import FMTSimGenDataset
from du2vox.models.stage1.gcain import GCAIN_full
from du2vox.utils.frame import FrameManifest


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_split(path: str | Path) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def build_activation(activation: str, leaky_slope: float):
    if activation == "sigmoid":
        return torch.sigmoid
    if activation == "leaky_relu":
        return lambda x: torch.nn.functional.leaky_relu(x, negative_slope=leaky_slope).clamp(max=1.0)
    return lambda x: x.clamp(0.0, 1.0)


def load_dataset(cfg: dict, split_file: Path, samples_dir: Path, shared_dir: Path) -> FMTSimGenDataset:
    data_cfg = cfg["data"]
    return FMTSimGenDataset(
        shared_dir=shared_dir,
        samples_dir=samples_dir,
        split_file=split_file,
        normalize_b=data_cfg.get("normalize_b", True),
        normalize_gt=data_cfg.get("normalize_gt", True),
        normalize_gt_mode=data_cfg.get("normalize_gt_mode", "per_sample"),
        binarize_gt=data_cfg.get("binarize_gt", False),
        binarize_threshold=data_cfg.get("binarize_threshold", 0.05),
        use_visible_mask=data_cfg.get("use_visible_mask", False),
    )


def build_model(cfg: dict, dataset: FMTSimGenDataset, device: torch.device) -> GCAIN_full:
    model_cfg = cfg["model"]
    return GCAIN_full(
        L=dataset.L.to(device),
        A=dataset.A.to(device),
        LTL=None,
        ATA=None,
        L0=dataset.L0.to(device),
        L1=dataset.L1.to(device),
        L2=dataset.L2.to(device),
        L3=dataset.L3.to(device),
        knn_idx=dataset.knn_idx.to(device),
        sens_w=dataset.sens_w.to(device),
        num_layer=model_cfg.get("num_layer", 6),
        feat_dim=model_cfg.get("feat_dim", 6),
    ).to(device)


def load_checkpoint(model: torch.nn.Module, checkpoint: Path, device: torch.device) -> None:
    ckpt = torch.load(checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Sweep] checkpoint epoch={ckpt.get('epoch', '?')}, primary={ckpt.get('primary_metric', '?')}")
    else:
        model.load_state_dict(ckpt)


def make_bbox_grid(bbox: dict[str, list[float]], spacing: float, padding: float) -> np.ndarray:
    lo = np.asarray(bbox["min"], dtype=np.float32) - float(padding)
    hi = np.asarray(bbox["max"], dtype=np.float32) + float(padding)
    axes = [np.arange(lo[i], hi[i] + spacing * 0.5, spacing, dtype=np.float32) for i in range(3)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=1).astype(np.float32)


def sample_gt_values(frame: FrameManifest, gt_voxels: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = frame.world_to_gt_index(points)
    shape = np.asarray(gt_voxels.shape)
    inside = ~np.any((idx < 0) | (idx > shape - 1), axis=1)
    values = map_coordinates(gt_voxels, idx.T, order=1, mode="constant", cval=0.0, prefilter=False).astype(np.float32)
    values[~inside] = 0.0
    return values, inside


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred_bin = pred >= 0.5
    gt_bin = gt >= 0.5
    tp = float((pred_bin & gt_bin).sum())
    fp = float((pred_bin & ~gt_bin).sum())
    fn = float((~pred_bin & gt_bin).sum())
    return {
        "dice": 2.0 * tp / (float(pred_bin.sum() + gt_bin.sum()) + 1e-8),
        "precision": tp / (tp + fp + 1e-8),
        "recall": tp / (tp + fn + 1e-8),
    }


def mesh_binary(pred: np.ndarray, gt: np.ndarray, threshold: float) -> dict[str, float]:
    pred_bin = pred > threshold
    gt_bin = gt > 0.5
    tp = float((pred_bin & gt_bin).sum())
    fp = float((pred_bin & ~gt_bin).sum())
    fn = float((~pred_bin & gt_bin).sum())
    return {
        "mesh_dice_proxy": 2.0 * tp / (float(pred_bin.sum() + gt_bin.sum()) + 1e-8),
        "precision": tp / (tp + fp + 1e-8),
        "recall": tp / (tp + fn + 1e-8),
    }


def common_bbox(common_bridge_dir: Path | None, sample_id: str, fallback: dict[str, list[float]]) -> dict[str, list[float]]:
    if common_bridge_dir is None:
        return fallback
    info_path = common_bridge_dir / sample_id / "roi_info.json"
    if not info_path.exists():
        return fallback
    with open(info_path) as f:
        return json.load(f)["roi_bbox_mm"]


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep Stage1 bridge thresholds")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split_file", required=True)
    parser.add_argument("--shared_dir", required=True)
    parser.add_argument("--samples_dir", required=True)
    parser.add_argument("--common_bridge_dir", default="output/bridge_20k_val")
    parser.add_argument("--taus", type=float, nargs="+", required=True)
    parser.add_argument("--dilate_layers", type=int, nargs="+", required=True)
    parser.add_argument("--min_component_size", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grid_spacing_mm", type=float, default=1.0)
    parser.add_argument("--padding_mm", type=float, default=1.0)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    split_ids = load_split(args.split_file)
    if args.max_samples is not None:
        split_ids = split_ids[: args.max_samples]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset(cfg, Path(args.split_file), Path(args.samples_dir), Path(args.shared_dir))
    id_to_idx = {sid: i for i, sid in enumerate(dataset.sample_ids)}
    indices = [id_to_idx[sid] for sid in split_ids if sid in id_to_idx]
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size or cfg["training"].get("batch_size", 8), shuffle=False)

    nodes, elements = FrameManifest.load_mesh_nodes(args.shared_dir)
    frame = FrameManifest.load(args.shared_dir)
    model = build_model(cfg, dataset, device)
    load_checkpoint(model, Path(args.checkpoint), device)
    model.eval()
    apply_activation = build_activation(
        cfg["training"].get("activation", "leaky_relu"),
        cfg["training"].get("leaky_relu_slope", 0.01),
    )
    common_dir = Path(args.common_bridge_dir) if args.common_bridge_dir else None
    n_tets = len(elements)
    accum: dict[tuple[float, int], list[dict[str, float]]] = {(tau, dilate): [] for tau in args.taus for dilate in args.dilate_layers}

    sample_offset = 0
    with torch.no_grad():
        for batch in loader:
            b = batch["b"].to(device)
            gt = batch["gt"].to(device)
            x0 = torch.zeros(b.size(0), dataset.nodes.shape[0], 1, device=device)
            pred = apply_activation(model(x0, b)).clamp(0.0, 1.0).squeeze(-1).cpu().numpy()
            gt_nodes = gt.squeeze(-1).cpu().numpy()
            for i in range(pred.shape[0]):
                sample_id = dataset.sample_ids[indices[sample_offset + i]]
                gt_voxels = np.load(Path(args.samples_dir) / sample_id / "gt_voxels.npy").astype(np.float32)
                gt_pos_nodes = np.where(gt_nodes[i] > 0.5)[0]
                for tau in args.taus:
                    for dilate in args.dilate_layers:
                        roi = derive_roi(
                            pred[i],
                            nodes,
                            elements,
                            tau=tau,
                            dilate_layers=dilate,
                            min_component_size=args.min_component_size,
                        )
                        roi_tets = roi["roi_tet_indices"]
                        roi_nodes = np.unique(elements[roi_tets].ravel())
                        coverage = float(np.isin(gt_pos_nodes, roi_nodes).mean()) if len(gt_pos_nodes) else 0.0
                        bbox = common_bbox(common_dir, sample_id, roi["roi_bbox_mm"])
                        points = make_bbox_grid(bbox, args.grid_spacing_mm, args.padding_mm)
                        bridge = FEMBridge(nodes, elements, roi_tets)
                        prior_8d, valid = bridge.get_prior_features(points, pred[i])
                        gt_values, gt_inside = sample_gt_values(frame, gt_voxels, points)
                        valid = valid & gt_inside
                        fem = (prior_8d[:, :4] * prior_8d[:, 4:8]).sum(axis=1)
                        fem_metrics = binary_metrics(fem[valid], gt_values[valid]) if valid.any() else {"dice": 0.0, "precision": 0.0, "recall": 0.0}
                        mesh_metrics = mesh_binary(pred[i], gt_nodes[i], tau)
                        accum[(tau, dilate)].append(
                            {
                                "roi_tet_ratio": len(roi_tets) / n_tets,
                                "gt_roi_coverage": coverage,
                                "common_domain_fem_dice": fem_metrics["dice"],
                                "common_domain_fem_precision": fem_metrics["precision"],
                                "common_domain_fem_recall": fem_metrics["recall"],
                                "mesh_dice_proxy": mesh_metrics["mesh_dice_proxy"],
                                "precision": mesh_metrics["precision"],
                                "recall": mesh_metrics["recall"],
                                "n_roi_tets": float(len(roi_tets)),
                            }
                        )
            sample_offset += pred.shape[0]

    rows = []
    for (tau, dilate), values in accum.items():
        row = {"tau": tau, "dilate_layers": dilate, "n_samples": len(values)}
        for key in [
            "roi_tet_ratio",
            "gt_roi_coverage",
            "common_domain_fem_dice",
            "common_domain_fem_precision",
            "common_domain_fem_recall",
            "mesh_dice_proxy",
            "precision",
            "recall",
            "n_roi_tets",
        ]:
            row[key] = mean([float(v[key]) for v in values])
        rows.append(row)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Sweep] wrote {len(rows)} rows to {out}")
    for row in sorted(rows, key=lambda r: (r["common_domain_fem_dice"], r["gt_roi_coverage"]), reverse=True):
        print(
            f"tau={row['tau']:.2f} d={row['dilate_layers']} fem={row['common_domain_fem_dice']:.4f} "
            f"coverage={row['gt_roi_coverage']:.4f} roi={row['roi_tet_ratio']:.4f}"
        )


if __name__ == "__main__":
    main()
