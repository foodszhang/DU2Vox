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
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))

from du2vox.bridge.roi_derivation import derive_roi
from du2vox.data.dataset import FMTSimGenDataset
from du2vox.evaluation.metrics import evaluate_batch, summarize_metrics
from du2vox.models.stage1.gcain import GCAIN_full
from du2vox.utils.frame import FrameManifest


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_activation(activation: str, leaky_slope: float):
    if activation == "sigmoid":
        return torch.sigmoid
    if activation == "leaky_relu":
        return lambda x: torch.nn.functional.leaky_relu(x, negative_slope=leaky_slope).clamp(max=1.0)
    return lambda x: x.clamp(0.0, 1.0)


def load_dataset(cfg: dict, split: str) -> FMTSimGenDataset:
    data_cfg = cfg["data"]
    return FMTSimGenDataset(
        shared_dir=data_cfg["shared_dir"],
        samples_dir=data_cfg["samples_dir"],
        split_file=Path(data_cfg["splits_dir"]) / f"{split}.txt",
        normalize_b=data_cfg.get("normalize_b", True),
        normalize_gt=data_cfg.get("normalize_gt", True),
        normalize_gt_mode=data_cfg.get("normalize_gt_mode", "per_sample"),
        binarize_gt=data_cfg.get("binarize_gt", False),
        binarize_threshold=data_cfg.get("binarize_threshold", 0.05),
        use_visible_mask=data_cfg.get("use_visible_mask", False),
    )


def build_model(cfg: dict, dataset: FMTSimGenDataset, device: torch.device) -> GCAIN_full:
    model_cfg = cfg["model"]
    model = GCAIN_full(
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
    return model


def load_checkpoint(model: torch.nn.Module, checkpoint: str, device: torch.device) -> None:
    ckpt = torch.load(checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Audit] checkpoint epoch={ckpt.get('epoch', '?')}, primary={ckpt.get('primary_metric', '?')}")
    else:
        model.load_state_dict(ckpt)


def binary_stats(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred_bin = pred > threshold
    gt_bin = gt > 0.5
    tp = float((pred_bin & gt_bin).sum())
    fp = float((pred_bin & ~gt_bin).sum())
    fn = float((~pred_bin & gt_bin).sum())
    return {
        "precision": tp / (tp + fp + 1e-8),
        "recall": tp / (tp + fn + 1e-8),
        "pred_pos_ratio": float(pred_bin.mean()),
        "gt_pos_ratio": float(gt_bin.mean()),
    }


def binary_dice(pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:
    pred_bin = pred > threshold
    gt_bin = gt > 0.5
    return float(2.0 * (pred_bin & gt_bin).sum() / (pred_bin.sum() + gt_bin.sum() + 1e-8))


def audit_roi(
    pred: np.ndarray,
    gt: np.ndarray,
    nodes: np.ndarray,
    elements: np.ndarray,
    tau_values: list[float],
    dilate_values: list[int],
) -> dict[str, float]:
    out = {}
    gt_nodes = np.where(gt > 0.5)[0]
    for tau in tau_values:
        for dilate in dilate_values:
            result = derive_roi(pred, nodes, elements, tau=tau, dilate_layers=dilate)
            roi_nodes = np.unique(elements[result["roi_tet_indices"]].ravel())
            coverage = 0.0
            if len(gt_nodes):
                coverage = float(np.isin(gt_nodes, roi_nodes).mean())
            key = f"tau{tau:g}_d{dilate}"
            out[f"roi_tet_ratio_{key}"] = float(result["roi_tet_ratio"])
            out[f"gt_roi_coverage_{key}"] = coverage
    return out


def mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    out = {}
    for key in keys:
        vals = [row[key] for row in rows if key in row and isinstance(row[key], int | float | np.floating)]
        if vals:
            out[key] = float(np.mean(vals))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Stage1 checkpoint mesh and ROI support metrics")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--tau_values", type=float, nargs="*", default=[0.1, 0.2, 0.3, 0.5, 0.6])
    parser.add_argument("--dilate_layers", type=int, nargs="*", default=[0, 1])
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset(cfg, args.split)
    indices = list(range(len(dataset)))
    if args.max_samples is not None:
        indices = indices[: args.max_samples]
    subset = Subset(dataset, indices)
    batch_size = args.batch_size or cfg["training"].get("batch_size", 8)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    nodes_np, elements = FrameManifest.load_mesh_nodes(cfg["data"]["shared_dir"])
    nodes = dataset.nodes.to(device)
    model = build_model(cfg, dataset, device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()
    apply_activation = build_activation(
        cfg["training"].get("activation", "leaky_relu"),
        cfg["training"].get("leaky_relu_slope", 0.01),
    )

    metric_batches = []
    rows = []
    sample_offset = 0
    with torch.no_grad():
        for batch in loader:
            b = batch["b"].to(device)
            gt = batch["gt"].to(device)
            x0 = torch.zeros(b.size(0), dataset.nodes.shape[0], 1, device=device)
            pred = apply_activation(model(x0, b)).clamp(0.0, 1.0)
            metric_batches.append(evaluate_batch(pred, gt, nodes))
            pred_np = pred.squeeze(-1).cpu().numpy()
            gt_np = gt.squeeze(-1).cpu().numpy()
            for i in range(pred_np.shape[0]):
                sample_id = dataset.sample_ids[indices[sample_offset + i]]
                row = {
                    "sample_id": sample_id,
                    "mesh_soft_dice": float(metric_batches[-1].get("dice", 0.0)),
                    "mesh_dice_pred03_gt05": binary_dice(pred_np[i], gt_np[i], 0.3),
                    "mesh_dice_pred05_gt05": binary_dice(pred_np[i], gt_np[i], 0.5),
                    "mesh_dice_pred06_gt05": binary_dice(pred_np[i], gt_np[i], 0.6),
                    "pred_mean": float(pred_np[i].mean()),
                    "pred_max": float(pred_np[i].max()),
                    "pred_std": float(pred_np[i].std()),
                }
                row.update({f"{key}_03": value for key, value in binary_stats(pred_np[i], gt_np[i], 0.3).items()})
                row.update({f"{key}_05": value for key, value in binary_stats(pred_np[i], gt_np[i], 0.5).items()})
                row.update(audit_roi(pred_np[i], gt_np[i], nodes_np, elements, args.tau_values, args.dilate_layers))
                rows.append(row)
            sample_offset += pred_np.shape[0]

    mesh_summary = summarize_metrics(metric_batches)
    roi_summary = mean_rows(rows)
    result = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "n_samples": len(rows),
        "mesh": mesh_summary,
        "roi": roi_summary,
        "per_sample_summary": roi_summary,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Audit] wrote JSON: {out_json}")
    print(f"[Audit] wrote CSV: {out_csv}")
    print(
        f"[Audit] dice@0.5={mesh_summary.get('dice_bin_0.5', 0):.4f}, "
        f"precision@0.5={mesh_summary.get('precision_pred05_gt05', 0):.4f}, "
        f"recall@0.5={mesh_summary.get('recall_pred05_gt05', 0):.4f}"
    )


if __name__ == "__main__":
    main()
