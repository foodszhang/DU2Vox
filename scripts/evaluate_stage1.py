#!/usr/bin/env python3
"""
Stage 1 comprehensive evaluation: grouped metrics + Tecplot-style visualization.

Generates:
  - 4 tables (CSV + LaTeX): overall, by_foci, by_depth, cross
  - fig1: training curves
  - fig2: Gaussian intensity (6x3)
  - fig3: Gaussian segmentation TP/FP/FN (6x3)
  - fig4: Uniform segmentation TP/FP/FN (6x3)
  - fig5: Gaussian vs Uniform comparison (3x4)
  - fig6: grouped bar charts

Usage:
    python scripts/evaluate_stage1.py \
        --checkpoint_g runs/gcain_gaussian_1000/checkpoints/best.pth \
        --checkpoint_u runs/gcain_uniform_1000/checkpoints/best.pth \
        --config_g configs/stage1/gaussian_1000.yaml \
        --config_u configs/stage1/uniform_1000.yaml \
        --shared_dir /home/foods/pro/FMT-SimGen/output/shared \
        --output_dir results/stage1/
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
import torch
import yaml
from torch.utils.data import DataLoader

from du2vox.data.dataset import FMTSimGenDataset
from du2vox.evaluation.metrics import evaluate_batch, summarize_metrics
from du2vox.models.stage1.gcain import GCAIN_full


# ── Paper-style matplotlib settings ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 11,
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 300,
})


# ─────────────────────────────────────────────────────────────────────────────
# Part A: Inference and Metrics
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg, checkpoint_path, device="cuda"):
    """Build model and load checkpoint."""
    model_cfg = cfg["model"]
    shared_dir = Path(cfg["data"]["shared_dir"])
    samples_dir = Path(cfg["data"]["samples_dir"])
    splits_dir = Path(cfg["data"]["splits_dir"])

    dataset = FMTSimGenDataset(
        shared_dir=shared_dir,
        samples_dir=samples_dir,
        split_file=splits_dir / "val.txt",
        normalize_b=cfg["data"].get("normalize_b", True),
        normalize_gt=cfg["data"].get("normalize_gt", True),
        binarize_gt=cfg["data"].get("binarize_gt", False),
        binarize_threshold=cfg["data"].get("binarize_threshold", 0.05),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    A = dataset.A.to(device)
    L = dataset.L.to(device)
    L0 = dataset.L0.to(device)
    L1 = dataset.L1.to(device)
    L2 = dataset.L2.to(device)
    L3 = dataset.L3.to(device)
    knn_idx = dataset.knn_idx.to(device)
    sens_w = dataset.sens_w.to(device)
    nodes = dataset.nodes.to(device)

    model = GCAIN_full(
        L=L, A=A,
        L0=L0, L1=L1, L2=L2, L3=L3,
        knn_idx=knn_idx, sens_w=sens_w,
        num_layer=model_cfg["num_layer"],
        feat_dim=model_cfg["feat_dim"],
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model, dataset, nodes


def derive_depth_tier(z_coord):
    """Derive depth tier from z-coordinate of tumor center."""
    # Based on quantiles observed in data: shallow < 7.5, medium < 13.5, deep >= 13.5
    if z_coord < 7.5:
        return "shallow"
    elif z_coord < 13.5:
        return "medium"
    else:
        return "deep"


def load_tumor_params(samples_dir, sample_id):
    """Load tumor_params.json for a sample."""
    path = samples_dir / sample_id / "tumor_params.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def get_depth_tier_from_sample(samples_dir, sample_id):
    """Get depth tier derived from foci z-coordinates."""
    params = load_tumor_params(samples_dir, sample_id)
    foci = params.get("foci", [])
    if not foci:
        return "unknown"
    # Use average z of all foci centers
    z_vals = []
    for focus in foci:
        center = focus.get("center", [])
        if len(center) >= 3:
            z_vals.append(center[2])
    if not z_vals:
        return "unknown"
    avg_z = sum(z_vals) / len(z_vals)
    return derive_depth_tier(avg_z)


def run_inference(model, dataset, nodes, device="cuda"):
    """Run inference on all val samples, return per-sample metrics + metadata."""
    all_metrics = []
    all_sample_ids = dataset.sample_ids

    with torch.no_grad():
        for i in range(len(dataset)):
            batch = dataset[i]
            b = batch["b"].unsqueeze(0).to(device)
            gt = batch["gt"].unsqueeze(0).to(device)
            X0 = torch.zeros(1, nodes.shape[0], 1, device=device)
            pred = model(X0, b)
            pred = torch.clamp(pred, min=0.0, max=1.0)
            metrics = evaluate_batch(pred, gt, nodes)
            all_metrics.append(metrics)

    return all_metrics, all_sample_ids


def compute_all_metrics_for_sample(pred, gt, nodes):
    """Compute comprehensive metrics for a single sample."""
    pred_squeeze = pred.squeeze().cpu().numpy() if torch.is_tensor(pred) else pred
    gt_squeeze = gt.squeeze().cpu().numpy() if torch.is_tensor(gt) else gt

    pred_clamp = np.clip(pred_squeeze, 0.0, 1.0)
    gt_clamp = np.clip(gt_squeeze, 0.0, 1.0)

    def binary_dice(pred, gt, thresh):
        p = (pred > thresh).astype(float)
        g = (gt > thresh).astype(float)
        intersect = (p * g).sum()
        return 2 * intersect / (p.sum() + g.sum() + 1e-8)

    def recall_prec(pred, gt, pred_th, gt_th):
        p_bin = (pred > pred_th).astype(float)
        g_bin = (gt > gt_th).astype(float)
        tp = (p_bin * g_bin).sum()
        fn = ((1 - p_bin) * g_bin).sum()
        fp = (p_bin * (1 - g_bin)).sum()
        recall = tp / (tp + fn + 1e-8)
        prec = tp / (tp + fp + 1e-8)
        return recall, prec

    def centroid_dist(pred, gt, nodes_arr):
        """Compute centroid distance in mm."""
        p_bin = (pred > 0.1).astype(float)
        g_bin = (gt > 0.05).astype(float)
        if p_bin.sum() == 0 or g_bin.sum() == 0:
            return 0.0
        p_centroid = (nodes_arr * p_bin[:, None]).sum(axis=0) / (p_bin.sum() + 1e-8)
        g_centroid = (nodes_arr * g_bin[:, None]).sum(axis=0) / (g_bin.sum() + 1e-8)
        return np.linalg.norm(p_centroid - g_centroid)

    nodes_np = nodes.cpu().numpy() if torch.is_tensor(nodes) else nodes

    dice_soft = binary_dice(pred_clamp, gt_clamp, thresh=0.0)  # Dice via correlation
    dice_05 = binary_dice(pred_clamp, gt_clamp, thresh=0.5)
    dice_03 = binary_dice(pred_clamp, gt_clamp, thresh=0.3)
    dice_01 = binary_dice(pred_clamp, gt_clamp, thresh=0.1)

    recall_01, prec_01 = recall_prec(pred_clamp, gt_clamp, 0.1, 0.05)
    recall_03, prec_03 = recall_prec(pred_clamp, gt_clamp, 0.3, 0.05)

    loc_err = centroid_dist(pred_clamp, gt_clamp, nodes_np)
    mse = np.mean((pred_clamp - gt_clamp) ** 2)

    return {
        "dice_soft": dice_soft,
        "dice_05": dice_05,
        "dice_03": dice_03,
        "dice_01": dice_01,
        "recall_01": recall_01,
        "recall_03": recall_03,
        "precision_01": prec_01,
        "precision_03": prec_03,
        "location_error": loc_err,
        "mse": mse,
        "gt_max": float(gt_clamp.max()),
        "pred_max": float(pred_clamp.max()),
        "pred_mean": float(pred_clamp.mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Part B: Table Generation
# ─────────────────────────────────────────────────────────────────────────────

# Note: evaluate_batch returns dice_bin_0.5, dice_bin_0.3, dice_bin_0.1
METRIC_COLS = ["Dice@0.5", "Dice@0.3", "Dice@0.1", "Recall@0.1", "Recall@0.3", "Precision@0.3", "LocErr", "MSE"]
METRIC_KEYS = ["dice_bin_0.5", "dice_bin_0.3", "dice_bin_0.1", "recall_0.1", "recall_0.3", "precision_0.3", "location_error", "mse"]


def build_metrics_dataframe(all_metrics, all_sample_ids, samples_dir, source_type):
    """Build DataFrame with per-sample metrics and grouping info."""
    rows = []
    for i, (metrics, sample_id) in enumerate(zip(all_metrics, all_sample_ids)):
        params = load_tumor_params(Path(samples_dir), sample_id)
        num_foci = params.get("num_foci", 0)
        avg_z = None
        foci = params.get("foci", [])
        if foci:
            z_vals = [f.get("center", [None, None, 10])[2] for f in foci if f.get("center")]
            if z_vals:
                avg_z = sum(z_vals) / len(z_vals)
        depth_tier = derive_depth_tier(avg_z) if avg_z is not None else "unknown"

        row = {
            "sample_id": sample_id,
            "source": source_type,
            "num_foci": num_foci,
            "depth_tier": depth_tier,
            **metrics
        }
        rows.append(row)
    return pd.DataFrame(rows)


def table1_overall(df_g, df_u):
    """Table 1: Overall comparison."""
    rows = []
    for name, df in [("Gaussian", df_g), ("Uniform", df_u)]:
        row = {"Source": name}
        for col, key in zip(METRIC_COLS, METRIC_KEYS):
            row[col] = df[key].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def table2_by_foci(df_g, df_u):
    """Table 2: By num_foci."""
    rows = []
    for name, df in [("Gaussian", df_g), ("Uniform", df_u)]:
        for foci in [1, 2, 3]:
            sub = df[df["num_foci"] == foci]
            if len(sub) == 0:
                continue
            row = {"Source": name, "Foci": foci}
            for col, key in zip(METRIC_COLS, METRIC_KEYS):
                row[col] = sub[key].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def table3_by_depth(df_g, df_u):
    """Table 3: By depth tier."""
    rows = []
    for name, df in [("Gaussian", df_g), ("Uniform", df_u)]:
        for depth in ["shallow", "medium", "deep"]:
            sub = df[df["depth_tier"] == depth]
            if len(sub) == 0:
                continue
            row = {"Source": name, "Depth": depth.title()}
            for col, key in zip(METRIC_COLS, METRIC_KEYS):
                row[col] = sub[key].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def table4_cross(df_g, df_u):
    """Table 4: Cross-group (foci x depth)."""
    rows = []
    for name, df in [("Gaussian", df_g), ("Uniform", df_u)]:
        for foci in [1, 2, 3]:
            for depth in ["shallow", "medium", "deep"]:
                sub = df[(df["num_foci"] == foci) & (df["depth_tier"] == depth)]
                if len(sub) == 0:
                    continue
                row = {"Source": name, "Foci": foci, "Depth": depth.title()}
                for col, key in zip(METRIC_COLS, METRIC_KEYS):
                    row[col] = sub[key].mean()
                row["N"] = len(sub)
                rows.append(row)
    return pd.DataFrame(rows)


def save_csv_latex(df, base_path):
    """Save DataFrame as CSV and LaTeX."""
    csv_path = f"{base_path}.csv"
    tex_path = f"{base_path}.tex"

    df.to_csv(csv_path, index=False, float_format="%.4f")

    # Build LaTeX table
    col_format = "l" + "r" * len(df.columns)
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        f"\\begin{{tabular}}{{{col_format}}}",
        r"\hline",
    ]
    header = " & ".join(df.columns.tolist()) + r" \\"
    lines.append(header)
    lines.append(r"\hline")
    for _, row in df.iterrows():
        vals = []
        for v in row:
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append(" & ".join(vals) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(tex_path, "w") as f:
        f.write("\n".join(lines))

    return csv_path, tex_path


# ─────────────────────────────────────────────────────────────────────────────
# Part C: Training Curves
# ─────────────────────────────────────────────────────────────────────────────

def parse_training_logs(log_dir):
    """Parse training logs to extract CSV rows."""
    log_files = glob.glob(str(Path(log_dir) / "train_*.log"))
    rows = []
    for lf in log_files:
        with open(lf) as f:
            for line in f:
                if line.startswith("[CSV]") and "epoch," not in line:
                    parts = line.strip().replace("[CSV] ", "").split(",")
                    try:
                        rows.append({
                            "epoch": int(parts[0]),
                            "train_loss": float(parts[1]),
                            "val_loss": float(parts[2]),
                            "dice": float(parts[3]),
                            "dice_03": float(parts[4]),
                            "dice_01": float(parts[5]),
                            "recall_01": float(parts[6]),
                            "prec_03": float(parts[7]),
                            "loc_err": float(parts[8]),
                            "mse": float(parts[9]),
                            "lr": float(parts[13]) if len(parts) > 13 else 0.0,
                        })
                    except (ValueError, IndexError):
                        continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # De-duplicate by epoch (keep last)
    df = df.sort_values("epoch").drop_duplicates(subset=["epoch"], keep="last")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Part D: Visualization
# ─────────────────────────────────────────────────────────────────────────────

# Organ color map: tissue_label -> (r, g, b, alpha)
ORGAN_STYLE = {
    1: (0.95, 0.85, 0.75, 0.15),   # 皮肤 - skin
    2: (0.95, 0.95, 0.95, 0.30),    # 骨骼 - bone
    3: (0.70, 0.85, 0.95, 0.20),    # 肺 - lung
    4: (0.90, 0.30, 0.30, 0.25),    # 心脏 - heart
    5: (0.55, 0.15, 0.15, 0.25),    # 肝脏 - liver
    6: (0.60, 0.35, 0.20, 0.25),    # 肾脏 - kidney
    7: (0.90, 0.80, 0.80, 0.15),    # 肌肉 - muscle
    8: (0.80, 0.80, 0.90, 0.15),    # 脑 - brain
    9: (0.80, 0.20, 0.20, 0.30),    # 心脏 - heart (alternate)
    10: (0.70, 0.60, 0.50, 0.20),   # 脂肪 - fat
    11: (0.80, 0.70, 0.60, 0.15),   # 其他软组织
}


def build_tet_mesh(nodes, elements, scalars=None, name="value"):
    """Build PyVista UnstructuredGrid from tet mesh."""
    n_tets = elements.shape[0]
    cells = np.hstack([
        np.full((n_tets, 1), 4, dtype=elements.dtype),
        elements,
    ]).ravel()
    celltypes = np.full(n_tets, pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, celltypes, nodes.astype(np.float64))
    if scalars is not None:
        grid.point_data[name] = scalars
    return grid


def get_body_wireframe(nodes, elements, tissue_labels):
    """Extract mouse body wireframe."""
    elem_labels = np.zeros(elements.shape[0], dtype=int)
    for i, tet in enumerate(elements):
        vals, counts = np.unique(tissue_labels[tet], return_counts=True)
        elem_labels[i] = vals[counts.argmax()]
    mask = elem_labels > 0
    body_grid = build_tet_mesh(nodes, elements[mask])
    body_surf = body_grid.extract_surface()
    body_surf = body_surf.smooth(n_iter=30, relaxation_factor=0.1)
    return body_surf


def get_organ_surfaces(nodes, elements, tissue_labels, organ_ids):
    """Extract organ surfaces."""
    elem_labels = np.zeros(elements.shape[0], dtype=int)
    for i, tet in enumerate(elements):
        vals, counts = np.unique(tissue_labels[tet], return_counts=True)
        elem_labels[i] = vals[counts.argmax()]
    surfaces = {}
    for oid in organ_ids:
        mask = elem_labels == oid
        if mask.sum() < 10:
            continue
        sub = build_tet_mesh(nodes, elements[mask])
        surf = sub.extract_surface().smooth(n_iter=20, relaxation_factor=0.1)
        if surf.n_points > 0:
            surfaces[oid] = surf
    return surfaces


def render_intensity_plot(
    nodes, elements, tissue_labels,
    values, title, colormap, clim,
    save_path, body_surf, organ_surfs,
    show_colorbar=True
):
    """Render intensity on mesh with organ overlay."""
    plotter = pv.Plotter(off_screen=True, window_size=(800, 700))
    plotter.set_background("white")

    # Body wireframe
    plotter.add_mesh(
        body_surf,
        color=(0.85, 0.85, 0.85),
        style="wireframe",
        line_width=0.3,
        opacity=0.05,
    )

    # Organ surfaces
    for oid, surf in organ_surfs.items():
        style = ORGAN_STYLE.get(oid, (0.7, 0.7, 0.7, 0.1))
        plotter.add_mesh(
            surf, color=style[:3], opacity=style[3],
            smooth_shading=True,
        )

    # Active points
    active_mask = values > 0.05
    if active_mask.sum() > 0:
        pts = pv.PolyData(nodes[active_mask])
        pts["intensity"] = values[active_mask]
        scalar_args = {
            "scalars": "intensity",
            "cmap": colormap,
            "clim": clim,
            "point_size": 5,
            "render_points_as_spheres": True,
        }
        if show_colorbar:
            scalar_args["scalar_bar_args"] = {
                "title": "Intensity",
                "title_font_size": 11,
                "label_font_size": 9,
                "width": 0.05,
                "position_x": 0.92,
            }
        plotter.add_mesh(pts, **scalar_args)

    plotter.add_text(title, position="upper_left", font_size=12,
                     font="times", color="black")
    plotter.camera_position = [(18, 50, 40), (18, 50, 10), (0, 0, 1)]
    plotter.camera.zoom(1.2)
    plotter.screenshot(save_path, transparent_background=False)
    plotter.close()


def render_segmentation_overlay(
    nodes, elements, tissue_labels,
    gt, pred, threshold_gt, threshold_pred,
    save_path, body_surf, organ_surfs,
    title=""
):
    """Render TP/FP/FN overlay."""
    gt_bin = gt > threshold_gt
    pred_bin = pred > threshold_pred

    tp_mask = gt_bin & pred_bin
    fp_mask = (~gt_bin) & pred_bin
    fn_mask = gt_bin & (~pred_bin)

    plotter = pv.Plotter(off_screen=True, window_size=(800, 700))
    plotter.set_background("white")

    # Body wireframe
    plotter.add_mesh(
        body_surf,
        color=(0.85, 0.85, 0.85),
        style="wireframe",
        line_width=0.3,
        opacity=0.05,
    )

    # Organ surfaces
    for oid, surf in organ_surfs.items():
        style = ORGAN_STYLE.get(oid, (0.7, 0.7, 0.7, 0.1))
        plotter.add_mesh(
            surf, color=style[:3], opacity=style[3],
            smooth_shading=True,
        )

    # TP - green
    if tp_mask.any():
        tp_cloud = pv.PolyData(nodes[tp_mask])
        plotter.add_mesh(tp_cloud, color=(0.2, 0.8, 0.2), opacity=0.9,
                        point_size=5, render_points_as_spheres=True)

    # FP - blue
    if fp_mask.any():
        fp_cloud = pv.PolyData(nodes[fp_mask])
        plotter.add_mesh(fp_cloud, color=(0.2, 0.4, 0.9), opacity=0.7,
                        point_size=5, render_points_as_spheres=True)

    # FN - red
    if fn_mask.any():
        fn_cloud = pv.PolyData(nodes[fn_mask])
        plotter.add_mesh(fn_cloud, color=(0.9, 0.2, 0.2), opacity=0.9,
                        point_size=5, render_points_as_spheres=True)

    # Legend
    plotter.add_legend([
        ["TP", "(0.2,0.8,0.2)"],
        ["FP", "(0.2,0.4,0.9)"],
        ["FN", "(0.9,0.2,0.2)"],
    ], bcolor="white", size=(0.12, 0.10))

    plotter.add_text(title, position="upper_left", font_size=11,
                     font="times", color="black")
    plotter.camera_position = [(18, 50, 40), (18, 50, 10), (0, 0, 1)]
    plotter.camera.zoom(1.2)
    plotter.screenshot(save_path, transparent_background=False)
    plotter.close()


def select_representative_samples(df, n=6):
    """Select representative samples from each (foci, depth) group."""
    groups = [
        (1, "shallow"),
        (1, "deep"),
        (2, "medium"),
        (3, "shallow"),
        (3, "medium"),
        (3, "deep"),
    ]
    selected = {}
    for foci, depth in groups:
        sub = df[(df["num_foci"] == foci) & (df["depth_tier"] == depth)]
        if len(sub) == 0:
            # Try any depth
            sub = df[df["num_foci"] == foci]
        if len(sub) == 0:
            continue
        # Pick sample closest to median dice_03
        median_dice = sub["dice_bin_0.3"].median()
        idx = (sub["dice_bin_0.3"] - median_dice).abs().idxmin()
        selected[f"{foci}-foci-{depth}"] = {
            "sample_id": sub.loc[idx, "sample_id"],
            "dice_03": sub.loc[idx, "dice_bin_0.3"],
            "num_foci": foci,
            "depth_tier": depth,
        }
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 1 comprehensive evaluation")
    parser.add_argument("--checkpoint_g", required=True, help="Gaussian checkpoint path")
    parser.add_argument("--checkpoint_u", required=True, help="Uniform checkpoint path")
    parser.add_argument("--config_g", required=True, help="Gaussian config path")
    parser.add_argument("--config_u", required=True, help="Uniform config path")
    parser.add_argument("--shared_dir", required=True, help="Shared data directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STAGE 1 COMPREHENSIVE EVALUATION")
    print("=" * 70)

    # Load configs
    with open(args.config_g) as f:
        cfg_g = yaml.safe_load(f)
    with open(args.config_u) as f:
        cfg_u = yaml.safe_load(f)

    # Build models
    print("\n[1/8] Building models...")
    device = "cuda"
    model_g, dataset_g, nodes_g = build_model(cfg_g, args.checkpoint_g, device)
    model_u, dataset_u, nodes_u = build_model(cfg_u, args.checkpoint_u, device)
    print(f"  Gaussian model: {len(dataset_g)} val samples")
    print(f"  Uniform model:   {len(dataset_u)} val samples")

    # Run inference
    print("\n[2/8] Running inference...")
    metrics_g, sample_ids_g = run_inference(model_g, dataset_g, nodes_g, device)
    metrics_u, sample_ids_u = run_inference(model_u, dataset_u, nodes_u, device)

    # Build DataFrames
    df_g = build_metrics_dataframe(metrics_g, sample_ids_g, cfg_g["data"]["samples_dir"], "Gaussian")
    df_u = build_metrics_dataframe(metrics_u, sample_ids_u, cfg_u["data"]["samples_dir"], "Uniform")

    # Save per-sample CSVs
    df_g.to_csv(out_dir / "metrics_per_sample_gaussian.csv", index=False, float_format="%.6f")
    df_u.to_csv(out_dir / "metrics_per_sample_uniform.csv", index=False, float_format="%.6f")
    print(f"  Saved per-sample CSVs")

    # Generate tables
    print("\n[3/8] Generating tables...")
    t1 = table1_overall(df_g, df_u)
    t2 = table2_by_foci(df_g, df_u)
    t3 = table3_by_depth(df_g, df_u)
    t4 = table4_cross(df_g, df_u)

    save_csv_latex(t1, out_dir / "table1_overall")
    save_csv_latex(t2, out_dir / "table2_by_foci")
    save_csv_latex(t3, out_dir / "table3_by_depth")
    save_csv_latex(t4, out_dir / "table4_cross")
    print(f"  Saved table1-4 (CSV + LaTeX)")

    # Print tables
    print("\n" + "=" * 70)
    print("TABLE 1: Overall")
    print("=" * 70)
    print(t1.to_string(index=False))
    print("\nTABLE 2: By Foci")
    print("=" * 70)
    print(t2.to_string(index=False))
    print("\nTABLE 3: By Depth")
    print("=" * 70)
    print(t3.to_string(index=False))
    print("\nTABLE 4: Cross (foci x depth)")
    print("=" * 70)
    print(t4.to_string(index=False))

    # Training curves
    print("\n[4/8] Generating training curves (fig1)...")
    log_g = parse_training_logs(Path(args.checkpoint_g).parent.parent)
    log_u = parse_training_logs(Path(args.checkpoint_u).parent.parent)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Stage 1 Training Curves: Gaussian vs Uniform", fontfamily="serif", fontsize=13)

    # (a) Val Loss
    ax = axes[0, 0]
    if len(log_g) > 0:
        ax.plot(log_g["epoch"], log_g["val_loss"], "b-", label="Gaussian", linewidth=1.5)
    if len(log_u) > 0:
        ax.plot(log_u["epoch"], log_u["val_loss"], "orange", label="Uniform", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Loss")
    ax.set_title("(a) Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) Dice@0.3
    ax = axes[0, 1]
    if len(log_g) > 0:
        ax.plot(log_g["epoch"], log_g["dice_03"], "b-", label="Gaussian", linewidth=1.5)
        best_g = log_g["dice_03"].max()
        best_ep_g = log_g.loc[log_g["dice_03"].idxmax(), "epoch"]
        ax.scatter([best_ep_g], [best_g], color="blue", s=100, zorder=5, marker="*", label=f"Best={best_g:.4f}")
    if len(log_u) > 0:
        ax.plot(log_u["epoch"], log_u["dice_03"], "orange", label="Uniform", linewidth=1.5)
        best_u = log_u["dice_03"].max()
        best_ep_u = log_u.loc[log_u["dice_03"].idxmax(), "epoch"]
        ax.scatter([best_ep_u], [best_u], color="orange", s=100, zorder=5, marker="*", label=f"Best={best_u:.4f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice@0.3")
    ax.set_title("(b) Dice@0.3")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) Dice@0.1
    ax = axes[1, 0]
    if len(log_g) > 0:
        ax.plot(log_g["epoch"], log_g["dice_01"], "b-", label="Gaussian", linewidth=1.5)
    if len(log_u) > 0:
        ax.plot(log_u["epoch"], log_u["dice_01"], "orange", label="Uniform", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice@0.1")
    ax.set_title("(c) Dice@0.1")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (d) Precision@0.3 & Recall@0.1
    ax = axes[1, 1]
    if len(log_g) > 0:
        ax.plot(log_g["epoch"], log_g["prec_03"], "b--", label="Gaussian Prec@0.3", linewidth=1.2)
        ax.plot(log_g["epoch"], log_g["recall_01"], "b:", label="Gaussian Recall@0.1", linewidth=1.2)
    if len(log_u) > 0:
        ax.plot(log_u["epoch"], log_u["prec_03"], "orange", linestyle="--", label="Uniform Prec@0.3", linewidth=1.2)
        ax.plot(log_u["epoch"], log_u["recall_01"], "orange", linestyle=":", label="Uniform Recall@0.1", linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("(d) Precision@0.3 & Recall@0.1")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "fig1_training_curves.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  Saved fig1_training_curves.png")

    # ── Load mesh data for visualization ──
    print("\n[5/8] Loading mesh for visualization...")
    mesh_data = np.load(Path(args.shared_dir) / "mesh.npz")
    mesh_nodes = mesh_data["nodes"]
    elements = mesh_data["elements"]
    tissue_labels = mesh_data["tissue_labels"]

    body_surf = get_body_wireframe(mesh_nodes, elements, tissue_labels)
    organ_surfs = get_organ_surfaces(mesh_nodes, elements, tissue_labels,
                                     organ_ids=[1, 2, 3, 4, 5, 6, 7])

    # Select representative samples
    repr_g = select_representative_samples(df_g, n=6)
    repr_u = select_representative_samples(df_u, n=6)

    # Save representative samples info
    repr_data = {
        "gaussian": repr_g,
        "uniform": repr_u,
    }
    with open(out_dir / "representative_samples.json", "w") as f:
        json.dump(repr_data, f, indent=2)

    # ── Build index: sample_id -> (gt, pred) for Gaussian ──
    print("\n[6/8] Generating fig2-4 visualizations...")
    with torch.no_grad():
        pred_dict_g = {}
        gt_dict_g = {}
        for i in range(len(dataset_g)):
            batch = dataset_g[i]
            b = batch["b"].unsqueeze(0).to(device)
            gt = batch["gt"].to(device)
            X0 = torch.zeros(1, nodes_g.shape[0], 1, device=device)
            pred = model_g(X0, b)
            pred = torch.clamp(pred, min=0.0, max=1.0)
            sid = sample_ids_g[i]
            pred_dict_g[sid] = pred.squeeze().cpu().numpy()
            gt_dict_g[sid] = gt.squeeze().cpu().numpy()

        pred_dict_u = {}
        gt_dict_u = {}
        for i in range(len(dataset_u)):
            batch = dataset_u[i]
            b = batch["b"].unsqueeze(0).to(device)
            gt = batch["gt"].to(device)
            X0 = torch.zeros(1, nodes_u.shape[0], 1, device=device)
            pred = model_u(X0, b)
            pred = torch.clamp(pred, min=0.0, max=1.0)
            sid = sample_ids_u[i]
            pred_dict_u[sid] = pred.squeeze().cpu().numpy()
            gt_dict_u[sid] = gt.squeeze().cpu().numpy()

    # ── fig2: Gaussian intensity (6x3) ──
    # 6 rows (samples) x 3 cols (GT, Pred, Error)
    fig2_rows = list(repr_g.items())
    n_rows = len(fig2_rows)
    fig2, axes2 = plt.subplots(n_rows, 3, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes2 = axes2.reshape(1, -1)

    for row_idx, (key, info) in enumerate(fig2_rows):
        sid = info["sample_id"]
        gt_vals = gt_dict_g.get(sid, np.zeros(len(mesh_nodes)))
        pred_vals = pred_dict_g.get(sid, np.zeros(len(mesh_nodes)))
        error_vals = np.abs(gt_vals - pred_vals)

        for col_idx, (vals, cmap, clim, label) in enumerate([
            (gt_vals, "jet", (0, 1), "GT"),
            (pred_vals, "jet", (0, 1), "Pred"),
            (error_vals, "hot", (0, 0.5), "|GT-Pred|"),
        ]):
            ax = axes2[row_idx, col_idx]
            ax.set_facecolor("white")
            # Scatter in 2D (xz plane - dorsal view)
            scatter = ax.scatter(
                mesh_nodes[:, 0], mesh_nodes[:, 2],
                c=vals, cmap=cmap, clim=clim,
                s=0.5, alpha=0.7
            )
            ax.set_xlim([mesh_nodes[:, 0].min(), mesh_nodes[:, 0].max()])
            ax.set_ylim([mesh_nodes[:, 2].min(), mesh_nodes[:, 2].max()])
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(label, fontfamily="serif", fontsize=12)
            if col_idx == 0:
                ax.set_ylabel(f"{key}\nDice={info['dice_03']:.3f}", fontsize=9)
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
            cbar.set_label(label, fontsize=8)

    fig2.suptitle("Fig.2 Gaussian Intensity Distribution", fontfamily="serif", fontsize=13)
    plt.tight_layout()
    fig2.savefig(out_dir / "fig2_gaussian_intensity.png", dpi=300, bbox_inches="tight",
                 facecolor="white")
    plt.close(fig2)
    print(f"  Saved fig2_gaussian_intensity.png")

    # ── fig3: Gaussian segmentation TP/FP/FN (6x3) ──
    # 6 rows x 3 cols (GT mask, Pred mask, TP/FP/FN)
    # For Gaussian: GT threshold 0.05, Pred threshold 0.3
    fig3, axes3 = plt.subplots(n_rows, 3, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes3 = axes3.reshape(1, -1)

    for row_idx, (key, info) in enumerate(fig2_rows):
        sid = info["sample_id"]
        gt_vals = gt_dict_g.get(sid, np.zeros(len(mesh_nodes)))
        pred_vals = pred_dict_g.get(sid, np.zeros(len(mesh_nodes)))

        gt_bin = (gt_vals > 0.05).astype(int)
        pred_bin = (pred_vals > 0.3).astype(int)

        tp = gt_bin & pred_bin
        fp = (~gt_bin.astype(bool)) & pred_bin.astype(bool)
        fn = gt_bin.astype(bool) & (~pred_bin.astype(bool))

        # Col 0: GT mask (red)
        ax = axes3[row_idx, 0]
        ax.set_facecolor("white")
        mask_vals = np.where(gt_bin, 1.0, 0.0)
        ax.scatter(mesh_nodes[:, 0], mesh_nodes[:, 2],
                   c=mask_vals, cmap="Reds", clim=(0, 1), s=0.5, alpha=0.8)
        ax.set_xlim([mesh_nodes[:, 0].min(), mesh_nodes[:, 0].max()])
        ax.set_ylim([mesh_nodes[:, 2].min(), mesh_nodes[:, 2].max()])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if row_idx == 0:
            ax.set_title("GT (red)", fontfamily="serif", fontsize=12)
        if col_idx == 0:
            ax.set_ylabel(f"{key}", fontsize=9)

        # Col 1: Pred mask (blue)
        ax = axes3[row_idx, 1]
        ax.set_facecolor("white")
        mask_vals = np.where(pred_bin, 1.0, 0.0)
        ax.scatter(mesh_nodes[:, 0], mesh_nodes[:, 2],
                   c=mask_vals, cmap="Blues", clim=(0, 1), s=0.5, alpha=0.8)
        ax.set_xlim([mesh_nodes[:, 0].min(), mesh_nodes[:, 0].max()])
        ax.set_ylim([mesh_nodes[:, 2].min(), mesh_nodes[:, 2].max()])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if row_idx == 0:
            ax.set_title("Pred (blue)", fontfamily="serif", fontsize=12)

        # Col 2: TP/FP/FN overlay
        ax = axes3[row_idx, 2]
        ax.set_facecolor("white")
        # Background gray
        ax.scatter(mesh_nodes[:, 0], mesh_nodes[:, 2],
                   color=(0.9, 0.9, 0.9), s=0.3, alpha=0.3)
        # TP green
        ax.scatter(mesh_nodes[tp, 0], mesh_nodes[tp, 2],
                   color=(0.2, 0.8, 0.2), s=1.5, alpha=0.9)
        # FP blue
        ax.scatter(mesh_nodes[fp, 0], mesh_nodes[fp, 2],
                   color=(0.2, 0.4, 0.9), s=1.5, alpha=0.7)
        # FN red
        ax.scatter(mesh_nodes[fn, 0], mesh_nodes[fn, 2],
                   color=(0.9, 0.2, 0.2), s=1.5, alpha=0.9)
        ax.set_xlim([mesh_nodes[:, 0].min(), mesh_nodes[:, 0].max()])
        ax.set_ylim([mesh_nodes[:, 2].min(), mesh_nodes[:, 2].max()])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if row_idx == 0:
            ax.set_title("TP(gr)/FP(bl)/FN(rd)", fontfamily="serif", fontsize=12)
        if row_idx == n_rows - 1:
            ax.legend(handles=[
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=(0.2, 0.8, 0.2), markersize=6, label='TP'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=(0.2, 0.4, 0.9), markersize=6, label='FP'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=(0.9, 0.2, 0.2), markersize=6, label='FN'),
            ], loc='upper right', fontsize=7)

    fig3.suptitle("Fig.3 Gaussian Segmentation (GT>0.05, Pred>0.3)", fontfamily="serif", fontsize=13)
    plt.tight_layout()
    fig3.savefig(out_dir / "fig3_gaussian_segmentation.png", dpi=300, bbox_inches="tight",
                 facecolor="white")
    plt.close(fig3)
    print(f"  Saved fig3_gaussian_segmentation.png")

    # ── fig4: Uniform segmentation TP/FP/FN (6x3) ──
    # For Uniform: GT threshold 0.5, Pred threshold 0.5 (binary source)
    fig4, axes4 = plt.subplots(n_rows, 3, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes4 = axes4.reshape(1, -1)

    # Reconstruct repr_u from df_u if available
    repr_u_items = list(repr_u.items())

    for row_idx, (key, info) in enumerate(repr_u_items):
        sid = info["sample_id"]
        gt_vals = gt_dict_u.get(sid, np.zeros(len(mesh_nodes)))
        pred_vals = pred_dict_u.get(sid, np.zeros(len(mesh_nodes)))

        gt_bin = (gt_vals > 0.5).astype(int)
        pred_bin = (pred_vals > 0.5).astype(int)

        tp = gt_bin & pred_bin
        fp = (~gt_bin.astype(bool)) & pred_bin.astype(bool)
        fn = gt_bin.astype(bool) & (~pred_bin.astype(bool))

        # Col 0: GT mask
        ax = axes4[row_idx, 0]
        ax.set_facecolor("white")
        mask_vals = np.where(gt_bin, 1.0, 0.0)
        ax.scatter(mesh_nodes[:, 0], mesh_nodes[:, 2],
                   c=mask_vals, cmap="Reds", clim=(0, 1), s=0.5, alpha=0.8)
        ax.set_xlim([mesh_nodes[:, 0].min(), mesh_nodes[:, 0].max()])
        ax.set_ylim([mesh_nodes[:, 2].min(), mesh_nodes[:, 2].max()])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if row_idx == 0:
            ax.set_title("GT (red)", fontfamily="serif", fontsize=12)
        if col_idx == 0:
            ax.set_ylabel(f"{key}", fontsize=9)

        # Col 1: Pred mask
        ax = axes4[row_idx, 1]
        ax.set_facecolor("white")
        mask_vals = np.where(pred_bin, 1.0, 0.0)
        ax.scatter(mesh_nodes[:, 0], mesh_nodes[:, 2],
                   c=mask_vals, cmap="Blues", clim=(0, 1), s=0.5, alpha=0.8)
        ax.set_xlim([mesh_nodes[:, 0].min(), mesh_nodes[:, 0].max()])
        ax.set_ylim([mesh_nodes[:, 2].min(), mesh_nodes[:, 2].max()])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if row_idx == 0:
            ax.set_title("Pred (blue)", fontfamily="serif", fontsize=12)

        # Col 2: TP/FP/FN
        ax = axes4[row_idx, 2]
        ax.set_facecolor("white")
        ax.scatter(mesh_nodes[:, 0], mesh_nodes[:, 2],
                   color=(0.9, 0.9, 0.9), s=0.3, alpha=0.3)
        ax.scatter(mesh_nodes[tp, 0], mesh_nodes[tp, 2],
                   color=(0.2, 0.8, 0.2), s=1.5, alpha=0.9)
        ax.scatter(mesh_nodes[fp, 0], mesh_nodes[fp, 2],
                   color=(0.2, 0.4, 0.9), s=1.5, alpha=0.7)
        ax.scatter(mesh_nodes[fn, 0], mesh_nodes[fn, 2],
                   color=(0.9, 0.2, 0.2), s=1.5, alpha=0.9)
        ax.set_xlim([mesh_nodes[:, 0].min(), mesh_nodes[:, 0].max()])
        ax.set_ylim([mesh_nodes[:, 2].min(), mesh_nodes[:, 2].max()])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if row_idx == 0:
            ax.set_title("TP(gr)/FP(bl)/FN(rd)", fontfamily="serif", fontsize=12)

    fig4.suptitle("Fig.4 Uniform Segmentation (GT>0.5, Pred>0.5)", fontfamily="serif", fontsize=13)
    plt.tight_layout()
    fig4.savefig(out_dir / "fig4_uniform_segmentation.png", dpi=300, bbox_inches="tight",
                 facecolor="white")
    plt.close(fig4)
    print(f"  Saved fig4_uniform_segmentation.png")

    # ── fig5: Gaussian vs Uniform comparison (3x4) ──
    # 3 rows (selected samples) x 4 cols: GT(G), Pred(G), GT(U), Pred(U)
    print("\n[7/8] Generating fig5 source comparison...")
    comparison_keys = [
        "1-foci-shallow",
        "2-foci-medium",
        "3-foci-deep",
    ]

    fig5, axes5 = plt.subplots(3, 4, figsize=(16, 12))
    for row_idx, key in enumerate(comparison_keys):
        info_g = repr_g.get(key, {})
        info_u = repr_u.get(key, {})
        sid_g = info_g.get("sample_id", "")
        sid_u = info_u.get("sample_id", "")

        gt_g = gt_dict_g.get(sid_g, np.zeros(len(mesh_nodes)))
        pred_g = pred_dict_g.get(sid_g, np.zeros(len(mesh_nodes)))
        gt_u = gt_dict_u.get(sid_u, np.zeros(len(mesh_nodes)))
        pred_u = pred_dict_u.get(sid_u, np.zeros(len(mesh_nodes)))

        for col_idx, (vals, cmap, clim, title_prefix) in enumerate([
            (gt_g, "jet", (0, 1), "GT(G)"),
            (pred_g, "jet", (0, 1), "Pred(G)"),
            (gt_u, "jet", (0, 1), "GT(U)"),
            (pred_u, "jet", (0, 1), "Pred(U)"),
        ]):
            ax = axes5[row_idx, col_idx]
            ax.set_facecolor("white")
            scatter = ax.scatter(
                mesh_nodes[:, 0], mesh_nodes[:, 2],
                c=vals, cmap=cmap, clim=clim,
                s=0.5, alpha=0.7
            )
            ax.set_xlim([mesh_nodes[:, 0].min(), mesh_nodes[:, 0].max()])
            ax.set_ylim([mesh_nodes[:, 2].min(), mesh_nodes[:, 2].max()])
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(title_prefix, fontfamily="serif", fontsize=12)
            if col_idx == 0:
                ax.set_ylabel(f"{key}\nDice={info_g.get('dice_03', 0):.3f}", fontsize=9)
            plt.colorbar(scatter, ax=ax, shrink=0.6)

    fig5.suptitle("Fig.5 Gaussian vs Uniform Source Comparison", fontfamily="serif", fontsize=13)
    plt.tight_layout()
    fig5.savefig(out_dir / "fig5_source_comparison.png", dpi=300, bbox_inches="tight",
                 facecolor="white")
    plt.close(fig5)
    print(f"  Saved fig5_source_comparison.png")

    # ── fig6: Grouped bar charts (9 groups x 2 sources) ──
    print("\n[8/8] Generating fig6 grouped bar charts...")
    fig6, axes6 = plt.subplots(2, 1, figsize=(16, 10))

    # Prepare cross-group data for bar chart
    groups = ["1F-S", "1F-M", "1F-D", "2F-S", "2F-M", "2F-D", "3F-S", "3F-M", "3F-D"]
    foci_labels = {1: "1F", 2: "2F", 3: "3F"}
    depth_abbrev = {"shallow": "S", "medium": "M", "deep": "D"}

    dice_g_vals = []
    dice_u_vals = []
    recall_g_vals = []
    recall_u_vals = []

    for foci in [1, 2, 3]:
        for depth in ["shallow", "medium", "deep"]:
            sub_g = df_g[(df_g["num_foci"] == foci) & (df_g["depth_tier"] == depth)]
            sub_u = df_u[(df_u["num_foci"] == foci) & (df_u["depth_tier"] == depth)]
            dice_g_vals.append(sub_g["dice_bin_0.3"].mean() if len(sub_g) > 0 else 0.0)
            dice_u_vals.append(sub_u["dice_bin_0.3"].mean() if len(sub_u) > 0 else 0.0)
            recall_g_vals.append(sub_g["recall_0.1"].mean() if len(sub_g) > 0 else 0.0)
            recall_u_vals.append(sub_u["recall_0.1"].mean() if len(sub_u) > 0 else 0.0)

    x = np.arange(len(groups))
    width = 0.35

    # (a) Dice@0.3
    ax = axes6[0]
    bars_g = ax.bar(x - width/2, dice_g_vals, width, label="Gaussian", color="steelblue", alpha=0.85)
    bars_u = ax.bar(x + width/2, dice_u_vals, width, label="Uniform", color="darkorange", alpha=0.85)
    ax.set_ylabel("Dice@0.3")
    ax.set_title("(a) Dice@0.3 by Foci × Depth")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 1.0])
    # Value labels
    for bar, val in zip(bars_g, dice_g_vals):
        if val > 0.05:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    for bar, val in zip(bars_u, dice_u_vals):
        if val > 0.05:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    # (b) Recall@0.1
    ax = axes6[1]
    bars_g = ax.bar(x - width/2, recall_g_vals, width, label="Gaussian", color="steelblue", alpha=0.85)
    bars_u = ax.bar(x + width/2, recall_u_vals, width, label="Uniform", color="darkorange", alpha=0.85)
    ax.set_ylabel("Recall@0.1")
    ax.set_title("(b) Recall@0.1 by Foci × Depth")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 1.0])

    fig6.suptitle("Fig.6 Grouped Performance: Gaussian vs Uniform", fontfamily="serif", fontsize=13)
    plt.tight_layout()
    fig6.savefig(out_dir / "fig6_grouped_bar.png", dpi=300, bbox_inches="tight",
                 facecolor="white")
    plt.close(fig6)
    print(f"  Saved fig6_grouped_bar.png")

    print("\n" + "=" * 70)
    print(f"ALL OUTPUTS SAVED TO: {out_dir}/")
    print("=" * 70)
    print("Tables:")
    print(f"  - table1_overall.csv + .tex")
    print(f"  - table2_by_foci.csv + .tex")
    print(f"  - table3_by_depth.csv + .tex")
    print(f"  - table4_cross.csv + .tex")
    print("Figures:")
    print(f"  - fig1_training_curves.png")
    print(f"  - fig2_gaussian_intensity.png")
    print(f"  - fig3_gaussian_segmentation.png")
    print(f"  - fig4_uniform_segmentation.png")
    print(f"  - fig5_source_comparison.png")
    print(f"  - fig6_grouped_bar.png")
    print(f"  - representative_samples.json")


if __name__ == "__main__":
    main()
