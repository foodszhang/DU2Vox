#!/usr/bin/env python3
"""
Stage 1 Tecplot-style tetrahedral mesh visualization.

Renders full mouse anatomy with:
  - Semi-transparent body surface
  - Internal organs (liver, lung, kidney, heart) at low opacity
  - GT tumor: opaque tetrahedra with intensity colormap (Gaussian) or red (Uniform)
  - Pred tumor: intensity overlay or TP/FP/FN three-color
  - Yellow sphere annotations at tumor centers
  - Oblique camera view (45°)

Usage:
    python scripts/render_stage1_figures.py \
        --checkpoint_g runs/gcain_gaussian_1000/checkpoints/best.pth \
        --checkpoint_u runs/gcain_uniform_1000/checkpoints/best.pth \
        --config_g configs/stage1/gaussian_1000.yaml \
        --config_u configs/stage1/uniform_1000.yaml \
        --shared_dir /home/foods/pro/FMT-SimGen/output/shared \
        --samples_g /home/foods/pro/FMT-SimGen/data/gaussian_1000/samples \
        --samples_u /home/foods/pro/FMT-SimGen/data/uniform_1000/samples \
        --output_dir results/stage1/figures/
"""

import argparse
import json
import os
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pyvista as pv
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from du2vox.data.dataset import FMTSimGenDataset
from du2vox.models.stage1.gcain import GCAIN_full


# ─────────────────────────────────────────────────────────────────────────────
# Organ color scheme (Digimouse tissue labels)
# ─────────────────────────────────────────────────────────────────────────────
ORGAN_STYLE = {
    # label: (name, (R, G, B), opacity)
    1:  ("Skin",      (0.95, 0.85, 0.75), 0.06),
    2:  ("Skeleton",  (0.95, 0.95, 0.90), 0.18),
    3:  ("Brain",     (1.00, 0.85, 0.85), 0.10),
    4:  ("Heart",     (0.80, 0.15, 0.15), 0.22),
    5:  ("Lung",      (0.70, 0.85, 0.95), 0.18),
    6:  ("Liver",     (0.55, 0.15, 0.15), 0.22),
    7:  ("Kidney",    (0.65, 0.30, 0.20), 0.22),
    8:  ("Spleen",    (0.60, 0.20, 0.30), 0.15),
    9:  ("Stomach",   (0.90, 0.75, 0.50), 0.15),
    10: ("Pancreas",  (0.85, 0.70, 0.50), 0.12),
    11: ("Muscle",    (0.85, 0.70, 0.70), 0.05),
}


# ─────────────────────────────────────────────────────────────────────────────
# Mesh builders
# ─────────────────────────────────────────────────────────────────────────────

def build_tet_mesh(nodes, elements):
    """Build PyVista UnstructuredGrid from tetrahedral elements."""
    n_cells = len(elements)
    cells = np.column_stack([
        np.full(n_cells, 4, dtype=np.int64),
        elements.astype(np.int64)
    ]).ravel()
    cell_types = np.full(n_cells, pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, cell_types, nodes.astype(np.float64))
    return grid


def build_surface_mesh(nodes, surface_faces):
    """Build PyVista PolyData from triangular surface faces."""
    faces_pv = np.column_stack([
        np.full(len(surface_faces), 3, dtype=np.int64),
        surface_faces.astype(np.int64)
    ]).ravel()
    return pv.PolyData(nodes.astype(np.float64), faces_pv)


def elem_majority_vote(values_per_node, elements):
    """Map node values to element values via majority vote (for binarized masks)."""
    n_elem = len(elements)
    elem_vals = np.zeros(n_elem, dtype=np.int32)
    for i, elem in enumerate(elements):
        vals = values_per_node[elem]
        # Majority vote
        counts = np.bincount(vals.astype(int), minlength=2)
        elem_vals[i] = 1 if counts[1] >= 2 else 0
    return elem_vals


def elem_mean_values(values_per_node, elements):
    """Map node values to element values via mean."""
    return values_per_node[elements].mean(axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_single_sample(
    nodes, elements, surface_faces, tissue_labels,
    gt_nodes, pred_nodes,
    tumor_params,
    title,
    save_path,
    mode="intensity",
    source_type="gaussian",
    gt_threshold=0.05,
    pred_threshold=0.3,
    camera_position="oblique",
    window_size=(1200, 900),
):
    """Render one sample with full anatomy."""

    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("white")

    mesh_center = nodes.mean(axis=0)

    # ── Layer 1: Body surface ──────────────────────────────────────────
    surface = build_surface_mesh(nodes, surface_faces)
    plotter.add_mesh(
        surface,
        color=(0.92, 0.85, 0.78),
        opacity=0.07,
        show_edges=False,
        smooth_shading=True,
    )

    # ── Layer 2: Organs ────────────────────────────────────────────────
    tet_grid = build_tet_mesh(nodes, elements)

    for label, (name, color, opacity) in ORGAN_STYLE.items():
        organ_mask = tissue_labels == label
        if organ_mask.sum() < 10:
            continue
        organ = tet_grid.extract_cells(np.where(organ_mask)[0])
        plotter.add_mesh(
            organ,
            color=color,
            opacity=opacity,
            show_edges=False,
            smooth_shading=True,
            label=name,
        )

    # ── Layer 3: GT and Pred ───────────────────────────────────────────
    if mode == "intensity":
        if source_type == "gaussian":
            # GT: intensity colormap (YlOrRd)
            gt_active = gt_nodes > gt_threshold
            if gt_active.any():
                elem_gt = elem_mean_values(gt_nodes, elements)
                active_elems = elem_gt > gt_threshold
                if active_elems.any():
                    gt_tet = tet_grid.extract_cells(np.where(active_elems)[0])
                    gt_tet.cell_data["gt_int"] = elem_gt[active_elems]
                    plotter.add_mesh(
                        gt_tet,
                        scalars="gt_int",
                        cmap="YlOrRd",
                        clim=[0, gt_nodes.max()],
                        opacity=0.90,
                        show_edges=False,
                        smooth_shading=True,
                        scalar_bar_args={
                            "title": "GT",
                            "title_font_size": 9,
                            "label_font_size": 8,
                            "width": 0.04,
                            "position_x": 0.93,
                            "position_y": 0.35,
                        },
                    )

            # Pred: intensity colormap (Blues)
            pred_active = pred_nodes > 0.02
            if pred_active.any():
                elem_pred = elem_mean_values(pred_nodes, elements)
                active_elems_p = elem_pred > 0.02
                if active_elems_p.any():
                    pred_tet = tet_grid.extract_cells(np.where(active_elems_p)[0])
                    pred_tet.cell_data["pred_int"] = elem_pred[active_elems_p]
                    plotter.add_mesh(
                        pred_tet,
                        scalars="pred_int",
                        cmap="Blues",
                        clim=[0, pred_nodes.max()],
                        opacity=0.80,
                        show_edges=False,
                        smooth_shading=True,
                        scalar_bar_args={
                            "title": "Pred",
                            "title_font_size": 9,
                            "label_font_size": 8,
                            "width": 0.04,
                            "position_x": 0.93,
                            "position_y": 0.02,
                        },
                    )

        elif source_type == "uniform":
            # Uniform: GT is binary, render as red tetrahedra
            gt_bin = gt_nodes > 0.5
            if gt_bin.any():
                elem_gt = elem_mean_values(gt_nodes, elements)
                active_elems = elem_gt > 0.5
                if active_elems.any():
                    gt_tet = tet_grid.extract_cells(np.where(active_elems)[0])
                    plotter.add_mesh(
                        gt_tet,
                        color=(0.90, 0.15, 0.10),
                        opacity=0.90,
                        show_edges=False,
                        smooth_shading=True,
                        scalar_bar_args={
                            "title": "GT",
                            "title_font_size": 9,
                            "label_font_size": 8,
                            "width": 0.04,
                            "position_x": 0.93,
                            "position_y": 0.35,
                        },
                    )

            pred_active = pred_nodes > 0.05
            if pred_active.any():
                elem_pred = elem_mean_values(pred_nodes, elements)
                active_elems_p = elem_pred > 0.05
                if active_elems_p.any():
                    pred_tet = tet_grid.extract_cells(np.where(active_elems_p)[0])
                    pred_tet.cell_data["pred_int"] = elem_pred[active_elems_p]
                    plotter.add_mesh(
                        pred_tet,
                        scalars="pred_int",
                        cmap="Blues",
                        clim=[0, pred_nodes.max()],
                        opacity=0.75,
                        show_edges=False,
                        smooth_shading=True,
                        scalar_bar_args={
                            "title": "Pred",
                            "title_font_size": 9,
                            "label_font_size": 8,
                            "width": 0.04,
                            "position_x": 0.93,
                            "position_y": 0.02,
                        },
                    )

    elif mode == "segmentation":
        gt_thresh = 0.5 if source_type == "uniform" else gt_threshold
        pred_thresh = 0.5 if source_type == "uniform" else pred_threshold

        gt_bin = gt_nodes > gt_thresh
        pred_bin = pred_nodes > pred_thresh

        # Node → element (majority vote)
        elem_gt = elem_majority_vote(
            gt_bin.astype(np.int32), elements
        ).astype(bool)
        elem_pred = elem_majority_vote(
            pred_bin.astype(np.int32), elements
        ).astype(bool)

        tp_mask = elem_gt & elem_pred
        fp_mask = (~elem_gt) & elem_pred
        fn_mask = elem_gt & (~elem_pred)

        if tp_mask.any():
            tp_tet = tet_grid.extract_cells(np.where(tp_mask)[0])
            plotter.add_mesh(
                tp_tet, color=(0.15, 0.80, 0.20), opacity=0.88,
                show_edges=False, smooth_shading=True, label="TP",
            )
        if fp_mask.any():
            fp_tet = tet_grid.extract_cells(np.where(fp_mask)[0])
            plotter.add_mesh(
                fp_tet, color=(0.25, 0.45, 0.90), opacity=0.60,
                show_edges=False, smooth_shading=True, label="FP",
            )
        if fn_mask.any():
            fn_tet = tet_grid.extract_cells(np.where(fn_mask)[0])
            plotter.add_mesh(
                fn_tet, color=(0.90, 0.15, 0.10), opacity=0.88,
                show_edges=False, smooth_shading=True, label="FN",
            )

        plotter.add_legend(
            bcolor="white", face="circle", size=(0.13, 0.09)
        )

    # ── Layer 4: Tumor center annotations ─────────────────────────────
    for i, focus in enumerate(tumor_params.get("foci", [])):
        center = np.array(focus["center"])
        sphere = pv.Sphere(radius=0.35, center=center)
        plotter.add_mesh(sphere, color="yellow", opacity=0.95)
        plotter.add_point_labels(
            [center],
            [f"F{i+1}"],
            font_size=11,
            text_color="black",
            shape_opacity=0.8,
            shape_color="yellow",
            always_visible=True,
        )

    # ── Title ─────────────────────────────────────────────────────────
    plotter.add_text(
        title,
        position="upper_left",
        font_size=11,
        font="times",
        color="black",
    )

    # ── Camera ────────────────────────────────────────────────────────
    if camera_position == "oblique":
        plotter.camera.position = (
            mesh_center[0] + 25,
            mesh_center[1] - 18,
            mesh_center[2] + 32,
        )
        plotter.camera.focal_point = mesh_center
        plotter.camera.up = (0, 0, 1)
        plotter.camera.zoom(1.4)
    elif camera_position == "dorsal":
        plotter.camera.position = (
            mesh_center[0], mesh_center[1], mesh_center[2] + 50
        )
        plotter.camera.focal_point = mesh_center
        plotter.camera.up = (0, -1, 0)
        plotter.camera.zoom(1.5)
    elif camera_position == "lateral":
        plotter.camera.position = (
            mesh_center[0] + 50, mesh_center[1], mesh_center[2]
        )
        plotter.camera.focal_point = mesh_center
        plotter.camera.up = (0, 0, 1)
        plotter.camera.zoom(1.5)

    plotter.screenshot(save_path, transparent_background=False, scale=2)
    plotter.close()


def stitch_images(image_paths, grid_shape, output_path, cell_size=(1200, 900)):
    """Stitch individual renderings into a single image."""
    rows, cols = grid_shape
    w, h = cell_size
    canvas = Image.new("RGB", (cols * w, rows * h), "white")
    for i, path in enumerate(image_paths):
        if not os.path.exists(path):
            continue
        r, c = divmod(i, cols)
        img = Image.open(path).resize((w, h), Image.LANCZOS)
        canvas.paste(img, (c * w, r * h))
    canvas.save(output_path, dpi=(300, 300))


# ─────────────────────────────────────────────────────────────────────────────
# Model & data loading
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
    if z_coord is None:
        return "unknown"
    if z_coord < 7.5:
        return "shallow"
    elif z_coord < 13.5:
        return "medium"
    else:
        return "deep"


def get_tumor_params(samples_dir, sample_id):
    path = Path(samples_dir) / sample_id / "tumor_params.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def select_representative_samples(metrics_df, n=6):
    """Select samples closest to median Dice@0.3 per (foci, depth) group."""
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
        sub = metrics_df[
            (metrics_df["num_foci"] == foci) & (metrics_df["depth_tier"] == depth)
        ]
        if len(sub) == 0:
            sub = metrics_df[metrics_df["num_foci"] == foci]
        if len(sub) == 0:
            continue
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
    parser = argparse.ArgumentParser(description="Stage 1 tetrahedral mesh visualization")
    parser.add_argument("--checkpoint_g", required=True)
    parser.add_argument("--checkpoint_u", required=True)
    parser.add_argument("--config_g", required=True)
    parser.add_argument("--config_u", required=True)
    parser.add_argument("--shared_dir", required=True)
    parser.add_argument("--samples_g", required=True)
    parser.add_argument("--samples_u", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STAGE 1 TETRAHEDRAL MESH VISUALIZATION")
    print("=" * 70)

    # Load configs
    with open(args.config_g) as f:
        cfg_g = yaml.safe_load(f)
    with open(args.config_u) as f:
        cfg_u = yaml.safe_load(f)

    device = "cuda"

    # Build models
    print("\n[1/5] Building models...")
    model_g, dataset_g, nodes_g = build_model(cfg_g, args.checkpoint_g, device)
    model_u, dataset_u, nodes_u = build_model(cfg_u, args.checkpoint_u, device)

    # Load mesh
    print("[2/5] Loading mesh...")
    mesh_data = np.load(Path(args.shared_dir) / "mesh.npz", allow_pickle=True)
    mesh_nodes = mesh_data["nodes"]
    elements = mesh_data["elements"]
    surface_faces = mesh_data["surface_faces"]
    tissue_labels = mesh_data["tissue_labels"]

    print(f"  nodes: {mesh_nodes.shape}, elements: {elements.shape}")
    print(f"  surface: {surface_faces.shape}, tissue_labels: {tissue_labels.shape}")
    print(f"  unique tissues: {np.unique(tissue_labels)}")

    # ── Run inference ──────────────────────────────────────────────────────
    print("\n[3/5] Running inference...")
    sample_ids_g = dataset_g.sample_ids
    sample_ids_u = dataset_u.sample_ids

    gt_dict_g = {}
    pred_dict_g = {}
    with torch.no_grad():
        for i in range(len(dataset_g)):
            batch = dataset_g[i]
            b = batch["b"].unsqueeze(0).to(device)
            gt = batch["gt"].squeeze().cpu().numpy()
            X0 = torch.zeros(1, nodes_g.shape[0], 1, device=device)
            pred = model_g(X0, b)
            pred = torch.clamp(pred, 0.0, 1.0).squeeze().cpu().numpy()
            sid = sample_ids_g[i]
            gt_dict_g[sid] = gt
            pred_dict_g[sid] = pred

    gt_dict_u = {}
    pred_dict_u = {}
    with torch.no_grad():
        for i in range(len(dataset_u)):
            batch = dataset_u[i]
            b = batch["b"].unsqueeze(0).to(device)
            gt = batch["gt"].squeeze().cpu().numpy()
            X0 = torch.zeros(1, nodes_u.shape[0], 1, device=device)
            pred = model_u(X0, b)
            pred = torch.clamp(pred, 0.0, 1.0).squeeze().cpu().numpy()
            sid = sample_ids_u[i]
            gt_dict_u[sid] = gt
            pred_dict_u[sid] = pred

    # ── Load representative samples from per-sample CSVs ─────────────────
    csv_g_path = Path(args.output_dir).parent / "metrics_per_sample_gaussian.csv"
    csv_u_path = Path(args.output_dir).parent / "metrics_per_sample_uniform.csv"

    # Derive depth tier and build metrics DataFrames
    import pandas as pd

    def build_df(metrics_dict, sample_ids, samples_dir):
        rows = []
        for sid, metrics in zip(sample_ids, metrics_dict):
            params = get_tumor_params(samples_dir, sid)
            foci = params.get("foci", [])
            z_vals = []
            for f in foci:
                c = f.get("center", [])
                if len(c) >= 3:
                    z_vals.append(c[2])
            avg_z = sum(z_vals) / len(z_vals) if z_vals else None
            depth = derive_depth_tier(avg_z)
            num_foci = params.get("num_foci", 0)
            row = {"sample_id": sid, "num_foci": num_foci, "depth_tier": depth}
            row.update(metrics)
            rows.append(row)
        return pd.DataFrame(rows)

    # Build dummy metrics dict (we already have gt/pred so we compute dice here)
    def compute_dice(pred, gt):
        p_bin = (pred > 0.3).astype(float)
        g_bin = (gt > 0.05).astype(float)
        tp = (p_bin * g_bin).sum()
        return 2 * tp / (p_bin.sum() + g_bin.sum() + 1e-8)

    metrics_list_g = []
    for sid in sample_ids_g:
        p = pred_dict_g[sid]
        g = gt_dict_g[sid]
        dice = compute_dice(p, g)
        metrics_list_g.append({"dice_bin_0.3": dice})
    df_g = build_df(metrics_list_g, sample_ids_g, args.samples_g)

    metrics_list_u = []
    for sid in sample_ids_u:
        p = pred_dict_u[sid]
        g = gt_dict_u[sid]
        dice = compute_dice(p, g)
        metrics_list_u.append({"dice_bin_0.3": dice})
    df_u = build_df(metrics_list_u, sample_ids_u, args.samples_u)

    repr_g = select_representative_samples(df_g, n=6)
    repr_u = select_representative_samples(df_u, n=6)

    # Save representative samples
    repr_data = {"gaussian": repr_g, "uniform": repr_u}
    with open(out_dir / "representative_samples.json", "w") as f:
        json.dump(repr_data, f, indent=2)

    print(f"  Gaussian repr: {list(repr_g.keys())}")
    print(f"  Uniform repr:  {list(repr_u.keys())}")

    # ── Render all individual panels ────────────────────────────────────
    print("\n[4/5] Rendering individual panels...")

    # Use a temp dir for individual renders
    tmpdir = Path(tempfile.mkdtemp(prefix="stage1_render_"))

    rendered_paths = []

    # ── Fig A: Gaussian intensity (6 samples × 3 cols) ─────────────────
    # Cols: GT intensity | Pred intensity | |GT-Pred| error
    # We'll render 3 separate views and stitch
    fig_a_keys = list(repr_g.keys())

    for col_idx, col_mode in enumerate(["gt", "pred", "error"]):
        for row_idx, key in enumerate(fig_a_keys):
            info = repr_g[key]
            sid = info["sample_id"]
            gt_vals = gt_dict_g.get(sid, np.zeros(len(mesh_nodes)))
            pred_vals = pred_dict_g.get(sid, np.zeros(len(mesh_nodes)))
            params = get_tumor_params(args.samples_g, sid)

            if col_mode == "gt":
                render_vals = gt_vals.copy()
                mode = "intensity"
                src_type = "gaussian"
                title = f"GT | {key} | Dice={info['dice_03']:.3f}"
                out_path = tmpdir / f"figA_{row_idx}_{col_mode}.png"
            elif col_mode == "pred":
                render_vals = pred_vals.copy()
                mode = "intensity"
                src_type = "gaussian"
                title = f"Pred | {key} | Dice={info['dice_03']:.3f}"
                out_path = tmpdir / f"figA_{row_idx}_{col_mode}.png"
            else:  # error
                # Create a combined array for error display
                render_vals = np.abs(gt_vals - pred_vals)
                mode = "intensity"
                src_type = "gaussian"
                title = f"|GT-Pred| | {key}"
                out_path = tmpdir / f"figA_{row_idx}_{col_mode}.png"

            # Render the sample
            render_single_sample(
                nodes=mesh_nodes,
                elements=elements,
                surface_faces=surface_faces,
                tissue_labels=tissue_labels,
                gt_nodes=gt_vals,
                pred_nodes=pred_vals if col_mode != "error" else np.abs(gt_vals - pred_vals),
                tumor_params=params,
                title=title,
                save_path=str(out_path),
                mode="intensity",
                source_type="gaussian",
                gt_threshold=0.05,
                pred_threshold=0.3,
                camera_position="oblique",
            )
            rendered_paths.append(str(out_path))

    # Stitch fig A
    fig_a_out = out_dir / "fig_gaussian_intensity.png"
    stitch_images(
        [str(tmpdir / f"figA_{r}_{c}.png")
         for r in range(len(fig_a_keys)) for c in ["gt", "pred", "error"]],
        grid_shape=(len(fig_a_keys), 3),
        output_path=str(fig_a_out),
        cell_size=(1200, 900),
    )
    print(f"  Saved fig_gaussian_intensity.png ({len(fig_a_keys)}×3 panels)")

    # ── Fig B: Gaussian segmentation (6 samples × 2 cols) ───────────────
    # Cols: GT mask (red) | TP/FP/FN
    for row_idx, key in enumerate(fig_a_keys):
        info = repr_g[key]
        sid = info["sample_id"]
        gt_vals = gt_dict_g.get(sid, np.zeros(len(mesh_nodes)))
        pred_vals = pred_dict_g.get(sid, np.zeros(len(mesh_nodes)))
        params = get_tumor_params(args.samples_g, sid)

        # Col 1: GT mask (intensity mode but showing GT only)
        title1 = f"GT | {key} | Dice={info['dice_03']:.3f}"
        out1 = tmpdir / f"figB_{row_idx}_gt.png"
        render_single_sample(
            nodes=mesh_nodes, elements=elements,
            surface_faces=surface_faces, tissue_labels=tissue_labels,
            gt_nodes=gt_vals, pred_nodes=np.zeros_like(pred_vals),
            tumor_params=params, title=title1, save_path=str(out1),
            mode="intensity", source_type="gaussian",
            gt_threshold=0.05, pred_threshold=0.3,
            camera_position="oblique",
        )

        # Col 2: TP/FP/FN
        title2 = f"TP/FP/FN | {key}"
        out2 = tmpdir / f"figB_{row_idx}_seg.png"
        render_single_sample(
            nodes=mesh_nodes, elements=elements,
            surface_faces=surface_faces, tissue_labels=tissue_labels,
            gt_nodes=gt_vals, pred_nodes=pred_vals,
            tumor_params=params, title=title2, save_path=str(out2),
            mode="segmentation", source_type="gaussian",
            gt_threshold=0.05, pred_threshold=0.3,
            camera_position="oblique",
        )

    fig_b_out = out_dir / "fig_gaussian_seg.png"
    stitch_images(
        [str(tmpdir / f"figB_{r}_{c}.png")
         for r in range(len(fig_a_keys)) for c in ["gt", "seg"]],
        grid_shape=(len(fig_a_keys), 2),
        output_path=str(fig_b_out),
        cell_size=(1200, 900),
    )
    print(f"  Saved fig_gaussian_seg.png ({len(fig_a_keys)}×2 panels)")

    # ── Fig C: Uniform segmentation (6 samples × 2 cols) ───────────────
    fig_c_keys = list(repr_u.keys())

    for row_idx, key in enumerate(fig_c_keys):
        info = repr_u[key]
        sid = info["sample_id"]
        gt_vals = gt_dict_u.get(sid, np.zeros(len(mesh_nodes)))
        pred_vals = pred_dict_u.get(sid, np.zeros(len(mesh_nodes)))
        params = get_tumor_params(args.samples_u, sid)
        dice_05 = compute_dice(pred_vals, gt_vals)

        # Col 1: GT mask (uniform, binary)
        title1 = f"GT (U) | {key} | Dice={dice_05:.3f}"
        out1 = tmpdir / f"figC_{row_idx}_gt.png"
        render_single_sample(
            nodes=mesh_nodes, elements=elements,
            surface_faces=surface_faces, tissue_labels=tissue_labels,
            gt_nodes=gt_vals, pred_nodes=np.zeros_like(pred_vals),
            tumor_params=params, title=title1, save_path=str(out1),
            mode="intensity", source_type="uniform",
            gt_threshold=0.5, pred_threshold=0.5,
            camera_position="oblique",
        )

        # Col 2: TP/FP/FN
        title2 = f"TP/FP/FN (U) | {key}"
        out2 = tmpdir / f"figC_{row_idx}_seg.png"
        render_single_sample(
            nodes=mesh_nodes, elements=elements,
            surface_faces=surface_faces, tissue_labels=tissue_labels,
            gt_nodes=gt_vals, pred_nodes=pred_vals,
            tumor_params=params, title=title2, save_path=str(out2),
            mode="segmentation", source_type="uniform",
            gt_threshold=0.5, pred_threshold=0.5,
            camera_position="oblique",
        )

    fig_c_out = out_dir / "fig_uniform_seg.png"
    stitch_images(
        [str(tmpdir / f"figC_{r}_{c}.png")
         for r in range(len(fig_c_keys)) for c in ["gt", "seg"]],
        grid_shape=(len(fig_c_keys), 2),
        output_path=str(fig_c_out),
        cell_size=(1200, 900),
    )
    print(f"  Saved fig_uniform_seg.png ({len(fig_c_keys)}×2 panels)")

    # ── Fig D: Gaussian vs Uniform comparison (3 samples × 4 cols) ─────
    comparison_keys = ["1-foci-shallow", "2-foci-medium", "3-foci-deep"]

    for row_idx, key in enumerate(comparison_keys):
        info_g = repr_g.get(key, {})
        info_u = repr_u.get(key, {})
        sid_g = info_g.get("sample_id", "")
        sid_u = info_u.get("sample_id", "")

        gt_g = gt_dict_g.get(sid_g, np.zeros(len(mesh_nodes)))
        pred_g = pred_dict_g.get(sid_g, np.zeros(len(mesh_nodes)))
        gt_u = gt_dict_u.get(sid_u, np.zeros(len(mesh_nodes)))
        pred_u = pred_dict_u.get(sid_u, np.zeros(len(mesh_nodes)))

        params_g = get_tumor_params(args.samples_g, sid_g)
        params_u = get_tumor_params(args.samples_u, sid_u)

        dice_g = info_g.get("dice_03", 0.0)
        dice_u = info_u.get("dice_03", 0.0)

        # Col 0: GT Gaussian (intensity)
        out0 = tmpdir / f"figD_{row_idx}_gtG.png"
        render_single_sample(
            nodes=mesh_nodes, elements=elements,
            surface_faces=surface_faces, tissue_labels=tissue_labels,
            gt_nodes=gt_g, pred_nodes=np.zeros_like(pred_g),
            tumor_params=params_g,
            title=f"GT-G | {key} | Dice={dice_g:.3f}",
            save_path=str(out0),
            mode="intensity", source_type="gaussian",
            gt_threshold=0.05, pred_threshold=0.3,
            camera_position="oblique",
        )

        # Col 1: Pred Gaussian (intensity)
        out1 = tmpdir / f"figD_{row_idx}_predG.png"
        render_single_sample(
            nodes=mesh_nodes, elements=elements,
            surface_faces=surface_faces, tissue_labels=tissue_labels,
            gt_nodes=np.zeros_like(gt_g), pred_nodes=pred_g,
            tumor_params=params_g,
            title=f"Pred-G | {key}",
            save_path=str(out1),
            mode="intensity", source_type="gaussian",
            gt_threshold=0.05, pred_threshold=0.3,
            camera_position="oblique",
        )

        # Col 2: GT Uniform (binary red)
        out2 = tmpdir / f"figD_{row_idx}_gtU.png"
        render_single_sample(
            nodes=mesh_nodes, elements=elements,
            surface_faces=surface_faces, tissue_labels=tissue_labels,
            gt_nodes=gt_u, pred_nodes=np.zeros_like(pred_u),
            tumor_params=params_u,
            title=f"GT-U | {key} | Dice={dice_u:.3f}",
            save_path=str(out2),
            mode="intensity", source_type="uniform",
            gt_threshold=0.5, pred_threshold=0.5,
            camera_position="oblique",
        )

        # Col 3: Pred Uniform (TP/FP/FN)
        out3 = tmpdir / f"figD_{row_idx}_segU.png"
        render_single_sample(
            nodes=mesh_nodes, elements=elements,
            surface_faces=surface_faces, tissue_labels=tissue_labels,
            gt_nodes=gt_u, pred_nodes=pred_u,
            tumor_params=params_u,
            title=f"TP/FP/FN-U | {key}",
            save_path=str(out3),
            mode="segmentation", source_type="uniform",
            gt_threshold=0.5, pred_threshold=0.5,
            camera_position="oblique",
        )

    fig_d_out = out_dir / "fig_source_comparison.png"
    stitch_images(
        [str(tmpdir / f"figD_{r}_{c}.png")
         for r in range(len(comparison_keys)) for c in ["gtG", "predG", "gtU", "segU"]],
        grid_shape=(len(comparison_keys), 4),
        output_path=str(fig_d_out),
        cell_size=(1200, 900),
    )
    print(f"  Saved fig_source_comparison.png ({len(comparison_keys)}×4 panels)")

    # Cleanup temp dir
    shutil.rmtree(tmpdir)

    print("\n" + "=" * 70)
    print(f"ALL OUTPUTS SAVED TO: {out_dir}/")
    print("=" * 70)
    print("Figures:")
    print("  - fig_gaussian_intensity.png  (6×3 tetrahedral intensity panels)")
    print("  - fig_gaussian_seg.png         (6×2 TP/FP/FN segmentation panels)")
    print("  - fig_uniform_seg.png          (6×2 TP/FP/FN segmentation panels)")
    print("  - fig_source_comparison.png    (3×4 source comparison panels)")
    print("  - representative_samples.json")


if __name__ == "__main__":
    main()
