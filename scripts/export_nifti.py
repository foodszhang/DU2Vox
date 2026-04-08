#!/usr/bin/env python3
"""
Stage 1 NIfTI export for ITK-SNAP visualization + real depth recomputation.

Exports:
  - FEM node values interpolated to atlas voxel space (NIfTI format)
  - Per-sample overlays: GT, Pred, TP/FP/FN segmentation
  - Real subcutaneous depth (distance to nearest surface node)
  - Depth-corrected evaluation tables and scatter plots

Usage:
    python scripts/export_nifti.py \
        --checkpoint_g runs/gcain_gaussian_1000/checkpoints/best.pth \
        --checkpoint_u runs/gcain_uniform_1000/checkpoints/best.pth \
        --config_g configs/stage1/gaussian_1000.yaml \
        --config_u configs/stage1/uniform_1000.yaml \
        --shared_dir /home/foods/pro/FMT-SimGen/output/shared \
        --atlas_path /home/foods/pro/mcx_simulation/ct_data/atlas_380x992x208.hdr \
        --samples_g /home/foods/pro/FMT-SimGen/data/gaussian_1000/samples \
        --samples_u /home/foods/pro/FMT-SimGen/data/uniform_1000/samples \
        --output_dir results/stage1/nifti/
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.sparse
import torch
import yaml
from scipy.interpolate import LinearNDInterpolator
from torch.utils.data import DataLoader

from du2vox.data.dataset import FMTSimGenDataset
from du2vox.models.stage1.gcain import GCAIN_full


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate transforms
# ─────────────────────────────────────────────────────────────────────────────

def phys_to_voxel_3D(phys_coord, voxel_size, offset):
    """Convert physical coordinate to voxel index."""
    return (phys_coord - offset) / voxel_size


def build_atlas_to_mesh_mapping(nodes, atlas_shape, voxel_size):
    """Build a lookup from atlas voxel centers to mesh nodes.

    Uses simple coordinate system: voxel_index * voxel_size = physical coord.
    This matches the mesh coordinate system (FMT-SimGen mesh uses
    voxel_index * voxel_size, NOT nibabel affine transforms).

    Returns arrays ix, iy, iz (voxel indices within atlas) and
    query_pts (corresponding physical coordinates in mesh space).
    """
    # Simple coordinate: voxel (i,j,k) -> physical (i*vx, j*vx, k*vx)
    gx = np.arange(atlas_shape[0]) * voxel_size + voxel_size / 2
    gy = np.arange(atlas_shape[1]) * voxel_size + voxel_size / 2
    gz = np.arange(atlas_shape[2]) * voxel_size + voxel_size / 2

    # Find mesh bounding box
    bbox_min = nodes.min(axis=0)
    bbox_max = nodes.max(axis=0)

    # Find overlapping voxel indices
    ix = np.where((gx >= bbox_min[0] - 1) & (gx <= bbox_max[0] + 1))[0]
    iy = np.where((gy >= bbox_min[1] - 1) & (gy <= bbox_max[1] + 1))[0]
    iz = np.where((gz >= bbox_min[2] - 1) & (gz <= bbox_max[2] + 1))[0]

    if len(ix) == 0 or len(iy) == 0 or len(iz) == 0:
        return None

    sub_x = gx[ix]
    sub_y = gy[iy]
    sub_z = gz[iz]
    sub_x_mesh, sub_y_mesh, sub_z_mesh = np.meshgrid(sub_x, sub_y, sub_z, indexing="ij")
    query_pts = np.column_stack([sub_x_mesh.ravel(), sub_y_mesh.ravel(), sub_z_mesh.ravel()])

    return ix, iy, iz, query_pts


# ─────────────────────────────────────────────────────────────────────────────
# FEM → Voxel interpolation
# ─────────────────────────────────────────────────────────────────────────────

def fem_to_voxel_volume(node_values, nodes, atlas_shape, voxel_size,
                         atlas_to_mesh_map):
    """Interpolate FEM node values onto atlas voxel grid.

    Returns volume of shape atlas_shape with values at each voxel center.
    """
    if atlas_to_mesh_map is None:
        return np.zeros(atlas_shape, dtype=np.float32)

    ix, iy, iz, query_pts = atlas_to_mesh_map

    # Build interpolator from mesh nodes to values
    interp = LinearNDInterpolator(nodes, node_values, fill_value=0.0)
    sub_values = interp(query_pts).reshape(len(ix), len(iy), len(iz))
    sub_values = np.clip(sub_values, 0.0, None).astype(np.float32)

    volume = np.zeros(atlas_shape, dtype=np.float32)
    volume[np.ix_(ix, iy, iz)] = sub_values
    return volume


def save_nifti(volume, voxel_size, path, atlas_affine=None):
    """Save volume as NIfTI with consistent voxel_size affine."""
    affine = np.diag([voxel_size, voxel_size, voxel_size, 1.0])
    img = nib.Nifti1Image(volume.astype(np.float32), affine)
    nib.save(img, str(path))


# ─────────────────────────────────────────────────────────────────────────────
# Model & data loading
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg, checkpoint_path, device="cuda"):
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
    L0, L1, L2, L3 = dataset.L0.to(device), dataset.L1.to(device), \
                      dataset.L2.to(device), dataset.L3.to(device)
    knn_idx = dataset.knn_idx.to(device)
    sens_w = dataset.sens_w.to(device)
    nodes = dataset.nodes.to(device)

    model = GCAIN_full(
        L=L, A=A, L0=L0, L1=L1, L2=L2, L3=L3,
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


# ─────────────────────────────────────────────────────────────────────────────
# Depth computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_real_depth(tumor_params, nodes, surface_node_indices):
    """Compute real subcutaneous depth: min distance from tumor center to surface."""
    surface_nodes = nodes[surface_node_indices]

    foci = tumor_params.get("foci", [])
    if not foci:
        return None

    center = np.array(foci[0]["center"])
    dists = np.linalg.norm(surface_nodes - center, axis=1)
    return float(dists.min())


def assign_depth_tier(depth_mm):
    """Assign depth tier based on real subcutaneous depth."""
    if depth_mm is None:
        return "unknown"
    if depth_mm < 3.5:
        return "shallow"
    elif depth_mm < 6.0:
        return "medium"
    else:
        return "deep"


# ─────────────────────────────────────────────────────────────────────────────
# Export per sample
# ─────────────────────────────────────────────────────────────────────────────

def export_sample(
    sample_id,
    nodes, surface_node_indices,
    atlas_volume, atlas_shape, voxel_size, atlas_to_mesh_map,
    gt_nodes_g, pred_nodes_g,
    gt_nodes_u, pred_nodes_u,
    output_dir,
):
    """Export 7 NIfTI files for one sample.

    GT values come from gt_voxels.npy (precomputed, already in correct grid).
    Pred values are interpolated from FEM nodes to atlas grid.
    """
    out = Path(output_dir) / sample_id
    out.mkdir(parents=True, exist_ok=True)

    # 1. Anatomy (atlas volume)
    save_nifti(atlas_volume.astype(np.float32), voxel_size, out / "anatomy.nii.gz")

    # 2. Gaussian GT — FEM interpolation to atlas grid
    gt_g_vol = fem_to_voxel_volume(gt_nodes_g, nodes, atlas_shape, voxel_size,
                                   atlas_to_mesh_map)
    save_nifti(gt_g_vol, voxel_size, out / "gt_gaussian.nii.gz")

    # 3. Gaussian Pred — FEM interpolation to atlas grid
    pred_g_vol = fem_to_voxel_volume(pred_nodes_g, nodes, atlas_shape, voxel_size,
                                      atlas_to_mesh_map)
    save_nifti(pred_g_vol, voxel_size, out / "pred_gaussian.nii.gz")

    # 4. Uniform GT — FEM interpolation to atlas grid
    gt_u_vol = fem_to_voxel_volume(gt_nodes_u, nodes, atlas_shape, voxel_size,
                                    atlas_to_mesh_map)
    save_nifti(gt_u_vol, voxel_size, out / "gt_uniform.nii.gz")

    # 5. Uniform Pred — FEM interpolation to atlas grid
    pred_u_vol = fem_to_voxel_volume(pred_nodes_u, nodes, atlas_shape, voxel_size,
                                      atlas_to_mesh_map)
    save_nifti(pred_u_vol, voxel_size, out / "pred_uniform.nii.gz")

    # 6. Gaussian TP/FP/FN overlay (label: 1=TP, 2=FP, 3=FN)
    gt_g_mask = (gt_g_vol > 0.05).astype(np.uint8)
    pred_g_mask = (pred_g_vol > 0.3).astype(np.uint8)
    overlay_g = np.zeros(atlas_shape, dtype=np.uint8)
    overlay_g[(gt_g_mask == 1) & (pred_g_mask == 1)] = 1   # TP
    overlay_g[(gt_g_mask == 0) & (pred_g_mask == 1)] = 2   # FP
    overlay_g[(gt_g_mask == 1) & (pred_g_mask == 0)] = 3   # FN
    save_nifti(overlay_g.astype(np.float32), voxel_size,
               out / "overlay_gaussian.nii.gz")

    # 7. Uniform TP/FP/FN overlay
    gt_u_mask = (gt_u_vol > 0.5).astype(np.uint8)
    pred_u_mask = (pred_u_vol > 0.5).astype(np.uint8)
    overlay_u = np.zeros(atlas_shape, dtype=np.uint8)
    overlay_u[(gt_u_mask == 1) & (pred_u_mask == 1)] = 1
    overlay_u[(gt_u_mask == 0) & (pred_u_mask == 1)] = 2
    overlay_u[(gt_u_mask == 1) & (pred_u_mask == 0)] = 3
    save_nifti(overlay_u.astype(np.float32), voxel_size,
               out / "overlay_uniform.nii.gz")

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Tables
# ─────────────────────────────────────────────────────────────────────────────

def compute_dice(pred, gt, pred_th=0.3, gt_th=0.05):
    p_bin = (pred > pred_th).astype(float)
    g_bin = (gt > gt_th).astype(float)
    tp = (p_bin * g_bin).sum()
    return 2 * tp / (p_bin.sum() + g_bin.sum() + 1e-8)


METRIC_COLS = ["Dice@0.5", "Dice@0.3", "Dice@0.1", "Recall@0.1", "Recall@0.3", "Precision@0.3", "LocErr", "MSE"]
METRIC_KEYS = ["dice_bin_0.5", "dice_bin_0.3", "dice_bin_0.1", "recall_0.1", "recall_0.3", "precision_0.3", "location_error", "mse"]


def table_depth_corrected(df_g, df_u):
    """Table with corrected real-depth grouping."""
    rows = []
    for name, df in [("Gaussian", df_g), ("Uniform", df_u)]:
        for depth in ["shallow", "medium", "deep"]:
            sub = df[df["real_depth_tier"] == depth]
            if len(sub) == 0:
                continue
            row = {"Source": name, "Depth": depth.title()}
            for col, key in zip(METRIC_COLS, METRIC_KEYS):
                row[col] = sub[key].mean()
            row["N"] = len(sub)
            rows.append(row)
    return pd.DataFrame(rows)


def save_csv_latex(df, base_path):
    csv_path = f"{base_path}.csv"
    tex_path = f"{base_path}.tex"
    df.to_csv(csv_path, index=False, float_format="%.4f")

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


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_g", required=True)
    parser.add_argument("--checkpoint_u", required=True)
    parser.add_argument("--config_g", required=True)
    parser.add_argument("--config_u", required=True)
    parser.add_argument("--shared_dir", required=True)
    parser.add_argument("--atlas_path", required=True)
    parser.add_argument("--samples_g", required=True)
    parser.add_argument("--samples_u", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STAGE 1 NIFTI EXPORT + DEPTH RECOMPUTATION")
    print("=" * 70)

    # Load configs
    with open(args.config_g) as f:
        cfg_g = yaml.safe_load(f)
    with open(args.config_u) as f:
        cfg_u = yaml.safe_load(f)

    device = "cuda"
    voxel_size = 0.1

    # Load atlas
    print("\n[1/6] Loading atlas...")
    atlas_img = nib.load(args.atlas_path)
    atlas_data = atlas_img.get_fdata()
    if atlas_data.ndim == 4:
        atlas_data = atlas_data.squeeze()
    atlas_shape = atlas_data.shape
    print(f"  Atlas shape: {atlas_shape}, dtype: {atlas_data.dtype}")

    # Load mesh
    print("[2/6] Loading mesh...")
    mesh_data = np.load(Path(args.shared_dir) / "mesh.npz", allow_pickle=True)
    mesh_nodes = mesh_data["nodes"]
    elements = mesh_data["elements"]
    surface_node_indices = mesh_data["surface_node_indices"]
    print(f"  nodes: {mesh_nodes.shape}, elements: {elements.shape}")
    print(f"  surface nodes: {surface_node_indices.shape}")

    # Build atlas → mesh mapping using simple voxel_size coordinate system
    print("[3/6] Building atlas-to-mesh coordinate mapping...")
    atlas_to_mesh_map = build_atlas_to_mesh_mapping(mesh_nodes, atlas_shape, voxel_size)
    if atlas_to_mesh_map is not None:
        ix, iy, iz, query_pts = atlas_to_mesh_map
        print(f"  Overlapping voxels: {len(ix)} × {len(iy)} × {len(iz)} = {len(ix)*len(iy)*len(iz):,} voxels")
    else:
        print("  WARNING: No overlapping voxels found between mesh and atlas!")
    if atlas_to_mesh_map is not None:
        ix, iy, iz, query_pts = atlas_to_mesh_map
        print(f"  Overlapping voxels: {len(ix)} × {len(iy)} × {len(iz)} = {len(ix)*len(iy)*len(iz):,} voxels")
    else:
        print("  WARNING: No overlapping voxels found between mesh and atlas!")

    # Build models and run inference
    print("[4/6] Running inference...")
    model_g, dataset_g, nodes_g = build_model(cfg_g, args.checkpoint_g, device)
    model_u, dataset_u, nodes_u = build_model(cfg_u, args.checkpoint_u, device)

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

    print(f"  Gaussian: {len(gt_dict_g)} samples")
    print(f"  Uniform:  {len(gt_dict_u)} samples")

    # ── Recompute real depth for all samples ─────────────────────────────
    print("\n[5/6] Recomputing real subcutaneous depth...")

    def recompute_depths(sample_ids, samples_dir, gt_dict, pred_dict, source_type):
        rows = []
        for sid in sample_ids:
            params_path = Path(samples_dir) / sid / "tumor_params.json"
            if params_path.exists():
                with open(params_path) as f:
                    params = json.load(f)
            else:
                params = {}

            real_depth = compute_real_depth(params, mesh_nodes, surface_node_indices)
            real_tier = assign_depth_tier(real_depth)

            pred = pred_dict.get(sid, np.zeros(len(mesh_nodes)))
            gt = gt_dict.get(sid, np.zeros(len(mesh_nodes)))
            dice_05 = compute_dice(pred, gt, pred_th=0.5, gt_th=0.05)
            dice_03 = compute_dice(pred, gt, pred_th=0.3, gt_th=0.05)
            dice_01 = compute_dice(pred, gt, pred_th=0.1, gt_th=0.05)

            # Also load tumor_params for foci count
            num_foci = params.get("num_foci", 0)
            foci = params.get("foci", [])
            z_vals = [f.get("center", [None, None, 10])[2] for f in foci if f.get("center")]
            avg_z = sum(z_vals) / len(z_vals) if z_vals else None
            z_tier = "unknown"
            if avg_z is not None:
                if avg_z < 7.5:
                    z_tier = "shallow"
                elif avg_z < 13.5:
                    z_tier = "medium"
                else:
                    z_tier = "deep"

            rows.append({
                "sample_id": sid,
                "source": source_type,
                "num_foci": num_foci,
                "z_based_tier": z_tier,
                "real_depth_mm": real_depth,
                "real_depth_tier": real_tier,
                "dice_bin_0.5": dice_05,
                "dice_bin_0.3": dice_03,
                "dice_bin_0.1": dice_01,
                "recall_0.1": float(np.nan),
                "recall_0.3": float(np.nan),
                "precision_0.3": float(np.nan),
                "location_error": float(np.nan),
                "mse": float(np.nan),
            })
        return pd.DataFrame(rows)

    df_g = recompute_depths(sample_ids_g, args.samples_g, gt_dict_g, pred_dict_g, "Gaussian")
    df_u = recompute_depths(sample_ids_u, args.samples_u, gt_dict_u, pred_dict_u, "Uniform")

    print(f"  Gaussian depth range: {df_g['real_depth_mm'].min():.2f} - {df_g['real_depth_mm'].max():.2f} mm")
    print(f"  Uniform depth range:  {df_u['real_depth_mm'].min():.2f} - {df_u['real_depth_mm'].max():.2f} mm")

    # Save corrected CSVs
    df_g.to_csv(out_dir.parent / "metrics_per_sample_gaussian_corrected.csv",
                index=False, float_format="%.6f")
    df_u.to_csv(out_dir.parent / "metrics_per_sample_uniform_corrected.csv",
                index=False, float_format="%.6f")

    # Compute depth-corrected table
    t3_corr = table_depth_corrected(df_g, df_u)
    save_csv_latex(t3_corr, out_dir.parent / "table3_by_depth_corrected")

    print("\n  Table 3 (corrected depth grouping):")
    print(t3_corr.to_string(index=False))

    # ── Select representative samples ───────────────────────────────────────
    repr_path = Path("results/stage1/representative_samples.json")
    if repr_path.exists():
        with open(repr_path) as f:
            repr_data = json.load(f)
        repr_g = repr_data.get("gaussian", {})
        repr_u = repr_data.get("uniform", {})
        print(f"\n  Using existing representative samples from {repr_path}")
    else:
        # Select from existing metrics with new depth tiers
        groups_g = [
            (1, "shallow"), (1, "deep"),
            (2, "medium"), (2, "deep"),
            (3, "shallow"), (3, "medium"),
        ]
        repr_g = {}
        for foci, depth in groups_g:
            sub = df_g[(df_g["num_foci"] == foci) & (df_g["real_depth_tier"] == depth)]
            if len(sub) == 0:
                sub = df_g[df_g["num_foci"] == foci]
            if len(sub) == 0:
                continue
            median_dice = sub["dice_bin_0.3"].median()
            idx = (sub["dice_bin_0.3"] - median_dice).abs().idxmin()
            repr_g[f"{foci}-foci-{depth}"] = {
                "sample_id": sub.loc[idx, "sample_id"],
                "dice_03": sub.loc[idx, "dice_bin_0.3"],
                "num_foci": foci,
                "real_depth_tier": depth,
            }

        groups_u = [
            (1, "shallow"), (1, "deep"),
            (2, "medium"), (2, "deep"),
            (3, "shallow"), (3, "medium"),
        ]
        repr_u = {}
        for foci, depth in groups_u:
            sub = df_u[(df_u["num_foci"] == foci) & (df_u["real_depth_tier"] == depth)]
            if len(sub) == 0:
                sub = df_u[df_u["num_foci"] == foci]
            if len(sub) == 0:
                continue
            median_dice = sub["dice_bin_0.3"].median()
            idx = (sub["dice_bin_0.3"] - median_dice).abs().idxmin()
            repr_u[f"{foci}-foci-{depth}"] = {
                "sample_id": sub.loc[idx, "sample_id"],
                "dice_03": sub.loc[idx, "dice_bin_0.3"],
                "num_foci": foci,
                "real_depth_tier": depth,
            }

    # Save representative samples
    with open(out_dir / "representative_samples.json", "w") as f:
        json.dump({"gaussian": repr_g, "uniform": repr_u}, f, indent=2)

    print(f"\n  Gaussian representative: {list(repr_g.keys())}")
    print(f"  Uniform representative:  {list(repr_u.keys())}")

    # ── Export NIfTI for representative samples ─────────────────────────────
    print("\n[6/6] Exporting NIfTI files for representative samples...")

    nifti_samples = set()
    for v in repr_g.values():
        nifti_samples.add(v["sample_id"])
    for v in repr_u.values():
        nifti_samples.add(v["sample_id"])

    for sid in nifti_samples:
        # FEM node values for GT and Pred (both in same mesh coordinate system)
        gt_g = gt_dict_g.get(sid, np.zeros(len(mesh_nodes)))
        pred_g = pred_dict_g.get(sid, np.zeros(len(mesh_nodes)))
        gt_u = gt_dict_u.get(sid, np.zeros(len(mesh_nodes)))
        pred_u = pred_dict_u.get(sid, np.zeros(len(mesh_nodes)))

        export_sample(
            sid,
            mesh_nodes, surface_node_indices,
            atlas_data, atlas_shape, voxel_size, atlas_to_mesh_map,
            gt_g, pred_g, gt_u, pred_u,
            out_dir,
        )
        print(f"  Exported {sid}")

    # ── Depth vs Dice scatter plot ─────────────────────────────────────────
    print("\n[7/6] Generating depth analysis plot...")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 300,
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Dice@0.3 vs real depth
    ax = axes[0]
    ax.scatter(df_g["real_depth_mm"], df_g["dice_bin_0.3"],
               alpha=0.5, s=20, c="steelblue", label="Gaussian")
    ax.scatter(df_u["real_depth_mm"], df_u["dice_bin_0.3"],
               alpha=0.5, s=20, c="darkorange", label="Uniform")
    # Trend lines
    x_range = [min(df_g["real_depth_mm"].min(), df_u["real_depth_mm"].min()),
               max(df_g["real_depth_mm"].max(), df_u["real_depth_mm"].max())]
    x_line = np.linspace(x_range[0], x_range[1], 100)
    # Gaussian trend
    mask_g = ~(df_g["real_depth_mm"].isna() | df_g["dice_bin_0.3"].isna())
    if mask_g.sum() > 2:
        z_g = np.polyfit(df_g.loc[mask_g, "real_depth_mm"], df_g.loc[mask_g, "dice_bin_0.3"], 1)
        ax.plot(x_line, np.polyval(z_g, x_line), "steelblue", lw=2, ls="--", alpha=0.7)
    # Uniform trend
    mask_u = ~(df_u["real_depth_mm"].isna() | df_u["dice_bin_0.3"].isna())
    if mask_u.sum() > 2:
        z_u = np.polyfit(df_u.loc[mask_u, "real_depth_mm"], df_u.loc[mask_u, "dice_bin_0.3"], 1)
        ax.plot(x_line, np.polyval(z_u, x_line), "darkorange", lw=2, ls="--", alpha=0.7)
    ax.set_xlabel("Surface-to-Tumor Depth (mm)")
    ax.set_ylabel("Dice@0.3")
    ax.legend()
    ax.set_title("(a) Dice@0.3 vs Real Depth")
    ax.set_ylim([0, 1.05])

    # (b) Dice@0.5 vs real depth
    ax = axes[1]
    ax.scatter(df_g["real_depth_mm"], df_g["dice_bin_0.5"],
               alpha=0.5, s=20, c="steelblue", label="Gaussian")
    ax.scatter(df_u["real_depth_mm"], df_u["dice_bin_0.5"],
               alpha=0.5, s=20, c="darkorange", label="Uniform")
    x_range = [min(df_g["real_depth_mm"].min(), df_u["real_depth_mm"].min()),
               max(df_g["real_depth_mm"].max(), df_u["real_depth_mm"].max())]
    x_line = np.linspace(x_range[0], x_range[1], 100)
    mask_g = ~(df_g["real_depth_mm"].isna() | df_g["dice_bin_0.5"].isna())
    if mask_g.sum() > 2:
        z_g = np.polyfit(df_g.loc[mask_g, "real_depth_mm"], df_g.loc[mask_g, "dice_bin_0.5"], 1)
        ax.plot(x_line, np.polyval(z_g, x_line), "steelblue", lw=2, ls="--", alpha=0.7)
    mask_u = ~(df_u["real_depth_mm"].isna() | df_u["dice_bin_0.5"].isna())
    if mask_u.sum() > 2:
        z_u = np.polyfit(df_u.loc[mask_u, "real_depth_mm"], df_u.loc[mask_u, "dice_bin_0.5"], 1)
        ax.plot(x_line, np.polyval(z_u, x_line), "darkorange", lw=2, ls="--", alpha=0.7)
    ax.set_xlabel("Surface-to-Tumor Depth (mm)")
    ax.set_ylabel("Dice@0.5")
    ax.legend()
    ax.set_title("(b) Dice@0.5 vs Real Depth")
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    fig.savefig(out_dir.parent / "fig_depth_analysis.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)

    print(f"\n" + "=" * 70)
    print(f"ALL OUTPUTS SAVED TO: {out_dir.parent}/")
    print("=" * 70)
    print(f"NIfTI dir: {out_dir}/ ({len(nifti_samples)} samples × 7 files)")
    print(f"Corrected CSVs: metrics_per_sample_*_corrected.csv")
    print(f"Table: table3_by_depth_corrected.csv + .tex")
    print(f"Figure: fig_depth_analysis.png")


if __name__ == "__main__":
    main()
