#!/usr/bin/env python3
"""
Tecplot-style 3D visualization: GT vs Prediction side-by-side.

Features:
  - Mouse body as gray wireframe silhouette
  - GT (left) and Pred (right) as smooth isosurfaces with jet colormap
  - Shared colorbar range [0, 1] for direct comparison
  - Dark gradient background, clean Tecplot aesthetic
  - Multi-angle PNG output (no HTML)
  - Per-foci-type sample selection and annotation

Usage:
  python scripts/visualize.py --n_samples 4
  python scripts/visualize.py --n_samples 1 --interactive
  python scripts/visualize.py --n_samples 6 --foci_balance
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv
import torch
import yaml
from torch.utils.data import DataLoader

from du2vox.models.stage1.gcain import GCAIN_full
from du2vox.data.dataset import FMTSimGenDataset


# ── Tecplot-style 全局设置 ──
pv.global_theme.font.family = "arial"
pv.global_theme.font.size = 14
pv.global_theme.font.label_size = 12


def load_model_and_data(cfg):
    """Load trained model and validation data."""
    model_cfg = cfg["model"]
    paths_cfg = cfg["paths"]
    shared_dir = Path(paths_cfg["shared_dir"])
    samples_dir = Path(paths_cfg["samples_dir"])
    splits_dir = Path(paths_cfg["splits_dir"])

    val_set = FMTSimGenDataset(
        shared_dir=shared_dir,
        samples_dir=samples_dir,
        split_file=splits_dir / "val.txt",
    )

    A = val_set.A.cuda()
    L = val_set.L.cuda()
    L0, L1, L2, L3 = (
        val_set.L0.cuda(), val_set.L1.cuda(),
        val_set.L2.cuda(), val_set.L3.cuda(),
    )
    knn_idx = val_set.knn_idx.cuda()
    sens_w = val_set.sens_w.cuda()

    model = GCAIN_full(
        L=L, A=A, L0=L0, L1=L1, L2=L2, L3=L3,
        knn_idx=knn_idx, sens_w=sens_w,
        num_layer=model_cfg["num_layer"],
        feat_dim=model_cfg["feat_dim"],
    ).cuda()

    ckpt = torch.load(
        paths_cfg.get("checkpoint", "checkpoints/best.pth"), map_location="cuda"
    )
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        epoch = ckpt.get("epoch", "?")
    else:
        model.load_state_dict(ckpt)
        epoch = "?"
    model.eval()
    print(f"Loaded best.pth (epoch {epoch})")

    # Load manifest for foci info
    manifest = None
    manifest_path = Path("output/dataset_manifest.json")
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"Loaded manifest with {len(manifest['samples'])} samples")

    return model, val_set, manifest


def build_tet_mesh(nodes, elements, scalars=None, name="value"):
    """Build PyVista UnstructuredGrid from tet mesh."""
    n_tets = elements.shape[0]
    cells = np.hstack([
        np.full((n_tets, 1), 4, dtype=elements.dtype),
        elements,
    ]).ravel()
    celltypes = np.full(n_tets, pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(
        cells, celltypes, nodes.astype(np.float64)
    )
    if scalars is not None:
        grid.point_data[name] = scalars
    return grid


def get_body_wireframe(nodes, elements, tissue_labels):
    """提取小鼠整体外轮廓（skin + muscle），返回 wireframe surface."""
    # 用所有非背景单元
    elem_labels = np.zeros(elements.shape[0], dtype=int)
    for i, tet in enumerate(elements):
        vals, counts = np.unique(
            tissue_labels[tet], return_counts=True
        )
        elem_labels[i] = vals[counts.argmax()]

    mask = elem_labels > 0  # 所有非背景
    body_grid = build_tet_mesh(nodes, elements[mask])
    body_surf = body_grid.extract_surface()
    # 轻度平滑让轮廓更自然
    body_surf = body_surf.smooth(n_iter=30, relaxation_factor=0.1)
    return body_surf


def get_organ_surfaces(nodes, elements, tissue_labels, organ_ids):
    """提取指定器官的外表面."""
    elem_labels = np.zeros(elements.shape[0], dtype=int)
    for i, tet in enumerate(elements):
        vals, counts = np.unique(
            tissue_labels[tet], return_counts=True
        )
        elem_labels[i] = vals[counts.argmax()]

    surfaces = {}
    for oid in organ_ids:
        mask = elem_labels == oid
        if mask.sum() < 10:
            continue
        sub = build_tet_mesh(nodes, elements[mask])
        surf = sub.extract_surface().smooth(
            n_iter=20, relaxation_factor=0.1
        )
        if surf.n_points > 0:
            surfaces[oid] = surf
    return surfaces


def add_scene_to_subplot(
    plotter, nodes, elements, tissue_labels,
    values, title, body_surf, organ_surfs,
    clim=(0.0, 1.0),
    iso_levels=None,
):
    """在一个 subplot 中渲染：小鼠轮廓 + 器官 + 等值面."""

    # ── 小鼠外轮廓：灰色半透明 wireframe ──
    plotter.add_mesh(
        body_surf,
        color=(0.75, 0.75, 0.75),
        style="wireframe",
        line_width=0.3,
        opacity=0.08,
    )

    # ── 关键器官：极淡半透明 surface ──
    ORGAN_STYLE = {
        # id: (color, opacity)
        3: ((0.6, 0.8, 1.0), 0.08),   # lung - 淡蓝
        4: ((0.9, 0.3, 0.3), 0.12),   # heart - 淡红
        5: ((0.55, 0.25, 0.15), 0.10), # liver - 深棕
        6: ((0.7, 0.4, 0.3), 0.10),   # kidney - 棕
    }
    for oid, surf in organ_surfs.items():
        style = ORGAN_STYLE.get(oid, ((0.5, 0.5, 0.5), 0.06))
        plotter.add_mesh(
            surf, color=style[0], opacity=style[1],
            smooth_shading=True,
        )

    # ── 值等值面（Tecplot 风格：jet colormap）──
    if iso_levels is None:
        iso_levels = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

    grid = build_tet_mesh(nodes, elements, values, "intensity")

    # 只取有效等值
    vmin, vmax = values.min(), values.max()
    valid_levels = [lv for lv in iso_levels if vmin < lv < vmax]

    if len(valid_levels) > 0:
        contours = grid.contour(valid_levels, scalars="intensity")
        if contours.n_points > 0:
            plotter.add_mesh(
                contours,
                scalars="intensity",
                cmap="jet",
                clim=clim,
                opacity=0.8,
                smooth_shading=True,
                scalar_bar_args={
                    "title": "Fluorescence Intensity",
                    "title_font_size": 12,
                    "label_font_size": 10,
                    "width": 0.4,
                    "height": 0.06,
                    "position_x": 0.3,
                    "position_y": 0.02,
                    "fmt": "%.1f",
                },
            )
    else:
        # 如果没有等值面（值太低），用散点显示
        active = values > 0.05
        if active.sum() > 0:
            pts = pv.PolyData(nodes[active])
            pts["intensity"] = values[active]
            plotter.add_mesh(
                pts,
                scalars="intensity",
                cmap="jet",
                clim=clim,
                point_size=5,
                render_points_as_spheres=True,
                scalar_bar_args={
                    "title": "Intensity",
                    "width": 0.4,
                    "height": 0.06,
                    "position_x": 0.3,
                    "position_y": 0.02,
                },
            )

    # ── 标题 ──
    plotter.add_text(
        title,
        position="upper_edge",
        font_size=14,
        color="white",
        shadow=True,
    )

    # ── Tecplot 风格深色背景 ──
    plotter.set_background(
        color=(0.12, 0.12, 0.18),  # 深蓝灰
        top=(0.22, 0.22, 0.30),    # 顶部稍亮 → 渐变
    )


def compute_metrics(gt_np, pred_np, gt_thr=0.05, pred_thr=0.3):
    gt_bin = (gt_np > gt_thr).astype(float)
    pred_bin = (pred_np > pred_thr).astype(float)
    tp = (pred_bin * gt_bin).sum()
    fp = (pred_bin * (1 - gt_bin)).sum()
    fn = ((1 - pred_bin) * gt_bin).sum()
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    return dice, recall, precision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/stage1/gcain_full.yaml")
    parser.add_argument(
        "--n_samples", type=int, default=4,
        help="Number of val samples to visualize",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Open interactive window (needs display)",
    )
    parser.add_argument(
        "--foci_balance", action="store_true",
        help="Select balanced samples from each foci type (1/2/3-foci)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best.pth",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Override checkpoint path if provided
    if args.checkpoint:
        cfg.setdefault("paths", {})["checkpoint"] = args.checkpoint

    model, val_set, manifest = load_model_and_data(cfg)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    nodes_np = val_set.nodes.numpy()
    n_nodes = nodes_np.shape[0]

    mesh_data = np.load(
        Path(cfg["paths"]["shared_dir"]) / "mesh.npz"
    )
    elements = mesh_data["elements"]
    tissue_labels = mesh_data["tissue_labels"]

    with open(Path(cfg["paths"]["splits_dir"]) / "val.txt") as f:
        val_names = [l.strip() for l in f if l.strip()]

    out_dir = Path("vis_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 预计算共享几何（只算一次）──
    print("Extracting body wireframe and organ surfaces...")
    body_surf = get_body_wireframe(nodes_np, elements, tissue_labels)
    organ_surfs = get_organ_surfaces(
        nodes_np, elements, tissue_labels,
        organ_ids=[3, 4, 5, 6],  # lung, heart, liver, kidney
    )

    # 相机预设（名称, camera_position）
    camera_angles = [
        ("side",  "xz"),
        ("front", "yz"),
        ("top",   "xy"),
        ("iso",   [(80, 120, 60), (19, 50, 10), (0, 0, 1)]),
    ]

    # Build index mapping for balanced foci selection
    if args.foci_balance and manifest:
        foci_indices = {1: [], 2: [], 3: []}
        for idx, name in enumerate(val_names):
            if name in manifest["samples"]:
                n_foci = manifest["samples"][name]["num_foci"]
                if n_foci in foci_indices:
                    foci_indices[n_foci].append(idx)
        # Take up to n_samples // 3 from each foci type
        per_foci = max(1, args.n_samples // 3)
        balanced_indices = []
        for n in [1, 2, 3]:
            balanced_indices.extend(foci_indices[n][:per_foci])
        balanced_indices = sorted(balanced_indices)[:args.n_samples]
        print(f"\nBalanced foci selection: {per_foci} samples each from 1/2/3-foci types")
    else:
        balanced_indices = list(range(min(args.n_samples, len(val_names))))

    print(f"\nVisualizing {len(balanced_indices)} val samples...\n")

    # Build index mapping for balanced foci selection
    needed_indices = set(balanced_indices)
    cached_results = {}  # idx -> (gt_np, pred_np, dice, recall, prec)

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            if idx not in needed_indices:
                continue

            b = batch["b"].cuda()
            gt = batch["gt"].cuda()
            X0 = torch.zeros(1, n_nodes, 1, device="cuda")
            pred = model(X0, b)
            pred = torch.clamp(pred, min=0.0, max=1.0)

            gt_np = gt.squeeze().cpu().numpy()
            pred_np = pred.squeeze().cpu().numpy()
            dice, recall, prec = compute_metrics(gt_np, pred_np)
            cached_results[idx] = (gt_np, pred_np, dice, recall, prec)

            if len(cached_results) == len(needed_indices):
                break

    # Render cached results
    for count_idx, idx in enumerate(balanced_indices):
        gt_np, pred_np, dice, recall, prec = cached_results[idx]
        sample_name = val_names[idx]

        # Get foci count for title
        foci_label = ""
        n_foci = None
        if manifest and sample_name in manifest["samples"]:
            n_foci = manifest["samples"][sample_name]["num_foci"]
            foci_label = f" [{n_foci}-Foci]"

        print(
            f"{sample_name}{foci_label}: "
            f"Dice={dice:.3f} Recall={recall:.3f} "
            f"Prec={prec:.3f} "
            f"GT>0.1={( gt_np > 0.1).sum()} "
            f"Pred>0.3={(pred_np > 0.3).sum()} "
            f"pred_max={pred_np.max():.3f}"
        )

        # ── 对每个角度生成一张 GT vs Pred 对比图 ──
        for angle_name, cam_pos in camera_angles:
            off = not args.interactive
            pv.OFF_SCREEN = off

            # 1x2 subplot: left=GT, right=Pred
            pl = pv.Plotter(
                shape=(1, 2),
                off_screen=off,
                window_size=[2400, 1000],
                border=False,
            )

            # ── Left: Ground Truth ──
            pl.subplot(0, 0)
            add_scene_to_subplot(
                pl, nodes_np, elements, tissue_labels,
                gt_np,
                title=f"{sample_name}{foci_label} | GT",
                body_surf=body_surf,
                organ_surfs=organ_surfs,
            )
            pl.camera_position = cam_pos

            # ── Right: Prediction ──
            pl.subplot(0, 1)
            add_scene_to_subplot(
                pl, nodes_np, elements, tissue_labels,
                pred_np,
                title=f"{sample_name}{foci_label} | Pred | Dice={dice:.3f}",
                body_surf=body_surf,
                organ_surfs=organ_surfs,
            )
            pl.camera_position = cam_pos

            # ── 同步相机（让两边完全一致）──
            pl.subplot(0, 0)
            cam = pl.camera_position
            pl.subplot(0, 1)
            pl.camera_position = cam

            if args.interactive:
                # 交互模式只看第一个角度
                pl.show()
                break
            else:
                foci_suffix = f"_{n_foci}f" if manifest and sample_name in manifest["samples"] else ""
                out_path = (
                    out_dir
                    / f"{sample_name}{foci_suffix}_{angle_name}.png"
                )
                pl.screenshot(
                    str(out_path),
                    transparent_background=False,
                )
                pl.close()

    # ── 打印汇总 ──
    print(f"\n{'='*60}")
    print(f"完成！输出目录: {out_dir}/")
    print(f"每个样本 × 4 角度 = {args.n_samples * 4} 张对比图")
    print(f"文件格式: <sample>_<side|front|top|iso>.png")
    print(f"  左半 = Ground Truth (GT)")
    print(f"  右半 = Prediction (Pred)")
    print(f"  colormap = jet [0, 1]")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
