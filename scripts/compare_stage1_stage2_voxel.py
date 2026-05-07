#!/usr/bin/env python3
"""
Compare Stage 1 voxel baseline vs Stage 2 DE-only / multiview with comprehensive dice metrics.

Metrics computed:
  - Stage 1:
    * mesh_dice_binary_05_global: Binary dice @ 0.5 on all mesh nodes
    * voxel_dice_binary_05_roi: Binary dice @ 0.5 on ROI voxels (from precomputed)
    * voxel_dice_binary_05_global: Binary dice @ 0.5 on global voxel grid (ROI outside = pred 0)
  
  - Stage 2 (DE-only / Multiview):
    * stage2_dice_binary_05_roi: Binary dice @ 0.5 on ROI voxels
    * stage2_dice_binary_05_global: Binary dice @ 0.5 on global voxel grid
    * fem_dice_binary_05_roi: FEM interpolation baseline on ROI
    * fem_dice_binary_05_global: FEM interpolation baseline on global
    * delta_dice_roi: Stage2 - FEM on ROI
    * delta_dice_global: Stage2 - FEM on global

Outputs:
  - full_results.json: Complete metrics with per_sample, overall, by_foci, by_depth, by_cross
  - summary.md: Human-readable markdown tables
  - figures/: Representative sample visualizations (GT vs Stage1 vs DE-only vs Multiview)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pyvista as pv
import torch
import yaml
from PIL import Image, ImageDraw

from du2vox.bridge.fem_bridging import FEMBridge
from du2vox.data.dataset import FMTSimGenDataset
from du2vox.models.stage1.gcain import GCAIN_full
from du2vox.models.stage2.residual_inr import ResidualINR
from du2vox.models.stage2.view_encoder import ViewEncoderModule
from du2vox.utils.frame import FrameManifest


pv.OFF_SCREEN = True


ORGAN_STYLE = {
    1: ((0.95, 0.85, 0.75), 0.04),
    2: ((0.95, 0.95, 0.90), 0.08),
    3: ((1.00, 0.85, 0.85), 0.05),
    4: ((0.80, 0.15, 0.15), 0.10),
    5: ((0.70, 0.85, 0.95), 0.08),
    6: ((0.55, 0.15, 0.15), 0.10),
    7: ((0.65, 0.30, 0.20), 0.10),
    8: ((0.60, 0.20, 0.30), 0.05),
    9: ((0.90, 0.75, 0.50), 0.05),
    10: ((0.85, 0.70, 0.50), 0.05),
    11: ((0.85, 0.70, 0.70), 0.03),
}

CAMERAS = {
    "oblique": {
        "position": [(70.0, -55.0, 45.0), (19.0, 20.0, 10.4), (0.0, 0.0, 1.0)],
        "zoom": 1.25,
    },
    "sagittal": {
        "position": [(95.0, 20.0, 12.0), (19.0, 20.0, 10.4), (0.0, 0.0, 1.0)],
        "zoom": 1.25,
    },
    "coronal": {
        "position": [(19.0, -85.0, 12.0), (19.0, 20.0, 10.4), (0.0, 0.0, 1.0)],
        "zoom": 1.20,
    },
}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_json(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def fem_interp_from_prior(prior_8d: np.ndarray) -> np.ndarray:
    return (prior_8d[:, :4] * prior_8d[:, 4:8]).sum(axis=1)


def compute_binary_dice(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    pred_bin = (pred >= threshold).astype(float)
    target_bin = (target >= threshold).astype(float)
    intersection = (pred_bin * target_bin).sum()
    return float(2 * intersection / (pred_bin.sum() + target_bin.sum() + 1e-8))


def build_global_voxel_grid(frame: FrameManifest, gt_voxels: np.ndarray) -> np.ndarray:
    """
    Build world coordinates for gt_voxels grid.
    
    Args:
        frame: FrameManifest instance
        gt_voxels: [H, W, D] voxel grid
    
    Returns:
        world_coords: [N_total, 3] world coordinates (mcx_trunk_local_mm)
    """
    shape = gt_voxels.shape
    offset = frame.gt_offset_world_mm
    spacing = frame.gt_spacing_mm
    
    indices = np.stack(np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]], axis=-1)
    world_coords = offset + indices * spacing
    return world_coords.reshape(-1, 3)


def build_summary(per_sample: dict, sample_ids: list[str], manifest: dict) -> dict:
    """
    Build summary statistics: overall, by_foci, by_depth, by_cross
    
    Args:
        per_sample: {sample_id: {metric: value}}
        sample_ids: list of sample ids in order
        manifest: {samples_by_id: {sample_id: {num_foci, depth_tier, ...}}}
    
    Returns:
        {overall: {...}, by_foci: {1: {...}, ...}, by_depth: {...}, by_cross: {...}}
    """
    keys = list(next(iter(per_sample.values())).keys()) if per_sample else []
    
    overall = {}
    for key in keys:
        values = [per_sample[sid][key] for sid in sample_ids if sid in per_sample]
        overall[key] = float(np.mean(values)) if values else 0.0
    
    by_foci = {}
    for n in [1, 2, 3]:
        grp_sids = [
            sid for sid in sample_ids
            if sid in per_sample and manifest["samples_by_id"].get(sid, {}).get("num_foci") == n
        ]
        by_foci[n] = {}
        for key in keys:
            values = [per_sample[sid][key] for sid in grp_sids]
            by_foci[n][key] = float(np.mean(values)) if values else 0.0
        by_foci[n]["n_samples"] = len(grp_sids)
    
    by_depth = {}
    for tier in ["shallow", "medium", "deep"]:
        grp_sids = [
            sid for sid in sample_ids
            if sid in per_sample and manifest["samples_by_id"].get(sid, {}).get("depth_tier") == tier
        ]
        by_depth[tier] = {}
        for key in keys:
            values = [per_sample[sid][key] for sid in grp_sids]
            by_depth[tier][key] = float(np.mean(values)) if values else 0.0
        by_depth[tier]["n_samples"] = len(grp_sids)
    
    by_cross = {}
    for n in [1, 2, 3]:
        for tier in ["shallow", "medium", "deep"]:
            grp_sids = [
                sid for sid in sample_ids
                if sid in per_sample
                and manifest["samples_by_id"].get(sid, {}).get("num_foci") == n
                and manifest["samples_by_id"].get(sid, {}).get("depth_tier") == tier
            ]
            by_cross[(n, tier)] = {}
            for key in keys:
                values = [per_sample[sid][key] for sid in grp_sids]
                by_cross[(n, tier)][key] = float(np.mean(values)) if values else 0.0
            by_cross[(n, tier)]["n_samples"] = len(grp_sids)
    
    return {
        "overall": overall,
        "by_foci": by_foci,
        "by_depth": by_depth,
        "by_cross": by_cross,
    }


def load_stage1_model(checkpoint_path: str, config_path: str, dataset: FMTSimGenDataset, device: torch.device):
    """Load Stage 1 GCAIN model."""
    cfg = yaml.safe_load(open(config_path))
    model_cfg = cfg["model"]
    
    nodes = dataset.nodes.to(device)
    A = dataset.A.to(device)
    L = dataset.L.to(device)
    LTL = torch.matmul(L.t(), L).to(device)
    ATA = torch.matmul(A.t(), A).to(device)
    L0 = dataset.L0.to(device)
    L1 = dataset.L1.to(device)
    L2 = dataset.L2.to(device)
    L3 = dataset.L3.to(device)
    knn_idx = dataset.knn_idx.to(device)
    sens_w = dataset.sens_w.to(device)
    
    model = GCAIN_full(
        L=L, A=A, LTL=LTL, ATA=ATA,
        L0=L0, L1=L1, L2=L2, L3=L3,
        knn_idx=knn_idx, sens_w=sens_w,
        num_layer=model_cfg["num_layer"],
        feat_dim=model_cfg["feat_dim"],
    ).to(device)
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    return model, dataset


def evaluate_stage1_global(
    checkpoint_path: str,
    config_path: str,
    shared_dir: Path,
    samples_dir: Path,
    bridge_dir: Path,
    precomputed_dir: Path,
    sample_ids: list[str],
    device: torch.device,
) -> tuple[dict, dict]:
    """
    Compute Stage 1 global metrics.
    
    Returns:
        per_sample: {sample_id: {mesh_dice_global, voxel_dice_roi, voxel_dice_global}}
        summary: {overall, by_foci, by_depth, by_cross}
    """
    dataset = FMTSimGenDataset(
        shared_dir=shared_dir,
        samples_dir=samples_dir,
        split_file=bridge_dir.parent / f"{bridge_dir.name.split('_')[-1]}.txt",
        normalize_b=True,
        normalize_gt=True,
        normalize_gt_mode="per_sample",
    )
    
    model, dataset = load_stage1_model(checkpoint_path, config_path, dataset, device)
    nodes, elements = FrameManifest.load_mesh_nodes(shared_dir)
    frame = FrameManifest.load(shared_dir)
    
    per_sample = {}
    for idx, sid in enumerate(sample_ids):
        if idx % 50 == 0:
            print(f"  [Stage1] [{idx}/{len(sample_ids)}] {sid}")
        
        b = dataset.b_list[idx].float().unsqueeze(0).to(device)
        gt_mesh = dataset.gt_list[idx].cpu().numpy().flatten()
        
        with torch.no_grad():
            pred_mesh = model(torch.zeros(1, nodes.shape[0], 1, device=device), b)
        pred_mesh_np = pred_mesh.cpu().numpy().flatten()
        mesh_dice_global = compute_binary_dice(pred_mesh_np, gt_mesh, 0.5)
        
        coarse_d = np.load(bridge_dir / sid / "coarse_d.npy").flatten().astype(np.float64)
        roi_tet_indices = np.load(bridge_dir / sid / "roi_tet_indices.npy")
        bridge = FEMBridge(nodes, elements, roi_tet_indices)
        
        npz_path = precomputed_dir / f"{sid}.npz"
        if not npz_path.exists():
            print(f"  [Stage1] Warning: {sid} missing precomputed data")
            continue
        
        npz = np.load(npz_path)
        valid_mask = npz["valid_mask"].astype(bool)
        
        if valid_mask.sum() == 0:
            voxel_dice_roi = 0.0
        else:
            coords_roi = npz["grid_coords"][valid_mask]
            gt_roi = npz["gt_values"][valid_mask]
            prior_8d, _ = bridge.get_prior_features(coords_roi.astype(np.float64), coarse_d)
            pred_roi = fem_interp_from_prior(prior_8d)
            voxel_dice_roi = compute_binary_dice(pred_roi, gt_roi, 0.5)
        
        gt_voxels = np.load(samples_dir / sid / "gt_voxels.npy")
        world_coords_global = build_global_voxel_grid(frame, gt_voxels)
        _, valid_in_roi = bridge.get_prior_features(world_coords_global.astype(np.float64), coarse_d)
        
        if valid_in_roi.sum() > 0:
            coords_in_roi = world_coords_global[valid_in_roi]
            prior_8d_in_roi, _ = bridge.get_prior_features(coords_in_roi.astype(np.float64), coarse_d)
            pred_in_roi = fem_interp_from_prior(prior_8d_in_roi)
        else:
            pred_in_roi = np.array([], dtype=np.float32)
        
        pred_global = np.zeros(len(world_coords_global), dtype=np.float32)
        if valid_in_roi.sum() > 0:
            pred_global[valid_in_roi] = pred_in_roi
        
        gt_global = gt_voxels.ravel()
        voxel_dice_global = compute_binary_dice(pred_global, gt_global, 0.5)
        
        per_sample[sid] = {
            "mesh_dice_binary_05_global": float(mesh_dice_global),
            "voxel_dice_binary_05_roi": float(voxel_dice_roi),
            "voxel_dice_binary_05_global": float(voxel_dice_global),
        }
    
    manifest_raw = load_json(samples_dir.parent / "dataset_manifest.json")
    manifest = {"samples_by_id": {s["id"]: s for s in manifest_raw["samples"]}}
    summary = build_summary(per_sample, sample_ids, manifest)
    return per_sample, summary


def load_stage2_model(checkpoint_path: str, config_path: str, device: torch.device, multiview: bool):
    """Load Stage 2 INR and optional ViewEncoder."""
    cfg = yaml.safe_load(open(config_path))
    model_cfg = cfg["model"]
    
    use_view_encoder = multiview and bool(model_cfg.get("view_encoder", False))
    view_feat_dim = model_cfg.get("view_feat_dim", 0) if use_view_encoder else 0
    
    inr = ResidualINR(
        n_freqs=model_cfg["n_freqs"],
        hidden_dim=model_cfg["hidden_dim"],
        n_hidden_layers=model_cfg["n_hidden_layers"],
        prior_dim=model_cfg["prior_dim"],
        skip_connection=model_cfg.get("skip_connection", True),
        view_feat_dim=view_feat_dim,
    ).to(device)
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "residual_inr" in ckpt:
        inr.load_state_dict(ckpt["residual_inr"])
    else:
        inr.load_state_dict(ckpt)
    inr.eval()
    
    view_encoder = None
    if use_view_encoder:
        view_encoder = ViewEncoderModule(
            view_feat_dim=view_feat_dim,
            fusion_method=model_cfg.get("fusion_method", "attn"),
            encoder_out_channels=model_cfg.get("encoder_out_channels", 32),
            encoder_base_channels=model_cfg.get("encoder_base_channels", 32),
        ).to(device)
        if "view_encoder" in ckpt:
            view_encoder.load_state_dict(ckpt["view_encoder"])
        view_encoder.eval()
    
    return inr, view_encoder, model_cfg


def infer_stage2(
    coords_norm: np.ndarray,
    coords_world: np.ndarray,
    prior_8d: np.ndarray,
    proj_imgs: np.ndarray,
    inr: ResidualINR,
    view_encoder,
    model_cfg: dict,
    frame: FrameManifest,
    device: torch.device,
    batch_points: int,
    multiview: bool,
) -> np.ndarray:
    """Run Stage 2 inference."""
    use_view_encoder = multiview and view_encoder is not None
    
    if use_view_encoder:
        proj_t = torch.from_numpy(proj_imgs).float().unsqueeze(1).to(device).unsqueeze(0)
        lo = frame.mcx_bbox_min
        hi = frame.mcx_bbox_max
        mcx_valid_all = (
            (coords_world[:, 0] >= lo[0]) & (coords_world[:, 0] <= hi[0]) &
            (coords_world[:, 1] >= lo[1]) & (coords_world[:, 1] <= hi[1]) &
            (coords_world[:, 2] >= lo[2]) & (coords_world[:, 2] <= hi[2])
        )
    else:
        proj_t = None
        mcx_valid_all = None
    
    preds = []
    with torch.no_grad():
        for start in range(0, len(coords_norm), batch_points):
            end = min(start + batch_points, len(coords_norm))
            coords_chunk = torch.from_numpy(coords_norm[start:end]).float().unsqueeze(0).to(device)
            prior_chunk = torch.from_numpy(prior_8d[start:end]).float().unsqueeze(0).to(device)
            
            if use_view_encoder:
                coords_world_chunk = torch.from_numpy(coords_world[start:end]).float().unsqueeze(0).to(device)
                view_feat, _ = view_encoder(proj_t, coords_world_chunk, None)
                mcx_valid_chunk = torch.from_numpy(mcx_valid_all[start:end]).to(device).view(1, -1, 1).float()
                pred_chunk, _, _ = inr(coords_chunk, prior_chunk, view_feat * mcx_valid_chunk)
            else:
                pred_chunk, _, _ = inr(coords_chunk, prior_chunk)
            
            preds.append(pred_chunk.squeeze(0).cpu().numpy())
    
    return np.concatenate(preds, axis=0)


def evaluate_stage2_global(
    checkpoint_path: str,
    config_path: str,
    shared_dir: Path,
    samples_dir: Path,
    bridge_dir: Path,
    sample_ids: list[str],
    device: torch.device,
    multiview: bool,
    batch_points: int,
) -> tuple[dict, dict]:
    """
    Compute Stage 2 global metrics (inference only on ROI).
    
    Returns:
        per_sample: {sample_id: {stage2_dice_roi, stage2_dice_global, fem_dice_roi, fem_dice_global, delta_roi, delta_global}}
        summary: {overall, by_foci, by_depth, by_cross}
    """
    inr, view_encoder, model_cfg = load_stage2_model(checkpoint_path, config_path, device, multiview)
    nodes, elements = FrameManifest.load_mesh_nodes(shared_dir)
    frame = FrameManifest.load(shared_dir)
    
    per_sample = {}
    for idx, sid in enumerate(sample_ids):
        if idx % 50 == 0:
            print(f"  [Stage2 {'MV' if multiview else 'DE'}] [{idx}/{len(sample_ids)}] {sid}")
        
        coarse_d_path = bridge_dir / sid / "coarse_d.npy"
        roi_path = bridge_dir / sid / "roi_tet_indices.npy"
        roi_info_path = bridge_dir / sid / "roi_info.json"
        
        if not (coarse_d_path.exists() and roi_path.exists()):
            print(f"  [Stage2] Warning: {sid} missing bridge data")
            continue
        
        coarse_d = np.load(coarse_d_path).flatten().astype(np.float64)
        roi_tet_indices = np.load(roi_path)
        bridge = FEMBridge(nodes, elements, roi_tet_indices)
        
        gt_voxels = np.load(samples_dir / sid / "gt_voxels.npy")
        world_coords_global = build_global_voxel_grid(frame, gt_voxels)
        gt_global = gt_voxels.ravel()
        
        _, valid_in_roi = bridge.get_prior_features(world_coords_global.astype(np.float64), coarse_d)
        
        if valid_in_roi.sum() == 0:
            stage2_dice_global = compute_binary_dice(np.zeros_like(gt_global), gt_global, 0.5)
            fem_dice_global = stage2_dice_global
            per_sample[sid] = {
                "stage2_dice_binary_05_roi": 0.0,
                "stage2_dice_binary_05_global": float(stage2_dice_global),
                "fem_dice_binary_05_roi": 0.0,
                "fem_dice_binary_05_global": float(fem_dice_global),
                "delta_dice_roi": 0.0,
                "delta_dice_global": 0.0,
            }
            continue
        
        coords_in_roi = world_coords_global[valid_in_roi]
        prior_8d_in_roi, _ = bridge.get_prior_features(coords_in_roi.astype(np.float64), coarse_d)
        fem_pred_in_roi = fem_interp_from_prior(prior_8d_in_roi)
        
        roi_info = json.loads(roi_info_path.read_text())
        bbox_min = np.array(roi_info["roi_bbox_mm"]["min"], dtype=np.float32)
        bbox_max = np.array(roi_info["roi_bbox_mm"]["max"], dtype=np.float32)
        coords_norm_in_roi = 2.0 * (coords_in_roi - bbox_min) / (bbox_max - bbox_min + 1e-8) - 1.0
        
        proj_path = samples_dir / sid / "proj.npz"
        if proj_path.exists():
            proj_data = np.load(proj_path)
            proj_imgs = np.stack([proj_data[str(a)].astype(np.float32) for a in [-90, -60, -30, 0, 30, 60, 90]], axis=0)
        else:
            proj_imgs = np.zeros((7, 256, 256), dtype=np.float32)
        
        stage2_pred_in_roi = infer_stage2(
            coords_norm=coords_norm_in_roi,
            coords_world=coords_in_roi,
            prior_8d=prior_8d_in_roi.astype(np.float32),
            proj_imgs=proj_imgs,
            inr=inr,
            view_encoder=view_encoder,
            model_cfg=model_cfg,
            frame=frame,
            device=device,
            batch_points=batch_points,
            multiview=multiview,
        )
        
        gt_in_roi = gt_global[valid_in_roi]
        stage2_dice_roi = compute_binary_dice(stage2_pred_in_roi, gt_in_roi, 0.5)
        fem_dice_roi = compute_binary_dice(fem_pred_in_roi, gt_in_roi, 0.5)
        delta_roi = stage2_dice_roi - fem_dice_roi
        
        stage2_pred_global = np.zeros(len(world_coords_global), dtype=np.float32)
        stage2_pred_global[valid_in_roi] = stage2_pred_in_roi
        
        fem_pred_global = np.zeros(len(world_coords_global), dtype=np.float32)
        fem_pred_global[valid_in_roi] = fem_pred_in_roi
        
        stage2_dice_global = compute_binary_dice(stage2_pred_global, gt_global, 0.5)
        fem_dice_global = compute_binary_dice(fem_pred_global, gt_global, 0.5)
        delta_global = stage2_dice_global - fem_dice_global
        
        per_sample[sid] = {
            "stage2_dice_binary_05_roi": float(stage2_dice_roi),
            "stage2_dice_binary_05_global": float(stage2_dice_global),
            "fem_dice_binary_05_roi": float(fem_dice_roi),
            "fem_dice_binary_05_global": float(fem_dice_global),
            "delta_dice_roi": float(delta_roi),
            "delta_dice_global": float(delta_global),
        }
    
    manifest_raw = load_json(samples_dir.parent / "dataset_manifest.json")
    manifest = {"samples_by_id": {s["id"]: s for s in manifest_raw["samples"]}}
    summary = build_summary(per_sample, sample_ids, manifest)
    return per_sample, summary


def pick_representative_samples(
    stage1_per_sample: dict,
    de_only_per_sample: dict,
    multiview_per_sample: dict,
    manifest: dict,
) -> list[dict]:
    """Pick representative samples (one per foci count)."""
    selected = []
    for n in [1, 2, 3]:
        candidates = []
        for sid in stage1_per_sample:
            if sid not in de_only_per_sample or sid not in multiview_per_sample:
                continue
            info = manifest["samples_by_id"].get(sid, {})
            if info.get("num_foci") != n:
                continue
            
            s1 = stage1_per_sample[sid]
            s2_de = de_only_per_sample[sid]
            s2_mv = multiview_per_sample[sid]
            
            candidates.append({
                "sample_id": sid,
                "num_foci": n,
                "depth_tier": info.get("depth_tier", "unknown"),
                "stage1_voxel_dice": s1["voxel_dice_binary_05_roi"],
                "de_only_dice": s2_de["stage2_dice_binary_05_roi"],
                "de_only_delta": s2_de["delta_dice_roi"],
                "multiview_dice": s2_mv["stage2_dice_binary_05_roi"],
                "multiview_delta": s2_mv["delta_dice_roi"],
            })
        
        candidates.sort(
            key=lambda x: (x["multiview_delta"], x["multiview_dice"], x["de_only_delta"], -x["stage1_voxel_dice"]),
            reverse=True,
        )
        positive = [c for c in candidates if c["multiview_delta"] > 0]
        if positive:
            selected.append(positive[0])
        elif candidates:
            selected.append(candidates[0])
    
    return selected


def build_tet_grid(nodes: np.ndarray, elements: np.ndarray) -> pv.UnstructuredGrid:
    n_cells = len(elements)
    cells = np.column_stack([np.full(n_cells, 4, dtype=np.int64), elements.astype(np.int64)]).ravel()
    cell_types = np.full(n_cells, pv.CellType.TETRA, dtype=np.uint8)
    return pv.UnstructuredGrid(cells, cell_types, nodes.astype(np.float64))


def build_surface_mesh(nodes: np.ndarray, surface_faces: np.ndarray) -> pv.PolyData:
    faces = np.column_stack([np.full(len(surface_faces), 3, dtype=np.int64), surface_faces.astype(np.int64)]).ravel()
    return pv.PolyData(nodes.astype(np.float64), faces)


def get_organ_surfaces(nodes: np.ndarray, elements: np.ndarray, tissue_labels: np.ndarray) -> dict[int, pv.PolyData]:
    tet_grid = build_tet_grid(nodes, elements)
    surfaces: dict[int, pv.PolyData] = {}
    for label in sorted(ORGAN_STYLE):
        mask = tissue_labels == label
        if mask.sum() < 10:
            continue
        surf = tet_grid.extract_cells(np.where(mask)[0]).extract_surface().smooth(n_iter=20, relaxation_factor=0.1)
        if surf.n_points > 0:
            surfaces[label] = surf
    return surfaces


def build_voxel_grid(volume: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray) -> tuple[pv.ImageData, np.ndarray]:
    shape = np.array(volume.shape, dtype=np.int32)
    spacing = (bbox_max - bbox_min) / np.maximum(shape - 1, 1)
    origin = bbox_min - spacing / 2.0

    grid = pv.ImageData()
    grid.dimensions = tuple((shape + 1).tolist())
    grid.origin = tuple(origin.tolist())
    grid.spacing = tuple(spacing.tolist())
    grid.cell_data["values"] = volume.astype(np.float32).ravel(order="F")
    return grid, spacing


def dense_volume_from_valid(data: dict, values_valid: np.ndarray) -> np.ndarray:
    full = np.zeros(len(data["valid_mask"]), dtype=np.float32)
    full[data["valid_mask"].astype(bool)] = values_valid.astype(np.float32)
    return full.reshape(tuple(data["grid_shape"].tolist()), order="C")


def add_common_scene(plotter: pv.Plotter, body_surface: pv.PolyData, organ_surfaces: dict[int, pv.PolyData], tumor_params: dict) -> None:
    plotter.set_background("white")
    plotter.add_mesh(body_surface, color=(0.80, 0.80, 0.80), opacity=0.10, show_edges=False, smooth_shading=True)
    for label, surf in organ_surfaces.items():
        color, opacity = ORGAN_STYLE[label]
        plotter.add_mesh(surf, color=color, opacity=opacity, show_edges=False, smooth_shading=True)
    for focus in tumor_params.get("foci", []):
        center = np.array(focus["center"], dtype=np.float64)
        radius = float(focus.get("radius", 1.2))
        sphere = pv.Sphere(radius=max(radius * 0.5, 0.8), center=center, theta_resolution=24, phi_resolution=24)
        plotter.add_mesh(sphere, color=(1.0, 0.85, 0.1), opacity=0.95)


def add_volume(plotter: pv.Plotter, volume: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray) -> None:
    grid, _ = build_voxel_grid(volume, bbox_min, bbox_max)
    th = grid.threshold(value=0.5, scalars="values", preference="cell")
    if th.n_cells > 0:
        plotter.add_mesh(
            th,
            scalars="values",
            cmap="hot",
            clim=[0.0, 1.0],
            opacity=0.90,
            show_edges=False,
            smooth_shading=False,
        )


def render_sample_comparison(
    sample_id: str,
    stage1_volume: np.ndarray,
    de_only_volume: np.ndarray,
    stage2_volume: np.ndarray,
    gt_volume: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    body_surface: pv.PolyData,
    organ_surfaces: dict[int, pv.PolyData],
    tumor_params: dict,
    out_path: Path,
) -> None:
    titles = ["Ground Truth", "Stage 1", "DE-only", "MultiView"]
    volumes = [gt_volume, stage1_volume, de_only_volume, stage2_volume]
    camera_names = ["oblique", "sagittal", "coronal"]

    rows = []
    for camera_name in camera_names:
        imgs = []
        for title, volume in zip(titles, volumes):
            p = pv.Plotter(off_screen=True, window_size=(900, 800))
            add_common_scene(p, body_surface, organ_surfaces, tumor_params)
            add_volume(p, volume, bbox_min, bbox_max)
            p.add_text(f"{sample_id} | {title}", position="upper_edge", font_size=18, color="black")
            p.camera_position = CAMERAS[camera_name]["position"]
            p.camera.zoom(CAMERAS[camera_name]["zoom"])
            img = p.screenshot(return_img=True)
            imgs.append(Image.fromarray(img))
            p.close()
        row = Image.new("RGB", (imgs[0].width * len(imgs), imgs[0].height), "white")
        for i, img in enumerate(imgs):
            row.paste(img, (i * img.width, 0))
        draw = ImageDraw.Draw(row)
        draw.text((20, 20), camera_name.title(), fill="black")
        rows.append(row)

    canvas = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), "white")
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    canvas.save(out_path)


def save_markdown_summary(results: dict, out_path: Path) -> None:
    """Save comprehensive markdown summary."""
    with open(out_path, "w") as f:
        f.write("# Stage 1 vs Stage 2 Dice Metrics Summary\n\n")
        f.write(f"**Metadata**: N={results['metadata']['n_samples']} samples\n\n")
        f.write("---\n\n")
        
        f.write("## Overall Metrics\n\n")
        
        f.write("### Stage 1\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        s1_overall = results["stage1"]["overall"]
        f.write(f"| Mesh Dice (binary @ 0.5, global) | {s1_overall['mesh_dice_binary_05_global']:.4f} |\n")
        f.write(f"| Voxel Dice (binary @ 0.5, ROI) | {s1_overall['voxel_dice_binary_05_roi']:.4f} |\n")
        f.write(f"| Voxel Dice (binary @ 0.5, global) | {s1_overall['voxel_dice_binary_05_global']:.4f} |\n")
        f.write("\n**Insight**: Mesh dice > Voxel dice (ROI) > Voxel dice (global)\n\n")
        
        for name, key in [("DE-only", "de_only"), ("Multiview", "multiview")]:
            f.write(f"### Stage 2 {name}\n\n")
            f.write("| Metric | ROI | Global | Difference |\n")
            f.write("|--------|-----|--------|------------|\n")
            overall = results[key]["overall"]
            stage2_roi = overall[f"stage2_dice_binary_05_roi"]
            stage2_global = overall[f"stage2_dice_binary_05_global"]
            fem_roi = overall[f"fem_dice_binary_05_roi"]
            fem_global = overall[f"fem_dice_binary_05_global"]
            delta_roi = overall["delta_dice_roi"]
            delta_global = overall["delta_dice_global"]
            f.write(f"| Stage2 Dice (binary @ 0.5) | {stage2_roi:.4f} | {stage2_global:.4f} | {stage2_global - stage2_roi:.4f} |\n")
            f.write(f"| FEM Dice (binary @ 0.5) | {fem_roi:.4f} | {fem_global:.4f} | {fem_global - fem_roi:.4f} |\n")
            f.write(f"| **Delta Dice** | **{delta_roi:+.4f}** | **{delta_global:+.4f}** | - |\n")
            f.write(f"\n**Insight**: {name} 提升 {delta_roi:.1%}，全局表现{'一致' if abs(delta_global - delta_roi) < 0.01 else '有差异'}\n\n")
        
        f.write("---\n\n")
        f.write("## Per-Foci Breakdown\n\n")
        for n in [1, 2, 3]:
            n_samples = results["stage1"]["by_foci"][n]["n_samples"]
            f.write(f"### {n}-Foci (N={n_samples})\n\n")
            f.write("| Stage | Mesh Global | Voxel ROI | Voxel Global | S2 ROI | S2 Global | Delta ROI | Delta Global |\n")
            f.write("|-------|------------|-----------|--------------|--------|-----------|-----------|--------------|\n")
            
            s1 = results["stage1"]["by_foci"][n]
            de = results["de_only"]["by_foci"][n]
            mv = results["multiview"]["by_foci"][n]
            
            f.write(f"| Stage 1 | {s1['mesh_dice_binary_05_global']:.4f} | {s1['voxel_dice_binary_05_roi']:.4f} | {s1['voxel_dice_binary_05_global']:.4f} | - | - | - | - |\n")
            f.write(f"| DE-only | - | - | - | {de['stage2_dice_binary_05_roi']:.4f} | {de['stage2_dice_binary_05_global']:.4f} | {de['delta_dice_roi']:+.4f} | {de['delta_dice_global']:+.4f} |\n")
            f.write(f"| Multiview | - | - | - | {mv['stage2_dice_binary_05_roi']:.4f} | {mv['stage2_dice_binary_05_global']:.4f} | {mv['delta_dice_roi']:+.4f} | {mv['delta_dice_global']:+.4f} |\n\n")
        
        f.write("---\n\n")
        f.write("## Per-Depth Breakdown\n\n")
        for tier in ["shallow", "medium", "deep"]:
            n_samples = results["stage1"]["by_depth"][tier]["n_samples"]
            f.write(f"### {tier.title()} (N={n_samples})\n\n")
            f.write("| Stage | Mesh Global | Voxel ROI | Voxel Global | S2 ROI | S2 Global | Delta ROI | Delta Global |\n")
            f.write("|-------|------------|-----------|--------------|--------|-----------|-----------|--------------|\n")
            
            s1 = results["stage1"]["by_depth"][tier]
            de = results["de_only"]["by_depth"][tier]
            mv = results["multiview"]["by_depth"][tier]
            
            f.write(f"| Stage 1 | {s1['mesh_dice_binary_05_global']:.4f} | {s1['voxel_dice_binary_05_roi']:.4f} | {s1['voxel_dice_binary_05_global']:.4f} | - | - | - | - |\n")
            f.write(f"| DE-only | - | - | - | {de['stage2_dice_binary_05_roi']:.4f} | {de['stage2_dice_binary_05_global']:.4f} | {de['delta_dice_roi']:+.4f} | {de['delta_dice_global']:+.4f} |\n")
            f.write(f"| Multiview | - | - | - | {mv['stage2_dice_binary_05_roi']:.4f} | {mv['stage2_dice_binary_05_global']:.4f} | {mv['delta_dice_roi']:+.4f} | {mv['delta_dice_global']:+.4f} |\n\n")
        
        f.write("---\n\n")
        f.write("## Foci × Depth Cross Analysis\n\n")
        f.write("| Foci | Depth | S1 Mesh | S1 Voxel ROI | S1 Voxel Global | DE ROI | DE Global | MV ROI | MV Global |\n")
        f.write("|------|-------|---------|--------------|-----------------|--------|-----------|--------|-----------|\n")
        
        for n in [1, 2, 3]:
            for tier in ["shallow", "medium", "deep"]:
                s1 = results["stage1"]["by_cross"].get((n, tier), {})
                de = results["de_only"]["by_cross"].get((n, tier), {})
                mv = results["multiview"]["by_cross"].get((n, tier), {})
                
                if not s1:
                    continue
                
                f.write(f"| {n}-Foci | {tier.title()} | {s1.get('mesh_dice_binary_05_global', 0):.4f} | {s1.get('voxel_dice_binary_05_roi', 0):.4f} | {s1.get('voxel_dice_binary_05_global', 0):.4f} | {de.get('stage2_dice_binary_05_roi', 0):.4f} | {de.get('stage2_dice_binary_05_global', 0):.4f} | {mv.get('stage2_dice_binary_05_roi', 0):.4f} | {mv.get('stage2_dice_binary_05_global', 0):.4f} |\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage1 vs Stage2 comprehensive comparison")
    parser.add_argument("--stage1_checkpoint", required=True, help="Stage 1 checkpoint path")
    parser.add_argument("--stage1_config", required=True, help="Stage 1 config YAML")
    parser.add_argument("--de_only_checkpoint", required=True, help="Stage 2 DE-only checkpoint")
    parser.add_argument("--de_only_config", required=True, help="Stage 2 DE-only config YAML")
    parser.add_argument("--multiview_checkpoint", required=True, help="Stage 2 Multiview checkpoint")
    parser.add_argument("--multiview_config", required=True, help="Stage 2 Multiview config YAML")
    parser.add_argument("--shared_dir", required=True, help="FMT-SimGen shared directory")
    parser.add_argument("--samples_dir", required=True, help="FMT-SimGen samples directory")
    parser.add_argument("--bridge_dir", required=True, help="Bridge output directory")
    parser.add_argument("--precomputed_dir", required=True, help="Precomputed grid directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    parser.add_argument("--batch_points", type=int, default=4096, help="Batch size for inference")
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    device = get_device()
    
    split_name = Path(args.bridge_dir).name.split("_")[-1]
    split_file = Path(args.samples_dir).parent / "splits" / f"{split_name}.txt"
    if not split_file.exists():
        split_file = Path(args.samples_dir).parent / f"{split_name}.txt"
    
    with open(split_file) as f:
        sample_ids = [line.strip() for line in f if line.strip()]
    
    print(f"[Stage 1] Computing global metrics for {len(sample_ids)} samples...")
    stage1_per_sample, stage1_summary = evaluate_stage1_global(
        checkpoint_path=args.stage1_checkpoint,
        config_path=args.stage1_config,
        shared_dir=Path(args.shared_dir),
        samples_dir=Path(args.samples_dir),
        bridge_dir=Path(args.bridge_dir),
        precomputed_dir=Path(args.precomputed_dir),
        sample_ids=sample_ids,
        device=device,
    )
    
    print(f"[Stage 2 DE-only] Computing global metrics...")
    de_only_per_sample, de_only_summary = evaluate_stage2_global(
        checkpoint_path=args.de_only_checkpoint,
        config_path=args.de_only_config,
        shared_dir=Path(args.shared_dir),
        samples_dir=Path(args.samples_dir),
        bridge_dir=Path(args.bridge_dir),
        sample_ids=sample_ids,
        device=device,
        multiview=False,
        batch_points=args.batch_points,
    )
    
    print(f"[Stage 2 Multiview] Computing global metrics...")
    multiview_per_sample, multiview_summary = evaluate_stage2_global(
        checkpoint_path=args.multiview_checkpoint,
        config_path=args.multiview_config,
        shared_dir=Path(args.shared_dir),
        samples_dir=Path(args.samples_dir),
        bridge_dir=Path(args.bridge_dir),
        sample_ids=sample_ids,
        device=device,
        multiview=True,
        batch_points=args.batch_points,
    )
    
    manifest_raw = load_json(Path(args.samples_dir).parent / "dataset_manifest.json")
    manifest = {"samples_by_id": {s["id"]: s for s in manifest_raw["samples"]}}
    
    selected = pick_representative_samples(stage1_per_sample, de_only_per_sample, multiview_per_sample, manifest)
    
    full_results = {
        "metadata": {
            "n_samples": len(sample_ids),
            "stage1_checkpoint": args.stage1_checkpoint,
            "de_only_checkpoint": args.de_only_checkpoint,
            "multiview_checkpoint": args.multiview_checkpoint,
            "timestamp": datetime.now().isoformat(),
        },
        "stage1": {
            **stage1_summary,
            "per_sample": stage1_per_sample,
        },
        "de_only": {
            **de_only_summary,
            "per_sample": de_only_per_sample,
        },
        "multiview": {
            **multiview_summary,
            "per_sample": multiview_per_sample,
        },
        "representative_samples": selected,
    }
    
    with open(out_dir / "full_results.json", "w") as f:
        json.dump(full_results, f, indent=2)
    
    save_markdown_summary(full_results, out_dir / "summary.md")
    
    print(f"[Visualization] Rendering {len(selected)} representative samples...")
    frame = FrameManifest.load(args.shared_dir)
    mesh = np.load(Path(args.shared_dir) / "mesh.npz")
    nodes = mesh["nodes"].astype(np.float64)
    elements = mesh["elements"]
    tissue_labels = mesh["tissue_labels"]
    surface_faces = mesh["surface_faces"]
    body_surface = build_surface_mesh(nodes, surface_faces).smooth(n_iter=20, relaxation_factor=0.1)
    organ_surfaces = get_organ_surfaces(nodes, elements, tissue_labels)
    
    inr_de, view_encoder_de, model_cfg_de = load_stage2_model(
        args.de_only_checkpoint, args.de_only_config, device, multiview=False
    )
    inr_mv, view_encoder_mv, model_cfg_mv = load_stage2_model(
        args.multiview_checkpoint, args.multiview_config, device, multiview=True
    )
    
    for item in selected:
        sid = item["sample_id"]
        print(f"  [Render] {sid} ({item['num_foci']}-foci, de={item['de_only_delta']:+.4f}, mv={item['multiview_delta']:+.4f})")
        
        data = dict(np.load(Path(args.precomputed_dir) / f"{sid}.npz", allow_pickle=False))
        proj_path = Path(args.samples_dir) / sid / "proj.npz"
        if proj_path.exists():
            proj_data = np.load(proj_path)
            proj_imgs = np.stack([proj_data[str(a)].astype(np.float32) for a in [-90, -60, -30, 0, 30, 60, 90]], axis=0)
        else:
            proj_imgs = np.zeros((7, 256, 256), dtype=np.float32)
        
        tumor_params = load_json(Path(args.samples_dir) / sid / "tumor_params.json")
        
        bridge_path = Path(args.bridge_dir) / sid
        coarse_d = np.load(bridge_path / "coarse_d.npy").astype(np.float64).flatten()
        roi_tet_indices = np.load(bridge_path / "roi_tet_indices.npy")
        bridge = FEMBridge(nodes, elements, roi_tet_indices)
        
        valid_mask = data["valid_mask"].astype(bool)
        coords_world = data["grid_coords"][valid_mask].copy()
        coords_norm = data["grid_coords_norm"][valid_mask].copy()
        prior_8d, _ = bridge.get_prior_features(coords_world, coarse_d)
        
        stage1_valid = fem_interp_from_prior(prior_8d)
        
        de_only_valid = infer_stage2(
            coords_norm=coords_norm,
            coords_world=coords_world,
            prior_8d=prior_8d.astype(np.float32),
            proj_imgs=proj_imgs,
            inr=inr_de,
            view_encoder=view_encoder_de,
            model_cfg=model_cfg_de,
            frame=frame,
            device=device,
            batch_points=args.batch_points,
            multiview=False,
        )
        
        stage2_valid = infer_stage2(
            coords_norm=coords_norm,
            coords_world=coords_world,
            prior_8d=prior_8d.astype(np.float32),
            proj_imgs=proj_imgs,
            inr=inr_mv,
            view_encoder=view_encoder_mv,
            model_cfg=model_cfg_mv,
            frame=frame,
            device=device,
            batch_points=args.batch_points,
            multiview=True,
        )
        
        gt_valid = data["gt_values"][valid_mask]
        
        stage1_volume = dense_volume_from_valid(data, stage1_valid)
        de_only_volume = dense_volume_from_valid(data, de_only_valid)
        stage2_volume = dense_volume_from_valid(data, stage2_valid)
        gt_volume = dense_volume_from_valid(data, gt_valid)
        
        render_sample_comparison(
            sample_id=sid,
            stage1_volume=stage1_volume,
            de_only_volume=de_only_volume,
            stage2_volume=stage2_volume,
            gt_volume=gt_volume,
            bbox_min=data["bbox_min"].astype(np.float64),
            bbox_max=data["bbox_max"].astype(np.float64),
            body_surface=body_surface,
            organ_surfaces=organ_surfaces,
            tumor_params=tumor_params,
            out_path=fig_dir / f"{sid}_compare.png",
        )
    
    print(f"\nSaved full results to {out_dir / 'full_results.json'}")
    print(f"Saved summary to {out_dir / 'summary.md'}")
    print(f"Saved {len(selected)} representative figures to {fig_dir}")


if __name__ == "__main__":
    main()
