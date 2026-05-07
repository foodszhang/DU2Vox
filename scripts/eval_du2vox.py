#!/usr/bin/env python3
"""
DU2Vox unified evaluation: Stage 1 (mesh + voxel Dice) and Stage 2 (DE-only + Multiview).

Usage:
    # Stage 1
    python scripts/eval_du2vox.py stage1 --config configs/stage1/uniform_1000_20k.yaml \\
        --checkpoint runs/stage1_uniform_1000_20k/checkpoints/best.pth

    # Stage 1 voxel Dice (coarse_d from bridge output, interpolated to precomputed grid)
    python scripts/eval_du2vox.py stage1 --config configs/stage1/uniform_1000_20k.yaml \\
        --checkpoint runs/stage1_uniform_1000_20k/checkpoints/best.pth --voxel

    # Stage 2 DE-only
    python scripts/eval_du2vox.py stage2 --config configs/stage2/uniform_1000_20k.yaml \\
        --checkpoint checkpoints/stage2/baseline_de_only_20k/best.pth

    # Stage 2 Multiview
    python scripts/eval_du2vox.py stage2 --config configs/stage2/full_multiview_20k.yaml \\
        --checkpoint checkpoints/stage2/mv_fixed_ext/best.pth --multiview
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from du2vox.models.stage1.gcain import GCAIN_full
from du2vox.data.dataset import FMTSimGenDataset
from du2vox.bridge.fem_bridging import FEMBridge
from du2vox.utils.frame import FrameManifest
from du2vox.evaluation.per_foci import group_metrics_by_foci, group_metrics_by_depth, group_metrics_by_cross
from du2vox.evaluation.metrics import evaluate_batch


def load_split(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def compute_dice(pred, target, threshold=0.5):
    pred_bin = (pred >= threshold).astype(float)
    target_bin = (target >= threshold).astype(float)
    intersection = (pred_bin * target_bin).sum()
    return float(2 * intersection / (pred_bin.sum() + target_bin.sum() + 1e-8))


def fem_interp_from_prior(prior_8d):
    """Convert FEM prior features [d_v0..d_v3, lambda0..lambda3] to scalar values."""
    return (prior_8d[:, :4] * prior_8d[:, 4:8]).sum(axis=1)


def resolve_bridge_dir(data_cfg, split):
    """Resolve bridge output dir from config, falling back to legacy 20k paths."""
    return Path(
        data_cfg.get(f"{split}_bridge_dir")
        or data_cfg.get("bridge_dir")
        or f"output/bridge_20k_{split}"
    )


def resolve_precomputed_dir(data_cfg, split):
    """Resolve precomputed grid dir from config, falling back to legacy 20k paths."""
    return Path(
        data_cfg.get(f"precomputed_{split}_dir")
        or data_cfg.get("precomputed_dir")
        or f"precomputed/{split}_20k"
    )


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Stage 1 ──────────────────────────────────────────────────────────────────

def eval_stage1(cfg, checkpoint_path, split="val", voxel_mode=False, output=None):
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    device = get_device()

    shared_dir = Path(data_cfg["shared_dir"])
    samples_dir = Path(data_cfg["samples_dir"])
    splits_dir = Path(data_cfg["splits_dir"])
    split_file = splits_dir / f"{split}.txt"
    sample_ids = load_split(split_file)

    manifest_path = splits_dir.parent / "dataset_manifest.json"
    manifest_data = json.load(open(manifest_path)) if manifest_path.exists() else None
    manifest = {"samples": {s["id"]: s for s in manifest_data["samples"]}} if manifest_data else None

    print(f"[Stage 1] Config: {cfg['experiment']['name']}, Split: {split}, n_samples={len(sample_ids)}")

    # Load Stage 1 model
    dataset = FMTSimGenDataset(
        shared_dir=shared_dir,
        samples_dir=samples_dir,
        split_file=split_file,
        normalize_b=data_cfg.get("normalize_b", True),
        normalize_gt=data_cfg.get("normalize_gt", True),
        normalize_gt_mode=data_cfg.get("normalize_gt_mode", "per_sample"),
        binarize_gt=data_cfg.get("binarize_gt", False),
        binarize_threshold=data_cfg.get("binarize_threshold", 0.05),
    )
    nodes = dataset.nodes.to(device)
    A, L = dataset.A.to(device), dataset.L.to(device)
    L0, L1, L2, L3 = dataset.L0.to(device), dataset.L1.to(device), dataset.L2.to(device), dataset.L3.to(device)
    knn_idx, sens_w = dataset.knn_idx.to(device), dataset.sens_w.to(device)
    LTL, ATA = torch.matmul(L.t(), L).to(device), torch.matmul(A.t(), A).to(device)

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

    if voxel_mode:
        bridge_dir = resolve_bridge_dir(data_cfg, split)
        precomputed_dir = resolve_precomputed_dir(data_cfg, split)
        _nodes, _elements = FrameManifest.load_mesh_nodes(shared_dir)
        nodes_np = _nodes.astype(np.float64)
        elements = _elements
    else:
        bridge_dir = None
        precomputed_dir = None
        nodes_np = None
        elements = None

    per_sample_mesh = []
    per_sample_voxel = []

    for idx, sid in enumerate(sample_ids):
        if idx % 50 == 0:
            print(f"  [{idx}/{len(sample_ids)}]")

        # ── Mesh Dice ──────────────────────────────────────────────────
        b = dataset.b_list[idx].float().unsqueeze(0).to(device)
        gt_mesh = dataset.gt_list[idx].float().unsqueeze(0).to(device)
        X0 = torch.zeros(1, nodes.shape[0], 1, device=device)
        with torch.no_grad():
            pred_mesh = model(X0, b)
        mesh_metrics = evaluate_batch(pred_mesh, gt_mesh, nodes)
        mesh_dice = mesh_metrics["dice"]

        per_sample_mesh.append({"sample_id": sid, "mesh_dice": float(mesh_dice)})

        if not voxel_mode:
            continue

        # ── Voxel Dice ─────────────────────────────────────────────────
        bridge_sample_dir = bridge_dir / sid
        npz_path = precomputed_dir / f"{sid}.npz"
        if not bridge_sample_dir.exists() or not npz_path.exists():
            continue

        coarse_d = np.load(bridge_sample_dir / "coarse_d.npy").flatten().astype(np.float64)
        roi_tet_indices = np.load(bridge_sample_dir / "roi_tet_indices.npy")
        bridge = FEMBridge(nodes_np, elements, roi_tet_indices)

        npz = dict(np.load(npz_path))
        coords_grid = npz["grid_coords"]
        gt_values = npz["gt_values"]
        valid_mask = npz["valid_mask"]

        v_mask = valid_mask > 0
        if v_mask.sum() == 0:
            continue

        coords_valid = coords_grid[v_mask]
        coarse_prior, _ = bridge.get_prior_features(coords_valid, coarse_d)
        coarse_scalar = fem_interp_from_prior(coarse_prior)
        voxel_dice = compute_dice(coarse_scalar, gt_values[v_mask], 0.5)

        per_sample_voxel.append({
            "sample_id": sid,
            "voxel_dice": float(voxel_dice),
            "dice": float(voxel_dice),
        })

    # ── Summary ──────────────────────────────────────────────────────────
    mesh_dices = [m["mesh_dice"] for m in per_sample_mesh]
    mesh_overall = float(np.mean(mesh_dices))
    print(f"\n{'='*60}")
    print(f"Stage 1 Mesh Dice (FEM nodes, threshold=0.5)")
    print(f"{'='*60}")
    print(f"  Overall: {mesh_overall:.4f} ({len(mesh_dices)} samples)")

    if manifest and per_sample_voxel:
        by_foci = group_metrics_by_foci(per_sample_voxel, [m["sample_id"] for m in per_sample_voxel], manifest)
        by_depth = group_metrics_by_depth(per_sample_voxel, [m["sample_id"] for m in per_sample_voxel], manifest)
        by_cross = group_metrics_by_cross(per_sample_voxel, [m["sample_id"] for m in per_sample_voxel], manifest)

        voxel_samples = per_sample_voxel
        v_overall = float(np.mean([m["voxel_dice"] for m in voxel_samples]))

        print(f"\n{'='*60}")
        print(f"Stage 1 Voxel Dice (interpolated to voxel grid)")
        print(f"{'='*60}")
        print(f"  Overall: {v_overall:.4f} ({len(voxel_samples)} samples)")

        print(f"\n{'='*60}")
        print("PER-FOCI — Stage 1")
        print(f"{'='*60}")
        print(f"{'Metric':<20} {'Overall':>10} {'1-Foci':>10} {'2-Foci':>10} {'3-Foci':>10}")
        print("-" * 62)
        for k, disp in [("voxel_dice","Voxel Dice"), ("mesh_dice","Mesh Dice")]:
            vals = [m[k] for m in voxel_samples if m.get(k) is not None] if k == "voxel_dice" else [m[k] for m in per_sample_mesh if m.get(k) is not None]
            ov = float(np.mean(vals)) if vals else 0.0
            row = f"{disp:<20} {ov:>10.4f}"
            for n in [1, 2, 3]:
                grp = by_foci.get(n, [])
                if k == "voxel_dice":
                    v = np.mean([g[k] for g in grp]) if grp else 0
                else:
                    mesh_grp = [m for m in per_sample_mesh if m["sample_id"] in [g["sample_id"] for g in grp]] if grp else []
                    v = np.mean([m[k] for m in mesh_grp]) if mesh_grp else 0
                row += f" {v:>10.4f}"
            print(row)

        print(f"\n{'='*60}")
        print("PER-DEPTH — Stage 1 Voxel Dice")
        print(f"{'='*60}")
        print(f"{'Metric':<20} {'Overall':>10} {'Shallow':>10} {'Medium':>10} {'Deep':>10}")
        print("-" * 62)
        vals = [m["voxel_dice"] for m in voxel_samples]
        ov = float(np.mean(vals))
        row = f"{'Voxel Dice':<20} {ov:>10.4f}"
        for t in ["shallow", "medium", "deep"]:
            grp = by_depth.get(t, [])
            v = np.mean([g["voxel_dice"] for g in grp]) if grp else 0
            row += f" {v:>10.4f}"
        print(row)

        print(f"\n{'='*60}")
        print("FOCI × DEPTH — Stage 1 Voxel Dice")
        print(f"{'='*60}")
        print(f"{'Foci \\ Depth':<15} {'Shallow':>12} {'Medium':>12} {'Deep':>12}")
        print("-" * 55)
        for n in [1, 2, 3]:
            row = f"{n}-Foci" + " " * (15 - len(f"{n}-Foci"))
            for t in ["shallow", "medium", "deep"]:
                grp = by_cross.get((n, t), [])
                v = np.mean([g["voxel_dice"] for g in grp]) if grp else 0
                row += f" {v:>12.4f}"
            print(row)

    result = {
        "stage1_mesh_dice": mesh_overall,
        "stage1_voxel_dice": float(np.mean([m["voxel_dice"] for m in per_sample_voxel])) if per_sample_voxel else None,
        "per_sample_mesh": {m["sample_id"]: m for m in per_sample_mesh},
        "per_sample_voxel": {m["sample_id"]: m for m in per_sample_voxel},
    }
    if output:
        with open(output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {output}")
    return result


# ─── Stage 2 ──────────────────────────────────────────────────────────────────

def eval_stage2(cfg, checkpoint_path, split="val", multiview=False, output=None, batch_points=8192):
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    device = get_device()

    shared_dir = Path(data_cfg["shared_dir"])
    samples_dir = Path(data_cfg["samples_dir"])
    train_split = Path(data_cfg["train_split"])
    val_split = Path(data_cfg["val_split"])
    split_file = val_split if split == "val" else train_split
    sample_ids = load_split(split_file)

    manifest_path = train_split.parent.parent / "dataset_manifest.json"
    manifest_data = json.load(open(manifest_path)) if manifest_path.exists() else None
    manifest = {"samples": {s["id"]: s for s in manifest_data["samples"]}} if manifest_data else None

    print(f"[Stage 2] Config: {cfg['experiment']['name']}, Split: {split}, n_samples={len(sample_ids)}, multiview={multiview}")

    # Build INR model
    view_feat_dim = model_cfg.get("view_feat_dim", 0) if multiview else 0
    from du2vox.models.stage2.residual_inr import ResidualINR
    inr = ResidualINR(
        n_freqs=model_cfg["n_freqs"],
        hidden_dim=model_cfg["hidden_dim"],
        n_hidden_layers=model_cfg["n_hidden_layers"],
        prior_dim=model_cfg["prior_dim"],
        skip_connection=model_cfg.get("skip_connection", True),
        view_feat_dim=view_feat_dim,
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    if multiview and "residual_inr" in state:
        inr.load_state_dict(state["residual_inr"])
    elif "residual_inr" in state:
        inr.load_state_dict(state["residual_inr"])
    else:
        inr.load_state_dict(state)

    if multiview:
        from du2vox.models.stage2.view_encoder import ViewEncoderModule
        view_encoder = ViewEncoderModule(
            view_feat_dim=view_feat_dim,
            fusion_method=model_cfg.get("fusion_method", "attn"),
            encoder_out_channels=model_cfg.get("encoder_out_channels", 32),
            encoder_base_channels=model_cfg.get("encoder_base_channels", 32),
        ).to(device)
        if "view_encoder" in state:
            view_encoder.load_state_dict(state["view_encoder"])
        view_encoder.eval()
        print(f"  Multiview loaded: view_feat_dim={view_feat_dim}, in_dim={inr.in_dim}")

    inr.eval()
    if multiview:
        view_encoder.eval()

    # Dataset
    precomputed_dir = resolve_precomputed_dir(data_cfg, split)
    if multiview:
        from du2vox.models.stage2.stage2_dataset import Stage2DatasetPrecomputedMultiview
        s2_dataset = Stage2DatasetPrecomputedMultiview(
            precomputed_dir=precomputed_dir,
            samples_dir=samples_dir,
            sample_ids=sample_ids,
            n_query_points=4096,
            cache_size=16,
            shared_dir=str(shared_dir),
        )
    else:
        from du2vox.models.stage2.stage2_dataset import Stage2DatasetPrecomputed
        s2_dataset = Stage2DatasetPrecomputed(
            precomputed_dir=precomputed_dir,
            sample_ids=sample_ids,
            n_query_points=4096,
            cache_size=32,
        )

    bridge_dir = resolve_bridge_dir(data_cfg, split)
    _nodes, _elements = FrameManifest.load_mesh_nodes(shared_dir)
    nodes_np = _nodes.astype(np.float64)
    elements = _elements

    per_sample = []

    for idx, sid in enumerate(sample_ids):
        if idx % 50 == 0:
            print(f"  [{idx}/{len(sample_ids)}]")

        # FEM baseline from bridge output
        bridge_sample_dir = bridge_dir / sid
        if not bridge_sample_dir.exists():
            print(f"  {sid}: missing bridge output at {bridge_sample_dir}, skipping")
            continue
        coarse_d = np.load(bridge_sample_dir / "coarse_d.npy").flatten().astype(np.float64)
        roi_tet_indices = np.load(bridge_sample_dir / "roi_tet_indices.npy")
        bridge = FEMBridge(nodes_np, elements, roi_tet_indices)

        data = s2_dataset._load_npz(sid)
        valid_mask = data["valid_mask"]
        v_mask = valid_mask > 0
        coords_norm = data["grid_coords_norm"][v_mask].copy()
        prior_8d = data["prior_8d"][v_mask].copy()
        gt_values = data["gt_values"][v_mask].copy()

        if len(coords_norm) == 0:
            continue

        # FEM interp baseline
        coords_world = data["grid_coords"][v_mask].copy()
        fem_interp, _ = bridge.get_prior_features(coords_world, coarse_d)
        fem_scalar = fem_interp_from_prior(fem_interp).astype(np.float64)
        fem_dice = compute_dice(fem_scalar, gt_values, 0.5)

        # Stage 2 prediction
        preds = []
        if multiview:
            MCX_ANGLES = [-90, -60, -30, 0, 30, 60, 90]
            proj_path = samples_dir / sid / "proj.npz"
            if proj_path.exists():
                proj_data = np.load(proj_path)
                proj_imgs = np.stack(
                    [proj_data[str(angle)].astype(np.float32) for angle in MCX_ANGLES], axis=0,
                )
            else:
                proj_imgs = np.zeros((7, 256, 256), dtype=np.float32)
            proj_t = torch.from_numpy(proj_imgs).float().unsqueeze(1).to(device).unsqueeze(0)
            if "mcx_valid" in data:
                mcx_valid_all = data["mcx_valid"][v_mask].astype(bool)
            else:
                frame = FrameManifest.load(shared_dir)
                lo, hi = frame.mcx_bbox_min, frame.mcx_bbox_max
                mcx_valid_all = (
                    (coords_world[:, 0] >= lo[0]) & (coords_world[:, 0] <= hi[0]) &
                    (coords_world[:, 1] >= lo[1]) & (coords_world[:, 1] <= hi[1]) &
                    (coords_world[:, 2] >= lo[2]) & (coords_world[:, 2] <= hi[2])
                )
        else:
            proj_t = None
            mcx_valid_all = None

        with torch.no_grad():
            for start in range(0, len(coords_norm), batch_points):
                end = min(start + batch_points, len(coords_norm))
                coords_chunk = torch.from_numpy(
                    coords_norm[start:end]
                ).float().unsqueeze(0).to(device)
                prior_chunk = torch.from_numpy(
                    prior_8d[start:end]
                ).float().unsqueeze(0).to(device)

                if multiview:
                    coords_world_chunk = torch.from_numpy(
                        coords_world[start:end]
                    ).float().unsqueeze(0).to(device)
                    view_feat, _ = view_encoder(proj_t, coords_world_chunk, None)
                    mcx_valid_chunk = torch.from_numpy(
                        mcx_valid_all[start:end]
                    ).to(device).view(1, -1, 1).float()
                    view_feat = view_feat * mcx_valid_chunk
                    pred_chunk, _, _ = inr(coords_chunk, prior_chunk, view_feat)
                else:
                    pred_chunk, _, _ = inr(coords_chunk, prior_chunk)

                preds.append(pred_chunk.squeeze(0).cpu().numpy())

        d_hat = np.concatenate(preds, axis=0)
        s2_dice = compute_dice(d_hat, gt_values, 0.5)

        per_sample.append({
            "sample_id": sid,
            "stage2_dice": float(s2_dice),
            "fem_dice": float(fem_dice),
            "delta_dice": float(s2_dice - fem_dice),
            "dice": float(s2_dice),
        })

    # ── Summary ──────────────────────────────────────────────────────────
    s2_dices = [m["stage2_dice"] for m in per_sample]
    fem_dices = [m["fem_dice"] for m in per_sample]
    delta_dices = [m["delta_dice"] for m in per_sample]

    print(f"\n{'='*60}")
    print(f"Stage 2 {'Multiview' if multiview else 'DE-only'} Evaluation")
    print(f"{'='*60}")
    print(f"  Stage2 Dice: {np.mean(s2_dices):.4f}")
    print(f"  FEM Dice:    {np.mean(fem_dices):.4f}")
    print(f"  Delta Dice:  {np.mean(delta_dices):.4f}")

    if manifest:
        by_foci = group_metrics_by_foci(per_sample, [m["sample_id"] for m in per_sample], manifest)
        by_depth = group_metrics_by_depth(per_sample, [m["sample_id"] for m in per_sample], manifest)
        by_cross = group_metrics_by_cross(per_sample, [m["sample_id"] for m in per_sample], manifest)

        print(f"\n{'='*60}")
        print("PER-FOCI")
        print(f"{'='*60}")
        print(f"{'Metric':<20} {'Overall':>10} {'1-Foci':>10} {'2-Foci':>10} {'3-Foci':>10}")
        print("-" * 62)
        for k, disp in [("stage2_dice","S2 Dice"), ("fem_dice","FEM Dice"), ("delta_dice","Δ Dice")]:
            vals = [m[k] for m in per_sample]
            ov = float(np.mean(vals))
            row = f"{disp:<20} {ov:>10.4f}"
            for n in [1, 2, 3]:
                grp = by_foci.get(n, [])
                v = np.mean([g[k] for g in grp]) if grp else 0
                row += f" {v:>10.4f}"
            print(row)

        print(f"\n{'='*60}")
        print("PER-DEPTH")
        print(f"{'='*60}")
        print(f"{'Metric':<20} {'Overall':>10} {'Shallow':>10} {'Medium':>10} {'Deep':>10}")
        print("-" * 62)
        for k, disp in [("stage2_dice","S2 Dice"), ("fem_dice","FEM Dice"), ("delta_dice","Δ Dice")]:
            vals = [m[k] for m in per_sample]
            ov = float(np.mean(vals))
            row = f"{disp:<20} {ov:>10.4f}"
            for t in ["shallow", "medium", "deep"]:
                grp = by_depth.get(t, [])
                v = np.mean([g[k] for g in grp]) if grp else 0
                row += f" {v:>10.4f}"
            print(row)

        print(f"\n{'='*60}")
        print("FOCI × DEPTH — S2 Dice")
        print(f"{'='*60}")
        print(f"{'Foci \\ Depth':<15} {'Shallow':>12} {'Medium':>12} {'Deep':>12}")
        print("-" * 55)
        for n in [1, 2, 3]:
            row = f"{n}-Foci" + " " * (15 - len(f"{n}-Foci"))
            for t in ["shallow", "medium", "deep"]:
                grp = by_cross.get((n, t), [])
                v = np.mean([g["stage2_dice"] for g in grp]) if grp else 0
                row += f" {v:>12.4f}"
            print(row)

    result = {
        "stage2_dice": float(np.mean(s2_dices)),
        "fem_dice": float(np.mean(fem_dices)),
        "delta_dice": float(np.mean(delta_dices)),
        "per_sample": {m["sample_id"]: m for m in per_sample},
    }
    if output:
        with open(output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {output}")
    return result


# ─── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("stage1", help="Stage 1 evaluation (mesh + voxel Dice)")
    p1.add_argument("--config", required=True)
    p1.add_argument("--checkpoint", required=True)
    p1.add_argument("--split", default="val")
    p1.add_argument("--voxel", action="store_true", help="Also compute voxel Dice")
    p1.add_argument("--output", type=str, default=None)

    p2 = sub.add_parser("stage2", help="Stage 2 evaluation (DE-only or Multiview)")
    p2.add_argument("--config", required=True)
    p2.add_argument("--checkpoint", required=True)
    p2.add_argument("--split", default="val")
    p2.add_argument("--multiview", action="store_true")
    p2.add_argument("--output", type=str, default=None)
    p2.add_argument("--batch_points", type=int, default=8192)

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.cmd == "stage1":
        eval_stage1(cfg, args.checkpoint, args.split, voxel_mode=args.voxel, output=args.output)
    elif args.cmd == "stage2":
        eval_stage2(
            cfg,
            args.checkpoint,
            args.split,
            multiview=args.multiview,
            output=args.output,
            batch_points=args.batch_points,
        )


if __name__ == "__main__":
    main()
