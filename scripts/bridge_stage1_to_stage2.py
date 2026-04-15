#!/usr/bin/env python3
"""
Stage 1→2 bridge pipeline.

Runs frozen Stage 1 inference, derives per-sample ROIs, and optionally
computes 8D FEM prior features for downstream Stage 2 training.

Usage:
    uv run python scripts/bridge_stage1_to_stage2.py \\
        --config configs/stage1/gaussian_1000.yaml \\
        --checkpoint runs/gcain_gaussian_1000/checkpoints/best.pth \\
        --split_file /path/to/splits/val.txt \\
        --output_dir bridge_output/ \\
        --tau 0.5 --dilate_layers 1

Output structure:
    bridge_output/
    ├── sample_0000/
    │   ├── coarse_d.npy         [N_nodes] float32, Stage 1 output
    │   ├── roi_tet_indices.npy  [M_roi] int64, ROI tet indices
    │   └── roi_info.json        ROI stats + bounding box
    ├── ...
    └── bridge_stats.json        Global statistics across all samples
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def load_shared_mesh(shared_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load nodes and elements from mesh.npz."""
    mesh = np.load(shared_dir / "mesh.npz")
    nodes = mesh["nodes"].astype(np.float32)      # [N, 3]
    elements = mesh["elements"].astype(np.int64)  # [N_tets, 4]
    return nodes, elements


def run_inference_stage(args, cfg, sample_ids: list[str]) -> None:
    from du2vox.bridge.stage1_inference import run_stage1_inference

    train_cfg = cfg["training"]
    run_stage1_inference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        shared_dir=cfg["data"]["shared_dir"],
        samples_dir=cfg["data"]["samples_dir"],
        split_file=args.split_file,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        activation=train_cfg.get("activation", "leaky_relu"),
        leaky_relu_slope=train_cfg.get("leaky_relu_slope", 0.01),
    )


def run_roi_stage(
    args,
    nodes: np.ndarray,
    elements: np.ndarray,
    sample_ids: list[str],
) -> list[dict]:
    from du2vox.bridge.roi_derivation import derive_roi, save_roi_results

    output_dir = Path(args.output_dir)
    roi_results = []

    print(f"\n[ROI] Deriving ROIs for {len(sample_ids)} samples (tau={args.tau}, dilate={args.dilate_layers})")

    for i, sid in enumerate(sample_ids):
        sample_out = output_dir / sid
        coarse_d_path = sample_out / "coarse_d.npy"
        if not coarse_d_path.exists():
            print(f"  [{i+1}/{len(sample_ids)}] {sid}: missing coarse_d.npy, skipping")
            continue

        coarse_d = np.load(coarse_d_path).astype(np.float64)
        result = derive_roi(coarse_d, nodes, elements, tau=args.tau, dilate_layers=args.dilate_layers)
        save_roi_results(result, sample_out)
        roi_results.append(result)

        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"  [{i+1}/{len(sample_ids)}] {sid}: "
                f"active={result['n_active_nodes']} ({result['activation_ratio']:.1%}), "
                f"roi_tets={result['n_roi_tets']} ({result['roi_tet_ratio']:.1%}), "
                f"bbox_size={[f'{v:.1f}' for v in [result['roi_bbox_mm']['max'][j]-result['roi_bbox_mm']['min'][j] for j in range(3)]]}mm"
            )

    return roi_results


def run_fem_stage(
    args,
    nodes: np.ndarray,
    elements: np.ndarray,
    sample_ids: list[str],
) -> None:
    """Compute and save 8D prior features for gt_nodes as query points."""
    from du2vox.bridge.fem_bridging import FEMBridge

    output_dir = Path(args.output_dir)
    samples_dir = Path(args.samples_dir or cfg["data"]["samples_dir"])

    print(f"\n[FEM] Computing 8D prior features for {len(sample_ids)} samples")

    for i, sid in enumerate(sample_ids):
        sample_out = output_dir / sid
        coarse_d_path = sample_out / "coarse_d.npy"
        roi_path = sample_out / "roi_tet_indices.npy"

        if not coarse_d_path.exists() or not roi_path.exists():
            continue

        coarse_d = np.load(coarse_d_path).astype(np.float64)
        roi_tet_indices = np.load(roi_path)

        # Use FEM node positions as query points (gt_nodes alignment)
        gt_path = samples_dir / sid / "gt_nodes.npy"
        if not gt_path.exists():
            continue
        # Query points: all mesh nodes (Stage 2 will query arbitrary points)
        # Here we demonstrate with a subset: ROI node positions
        roi_node_indices = np.unique(elements[roi_tet_indices].ravel())
        query_points = nodes[roi_node_indices].astype(np.float64)

        bridge = FEMBridge(nodes, elements, roi_tet_indices)
        prior_8d, valid_mask = bridge.get_prior_features(query_points, coarse_d)

        np.savez_compressed(
            sample_out / "prior_cache.npz",
            prior_8d=prior_8d.astype(np.float32),
            valid_mask=valid_mask,
            query_node_indices=roi_node_indices,
        )

        if (i + 1) % 20 == 0 or i == 0:
            n_valid = valid_mask.sum()
            print(
                f"  [{i+1}/{len(sample_ids)}] {sid}: "
                f"{n_valid}/{len(query_points)} query points valid ({n_valid/max(len(query_points),1):.1%})"
            )


def print_summary(roi_results: list[dict], output_dir: Path) -> None:
    if not roi_results:
        print("\n[Summary] No ROI results to summarize.")
        return

    activation_ratios = [r["activation_ratio"] for r in roi_results]
    roi_ratios = [r["roi_tet_ratio"] for r in roi_results]
    bbox_sizes = [
        [r["roi_bbox_mm"]["max"][j] - r["roi_bbox_mm"]["min"][j] for j in range(3)]
        for r in roi_results
    ]
    mean_bbox = [float(np.mean([s[j] for s in bbox_sizes])) for j in range(3)]

    stats = {
        "n_samples": len(roi_results),
        "activation_ratio": {
            "mean": float(np.mean(activation_ratios)),
            "std": float(np.std(activation_ratios)),
            "min": float(np.min(activation_ratios)),
            "max": float(np.max(activation_ratios)),
        },
        "roi_tet_ratio": {
            "mean": float(np.mean(roi_ratios)),
            "std": float(np.std(roi_ratios)),
            "min": float(np.min(roi_ratios)),
            "max": float(np.max(roi_ratios)),
        },
        "mean_bbox_size_mm": mean_bbox,
        "mean_n_active_nodes": float(np.mean([r["n_active_nodes"] for r in roi_results])),
        "mean_n_roi_tets": float(np.mean([r["n_roi_tets"] for r in roi_results])),
    }

    with open(output_dir / "bridge_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n[Summary]")
    print(f"  Samples processed:      {stats['n_samples']}")
    print(f"  Active nodes (mean±std): {stats['activation_ratio']['mean']:.1%} ± {stats['activation_ratio']['std']:.1%}")
    print(f"  ROI tet ratio (mean):    {stats['roi_tet_ratio']['mean']:.1%}")
    print(f"  Mean bbox size (mm):     {[f'{v:.1f}' for v in mean_bbox]}")
    print(f"  Mean active nodes:       {stats['mean_n_active_nodes']:.0f}")
    print(f"  Mean ROI tets:           {stats['mean_n_roi_tets']:.0f}")
    print(f"  Stats saved to {output_dir / 'bridge_stats.json'}")


def main():
    parser = argparse.ArgumentParser(description="Stage 1→2 bridge pipeline")
    parser.add_argument("--config", required=True, help="Stage 1 config yaml")
    parser.add_argument("--checkpoint", required=True, help="Stage 1 checkpoint .pth")
    parser.add_argument("--split_file", required=True, help="Split txt file (one sample_id per line)")
    parser.add_argument("--output_dir", default="bridge_output", help="Output root directory")
    parser.add_argument("--samples_dir", default=None, help="Override samples_dir from config")
    parser.add_argument("--tau", type=float, default=0.5, help="ROI activation threshold (default 0.5)")
    parser.add_argument("--dilate_layers", type=int, default=1, help="ROI dilation layers (default 1)")
    parser.add_argument("--device", default="cuda", help="Device for Stage 1 inference")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--skip_inference", action="store_true", help="Skip Stage 1 inference (reuse existing coarse_d.npy)")
    parser.add_argument("--skip_roi", action="store_true", help="Skip ROI derivation")
    parser.add_argument("--compute_prior_cache", action="store_true", help="Compute and save 8D prior features")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.samples_dir is not None:
        cfg["data"]["samples_dir"] = args.samples_dir

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shared_dir = Path(cfg["data"]["shared_dir"])
    samples_dir = Path(cfg["data"]["samples_dir"])

    # Load sample IDs from split file
    with open(args.split_file) as f:
        all_ids = [line.strip() for line in f if line.strip()]
    sample_ids = [sid for sid in all_ids if (samples_dir / sid).exists()]
    skipped = len(all_ids) - len(sample_ids)
    if skipped:
        print(f"[Bridge] Skipping {skipped} IDs not found in {samples_dir}")
    print(f"[Bridge] Processing {len(sample_ids)} samples → {output_dir}")

    # Load shared mesh (needed for ROI and FEM stages)
    print(f"[Bridge] Loading mesh from {shared_dir}")
    nodes, elements = load_shared_mesh(shared_dir)
    print(f"  Nodes: {nodes.shape[0]}, Tets: {elements.shape[0]}")

    # Stage 1 inference
    if not args.skip_inference:
        print("\n[Bridge] === Stage 1 Inference ===")
        run_inference_stage(args, cfg, sample_ids)
    else:
        print("\n[Bridge] Skipping Stage 1 inference (--skip_inference)")

    # ROI derivation
    roi_results = []
    if not args.skip_roi:
        print("\n[Bridge] === ROI Derivation ===")
        roi_results = run_roi_stage(args, nodes, elements, sample_ids)
    else:
        print("\n[Bridge] Skipping ROI derivation (--skip_roi)")

    # 8D prior features (optional)
    if args.compute_prior_cache:
        print("\n[Bridge] === FEM Prior Features ===")
        run_fem_stage(args, nodes, elements, sample_ids)

    # Summary
    print_summary(roi_results, output_dir)


if __name__ == "__main__":
    main()
