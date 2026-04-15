#!/usr/bin/env python3
"""
Evaluate trained MS-GDUN model on FMT-SimGen dataset.

Supports per-foci-type and per-depth-tier metric aggregation when dataset_manifest.json is available.

Usage:
    python scripts/eval_stage1.py --config configs/stage1/gcain_full_1000.yaml --checkpoint checkpoints/best.pth
    python scripts/eval_stage1.py --config configs/stage1/gcain_full_1000.yaml --checkpoint checkpoints/best.pth --output results/baseline.json
"""

import argparse
import json
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader

from du2vox.models.stage1.gcain import GCAIN_full
from du2vox.data.dataset import FMTSimGenDataset
from du2vox.evaluation.metrics import evaluate_batch, summarize_metrics
from du2vox.evaluation.per_foci import (
    load_manifest, get_sample_names, grouped_evaluation,
)


def evaluate(
    checkpoint_path: str | Path,
    split: str = "val",
    config_path: str | Path | None = None,
    output_path: Path | None = None,
):
    if config_path is None:
        config_path = Path("configs/stage1/gcain_full_1000.yaml")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    shared_dir = Path(data_cfg["shared_dir"])
    samples_dir = Path(data_cfg["samples_dir"])
    splits_dir = Path(data_cfg["splits_dir"])

    split_file = splits_dir / f"{split}.txt"
    dataset = FMTSimGenDataset(
        shared_dir=shared_dir,
        samples_dir=samples_dir,
        split_file=split_file,
    )
    # Use batch_size=1 so metrics align 1:1 with sample names for per-foci grouping
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    A = dataset.A.cuda()
    L = dataset.L.cuda()
    L0 = dataset.L0.cuda()
    L1 = dataset.L1.cuda()
    L2 = dataset.L2.cuda()
    L3 = dataset.L3.cuda()
    knn_idx = dataset.knn_idx.cuda()
    sens_w = dataset.sens_w.cuda()
    nodes = dataset.nodes.cuda()

    model = GCAIN_full(
        L=L, A=A,
        L0=L0, L1=L1, L2=L2, L3=L3,
        knn_idx=knn_idx,
        sens_w=sens_w,
        num_layer=model_cfg["num_layer"],
        feat_dim=model_cfg["feat_dim"],
    ).cuda()

    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Load manifest for per-foci/depth grouping
    manifest = load_manifest(samples_dir)
    sample_names = get_sample_names(split_file) if manifest else None

    n_nodes = dataset.nodes.shape[0]
    all_metrics = []
    all_sample_names = []

    with torch.no_grad():
        for batch in loader:
            b = batch["b"].cuda()
            gt = batch["gt"].cuda()
            X0 = torch.zeros(b.size(0), n_nodes, 1, device="cuda")
            pred = model(X0, b)
            batch_metrics = evaluate_batch(pred, gt, nodes)
            all_metrics.append(batch_metrics)

            # Track sample names for this batch
            if sample_names is not None:
                batch_size = b.size(0)
                all_sample_names.extend(sample_names[:batch_size])
                sample_names = sample_names[batch_size:]

    # Run grouped evaluation
    if manifest:
        grouped_evaluation(all_metrics, all_sample_names, manifest, output_path)
    else:
        # Just print overall metrics
        summary = summarize_metrics(all_metrics)
        print(f"\n=== {split.upper()} Results ===")
        print(f"Dice:            {summary['dice']:.4f}")
        print(f"Dice_bin@0.3:    {summary['dice_bin_0.3']:.4f}")
        print(f"Dice_bin@0.1:    {summary['dice_bin_0.1']:.4f}")
        print(f"Recall@0.1:      {summary['recall_0.1']:.4f}")
        print(f"Recall@0.3:      {summary['recall_0.3']:.4f}")
        print(f"Precision@0.3:   {summary['precision_0.3']:.4f}")
        print(f"Location Error:  {summary['location_error']:.4f} mm")
        print(f"MSE:             {summary['mse']:.6f}")

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump({"overall": summary, "n_samples": len(all_metrics)}, f, indent=2)
            print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/stage1/gcain_full_1000.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None
    evaluate(args.checkpoint, args.split, args.config, output_path)
