#!/usr/bin/env python3
"""
Evaluate trained MS-GDUN model on FMT-SimGen dataset.

Supports per-foci-type metric aggregation when dataset_manifest.json is available.

Usage:
    python scripts/eval_stage1.py --config configs/stage1/gcain_full.yaml --checkpoint checkpoints/best.pth
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
    load_manifest, get_sample_names, group_metrics_by_foci,
    format_metrics_table,
)


def evaluate(checkpoint_path: str | Path, split: str = "val", config_path: str | Path | None = None):
    if config_path is None:
        config_path = Path("configs/stage1/gcain_full.yaml")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    paths_cfg = cfg["paths"]

    shared_dir = Path(paths_cfg["shared_dir"])
    samples_dir = Path(paths_cfg["samples_dir"])
    splits_dir = Path(paths_cfg["splits_dir"])

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
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load manifest for per-foci grouping
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

    summary = summarize_metrics(all_metrics)
    print(f"\n=== {split.upper()} Results ===")
    print(f"Dice:            {summary['dice']:.4f}")
    print(f"Location Error:  {summary['location_error']:.4f} mm")
    print(f"MSE:             {summary['mse']:.6f}")

    # Per-foci breakdown
    if manifest and all_sample_names:
        # Group by foci
        metrics_by_foci = group_metrics_by_foci(all_metrics, all_sample_names, manifest)

        # Compute summaries
        overall = summarize_metrics(all_metrics)
        foci_summaries = {}
        for n in [1, 2, 3]:
            if metrics_by_foci[n]:
                foci_summaries[n] = summarize_metrics(metrics_by_foci[n])
            else:
                foci_summaries[n] = None

        print(f"\n=== Per-Foci Breakdown ===")
        print(format_metrics_table(overall, foci_summaries))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/stage1/gcain_full.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    args = parser.parse_args()
    evaluate(args.checkpoint, args.split, args.config)
