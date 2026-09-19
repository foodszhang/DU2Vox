#!/usr/bin/env python3
"""Dense canonical-domain evaluation for a frozen ESCB/baseline checkpoint."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.bridge.canonical_cross_discretization import (
    CanonicalCrossDiscretization,
)
from du2vox.evaluation.error_structured import evaluate_dense, save_dense_result
from scripts.train_error_structured_bridge import (
    build_dataset,
    build_model,
    load_ids,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=["escb", "baseline"], required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-predictions-dir", type=Path)
    parser.add_argument("--batch-points", type=int, default=32768)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    os.environ["DU2VOX_SHARED_DIR"] = str(cfg["data"]["shared_dir"])
    if cfg["data"].get("allow_stale_frame_manifest", False):
        os.environ["DU2VOX_ALLOW_STALE_FRAME_MANIFEST"] = "1"
    if cfg["data"].get("frame_manifest_sha256"):
        os.environ["DU2VOX_FRAME_MANIFEST_SHA256"] = str(
            cfg["data"]["frame_manifest_sha256"]
        )
    ids = load_ids(cfg["data"][f"{args.split}_split"])
    if args.max_samples is not None:
        ids = ids[: args.max_samples]
    dataset = build_dataset(
        cfg,
        args.split,
        ids,
        int(cfg["training"]["n_query_points"]),
    )
    canonical = CanonicalCrossDiscretization(
        cfg["data"]["operator_cache"],
        shared_dir=cfg["data"]["shared_dir"],
        factorize=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, args.model, cfg["data"]["shared_dir"]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("model_type") != args.model:
        raise RuntimeError(
            f"Checkpoint model_type={checkpoint.get('model_type')} != {args.model}"
        )
    model.load_state_dict(checkpoint["model"])
    result = evaluate_dense(
        model,
        dataset,
        canonical,
        device,
        batch_points=args.batch_points,
        save_predictions_dir=args.save_predictions_dir,
    )
    result["checkpoint"] = str(args.checkpoint.resolve())
    result["checkpoint_dense_val_dice"] = checkpoint.get("dense_val_dice")
    result["split"] = args.split
    result["model"] = args.model
    save_dense_result(result, args.output)


if __name__ == "__main__":
    main()
