#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.eval_stage2_unified import (  # noqa: E402
    METRIC_KEYS,
    build_model,
    evaluate_sample,
    get_num_foci,
    load_checkpoint,
    load_split,
    mean_dict,
    run_model_on_sample,
)
from du2vox.utils.frame import FrameManifest  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def sample_groups(samples_dir: Path | None, sample_id: str) -> dict[str, Any]:
    if samples_dir is None:
        return {"depth_tier": "unknown", "shape_class": "unknown"}
    meta = read_json(samples_dir / sample_id / "meta.json")
    tumor = read_json(samples_dir / sample_id / "tumor_params.json")
    merged = {**tumor, **meta}
    return {
        "depth_tier": merged.get("depth_tier", merged.get("real_depth_tier", "unknown")),
        "shape_class": merged.get("shape_class", merged.get("shape", "unknown")),
    }


def summarize_group(rows: list[dict[str, Any]], group_key: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    values = sorted({str(row.get(group_key, "unknown")) for row in rows})
    for value in values:
        group = [row for row in rows if str(row.get(group_key, "unknown")) == value]
        out[value] = mean_dict(group, METRIC_KEYS)
        out[value]["n_samples"] = len(group)
        out[value]["assd"] = None
        out[value]["hd95"] = None
    return out


def write_group_csv(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["group_type", "group", "n_samples", *METRIC_KEYS, "assd", "hd95"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for group_type in ["by_foci", "by_depth", "by_shape"]:
            for group, metrics in result[group_type].items():
                writer.writerow({"group_type": group_type, "group": group, **metrics})


def main() -> None:
    parser = argparse.ArgumentParser(description="DU2Vox v2 grouped Stage2 evaluator")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_points", type=int, default=8192)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if cfg.get("data", {}).get("shared_dir"):
        os.environ["DU2VOX_SHARED_DIR"] = str(cfg["data"]["shared_dir"])

    split_file = cfg["data"][f"{args.split}_split"]
    sample_ids = load_split(split_file)
    if args.max_samples is not None:
        sample_ids = sample_ids[: args.max_samples]

    precomputed_dir = Path(cfg["data"].get(f"precomputed_{args.split}_dir", ""))
    if not precomputed_dir.exists():
        raise SystemExit(f"precomputed_{args.split}_dir not found: {precomputed_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, view_encoder = build_model(cfg, device)
    load_checkpoint(Path(args.checkpoint), model, view_encoder, device)
    model.eval()
    if view_encoder is not None:
        view_encoder.eval()

    samples_dir = Path(cfg["data"]["samples_dir"]) if cfg["data"].get("samples_dir") else None
    frame = FrameManifest.load(cfg["data"]["shared_dir"]) if cfg["data"].get("shared_dir") else None

    print(f"[EvalV2] split={args.split}, split_file={split_file}, samples={len(sample_ids)}")
    print(f"[EvalV2] precomputed_dir={precomputed_dir}")

    rows = []
    with torch.no_grad():
        for sample_id in sample_ids:
            npz_path = precomputed_dir / f"{sample_id}.npz"
            if not npz_path.exists():
                print(f"[WARN] missing npz: {npz_path}")
                continue
            data = dict(np.load(npz_path, allow_pickle=False))
            d_hat, fem, residual, valid, diagnostics = run_model_on_sample(
                model=model,
                view_encoder=view_encoder,
                cfg=cfg,
                data=data,
                samples_dir=samples_dir,
                frame=frame,
                sample_id=sample_id,
                batch_points=args.batch_points,
                role_subset="all",
                device=device,
            )
            row = evaluate_sample(
                sample_id,
                "all",
                get_num_foci(precomputed_dir, samples_dir, sample_id),
                data,
                d_hat,
                fem,
                residual,
                valid,
                diagnostics,
            )
            row.update(sample_groups(samples_dir, sample_id))
            rows.append(row)

    result = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "n_samples": len(rows),
        "overall": {**mean_dict(rows, METRIC_KEYS), "assd": None, "hd95": None},
        "by_foci": summarize_group(rows, "num_foci"),
        "by_depth": summarize_group(rows, "depth_tier"),
        "by_shape": summarize_group(rows, "shape_class"),
        "per_sample": rows,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    write_group_csv(Path(args.out_csv), result)

    overall = result["overall"]
    print(f"[EvalV2] wrote JSON: {out_json}")
    print(f"[EvalV2] wrote CSV: {args.out_csv}")
    print(
        f"[EvalV2] dice={overall['s2_dice_05']:.4f}, "
        f"iou={overall['s2_iou_05']:.4f}, precision={overall['s2_precision_05']:.4f}, "
        f"recall={overall['s2_recall_05']:.4f}"
    )


if __name__ == "__main__":
    main()
