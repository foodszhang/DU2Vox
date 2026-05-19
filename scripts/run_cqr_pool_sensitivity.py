#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np


ROLE_NAMES = {0: "bg", 1: "core", 2: "halo", 3: "sentinel"}


def dice_at(pred: np.ndarray, gt: np.ndarray, thr: float = 0.5) -> float:
    p = pred >= thr
    g = gt >= thr
    return float(2 * (p & g).sum() / (p.sum() + g.sum() + 1e-8))


def mean_or_zero(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else 0.0


def summarize_dir(path: Path, pool_size: int, split: str, max_files: int | None) -> dict[str, float | int | str]:
    paths = sorted(path.glob("*.npz"))
    if max_files is not None:
        paths = paths[:max_files]
    gt_all = []
    fem_all = []
    role_all = []
    for npz_path in paths:
        with np.load(npz_path, allow_pickle=False) as data:
            valid = data["valid_mask"].astype(bool)
            prior = data["prior_ext"] if "prior_ext" in data.files else data["prior_8d"]
            fem = (prior[:, :4] * prior[:, 4:8]).sum(axis=1)
            role = data["role"] if "role" in data.files else np.zeros(len(valid), dtype=np.int64)
            gt_all.append(data["gt_values"][valid])
            fem_all.append(fem[valid])
            role_all.append(role[valid])

    gt = np.concatenate(gt_all) if gt_all else np.zeros((0,), dtype=np.float32)
    fem = np.concatenate(fem_all) if fem_all else np.zeros((0,), dtype=np.float32)
    role = np.concatenate(role_all) if role_all else np.zeros((0,), dtype=np.int64)
    row: dict[str, float | int | str] = {
        "pool_size": pool_size,
        "split": split,
        "max_samples": len(paths),
        "gt_pos_ratio_05": mean_or_zero(gt >= 0.5),
        "fem_pos_ratio_05": mean_or_zero(fem >= 0.5),
        "fem_dice_05": dice_at(fem, gt, 0.5),
    }
    total = max(len(role), 1)
    for rid, name in ROLE_NAMES.items():
        mask = role == rid
        row[f"{name}_gt_pos_05"] = mean_or_zero(gt[mask] >= 0.5)
        row[f"{name}_count_ratio"] = float(mask.sum() / total)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CQR pool-size sensitivity precompute and summary")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val"])
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--pool_sizes", nargs="+", type=int, required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--out_csv", default="results/cqr_pool_sensitivity.csv")
    args = parser.parse_args()

    rows = []
    for pool_size in args.pool_sizes:
        out_dir = Path(args.out_root) / f"{args.split}_{pool_size}"
        cmd = [
            sys.executable,
            "scripts/precompute_stage2_cqr.py",
            "--config",
            args.config,
            "--split",
            args.split,
            "--output_dir",
            str(out_dir),
            "--n_query_points",
            str(pool_size),
            "--max_samples",
            str(args.max_samples),
            "--overwrite",
        ]
        subprocess.run(cmd, check=True)
        subprocess.run(
            [sys.executable, "scripts/diagnose_cqr_npz.py", "--dir", str(out_dir), "--max_files", str(args.max_samples)],
            check=True,
        )
        rows.append(summarize_dir(out_dir, pool_size, args.split, args.max_samples))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CQR] wrote pool sensitivity summary: {out_csv}")


if __name__ == "__main__":
    main()
