#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


ROLE_NAMES = {
    0: "bg",
    1: "core",
    2: "halo",
    3: "sentinel",
}


FIELDNAMES = [
    "sample_id",
    "n_valid",
    "gt_pos_ratio_05",
    "fem_pos_ratio_05",
    "fem_dice_05",
    "gt_mean",
    "fem_mean",
    "bg_count",
    "core_count",
    "halo_count",
    "sentinel_count",
    "bg_gt_pos_05",
    "core_gt_pos_05",
    "halo_gt_pos_05",
    "sentinel_gt_pos_05",
    "bg_fem_pos_05",
    "core_fem_pos_05",
    "halo_fem_pos_05",
    "sentinel_fem_pos_05",
    "bg_gt_mean",
    "core_gt_mean",
    "halo_gt_mean",
    "sentinel_gt_mean",
    "bg_fem_mean",
    "core_fem_mean",
    "halo_fem_mean",
    "sentinel_fem_mean",
]


def dice_at(pred: np.ndarray, gt: np.ndarray, thr: float = 0.5) -> float:
    p = pred >= thr
    g = gt >= thr
    return float(2 * (p & g).sum() / (p.sum() + g.sum() + 1e-8))


def mean_or_zero(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else 0.0


def summarize_one(path: Path) -> dict[str, float | int | str]:
    with np.load(path, allow_pickle=False) as data:
        gt = data["gt_values"]
        prior = data["prior_ext"] if "prior_ext" in data.files else data["prior_8d"]
        role = data["role"] if "role" in data.files else np.zeros(len(gt), dtype=np.int64)
        valid = data["valid_mask"].astype(bool)

        fem = (prior[:, :4] * prior[:, 4:8]).sum(axis=1)

        gt_valid = gt[valid]
        fem_valid = fem[valid]
        row: dict[str, float | int | str] = {
            "sample_id": path.stem,
            "n_valid": int(valid.sum()),
            "gt_pos_ratio_05": mean_or_zero(gt_valid >= 0.5),
            "fem_pos_ratio_05": mean_or_zero(fem_valid >= 0.5),
            "fem_dice_05": dice_at(fem_valid, gt_valid, 0.5),
            "gt_mean": mean_or_zero(gt_valid),
            "fem_mean": mean_or_zero(fem_valid),
        }

        for rid, name in ROLE_NAMES.items():
            mask = valid & (role == rid)
            gt_role = gt[mask]
            fem_role = fem[mask]
            row[f"{name}_count"] = int(mask.sum())
            row[f"{name}_gt_pos_05"] = mean_or_zero(gt_role >= 0.5)
            row[f"{name}_fem_pos_05"] = mean_or_zero(fem_role >= 0.5)
            row[f"{name}_gt_mean"] = mean_or_zero(gt_role)
            row[f"{name}_fem_mean"] = mean_or_zero(fem_role)

        return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize CQR precomputed NPZ files to CSV")
    parser.add_argument("--dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_files", type=int, default=None)
    args = parser.parse_args()

    paths = sorted(Path(args.dir).glob("*.npz"))
    if args.max_files is not None:
        paths = paths[: args.max_files]
    if not paths:
        raise SystemExit(f"No npz files found in {args.dir}")

    rows = [summarize_one(path) for path in paths]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[CQR] wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
