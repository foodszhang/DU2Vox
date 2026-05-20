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


def dice_at(pred, gt, thr=0.5):
    p = pred >= thr
    g = gt >= thr
    return 2 * (p & g).sum() / (p.sum() + g.sum() + 1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--max_files", type=int, default=20)
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    paths = sorted(Path(args.dir).glob("*.npz"))[: args.max_files]
    if not paths:
        raise SystemExit(f"No npz files found in {args.dir}")

    all_rows = []
    for p in paths:
        d = np.load(p, allow_pickle=False)
        gt = d["gt_values"]
        prior = d["prior_ext"] if "prior_ext" in d.files else d["prior_8d"]
        role = d["role"] if "role" in d.files else np.zeros(len(gt), dtype=np.int64)
        valid = d["valid_mask"].astype(bool)

        fem = (prior[:, :4] * prior[:, 4:8]).sum(axis=1)

        print(f"\n[{p.name}] valid={valid.sum()}/{len(valid)}")
        print(f"  gt_pos@0.5={((gt[valid] >= 0.5).mean() if valid.any() else 0):.4f}")
        print(f"  fem_pos@0.5={((fem[valid] >= 0.5).mean() if valid.any() else 0):.4f}")
        print(f"  fem_dice@0.5={dice_at(fem[valid], gt[valid], 0.5):.4f}")
        print(f"  gt_mean={gt[valid].mean():.4f}, fem_mean={fem[valid].mean():.4f}")

        for rid in [0, 1, 2, 3]:
            m = valid & (role == rid)
            if m.sum() == 0:
                continue
            print(
                f"  role={ROLE_NAMES[rid]:8s} "
                f"n={m.sum():6d} "
                f"gt_pos={((gt[m] >= 0.5).mean()):.4f} "
                f"fem_pos={((fem[m] >= 0.5).mean()):.4f} "
                f"gt_mean={gt[m].mean():.4f} "
                f"fem_mean={fem[m].mean():.4f}"
            )

        all_rows.append((gt[valid], fem[valid], role[valid]))

    gt_all = np.concatenate([r[0] for r in all_rows])
    fem_all = np.concatenate([r[1] for r in all_rows])
    role_all = np.concatenate([r[2] for r in all_rows])

    print("\n[Aggregate]")
    print(f"  n={len(gt_all)}")
    print(f"  gt_pos@0.5={(gt_all >= 0.5).mean():.4f}")
    print(f"  fem_pos@0.5={(fem_all >= 0.5).mean():.4f}")
    print(f"  fem_dice@0.5={dice_at(fem_all, gt_all, 0.5):.4f}")
    print("  threshold sweep for FEM:")
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        print(f"    thr={thr:.1f} dice={dice_at(fem_all, gt_all, thr):.4f}")

    print("\n  role aggregate:")
    rows = [
        {
            "role": "all",
            "n": len(gt_all),
            "gt_pos_05": float((gt_all >= 0.5).mean()),
            "fem_pos_05": float((fem_all >= 0.5).mean()),
            "gt_mean": float(gt_all.mean()),
            "fem_mean": float(fem_all.mean()),
            "fem_dice_05": float(dice_at(fem_all, gt_all, 0.5)),
        }
    ]
    for rid in [0, 1, 2, 3]:
        m = role_all == rid
        if m.sum() == 0:
            continue
        rows.append(
            {
                "role": ROLE_NAMES[rid],
                "n": int(m.sum()),
                "gt_pos_05": float((gt_all[m] >= 0.5).mean()),
                "fem_pos_05": float((fem_all[m] >= 0.5).mean()),
                "gt_mean": float(gt_all[m].mean()),
                "fem_mean": float(fem_all[m].mean()),
                "fem_dice_05": float(dice_at(fem_all[m], gt_all[m], 0.5)),
            }
        )
        print(
            f"    {ROLE_NAMES[rid]:8s} "
            f"n={m.sum():7d} "
            f"gt_pos={(gt_all[m] >= 0.5).mean():.4f} "
            f"fem_pos={(fem_all[m] >= 0.5).mean():.4f} "
            f"gt_mean={gt_all[m].mean():.4f} "
            f"fem_mean={fem_all[m].mean():.4f}"
        )

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["role", "n", "gt_pos_05", "fem_pos_05", "gt_mean", "fem_mean", "fem_dice_05"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[Diagnose] wrote {out_path}")


if __name__ == "__main__":
    main()
