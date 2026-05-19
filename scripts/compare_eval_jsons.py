#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


BASE_FIELDS = [
    "name",
    "s2_dice_05",
    "fem_dice_05",
    "delta_dice_05",
    "s2_iou_05",
    "s2_precision_05",
    "s2_recall_05",
    "mse_s2",
    "residual_norm",
    "gt_pos_ratio_05",
    "s2_pos_ratio_05",
    "fem_pos_ratio_05",
]

FOCI_FIELDS = [
    "foci1_s2_dice_05",
    "foci2_s2_dice_05",
    "foci3_s2_dice_05",
    "foci1_delta_dice_05",
    "foci2_delta_dice_05",
    "foci3_delta_dice_05",
]


def load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def build_row(name: str, data: dict[str, Any]) -> dict[str, Any]:
    overall = data.get("overall", {})
    row = {"name": name}
    for field in BASE_FIELDS:
        if field == "name":
            continue
        row[field] = overall.get(field, "")

    by_foci = data.get("by_foci", {}) or {}
    for foci in [1, 2, 3]:
        group = by_foci.get(str(foci), {})
        row[f"foci{foci}_s2_dice_05"] = group.get("s2_dice_05", "")
        row[f"foci{foci}_delta_dice_05"] = group.get("delta_dice_05", "")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare unified Stage2 eval JSON files")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    if len(args.inputs) != len(args.names):
        raise SystemExit("--inputs and --names must have the same length")

    rows = [build_row(name, load_json(Path(path))) for name, path in zip(args.names, args.inputs)]
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BASE_FIELDS + FOCI_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Compare] wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
