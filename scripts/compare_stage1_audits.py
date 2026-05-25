#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "method",
    "checkpoint",
    "mesh_dice05",
    "mesh_dice03",
    "soft_dice",
    "precision05",
    "recall05",
    "pred_pos_ratio05",
    "pred_mean",
    "pred_max",
]


def load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def row_from_audit(method: str, data: dict[str, Any]) -> dict[str, Any]:
    mesh = data.get("mesh", {})
    summary = data.get("per_sample_summary", {})
    return {
        "method": method,
        "checkpoint": data.get("checkpoint", ""),
        "mesh_dice05": mesh.get("dice_bin_0.5", ""),
        "mesh_dice03": mesh.get("dice_bin_0.3", ""),
        "soft_dice": mesh.get("dice", ""),
        "precision05": mesh.get("precision_pred05_gt05", ""),
        "recall05": mesh.get("recall_pred05_gt05", ""),
        "pred_pos_ratio05": summary.get("pred_pos_ratio_05", ""),
        "pred_mean": summary.get("pred_mean", ""),
        "pred_max": summary.get("pred_max", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Stage1 audit JSON files")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()
    if len(args.inputs) != len(args.names):
        raise SystemExit("--inputs and --names must have the same length")
    rows = [row_from_audit(name, load_json(Path(path))) for name, path in zip(args.names, args.inputs)]
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CompareStage1] wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
