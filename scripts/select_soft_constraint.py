#!/usr/bin/env python3
"""Select the P0 soft-constraint checkpoint from validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument(
        "--output", type=Path,
        default=Path("diagnosis/p0_soft_constraint_selection.json"),
    )
    args = parser.parse_args()
    candidates = []
    for run in args.run:
        checkpoint_path = run / "checkpoints" / "best_dense_val_dice.pth"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("mode") != "soft":
            raise RuntimeError(f"{checkpoint_path} is not a soft checkpoint")
        history = json.loads((run / "history.json").read_text())
        epoch = int(checkpoint["epoch"])
        row = next(item for item in history if int(item["epoch"]) == epoch)
        candidates.append(
            {
                "lambda_coarse_soft": float(checkpoint["config"]["loss"]["lambda_coarse_soft"]),
                "checkpoint": str(checkpoint_path.resolve()),
                "epoch": epoch,
                "val_dice": float(checkpoint["dense_val_dice"]),
                "val_coarse_leakage": float(row["val_coarse_leakage"]),
            }
        )
    # Primary: Dice. Within 1e-4 of the maximum: lower leakage.
    maximum_dice = max(item["val_dice"] for item in candidates)
    tied = [item for item in candidates if maximum_dice - item["val_dice"] < 1e-4]
    selected = min(tied, key=lambda item: item["val_coarse_leakage"])
    result = {
        "selection_split": "validation",
        "rule": "maximum Dice; if difference <1e-4, lower coarse leakage",
        "candidates": sorted(candidates, key=lambda item: item["lambda_coarse_soft"]),
        "selected": selected,
        "development_test_used": False,
        "confirmation_data_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
