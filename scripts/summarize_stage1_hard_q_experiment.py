#!/usr/bin/env python3
"""Summarize frozen direct-Stage1 hard-Q development results without reselection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SEED = 20260901
ORACLE_DICE = 0.73534
TARGET_DICE = 0.73


def load_test(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text())
    if result.get("split") != "test" or result.get("n_samples") != 300:
        raise RuntimeError(f"{path} is not a complete development-test300 result")
    if result.get("coarse_source") != "stage1" or result.get("mode") != "hard":
        raise RuntimeError(f"{path} is not a direct Stage1 hard-Q result")
    if result.get("freeze_receipt") is None:
        raise RuntimeError(f"{path} was evaluated without a val freeze receipt")
    if result.get("freeze_receipt_sha256") is None:
        raise RuntimeError(f"{path} does not identify the freeze receipt content")
    return result


def paired_difference(
    first: dict[str, Any], second: dict[str, Any], key: str = "final_dice"
) -> dict[str, Any]:
    left = {row["sample_id"]: float(row[key]) for row in first["per_sample"]}
    right = {row["sample_id"]: float(row[key]) for row in second["per_sample"]}
    if left.keys() != right.keys():
        raise RuntimeError("Development results have different sample IDs")
    delta = np.asarray([right[sid] - left[sid] for sid in left], dtype=np.float64)
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(delta), size=(10_000, len(delta)))
    sampled = delta[indices].mean(axis=1)
    return {
        "mean": float(delta.mean()),
        "paired_bootstrap_95_ci": [
            float(np.quantile(sampled, 0.025)),
            float(np.quantile(sampled, 0.975)),
        ],
        "bootstrap_draws": 10_000,
        "seed": SEED,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--selected-test", type=Path, required=True)
    parser.add_argument("--no-views-test", type=Path)
    parser.add_argument("--views-test", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    receipt = json.loads(args.receipt.read_text())
    if receipt.get("status") != "frozen_on_val300":
        raise RuntimeError("Selection receipt is not frozen on val300")
    selected = load_test(args.selected_test)
    receipt_sha = hashlib.sha256(args.receipt.read_bytes()).hexdigest()
    if selected["freeze_receipt_sha256"] != receipt_sha:
        raise RuntimeError("Selected result does not match the supplied receipt content")
    if selected["checkpoint_sha256"] != receipt["selected_checkpoint_sha256"]:
        raise RuntimeError("Selected development result is not the val-selected checkpoint")

    summary = selected["summary"]
    stage1_dice = float(summary["stage1_dice"])
    final_dice = float(summary["final_dice"])
    oracle_headroom = ORACLE_DICE - stage1_dice
    result: dict[str, Any] = {
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "stage1_dice": stage1_dice,
        "final_dice": final_dice,
        "hard_q_delta_dice": final_dice - stage1_dice,
        "hard_q_paired_bootstrap": selected["paired_delta_dice"],
        "target_dice": TARGET_DICE,
        "reaches_0.73": final_dice >= TARGET_DICE,
        "oracle_dice": ORACLE_DICE,
        "distance_to_oracle": ORACLE_DICE - final_dice,
        "oracle_dice_space_recovered": (
            (final_dice - stage1_dice) / oracle_headroom
            if oracle_headroom > 0
            else float("nan")
        ),
        "reported_metrics": summary,
        "sealed_confirmation_accessed": False,
    }
    if (args.no_views_test is None) != (args.views_test is None):
        raise RuntimeError("Provide both --no-views-test and --views-test, or neither")
    if args.no_views_test is not None:
        no_views = load_test(args.no_views_test)
        views = load_test(args.views_test)
        if not (
            no_views["freeze_receipt_sha256"]
            == views["freeze_receipt_sha256"]
            == selected["freeze_receipt_sha256"]
        ):
            raise RuntimeError("Development results do not share one frozen receipt")
        direct_views = paired_difference(
            no_views, views
        )
        direct_views["passes_preregistered_gate"] = (
            direct_views["mean"] >= 0.002
            and direct_views["paired_bootstrap_95_ci"][0] > 0
        )
        result["direct_views_development_comparison"] = direct_views

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")


if __name__ == "__main__":
    main()
