#!/usr/bin/env python3
"""Select the direct Stage1 hard-Q candidate on val300 and write its receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


SEED = 20260901
BOOTSTRAP_DRAWS = 10_000


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text())
    if result.get("split") != "val" or result.get("n_samples") != 300:
        raise RuntimeError(f"{path} is not a complete val300 result")
    if result.get("mode") != "hard" or result.get("coarse_source") != "stage1":
        raise RuntimeError(f"{path} is not a direct Stage1 hard-Q result")
    for key in ("checkpoint", "config"):
        if result.get(f"{key}_sha256") != sha256_file(result[key]):
            raise RuntimeError(f"{path}: {key} SHA256 mismatch")
    return result


def aligned_delta(first: dict[str, Any], second: dict[str, Any]) -> np.ndarray:
    left = {row["sample_id"]: float(row["final_dice"]) for row in first["per_sample"]}
    right = {row["sample_id"]: float(row["final_dice"]) for row in second["per_sample"]}
    if left.keys() != right.keys():
        raise RuntimeError("Compared validation results have different sample IDs")
    return np.asarray([right[sid] - left[sid] for sid in left], dtype=np.float64)


def bootstrap(values: np.ndarray) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "paired_bootstrap_95_ci": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "seed": SEED,
    }


def assert_matched_baselines(no_views: dict[str, Any], views: dict[str, Any]) -> None:
    for key in ("initial_decoder_sha256", "initial_encoder_sha256", "parameter_counts"):
        if no_views.get(key) != views.get(key):
            raise RuntimeError(f"Baseline arms are not matched on {key}")
    if no_views.get("training_order_sha256") != views.get("training_order_sha256"):
        raise RuntimeError("Baseline arms used different sample orders")
    if no_views["training_lr"] != 1e-4 or views["training_lr"] != 1e-4:
        raise RuntimeError("Baseline learning rate must be 1e-4")
    if no_views["training_epochs"] != 5 or views["training_epochs"] != 5:
        raise RuntimeError("Baseline training budget must be 5 epochs")
    no_views_cfg = yaml.safe_load(Path(no_views["config"]).read_text())
    views_cfg = yaml.safe_load(Path(views["config"]).read_text())
    for cfg in (no_views_cfg, views_cfg):
        cfg["experiment"].pop("name", None)
        cfg["model"].pop("use_views", None)
    if no_views_cfg != views_cfg:
        raise RuntimeError("Baseline configs differ outside name and use_views")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-views", type=Path, required=True)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--tuning", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    no_views = load_result(args.no_views)
    views = load_result(args.views)
    assert_matched_baselines(no_views, views)
    baseline_winner = max((no_views, views), key=lambda item: item["summary"]["final_dice"])
    tuning = [load_result(path) for path in args.tuning]
    expected_lrs = {3e-5, 3e-4}
    if {item["training_lr"] for item in tuning} != expected_lrs:
        raise RuntimeError("Tuning trials must use exactly LR 3e-5 and 3e-4")
    for item in tuning:
        if item["training_epochs"] != 10:
            raise RuntimeError("Each tuning trial must declare 10 epochs")
        if item["use_views"] != baseline_winner["use_views"]:
            raise RuntimeError("Only the validation-best baseline arm may be tuned")
        if item["initial_decoder_sha256"] != baseline_winner["initial_decoder_sha256"]:
            raise RuntimeError("Tuning did not restart from the matched decoder initialization")
        if item["initial_encoder_sha256"] != baseline_winner["initial_encoder_sha256"]:
            raise RuntimeError("Tuning did not restart from the matched encoder initialization")
        winner_cfg = yaml.safe_load(Path(baseline_winner["config"]).read_text())
        trial_cfg = yaml.safe_load(Path(item["config"]).read_text())
        for candidate_cfg in (winner_cfg, trial_cfg):
            candidate_cfg["experiment"].pop("name", None)
            candidate_cfg["training"].pop("lr", None)
            candidate_cfg["training"].pop("epochs", None)
        if winner_cfg != trial_cfg:
            raise RuntimeError("Tuning changed settings beyond LR and epoch budget")

    direct_view = bootstrap(aligned_delta(no_views, views))
    low = direct_view["paired_bootstrap_95_ci"][0]
    direct_view["passes_preregistered_gate"] = direct_view["mean"] >= 0.002 and low > 0
    candidates = [no_views, views, *tuning]
    selected = max(candidates, key=lambda item: item["summary"]["final_dice"])
    receipt = {
        "status": "frozen_on_val300",
        "selection_metric": "final_dice_at_fixed_threshold_0.5",
        "selected_checkpoint": selected["checkpoint"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "selected_config": selected["config"],
        "selected_config_sha256": selected["config_sha256"],
        "selected_val_dice": selected["summary"]["final_dice"],
        "selected_use_views": selected["use_views"],
        "direct_views_baseline_comparison": direct_view,
        "candidates": [
            {
                "checkpoint": item["checkpoint"],
                "checkpoint_sha256": item["checkpoint_sha256"],
                "config": item["config"],
                "config_sha256": item["config_sha256"],
                "use_views": item["use_views"],
                "lr": item["training_lr"],
                "epochs": item["training_epochs"],
                "val_dice": item["summary"]["final_dice"],
            }
            for item in candidates
        ],
        "development_test_accessed_before_freeze": False,
        "sealed_confirmation_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
