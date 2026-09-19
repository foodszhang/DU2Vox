#!/usr/bin/env python3
"""Paired P0 frozen/joint comparison with a deterministic bootstrap CI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_rows(path: Path) -> tuple[dict, dict[str, dict]]:
    payload = json.loads(path.read_text())
    return payload, {row["sample_id"]: row for row in payload["per_sample"]}


def paired_delta(
    frozen: dict[str, dict], joint: dict[str, dict], key: str
) -> np.ndarray:
    ids = sorted(set(frozen) & set(joint))
    if len(ids) != len(frozen) or len(ids) != len(joint):
        raise RuntimeError("Frozen and joint sample identities differ")
    return np.asarray([joint[sid][key] - frozen[sid][key] for sid in ids], dtype=np.float64)


def bootstrap_ci(values: np.ndarray, seed: int = 20260902) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(10000, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-val", type=Path, required=True)
    parser.add_argument("--joint-val", type=Path, required=True)
    parser.add_argument("--frozen-test", type=Path)
    parser.add_argument("--joint-test", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=Path("diagnosis/p0_joint_vs_frozen_v4_comparison.json"),
    )
    args = parser.parse_args()
    frozen_val_payload, frozen_val = load_rows(args.frozen_val)
    joint_val_payload, joint_val = load_rows(args.joint_val)
    metrics = ("fem_dice", "fem_hd95", "fem_localization_error", "fem_mse", "final_dice")
    validation = {}
    for key in metrics:
        delta = paired_delta(frozen_val, joint_val, key)
        validation[key] = {
            "mean_delta": float(delta.mean()),
            "paired_bootstrap_95_ci": bootstrap_ci(delta),
        }
    result = {
        "n_validation": len(frozen_val),
        "validation": validation,
        "frozen_validation_summary": frozen_val_payload["summary"],
        "joint_validation_summary": joint_val_payload["summary"],
        "strong_evidence_rules": {
            "fem_dice_minimum_delta": 0.002,
            "fem_dice_ci_lower_gt_zero": True,
            "hd95_not_worse": True,
            "localization_not_worse": True,
            "development_test_same_direction": True,
        },
        "confirmation_data_used": False,
    }
    if args.frozen_test is not None and args.joint_test is not None:
        frozen_test_payload, frozen_test = load_rows(args.frozen_test)
        joint_test_payload, joint_test = load_rows(args.joint_test)
        result["n_development_test"] = len(frozen_test)
        result["development_test"] = {
            key: {"mean_delta": float(paired_delta(frozen_test, joint_test, key).mean())}
            for key in metrics
        }
        result["frozen_development_test_summary"] = frozen_test_payload["summary"]
        result["joint_development_test_summary"] = joint_test_payload["summary"]
    val_dice = validation["fem_dice"]
    hd95 = validation["fem_hd95"]["mean_delta"]
    localization = validation["fem_localization_error"]["mean_delta"]
    same_direction = (
        result.get("development_test", {}).get("fem_dice", {}).get("mean_delta", -np.inf) > 0
    )
    result["strong_evidence_supported"] = bool(
        val_dice["mean_delta"] >= 0.002
        and val_dice["paired_bootstrap_95_ci"][0] > 0
        and hd95 <= 0
        and localization <= 0
        and same_direction
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    print(json.dumps({"validation": validation, "strong_evidence_supported": result["strong_evidence_supported"]}, indent=2))


if __name__ == "__main__":
    main()
