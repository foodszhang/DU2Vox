#!/usr/bin/env python3
"""10,000-draw paired bootstrap for the frozen LPR continuous benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PRIMARY = {
    "ssim3d": "higher",
    "psnr_db": "higher",
    "ccc": "higher",
    "relative_l2": "lower",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()

    report = json.loads(args.input.read_text())
    if report.get("n_samples") != 300:
        raise RuntimeError("Paired bootstrap requires the complete aligned 300 cases")
    comparisons = [("M1", "M0"), ("M2", "M1"), ("M2", "M3")]
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, 300, size=(args.draws, 300))
    output = {}
    for candidate, baseline in comparisons:
        if candidate not in report["methods"] or baseline not in report["methods"]:
            raise RuntimeError(f"Missing {candidate} or {baseline} in benchmark report")
        comparison = {}
        for metric, direction in PRIMARY.items():
            candidate_values = np.asarray(
                [row["methods"][candidate][metric] for row in report["per_sample"]]
            )
            baseline_values = np.asarray(
                [row["methods"][baseline][metric] for row in report["per_sample"]]
            )
            delta = (
                candidate_values - baseline_values
                if direction == "higher"
                else baseline_values - candidate_values
            )
            sampled = delta[indices].mean(axis=1)
            comparison[metric] = {
                "directional_improvement_mean": float(delta.mean()),
                "paired_bootstrap_95_ci": [
                    float(np.quantile(sampled, 0.025)),
                    float(np.quantile(sampled, 0.975)),
                ],
                "probability_improvement_le_zero": float(np.mean(sampled <= 0.0)),
            }
        output[f"{candidate}_vs_{baseline}"] = comparison
    result = {
        "input": str(args.input.resolve()),
        "draws": args.draws,
        "seed": args.seed,
        "difference_orientation": "positive always favors the first method",
        "comparisons": output,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
