#!/usr/bin/env python3
"""Quantify V1/V3/V4 FEM-state correction agreement on development validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator > 1e-12 else 0.0


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--v1-predictions-dir", type=Path, required=True)
    parser.add_argument("--v3-predictions-dir", type=Path, required=True)
    parser.add_argument("--v4-predictions-dir", type=Path, required=True)
    parser.add_argument("--v4-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ids = [line.strip() for line in args.split.read_text().splitlines() if line.strip()]
    correction_cosines: dict[str, list[float]] = {
        "v1_v3": [],
        "v4_v1": [],
        "v4_v3": [],
    }
    for sid in ids:
        archives = [
            np.load(directory / f"{sid}.npz")
            for directory in (
                args.v1_predictions_dir,
                args.v3_predictions_dir,
                args.v4_predictions_dir,
            )
        ]
        stage1 = np.asarray(archives[0]["stage1_fem_nodes"], dtype=np.float64)
        corrections = [
            np.asarray(archive["step3_fem_nodes"], dtype=np.float64) - stage1
            for archive in archives
        ]
        correction_cosines["v1_v3"].append(_cosine(corrections[0], corrections[1]))
        correction_cosines["v4_v1"].append(_cosine(corrections[2], corrections[0]))
        correction_cosines["v4_v3"].append(_cosine(corrections[2], corrections[1]))

    v4_payload = json.loads(args.v4_result.read_text())
    physics: dict[str, dict[str, float]] = {}
    diagnostic_names = (
        "profiled_amplitude",
        "raw_residual_rms",
        "si_residual_rms",
        "raw_adjoint_rms",
        "si_adjoint_rms",
        "relative_forward_rms",
        "raw_si_adjoint_cosine",
        "update_rms",
        "state_rms",
        "dc_rms",
        "si_fusion_weight",
    )
    for iteration in range(1, 4):
        for name in diagnostic_names:
            values = [
                float(row[f"step{iteration}_{name}"])
                for row in v4_payload["per_sample"]
            ]
            physics[f"iteration_{iteration}_{name}"] = _summary(values)
    result = {
        "n_samples": len(ids),
        "correction_cosine": {
            key: _summary(values) for key, values in correction_cosines.items()
        },
        "physics_evidence": physics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
