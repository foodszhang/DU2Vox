#!/usr/bin/env python3
"""Paired paper-level statistics for the one-shot confirmation evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.utils.confirmation import require_confirmation_permission


def _rows(path: Path, prefix: str) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text())
    result: dict[str, dict[str, float]] = {}
    for row in payload["per_sample"]:
        result[row["sample_id"]] = {
            key[len(prefix) :]: float(value)
            for key, value in row.items()
            if key.startswith(prefix) and isinstance(value, (int, float))
        }
    return result


def _paired_statistics(
    first: np.ndarray, second: np.ndarray, *, seed: int = 20260901
) -> dict[str, Any]:
    delta = first - second
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(delta), size=(20_000, len(delta)))
    bootstrap = delta[draws].mean(axis=1)
    nonzero = delta[np.abs(delta) > 1e-12]
    p_value = (
        float(stats.wilcoxon(nonzero, alternative="two-sided").pvalue)
        if len(nonzero)
        else 1.0
    )
    std = float(delta.std(ddof=1)) if len(delta) > 1 else 0.0
    return {
        "mean_delta": float(delta.mean()),
        "median_delta": float(np.median(delta)),
        "bootstrap_95_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "paired_wilcoxon_p": p_value,
        "paired_cohen_dz": float(delta.mean() / std) if std > 0.0 else 0.0,
        "win_tie_loss": {
            "win": int((delta > 1e-12).sum()),
            "tie": int((np.abs(delta) <= 1e-12).sum()),
            "loss": int((delta < -1e-12).sum()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--v3", type=Path, required=True)
    parser.add_argument("--v4", type=Path, required=True)
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-confirmation-eval", action="store_true")
    args = parser.parse_args()
    require_confirmation_permission(
        samples_dir=args.samples_dir,
        allow_confirmation_eval=args.allow_confirmation_eval,
        manifest_path=args.manifest,
    )

    v1 = _rows(args.v1, "step3_")
    v3 = _rows(args.v3, "step3_")
    v4 = _rows(args.v4, "step3_")
    fem = _rows(args.v4, "stage1_")
    ensemble = _rows(args.ensemble, "")
    ids = sorted(v4)
    methods = {"stage1": fem, "v1": v1, "v3": v3, "v4": v4, "ensemble": ensemble}
    metrics = ("dice", "precision", "recall", "weak_recall", "hd95", "localization_error", "mse")
    summaries = {
        name: {
            metric: float(np.nanmean([rows[sid][metric] for sid in ids]))
            for metric in metrics
        }
        for name, rows in methods.items()
    }
    comparisons = {}
    for first_name, second_name in (
        ("v4", "stage1"),
        ("v4", "v3"),
        ("v4", "v1"),
        ("ensemble", "v3"),
    ):
        comparisons[f"{first_name}_vs_{second_name}"] = _paired_statistics(
            np.asarray([methods[first_name][sid]["dice"] for sid in ids]),
            np.asarray([methods[second_name][sid]["dice"] for sid in ids]),
        )

    groups: dict[str, list[str]] = {}
    for sid in ids:
        tumor = json.loads((args.samples_dir / sid / "tumor_params.json").read_text())
        groups.setdefault(f"sources_{tumor['num_foci']}", []).append(sid)
        groups.setdefault(f"depth_{tumor.get('depth_tier', 'unknown')}", []).append(sid)
    subgroup = {
        group: {
            "n": len(group_ids),
            "v4_dice": float(np.mean([v4[sid]["dice"] for sid in group_ids])),
            "ensemble_dice": float(
                np.mean([ensemble[sid]["dice"] for sid in group_ids])
            ),
        }
        for group, group_ids in sorted(groups.items())
    }
    result = {
        "n_samples": len(ids),
        "threshold": 0.5,
        "ensemble_weights": {"v1": 0.45, "v3": 0.55},
        "summary": summaries,
        "paired_dice_analysis": comparisons,
        "subgroups": subgroup,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
