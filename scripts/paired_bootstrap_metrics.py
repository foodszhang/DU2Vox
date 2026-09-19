#!/usr/bin/env python3
"""Paired bootstrap over per-sample metrics for a reconstruction comparison.

The scientific question is not whether the refined reconstruction has a higher
mean score, but whether it is better *on the same cases* and by how much. This
resamples samples with replacement (cases are the unit of resampling, and every
method is re-scored on the identical resampled set), so the resulting interval
reflects how consistent the per-case improvement is rather than how spread out
the cohort is.

Comparisons are formed from a single evaluation file, which holds per-sample
metrics for every layer, so the pairing is exact by construction.

CPU only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Metrics where a larger value is better. Everything else is treated as an
# error or distance, so the sign of the improvement is flipped for reporting.
HIGHER_IS_BETTER = {
    "pearson",
    "ccc",
    "dice_halfmax",
    "psnr_db",
    "contrast_ratio_pred_mean",
}

# Recovery ratios whose ideal value is 1.0: both over- and under-shoot are
# errors, so they are scored as |value - 1| and a *decrease* is an improvement.
# Reporting the raw ratio as "higher is better" would score the Exp A
# over-shoot (R_peak 1.03 -> 1.61, i.e. away from unity) as a gain.
TARGET_IS_ONE = {
    "integrated_ratio",
    "rpeak_mean",
    "rint_mean",
}


def oriented_delta(
    compare: np.ndarray, baseline: np.ndarray, metric: str
) -> np.ndarray:
    """Return per-sample improvement, positive meaning `compare` is better."""
    if metric in TARGET_IS_ONE:
        return np.abs(baseline - 1.0) - np.abs(compare - 1.0)
    delta = compare - baseline
    return delta if metric in HIGHER_IS_BETTER else -delta


def bootstrap_ci(
    deltas: np.ndarray, n_resamples: int, seed: int, alpha: float = 0.05
) -> dict[str, float]:
    """Percentile bootstrap CI for the mean of paired differences."""
    finite = deltas[np.isfinite(deltas)]
    if finite.size == 0:
        return {"n": 0, "mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, finite.size, size=(n_resamples, finite.size))
    means = finite[idx].mean(axis=1)
    return {
        "n": int(finite.size),
        "mean": float(finite.mean()),
        "lo": float(np.quantile(means, alpha / 2)),
        "hi": float(np.quantile(means, 1 - alpha / 2)),
        # Fraction of resamples where the improvement keeps its sign.
        "p_consistent": float(
            np.mean(means > 0) if finite.mean() > 0 else np.mean(means < 0)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="per-sample metrics JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", default="coarse", help="layer used as the baseline")
    parser.add_argument("--compare", nargs="+", default=["step1", "step2", "final"])
    parser.add_argument("--n-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "relative_l2",
            "pearson",
            "ccc",
            "dice_halfmax",
            "centroid_error_mm",
            "integrated_ratio",
            "rpeak_mean",
            "rint_mean",
            "contrast_ratio_log_err_mean",
        ],
    )
    args = parser.parse_args()

    payload = json.load(open(args.input))
    rows = payload["per_sample"]
    print(f"samples: {len(rows)}  baseline: {args.baseline}")

    results: dict = {"input": str(args.input), "baseline": args.baseline, "comparisons": {}}
    for compare in args.compare:
        key = f"{compare}_vs_{args.baseline}"
        per_metric: dict = {}
        for metric in args.metrics:
            a_key, b_key = f"{args.baseline}_{metric}", f"{compare}_{metric}"
            pairs = [
                (r[b_key], r[a_key])
                for r in rows
                if a_key in r and b_key in r
                and np.isfinite(r[a_key]) and np.isfinite(r[b_key])
            ]
            if not pairs:
                continue
            b = np.asarray([p[0] for p in pairs], dtype=np.float64)
            a = np.asarray([p[1] for p in pairs], dtype=np.float64)
            # Orient every metric so that positive means "compare is better".
            delta = oriented_delta(b, a, metric)
            per_metric[metric] = bootstrap_ci(delta, args.n_resamples, args.seed)
        results["comparisons"][key] = per_metric

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    for key, per_metric in results["comparisons"].items():
        print(f"\n=== {key}  (positive = {key.split('_')[0]} better) ===")
        print(f"{'metric':<32s} {'mean':>10s} {'95% CI':>22s} {'consistent':>11s} {'n':>5s}")
        print("-" * 84)
        for metric, s in per_metric.items():
            print(
                f"{metric:<32s} {s['mean']:>+10.4f} "
                f"[{s['lo']:>+8.4f},{s['hi']:>+8.4f}] "
                f"{s['p_consistent']:>10.1%} {s['n']:>5d}"
            )
    print(f"\n-> {args.output}")


if __name__ == "__main__":
    main()
