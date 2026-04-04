"""
Per-foci and per-depth metric grouping for evaluation.
"""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Any

from du2vox.evaluation.metrics import summarize_metrics


def load_manifest(samples_dir: Path) -> dict | None:
    """Load dataset manifest from multiple possible locations."""
    candidates = [
        Path("output/dataset_manifest.json"),
        Path("data/dataset_manifest.json"),
        samples_dir.parent / "dataset_manifest.json",
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


def get_sample_names(split_file: Path) -> list[str]:
    """Get ordered list of sample names from split file."""
    with open(split_file) as f:
        return [line.strip() for line in f if line.strip()]


def group_metrics_by_foci(
    all_metrics: list[dict],
    all_sample_names: list[str],
    manifest: dict,
) -> dict[int, list[dict]]:
    """Group metrics by foci count (1, 2, or 3)."""
    metrics_by_foci: dict[int | None, list[dict]] = {1: [], 2: [], 3: []}
    for i, metrics in enumerate(all_metrics):
        sample_name = all_sample_names[i] if i < len(all_sample_names) else None
        if sample_name and sample_name in manifest["samples"]:
            n_foci = manifest["samples"][sample_name].get("num_foci")
            if n_foci in metrics_by_foci:
                metrics_by_foci[n_foci].append(metrics)
    return metrics_by_foci


def group_metrics_by_depth(
    all_metrics: list[dict],
    all_sample_names: list[str],
    manifest: dict,
) -> dict[str, list[dict]]:
    """Group metrics by depth tier (shallow, medium, deep)."""
    metrics_by_depth: dict[str, list[dict]] = {"shallow": [], "medium": [], "deep": []}
    for i, metrics in enumerate(all_metrics):
        sample_name = all_sample_names[i] if i < len(all_sample_names) else None
        if sample_name and sample_name in manifest["samples"]:
            depth_tier = manifest["samples"][sample_name].get("depth_tier", "unknown")
            if depth_tier in metrics_by_depth:
                metrics_by_depth[depth_tier].append(metrics)
    return metrics_by_depth


def group_metrics_by_cross(
    all_metrics: list[dict],
    all_sample_names: list[str],
    manifest: dict,
) -> dict[tuple, list[dict]]:
    """Group metrics by foci count × depth tier."""
    metrics_by_cross: dict[tuple, list[dict]] = defaultdict(list)
    for i, metrics in enumerate(all_metrics):
        sample_name = all_sample_names[i] if i < len(all_sample_names) else None
        if sample_name and sample_name in manifest["samples"]:
            info = manifest["samples"][sample_name]
            n_foci = info.get("num_foci")
            depth_tier = info.get("depth_tier", "unknown")
            if n_foci is not None and depth_tier != "unknown":
                metrics_by_cross[(n_foci, depth_tier)].append(metrics)
    return dict(metrics_by_cross)


def format_metrics_table(
    overall: dict[str, Any],
    foci_summaries: dict[int, dict[str, Any] | None],
    keys: list[str] | None = None,
) -> str:
    """Format metrics as a table string."""
    if keys is None:
        keys = [
            "dice", "dice_bin_0.3", "dice_bin_0.1",
            "recall_0.3", "recall_0.1", "precision_0.3",
            "location_error", "mse",
        ]

    header = f"{'Metric':<20} {'Overall':>10} {'1-Foci':>10} {'2-Foci':>10} {'3-Foci':>10}"
    separator = "-" * len(header)

    lines = [header, separator]
    for key in keys:
        overall_val = overall.get(key, 0)
        vals = []
        for n in [1, 2, 3]:
            if foci_summaries.get(n):
                vals.append(f"{foci_summaries[n].get(key, 0):>10.4f}")
            else:
                vals.append(f"{'N/A':>10}")
        lines.append(f"{key:<20} {overall_val:>10.4f} {''.join(vals)}")

    return "\n".join(lines)


def format_depth_table(
    overall: dict[str, Any],
    depth_summaries: dict[str, dict[str, Any] | None],
    keys: list[str] | None = None,
) -> str:
    """Format metrics by depth tier as a table string."""
    if keys is None:
        keys = [
            "dice", "dice_bin_0.3", "dice_bin_0.1",
            "recall_0.3", "recall_0.1", "precision_0.3",
            "location_error", "mse",
        ]

    header = f"{'Metric':<20} {'Overall':>10} {'Shallow':>10} {'Medium':>10} {'Deep':>10}"
    separator = "-" * len(header)

    lines = [header, separator]
    for key in keys:
        overall_val = overall.get(key, 0)
        vals = []
        for tier in ["shallow", "medium", "deep"]:
            if depth_summaries.get(tier):
                vals.append(f"{depth_summaries[tier].get(key, 0):>10.4f}")
            else:
                vals.append(f"{'N/A':>10}")
        lines.append(f"{key:<20} {overall_val:>10.4f} {''.join(vals)}")

    return "\n".join(lines)


def format_cross_table(
    cross_metrics: dict[tuple, list[dict]],
    key: str,
    foci_list: list[int] | None = None,
    depth_list: list[str] | None = None,
) -> str:
    """Format cross (foci × depth) metrics as a markdown table."""
    if foci_list is None:
        foci_list = [1, 2, 3]
    if depth_list is None:
        depth_list = ["shallow", "medium", "deep"]

    header = f"| **Metric** |" + "".join(f" **{t.title()}** |" for t in depth_list)
    separator = "|" + "|".join(["---" for _ in range(len(depth_list) + 1)]) + "|"

    lines = [header, separator]

    for n_foci in foci_list:
        row = f"| {n_foci}-Foci |"
        for tier in depth_list:
            group = cross_metrics.get((n_foci, tier), [])
            if group:
                vals = [m.get(key, 0) for m in group]
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                row += f" {mean_val:.3f}±{std_val:.3f} |"
            else:
                row += " N/A |"
        lines.append(row)

    return "\n".join(lines)


def compute_group_stats(group: list[dict], key: str) -> tuple[float, float]:
    """Compute mean and std for a metric key across a group of metrics."""
    if not group:
        return 0.0, 0.0
    vals = [m.get(key, 0) for m in group]
    return np.mean(vals), np.std(vals)


def grouped_evaluation(
    all_metrics: list[dict],
    all_sample_names: list[str],
    manifest: dict,
    output_json: Path | None = None,
) -> dict:
    """
    Compute and print all grouped metrics tables.

    Returns a dict with all grouped summaries for JSON export.
    """
    # Group by foci
    by_foci = group_metrics_by_foci(all_metrics, all_sample_names, manifest)
    # Group by depth
    by_depth = group_metrics_by_depth(all_metrics, all_sample_names, manifest)
    # Group by cross
    by_cross = group_metrics_by_cross(all_metrics, all_sample_names, manifest)

    # Overall
    overall = summarize_metrics(all_metrics)

    # Summaries for each group
    foci_summaries = {n: summarize_metrics(by_foci[n]) if by_foci[n] else None for n in [1, 2, 3]}
    depth_summaries = {t: summarize_metrics(by_depth[t]) if by_depth[t] else None for t in ["shallow", "medium", "deep"]}

    # Keys for tables
    main_keys = ["dice", "dice_bin_0.3", "dice_bin_0.1", "recall_0.3", "recall_0.1", "precision_0.3", "location_error", "mse"]
    cross_keys = ["dice_bin_0.1", "recall_0.1"]

    # ===== Print BY FOCI table =====
    print("\n" + "=" * 70)
    print("EVALUATION BY FOCI COUNT")
    print("=" * 70)
    print(f"Overall ({len(all_metrics)} samples)")
    print(f"  1-Foci: {len(by_foci[1])} samples")
    print(f"  2-Foci: {len(by_foci[2])} samples")
    print(f"  3-Foci: {len(by_foci[3])} samples")
    print()
    print(format_metrics_table(overall, foci_summaries, keys=main_keys))

    # ===== Print BY DEPTH table =====
    print("\n" + "=" * 70)
    print("EVALUATION BY DEPTH TIER")
    print("=" * 70)
    print(f"Overall ({len(all_metrics)} samples)")
    print(f"  Shallow: {len(by_depth['shallow'])} samples")
    print(f"  Medium: {len(by_depth['medium'])} samples")
    print(f"  Deep: {len(by_depth['deep'])} samples")
    print()
    print(format_depth_table(overall, depth_summaries, keys=main_keys))

    # ===== Print CROSS tables =====
    for ck in cross_keys:
        print("\n" + "=" * 70)
        print(f"CROSS TABLE: {ck.upper()} (Dice@0.1 / Recall@0.1)")
        print("=" * 70)
        print(format_cross_table(by_cross, ck))

    # Build result dict for JSON export
    result = {
        "overall": overall,
        "by_foci": {},
        "by_depth": {},
        "by_cross": {},
        "n_samples": {
            "total": len(all_metrics),
            "by_foci": {str(k): len(v) for k, v in by_foci.items()},
            "by_depth": {k: len(v) for k, v in by_depth.items()},
        },
    }

    for n in [1, 2, 3]:
        if foci_summaries[n]:
            result["by_foci"][f"{n}_foci"] = foci_summaries[n]

    for tier in ["shallow", "medium", "deep"]:
        if depth_summaries[tier]:
            result["by_depth"][tier] = depth_summaries[tier]

    for (n_foci, depth_tier), group in by_cross.items():
        if group:
            key = f"{n_foci}_foci_{depth_tier}"
            result["by_cross"][key] = {
                "n_samples": len(group),
                "mean": {},
                "std": {},
            }
            for k in main_keys:
                mean_val, std_val = compute_group_stats(group, k)
                result["by_cross"][key]["mean"][k] = mean_val
                result["by_cross"][key]["std"][k] = std_val

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {output_json}")

    return result
