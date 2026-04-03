"""
Per-foci and per-depth metric grouping for evaluation.
"""

import json
from pathlib import Path
from typing import Any

from du2vox.evaluation.metrics import evaluate_batch, summarize_metrics


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
