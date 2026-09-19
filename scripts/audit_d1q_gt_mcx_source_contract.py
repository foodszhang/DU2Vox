#!/usr/bin/env python3
"""Compare historical D1-Q voxel GT directly with saved MCX source patterns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from du2vox.utils.gt_io import load_gt_volume


def describe(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


def load_mcx_source(sample_dir: Path, shape: tuple[int, ...]) -> np.ndarray:
    sample_id = sample_dir.name
    config = json.loads((sample_dir / f"{sample_id}.json").read_text())
    source = config["Optode"]["Source"]
    pattern_meta = source["Pattern"]
    nz, ny, nx = (int(value) for value in source["Param1"][:3])
    pos_z, pos_y, pos_x = (int(value) for value in source["Pos"])
    raw = np.fromfile(sample_dir / pattern_meta["Data"], dtype=np.float32)
    expected = nx * ny * nz
    if raw.size != expected:
        raise RuntimeError(f"{sample_id}: source size {raw.size} != {expected}")
    # The writer transposes pattern ZYX to XYZ before C-order tofile().
    pattern_xyz = raw.reshape(nx, ny, nz)
    full = np.zeros(shape, dtype=np.float32)
    full[pos_x : pos_x + nx, pos_y : pos_y + ny, pos_z : pos_z + nz] = pattern_xyz
    return full


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "/home/foods/pro/FMT-SimGen/data/"
            "fmt_simgen_v2_quantitative_gaussian_3k_20k"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnosis/d1q_gt_mcx_source_contract3000.json"),
    )
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    sample_dirs = sorted((args.dataset_root / "samples").glob("sample_*"))
    if args.max_samples is not None:
        sample_dirs = sample_dirs[: args.max_samples]

    gt_only_fraction: list[float] = []
    source_only_fraction: list[float] = []
    common_relative_l2: list[float] = []
    full_relative_l2: list[float] = []
    gt_below_source_one_percent_fraction: list[float] = []
    gt_nonzero_min_over_peak: list[float] = []
    exact_support_matches = 0
    exact_value_matches = 0
    per_case = {}

    for index, sample_dir in enumerate(sample_dirs, start=1):
        gt = np.asarray(load_gt_volume(sample_dir), dtype=np.float32)
        source = load_mcx_source(sample_dir, gt.shape)
        gt_mask = gt != 0.0
        source_mask = source != 0.0
        common = gt_mask & source_mask
        gt_only = gt_mask & ~source_mask
        source_only = source_mask & ~gt_mask

        support_match = bool(np.array_equal(gt_mask, source_mask))
        value_match = bool(np.array_equal(gt, source))
        exact_support_matches += int(support_match)
        exact_value_matches += int(value_match)

        gt_count = max(int(gt_mask.sum()), 1)
        gt_only_value = float(gt_only.sum() / gt_count)
        source_only_value = float(source_only.sum() / max(int(source_mask.sum()), 1))
        gt_only_fraction.append(gt_only_value)
        source_only_fraction.append(source_only_value)

        if np.any(common):
            common_error = np.linalg.norm(source[common] - gt[common]) / max(
                np.linalg.norm(gt[common]), 1e-30
            )
        else:
            common_error = float("nan")
        full_error = np.linalg.norm(source - gt) / max(np.linalg.norm(gt), 1e-30)
        common_relative_l2.append(float(common_error))
        full_relative_l2.append(float(full_error))

        gt_peak = float(gt.max())
        source_peak = float(source.max())
        gt_values = gt[gt_mask]
        gt_nonzero_min_over_peak.append(float(gt_values.min() / max(gt_peak, 1e-30)))
        below = gt_mask & (gt < 0.01 * source_peak)
        below_fraction = float(below.sum() / gt_count)
        gt_below_source_one_percent_fraction.append(below_fraction)

        if index <= 12 or not support_match:
            per_case[sample_dir.name] = {
                "gt_nonzero": int(gt_mask.sum()),
                "source_nonzero": int(source_mask.sum()),
                "gt_only": int(gt_only.sum()),
                "source_only": int(source_only.sum()),
                "gt_only_fraction_of_gt": gt_only_value,
                "source_only_fraction_of_source": source_only_value,
                "gt_peak": gt_peak,
                "source_peak": source_peak,
                "gt_min_nonzero_over_gt_peak": gt_nonzero_min_over_peak[-1],
                "gt_below_1pct_source_peak_fraction": below_fraction,
                "common_relative_l2": float(common_error),
                "full_relative_l2": float(full_error),
            }
        if index % 250 == 0:
            print(f"[audit] {index}/{len(sample_dirs)}", flush=True)

    result = {
        "dataset_root": str(args.dataset_root.resolve()),
        "n_samples": len(sample_dirs),
        "method": "saved gt_voxels versus saved source binary embedded using MCX JSON",
        "exact_support_matches": exact_support_matches,
        "exact_value_matches": exact_value_matches,
        "gt_only_fraction_of_gt_support": describe(gt_only_fraction),
        "source_only_fraction_of_source_support": describe(source_only_fraction),
        "common_support_relative_l2": describe(common_relative_l2),
        "full_relative_l2": describe(full_relative_l2),
        "gt_below_1pct_source_peak_fraction": describe(
            gt_below_source_one_percent_fraction
        ),
        "gt_min_nonzero_over_gt_peak": describe(gt_nonzero_min_over_peak),
        "per_case_nonmatching": per_case,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "per_case_nonmatching"}, indent=2))


if __name__ == "__main__":
    main()
