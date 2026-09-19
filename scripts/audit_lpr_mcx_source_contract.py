#!/usr/bin/env python3
"""Verify every MCX source binary is identical to the frozen voxel GT field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from du2vox.utils.gt_io import load_gt_volume


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    sample_dirs = sorted((args.dataset_root / "samples").glob("sample_*"))
    for index, sample_dir in enumerate(sample_dirs, start=1):
        config = json.loads((sample_dir / f"{sample_dir.name}.json").read_text())
        source = config["Optode"]["Source"]
        pattern = source["Pattern"]
        source_xyz = np.fromfile(
            sample_dir / pattern["Data"], dtype=np.float32
        ).reshape(pattern["Nz"], pattern["Ny"], pattern["Nx"])
        gt = load_gt_volume(sample_dir)
        z0, y0, x0 = (int(value) for value in source["Pos"])
        target_xyz = gt[
            x0 : x0 + pattern["Nz"],
            y0 : y0 + pattern["Ny"],
            z0 : z0 + pattern["Nx"],
        ]
        difference = source_xyz - target_xyz
        outside = np.array(gt, copy=True)
        outside[
            x0 : x0 + pattern["Nz"],
            y0 : y0 + pattern["Ny"],
            z0 : z0 + pattern["Nx"],
        ] = 0.0
        outside_mass = float(np.clip(outside, 0.0, None).sum(dtype=np.float64))
        rows.append(
            {
                "sample_id": sample_dir.name,
                "max_abs_error": float(np.max(np.abs(difference))),
                "relative_l2": float(
                    np.linalg.norm(difference)
                    / max(np.linalg.norm(target_xyz), 1e-30)
                ),
                "mass_outside_source_bbox": outside_mass,
            }
        )
        if index % 250 == 0:
            print(f"[{index}/{len(sample_dirs)}]", flush=True)
    result = {
        "n_samples": len(rows),
        "max_abs_error": max(row["max_abs_error"] for row in rows),
        "max_relative_l2": max(row["relative_l2"] for row in rows),
        "max_mass_outside_source_bbox": max(
            row["mass_outside_source_bbox"] for row in rows
        ),
        "numerically_identical": all(
            row["max_abs_error"] <= 1e-7
            and row["relative_l2"] <= 1e-8
            and row["mass_outside_source_bbox"] == 0.0
            for row in rows
        ),
        "per_sample": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "per_sample"}, indent=2))


if __name__ == "__main__":
    main()
