#!/usr/bin/env python3
"""Sanity and identity audit for an LPR continuous Gaussian-mixture cohort."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from du2vox.bridge.canonical_cross_discretization import (  # noqa: E402
    CanonicalCrossDiscretization,
)
from du2vox.utils.frame import FrameManifest  # noqa: E402
from du2vox.utils.gt_io import load_gt_volume  # noqa: E402

_MIXTURE_PATH = Path(
    "/home/foods/pro/FMT-SimGen/fmt_simgen/tumor/gaussian_mixture.py"
)
_SPEC = importlib.util.spec_from_file_location("lpr_gaussian_mixture", _MIXTURE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load {_MIXTURE_PATH}")
_MIXTURE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MIXTURE)
evaluate_mixture = _MIXTURE.evaluate_mixture
focus_sigmas = _MIXTURE.focus_sigmas
rotation_matrix = _MIXTURE.rotation_matrix


def stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "max": float(array.max()),
        "n": int(array.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--shared-dir",
        type=Path,
        default=Path("/home/foods/pro/FMT-SimGen/output/shared_mesh_20k"),
    )
    parser.add_argument(
        "--operator-cache",
        type=Path,
        default=Path("experiments/cross_discretization_decomposition/artifacts/operator_cache"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--voxel-probes", type=int, default=100_000)
    parser.add_argument("--require-mcx-source", action="store_true")
    args = parser.parse_args()

    sample_dirs = sorted((args.dataset_root / "samples").glob("sample_*"))
    if args.max_samples is not None:
        sample_dirs = sample_dirs[: args.max_samples]
    canonical = CanonicalCrossDiscretization(
        args.operator_cache,
        shared_dir=args.shared_dir,
        factorize=False,
        allow_stale_frame_manifest=True,
    )
    frame = FrameManifest.load(args.shared_dir)
    nodes, _ = FrameManifest.load_mesh_nodes(args.shared_dir)
    forward = np.load(args.shared_dir / "system_matrix.A.npz")["forward_matrix"]
    shape = tuple(frame.gt_shape)
    spacing = float(frame.gt_spacing_mm)
    offset = np.asarray(frame.gt_offset_world_mm, dtype=np.float64)
    rng = np.random.default_rng(20260915)

    rows = []
    all_amplitudes: list[float] = []
    all_sigma: list[float] = []
    all_fwhm: list[float] = []
    all_separations: list[float] = []
    for sample_dir in sample_dirs:
        tumor = json.loads((sample_dir / "tumor_params.json").read_text())
        foci = tumor["foci"]
        rotations = [rotation_matrix(focus) for focus in foci]
        sigmas = [focus_sigmas(focus) for focus in foci]
        centers = [np.asarray(focus["center"], dtype=np.float64) for focus in foci]
        all_amplitudes.extend(float(focus["params"]["intensity"]) for focus in foci)
        all_sigma.extend(float(value) for item in sigmas for value in item)
        all_fwhm.extend(float(2.354820045 * value) for item in sigmas for value in item)
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                all_separations.append(float(np.linalg.norm(centers[i] - centers[j])))

        gt_nodes = np.load(sample_dir / "gt_nodes.npy").astype(np.float64)
        analytic_nodes = evaluate_mixture(nodes, foci).astype(np.float64)
        node_abs = np.abs(gt_nodes - analytic_nodes)
        gt_volume = np.asarray(load_gt_volume(sample_dir), dtype=np.float64)
        flat = gt_volume.ravel()
        n_probe = min(int(args.voxel_probes), flat.size)
        probe = rng.choice(flat.size, size=n_probe, replace=False)
        # Mirror DualSampler's float32 voxel-center convention exactly
        # (FMT-SimGen fmt_simgen/sampling/dual_sampler.py):
        #     arange(n, float32) * float32(spacing) + float32(offset) + float32(spacing/2)
        # The stored volume was generated at float32 centers, so recomputing the
        # same mixture at float64 centers disagrees about 3.5-sigma truncation
        # membership on the boundary surface. That disagreement equals the
        # Gaussian value at 3.5 sigma (~2.2e-3), far above the 5e-6 acceptance
        # bound, and reflects the comparison convention rather than the data.
        unravel = np.stack(np.unravel_index(probe, shape), axis=1).astype(np.float32)
        probe_coords = (
            unravel * np.float32(spacing)
            + np.asarray(offset, dtype=np.float32)
            + np.float32(spacing / 2)
        ).astype(np.float32)
        analytic_probe = evaluate_mixture(probe_coords, foci).astype(np.float64)
        valid = canonical.operator.valid_flat_indices
        total_positive_mass = float(np.clip(flat, 0.0, None).sum())
        valid_positive_mass = float(np.clip(flat[valid], 0.0, None).sum())
        measurement = np.load(sample_dir / "measurement_b.npy")
        measurement_from_saved_a = np.maximum(forward @ gt_nodes, 0.0)
        measurement_error = measurement_from_saved_a - measurement
        mcx_json_path = sample_dir / f"{sample_dir.name}.json"
        if args.require_mcx_source and not mcx_json_path.exists():
            raise FileNotFoundError(mcx_json_path)
        mcx_source_error = None
        mcx_source_outside_mass = None
        if mcx_json_path.exists():
            mcx_json = json.loads(mcx_json_path.read_text())
            source = mcx_json["Optode"]["Source"]
            pattern = source["Pattern"]
            source_xyz = np.fromfile(
                sample_dir / pattern["Data"], dtype=np.float32
            ).reshape(pattern["Nz"], pattern["Ny"], pattern["Nx"])
            z0, y0, x0 = (int(value) for value in source["Pos"])
            target_xyz = gt_volume[
                x0 : x0 + pattern["Nz"],
                y0 : y0 + pattern["Ny"],
                z0 : z0 + pattern["Nx"],
            ]
            mcx_source_error = float(
                np.linalg.norm(source_xyz - target_xyz)
                / max(np.linalg.norm(target_xyz), 1e-30)
            )
            mcx_source_outside_mass = float(
                max(total_positive_mass - np.clip(target_xyz, 0.0, None).sum(), 0.0)
                / max(total_positive_mass, 1e-30)
            )
        row = {
            "sample_id": sample_dir.name,
            "num_foci": len(foci),
            "node_max_abs_error": float(node_abs.max()),
            "node_rmse": float(np.sqrt(np.mean(node_abs**2))),
            "voxel_probe_max_abs_error": float(
                np.max(np.abs(flat[probe] - analytic_probe))
            ),
            "gt_peak": float(flat.max()),
            "gt_positive_mass": total_positive_mass,
            "valid_mass_fraction": valid_positive_mass / max(total_positive_mass, 1e-30),
            "measurement_max": float(np.max(measurement)),
            "measurement_min": float(np.min(measurement)),
            "measurement_forward_relative_l2": float(
                np.linalg.norm(measurement_error)
                / max(np.linalg.norm(measurement), 1e-30)
            ),
            "rotation_orthogonality_error": float(
                max(np.max(np.abs(value.T @ value - np.eye(3))) for value in rotations)
            ),
            "rotation_det_error": float(
                max(abs(np.linalg.det(value) - 1.0) for value in rotations)
            ),
        }
        if mcx_source_error is not None:
            row["mcx_source_gt_relative_l2"] = mcx_source_error
            row["gt_mass_outside_mcx_source_bbox"] = mcx_source_outside_mass
        rows.append(row)

    numeric = {
        key: stats([float(row[key]) for row in rows])
        for key in rows[0]
        if key not in {"sample_id", "num_foci"}
    }
    result = {
        "dataset_root": str(args.dataset_root.resolve()),
        "n_samples": len(rows),
        "num_foci": {
            str(k): sum(row["num_foci"] == k for row in rows) for k in (1, 2, 3)
        },
        "amplitude": stats(all_amplitudes),
        "principal_sigma_mm": stats(all_sigma),
        "principal_fwhm_mm": stats(all_fwhm),
        "pair_separation_mm": stats(all_separations) if all_separations else None,
        "summary": numeric,
        "acceptance": {
            "analytic_node_max_abs_error_le_1e-6": numeric["node_max_abs_error"]["max"] <= 1e-6,
            # GT coordinates are constructed in float32 by DualSampler; the
            # independent audit reconstructs them in float64. A 5e-6 absolute
            # tolerance covers only that coordinate-rounding difference.
            "analytic_voxel_probe_max_abs_error_le_5e-6": numeric["voxel_probe_max_abs_error"]["max"] <= 5e-6,
            "measurement_min_nonnegative": numeric["measurement_min"]["min"] >= 0.0,
            "measurement_max_min_ge_5e-4": numeric["measurement_max"]["min"] >= 5e-4,
            "measurement_forward_relative_l2_max_le_1e-5": numeric[
                "measurement_forward_relative_l2"
            ]["max"]
            <= 1e-5,
            "gt_peak_max_le_fixed_data_range_2": numeric["gt_peak"]["max"] <= 2.0,
            "valid_mass_fraction_min_ge_0_95": numeric["valid_mass_fraction"]["min"] >= 0.95,
            "rotation_contract": numeric["rotation_orthogonality_error"]["max"] <= 1e-6
            and numeric["rotation_det_error"]["max"] <= 1e-6,
        },
        "per_sample": rows,
    }
    if args.require_mcx_source:
        result["acceptance"].update(
            {
                "mcx_source_gt_relative_l2_max_le_5e-6": numeric[
                    "mcx_source_gt_relative_l2"
                ]["max"]
                <= 5e-6,
                "gt_mass_outside_mcx_source_bbox_max_le_1e-8": numeric[
                    "gt_mass_outside_mcx_source_bbox"
                ]["max"]
                <= 1e-8,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
