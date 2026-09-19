#!/usr/bin/env python3
"""Independently audit the derived MCX-canonical D1-Q dataset contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.bridge.canonical_cross_discretization import CanonicalCrossDiscretization
from du2vox.utils.frame import FrameManifest
from du2vox.utils.gt_io import load_gt_volume


def read_ids(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if line]


def embed_source(sample_dir: Path, shape: tuple[int, ...]) -> np.ndarray:
    sample_id = sample_dir.name
    config = json.loads((sample_dir / f"{sample_id}.json").read_text())
    source = config["Optode"]["Source"]
    nz, ny, nx = (int(value) for value in source["Param1"][:3])
    pos_z, pos_y, pos_x = (int(value) for value in source["Pos"])
    raw = np.fromfile(sample_dir / source["Pattern"]["Data"], dtype=np.float32)
    if raw.size != nx * ny * nz:
        raise RuntimeError(f"{sample_id}: invalid source binary size")
    full = np.zeros(shape, dtype=np.float32)
    full[pos_x : pos_x + nx, pos_y : pos_y + ny, pos_z : pos_z + nz] = raw.reshape(
        nx, ny, nz
    )
    return full


def describe(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--operator-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    canonical = CanonicalCrossDiscretization(
        args.operator_cache,
        shared_dir=args.shared_dir,
        factorize=False,
        allow_stale_frame_manifest=True,
    )
    frame = FrameManifest.load(args.shared_dir)
    nodes, _ = FrameManifest.load_mesh_nodes(args.shared_dir)
    node_indices = frame.world_to_gt_index(nodes).T
    matrix = np.asarray(
        np.load(args.shared_dir / "system_matrix.A.npz")["forward_matrix"],
        dtype=np.float32,
    )

    source_splits = {
        name: read_ids(args.source_root / "splits" / f"{name}.txt")
        for name in ("train", "val", "test")
    }
    derived_splits = {
        name: read_ids(args.dataset_root / "splits" / f"{name}.txt")
        for name in ("train", "val", "test")
    }
    if source_splits != derived_splits:
        raise RuntimeError("Derived splits differ from historical D1-Q")
    ids = source_splits["train"] + source_splits["val"] + source_splits["test"]
    if args.max_samples is not None:
        ids = ids[: args.max_samples]

    voxel_errors = []
    node_errors = []
    measurement_errors = []
    scale_errors = []
    negative_node_fractions = []
    negative_measurement_fractions = []
    measurement_maxima = []
    node_to_voxel_peak_ratios = []
    outside_valid_mass_fractions = []
    hardlink_failures = []
    for index, sample_id in enumerate(ids, start=1):
        sample_dir = args.dataset_root / "samples" / sample_id
        source_dir = args.source_root / "samples" / sample_id
        expected_voxel = embed_source(source_dir, canonical.operator.grid_shape)
        voxel = np.asarray(load_gt_volume(sample_dir), dtype=np.float32)
        voxel_error = float(np.max(np.abs(voxel - expected_voxel)))
        voxel_errors.append(voxel_error)

        expected_nodes = map_coordinates(
            expected_voxel,
            node_indices,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ).astype(np.float32)
        node = np.asarray(np.load(sample_dir / "gt_nodes.npy"), dtype=np.float32)
        node_errors.append(float(np.max(np.abs(node - expected_nodes))))

        expected_measurement = np.maximum(matrix @ expected_nodes, 0.0).astype(np.float32)
        measurement = np.asarray(
            np.load(sample_dir / "measurement_b.npy"), dtype=np.float32
        )
        measurement_errors.append(
            float(np.max(np.abs(measurement - expected_measurement)))
        )
        scale = float(np.asarray(np.load(sample_dir / "gt_scale.npy")).reshape(()))
        scale_errors.append(abs(scale - float(expected_voxel.max())))
        negative_node_fractions.append(float(np.mean(node < 0.0)))
        negative_measurement_fractions.append(float(np.mean(measurement < 0.0)))
        measurement_maxima.append(float(measurement.max()))
        node_to_voxel_peak_ratios.append(
            float(node.max()) / max(float(expected_voxel.max()), 1e-30)
        )

        valid_mass = float(
            expected_voxel.ravel()[canonical.operator.valid_flat_indices].sum(
                dtype=np.float64
            )
        )
        total_mass = float(expected_voxel.sum(dtype=np.float64))
        outside_valid_mass_fractions.append(
            max(total_mass - valid_mass, 0.0) / max(total_mass, 1e-30)
        )
        for name in (
            "tumor_params.json",
            f"{sample_id}.json",
            f"source-{sample_id}.bin",
            "proj.npz",
        ):
            if not (sample_dir / name).samefile(source_dir / name):
                hardlink_failures.append(f"{sample_id}/{name}")
        if index % 250 == 0:
            print(f"[audit] {index}/{len(ids)}", flush=True)

    result = {
        "status": "passed",
        "dataset_root": str(args.dataset_root.resolve()),
        "source_root": str(args.source_root.resolve()),
        "n_samples": len(ids),
        "split_counts": {name: len(values) for name, values in derived_splits.items()},
        "max_abs_voxel_error": max(voxel_errors),
        "max_abs_node_error": max(node_errors),
        "max_abs_measurement_error": max(measurement_errors),
        "max_abs_scale_error": max(scale_errors),
        "negative_node_fraction": describe(negative_node_fractions),
        "negative_measurement_fraction": describe(negative_measurement_fractions),
        "measurement_max": describe(measurement_maxima),
        "node_to_voxel_peak_ratio": describe(node_to_voxel_peak_ratios),
        "outside_valid_mass_fraction": describe(outside_valid_mass_fractions),
        "outside_valid_mass_counts": {
            f"greater_than_{threshold:g}": int(
                np.count_nonzero(np.asarray(outside_valid_mass_fractions) > threshold)
            )
            for threshold in (0.001, 0.01, 0.05, 0.1, 0.2)
        },
        "hardlink_failures": hardlink_failures,
        "checks": {
            "voxel_exact": max(voxel_errors) == 0.0,
            "nodes_exact": max(node_errors) == 0.0,
            "measurements_exact": max(measurement_errors) == 0.0,
            "scale_exact": max(scale_errors) == 0.0,
            "nodes_nonnegative": max(negative_node_fractions) == 0.0,
            "measurements_nonnegative": max(negative_measurement_fractions) == 0.0,
            "every_case_has_nonzero_measurement": min(measurement_maxima) > 0.0,
            "outside_domain_source_mass_characterized": True,
            "reused_assets_are_hardlinks": not hardlink_failures,
        },
    }
    if not all(result["checks"].values()):
        result["status"] = "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise RuntimeError("Derived dataset audit failed")


if __name__ == "__main__":
    main()
