#!/usr/bin/env python3
"""Build a derived D1-Q cohort whose voxel GT is the saved MCX source pattern."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.bridge.canonical_cross_discretization import CanonicalCrossDiscretization
from du2vox.utils.frame import FrameManifest

SPARSE_FORMAT = b"sparse_flat_c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_ids(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if line]


def embed_source(sample_dir: Path, shape: tuple[int, ...]) -> tuple[np.ndarray, dict]:
    sample_id = sample_dir.name
    config_path = sample_dir / f"{sample_id}.json"
    config = json.loads(config_path.read_text())
    source = config["Optode"]["Source"]
    pattern_meta = source["Pattern"]
    nz, ny, nx = (int(value) for value in source["Param1"][:3])
    pos_z, pos_y, pos_x = (int(value) for value in source["Pos"])
    source_path = sample_dir / pattern_meta["Data"]
    raw = np.fromfile(source_path, dtype=np.float32)
    if raw.size != nx * ny * nz:
        raise RuntimeError(
            f"{sample_id}: source size {raw.size} != pattern size {nx * ny * nz}"
        )
    full = np.zeros(shape, dtype=np.float32)
    full[pos_x : pos_x + nx, pos_y : pos_y + ny, pos_z : pos_z + nz] = raw.reshape(
        nx, ny, nz
    )
    identity = {
        "source_binary": str(source_path.resolve()),
        "source_binary_sha256": sha256(source_path),
        "mcx_json": str(config_path.resolve()),
        "mcx_json_sha256": sha256(config_path),
        "pos_zyx": [pos_z, pos_y, pos_x],
        "pattern_shape_zyx": [nz, ny, nx],
    }
    return full, identity


def save_sparse_volume(path: Path, volume: np.ndarray) -> None:
    array = np.asarray(volume, dtype=np.float32)
    flat = array.ravel()
    indices = np.flatnonzero(flat).astype(np.int32)
    np.savez_compressed(
        path,
        format=np.asarray(SPARSE_FORMAT),
        shape=np.asarray(array.shape, dtype=np.int32),
        indices=indices,
        values=flat[indices],
    )


def hardlink(source: Path, destination: Path) -> None:
    if destination.exists():
        if os.path.samefile(source, destination):
            return
        raise FileExistsError(f"Refusing to replace non-linked file: {destination}")
    os.link(source, destination)


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
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--operator-cache", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    if args.source_root.resolve() == args.output_root.resolve():
        raise RuntimeError("Derived dataset must not overwrite the historical dataset")
    if args.output_root.exists() and not args.output_root.is_dir():
        raise RuntimeError(f"Output is not a directory: {args.output_root}")

    canonical = CanonicalCrossDiscretization(
        args.operator_cache,
        shared_dir=args.shared_dir,
        factorize=False,
        allow_stale_frame_manifest=True,
    )
    frame = FrameManifest.load(args.shared_dir)
    nodes, _ = FrameManifest.load_mesh_nodes(args.shared_dir)
    node_grid_indices = frame.world_to_gt_index(nodes).T
    shape = tuple(canonical.operator.grid_shape)
    valid = canonical.operator.valid_flat_indices
    matrix_archive = np.load(args.shared_dir / "system_matrix.A.npz", allow_pickle=True)
    matrix = np.asarray(matrix_archive["forward_matrix"], dtype=np.float32)
    if matrix.shape[1] != canonical.p.shape[1]:
        raise RuntimeError(
            f"A columns {matrix.shape[1]} != FEM nodes {canonical.p.shape[1]}"
        )

    splits = {
        split: read_ids(args.source_root / "splits" / f"{split}.txt")
        for split in ("train", "val", "test")
    }
    all_ids = [sample_id for split in ("train", "val", "test") for sample_id in splits[split]]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("Source splits contain duplicate sample IDs")
    selected = all_ids[: args.max_samples] if args.max_samples is not None else all_ids

    args.output_root.mkdir(parents=True, exist_ok=True)
    split_output = args.output_root / "splits"
    sample_output = args.output_root / "samples"
    split_output.mkdir(exist_ok=True)
    sample_output.mkdir(exist_ok=True)
    for split in splits:
        shutil.copy2(
            args.source_root / "splits" / f"{split}.txt",
            split_output / f"{split}.txt",
        )

    projection_errors: list[float] = []
    negative_node_fractions: list[float] = []
    negative_node_mass_ratios: list[float] = []
    negative_measurement_fractions: list[float] = []
    outside_valid_mass_fractions: list[float] = []
    records = {}

    for index, sample_id in enumerate(selected, start=1):
        source_dir = args.source_root / "samples" / sample_id
        destination = sample_output / sample_id
        destination.mkdir(exist_ok=True)
        voxel_gt, source_identity = embed_source(source_dir, shape)
        full_mass = float(np.sum(voxel_gt, dtype=np.float64))
        valid_values = voxel_gt.ravel()[valid].astype(np.float64)
        valid_mass = float(valid_values.sum())
        outside_fraction = max(full_mass - valid_mass, 0.0) / max(full_mass, 1e-30)
        outside_valid_mass_fractions.append(outside_fraction)

        # Define a continuous, positivity-preserving trilinear interpolant through
        # the exact MCX voxel-center source samples, then sample it at FEM nodes.
        # This mirrors the historical analytic-field-at-nodes forward contract.
        nodal_gt64 = map_coordinates(
            voxel_gt,
            node_grid_indices,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ).astype(np.float64)
        projected = canonical.prolong(nodal_gt64)
        projection_error = float(
            np.linalg.norm(projected - valid_values)
            / max(np.linalg.norm(valid_values), 1e-30)
        )
        negative_nodes = nodal_gt64 < 0.0
        negative_fraction = float(np.mean(negative_nodes))
        negative_mass_ratio = float(
            -np.minimum(nodal_gt64, 0.0).sum()
            / max(np.maximum(nodal_gt64, 0.0).sum(), 1e-30)
        )
        nodal_gt = nodal_gt64.astype(np.float32)
        # Match FMT-SimGen FEMSolver.forward, which clips small signed numerical
        # surface responses after the linear solve/matrix application.
        measurement = np.maximum(
            np.asarray(matrix @ nodal_gt, dtype=np.float32), 0.0
        )
        negative_measurement_fraction = float(np.mean(measurement < 0.0))

        projection_errors.append(projection_error)
        negative_node_fractions.append(negative_fraction)
        negative_node_mass_ratios.append(negative_mass_ratio)
        negative_measurement_fractions.append(negative_measurement_fraction)

        save_sparse_volume(destination / "gt_voxels.npz", voxel_gt)
        np.save(destination / "gt_nodes.npy", nodal_gt)
        np.save(destination / "measurement_b.npy", measurement)
        np.save(destination / "gt_scale.npy", np.asarray(voxel_gt.max(), dtype=np.float32))
        for name in (
            "tumor_params.json",
            f"{sample_id}.json",
            f"source-{sample_id}.bin",
            "proj.npz",
        ):
            hardlink(source_dir / name, destination / name)

        record = {
            "sample_id": sample_id,
            "canonical_voxel_gt": "saved MCX source pattern embedded in full XYZ grid",
            "nodal_gt": (
                "positive trilinear voxel-center field sampled at FEM nodes"
            ),
            "measurement": "maximum(system_matrix.A @ gt_nodes, 0)",
            "source_identity": source_identity,
            "voxel_nonzero": int(np.count_nonzero(voxel_gt)),
            "voxel_peak": float(voxel_gt.max()),
            "shared_normalization_scale": "gt_scale.npy = canonical voxel GT peak",
            "outside_valid_mass_fraction": outside_fraction,
            "projection_relative_l2": projection_error,
            "negative_node_fraction": negative_fraction,
            "negative_node_mass_ratio": negative_mass_ratio,
            "negative_measurement_fraction": negative_measurement_fraction,
        }
        (destination / "derivation.json").write_text(json.dumps(record, indent=2) + "\n")
        records[sample_id] = record
        if index % 25 == 0 or index == len(selected):
            print(f"[derive] {index}/{len(selected)} {sample_id}", flush=True)

    manifest = {
        "contract": "D1-Q MCX-canonical derived field v1",
        "status": "complete" if len(selected) == len(all_ids) else "smoke_incomplete",
        "source_dataset": str(args.source_root.resolve()),
        "source_distribution": (
            "historical D1-Q max-composed axis-aligned Gaussian/irregular fields"
        ),
        "canonical_voxel_gt": "saved MCX source pattern actually used for proj.npz",
        "nodal_contract": (
            "positive trilinear interpolation of MCX voxel-center field at FEM nodes"
        ),
        "measurement_contract": "measurement_b=max(system_matrix.A@gt_nodes, 0)",
        "normalization": (
            "raw files stored; requested per-case normalization is applied by loaders"
        ),
        "split_counts": {name: len(ids) for name, ids in splits.items()},
        "n_derived": len(selected),
        "shared_dir": str(args.shared_dir.resolve()),
        "operator_cache": str(args.operator_cache.resolve()),
        "operator_hashes": canonical.cache_hashes,
        "system_matrix_sha256": sha256(args.shared_dir / "system_matrix.A.npz"),
        "summary": {
            "projection_relative_l2": describe(projection_errors),
            "negative_node_fraction": describe(negative_node_fractions),
            "negative_node_mass_ratio": describe(negative_node_mass_ratios),
            "negative_measurement_fraction": describe(negative_measurement_fractions),
            "outside_valid_mass_fraction": describe(outside_valid_mass_fractions),
        },
        "samples": records,
    }
    (args.output_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
