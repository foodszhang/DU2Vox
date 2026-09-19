#!/usr/bin/env python3
"""Create an immutable identity receipt for the formal continuous benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_ids(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if line]


def file_record(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--generator-config", type=Path, required=True)
    parser.add_argument("--generator-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-projections", action="store_true")
    args = parser.parse_args()

    split_paths = {
        name: args.dataset_root / "splits" / f"{name}.txt"
        for name in ("train", "val", "test")
    }
    splits = {name: read_ids(path) for name, path in split_paths.items()}
    expected = {"train": 2400, "val": 300, "test": 300}
    actual = {name: len(ids) for name, ids in splits.items()}
    if actual != expected:
        raise RuntimeError(f"Split sizes {actual} != frozen contract {expected}")
    for name, ids in splits.items():
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"Duplicate IDs in {name}")
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(splits[first]) & set(splits[second])
        if overlap:
            raise RuntimeError(f"{first}/{second} overlap: {sorted(overlap)[:5]}")

    sample_dirs = sorted(
        path.name for path in (args.dataset_root / "samples").glob("sample_*")
    )
    declared = sorted(sid for ids in splits.values() for sid in ids)
    if sample_dirs != declared:
        raise RuntimeError("Split union does not exactly match the sample directories")

    required = ["tumor_params.json", "measurement_b.npy", "gt_nodes.npy", "gt_voxels.npz"]
    if args.require_projections:
        required.append("proj.npz")
    cases: dict[str, Any] = {}
    for index, sid in enumerate(declared, start=1):
        sample_dir = args.dataset_root / "samples" / sid
        params = json.loads((sample_dir / "tumor_params.json").read_text())
        if params.get("source_type") != "gaussian_mixture":
            raise RuntimeError(f"{sid}: unexpected source_type={params.get('source_type')}")
        cases[sid] = {
            "split": next(name for name, ids in splits.items() if sid in ids),
            "num_foci": int(params["num_foci"]),
            "files": {name: file_record(sample_dir / name) for name in required},
        }
        if index % 250 == 0:
            print(f"[hash] {index}/{len(declared)}", flush=True)

    shared_names = [
        "mesh.npz",
        "system_matrix.A.npz",
        "frame_manifest.json",
        "view_config.json",
        "mcx_volume_trunk.bin",
        "mcx_material.yaml",
    ]
    generator_files = [
        "fmt_simgen/tumor/gaussian_mixture.py",
        "fmt_simgen/tumor/tumor_generator.py",
        "fmt_simgen/mcx_source.py",
        "fmt_simgen/dataset/builder.py",
    ]
    receipt = {
        "contract": "LPR continuous anisotropic additive Gaussian mixture v1",
        "amplitude_contract": "raw GT and raw DE measurement; no per-case normalization",
        "split_counts": actual,
        "split_files": {name: file_record(path) for name, path in split_paths.items()},
        "dataset_manifest": file_record(args.dataset_root / "dataset_manifest.json"),
        "generator_config": file_record(args.generator_config),
        "generator_code": {
            name: file_record(args.generator_root / name) for name in generator_files
        },
        "shared_assets": {
            name: file_record(args.shared_dir / name) for name in shared_names
        },
        "projections_required": args.require_projections,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(cases)}, indent=2))


if __name__ == "__main__":
    main()
