#!/usr/bin/env python3
"""Freeze identity of the derived MCX-canonical D1-Q benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, str | int]:
    if not path.exists():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def read_ids(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--operator-cache", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text())
    if audit.get("status") != "passed" or audit.get("n_samples") != 3000:
        raise RuntimeError("Dataset requires a passing independent 3000-case audit")
    manifest = json.loads((args.dataset_root / "dataset_manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("n_derived") != 3000:
        raise RuntimeError("Derived dataset manifest is incomplete")

    split_paths = {
        name: args.dataset_root / "splits" / f"{name}.txt"
        for name in ("train", "val", "test")
    }
    splits = {name: read_ids(path) for name, path in split_paths.items()}
    counts = {name: len(ids) for name, ids in splits.items()}
    if counts != {"train": 2400, "val": 300, "test": 300}:
        raise RuntimeError(f"Unexpected split counts: {counts}")
    if any(len(ids) != len(set(ids)) for ids in splits.values()):
        raise RuntimeError("Duplicate IDs within a split")
    if set(splits["train"]) & set(splits["val"]):
        raise RuntimeError("train/val overlap")
    if set(splits["train"]) & set(splits["test"]):
        raise RuntimeError("train/test overlap")
    if set(splits["val"]) & set(splits["test"]):
        raise RuntimeError("val/test overlap")

    cases = {}
    for index, (split, ids) in enumerate(splits.items()):
        for case_index, sample_id in enumerate(ids, start=1):
            sample_dir = args.dataset_root / "samples" / sample_id
            source_dir = args.source_root / "samples" / sample_id
            generated = (
                "gt_voxels.npz",
                "gt_nodes.npy",
                "measurement_b.npy",
                "gt_scale.npy",
                "derivation.json",
            )
            reused = (
                "tumor_params.json",
                f"{sample_id}.json",
                f"source-{sample_id}.bin",
                "proj.npz",
            )
            for name in reused:
                if not (sample_dir / name).samefile(source_dir / name):
                    raise RuntimeError(f"{sample_id}/{name} is not the frozen hardlink")
            cases[sample_id] = {
                "split": split,
                "generated": {name: record(sample_dir / name) for name in generated},
                "reused": {name: record(sample_dir / name) for name in reused},
            }
            if case_index % 250 == 0:
                print(f"[hash] {split} {case_index}/{len(ids)}", flush=True)

    shared_names = (
        "mesh.npz",
        "system_matrix.A.npz",
        "frame_manifest.json",
        "view_config.json",
        "mcx_volume_trunk.bin",
        "mcx_material.yaml",
    )
    operator_names = (
        "P_full_fem_gt_centers.npz",
        "domain_arrays.npz",
        "operator_metadata.json",
    )
    receipt = {
        "status": "frozen",
        "contract": "D1-Q MCX-canonical derived field v1",
        "normalization": (
            "raw storage; shared per-case voxel peak; prediction is not normalized post hoc"
        ),
        "sealed_confirmation_accessed": False,
        "development_test_predictions_accessed": False,
        "dataset_root": str(args.dataset_root.resolve()),
        "source_root": str(args.source_root.resolve()),
        "split_counts": counts,
        "split_files": {name: record(path) for name, path in split_paths.items()},
        "manifest": record(args.dataset_root / "dataset_manifest.json"),
        "audit": record(args.audit),
        "shared_assets": {
            name: record(args.shared_dir / name) for name in shared_names
        },
        "operator_cache": {
            name: record(args.operator_cache / name) for name in operator_names
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(cases)}, indent=2))


if __name__ == "__main__":
    main()
