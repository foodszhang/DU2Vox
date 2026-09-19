#!/usr/bin/env python3
"""Create a hash-sealed metadata manifest without loading confirmation GT arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.utils.confirmation import canonical_json_bytes, sha256_file, write_sealed_manifest


def _digest_json_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.parent.name.encode("utf-8"))
        digest.update(canonical_json_bytes(json.loads(path.read_text())))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--generator-root", type=Path, required=True)
    parser.add_argument("--generator-config", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--global-seed", type=int, required=True)
    parser.add_argument("--expected-samples", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development-samples-dir", type=Path)
    args = parser.parse_args()

    samples_dir = args.dataset_root / "samples"
    sample_dirs = sorted(path for path in samples_dir.glob("sample_*") if path.is_dir())
    if len(sample_dirs) != args.expected_samples:
        raise RuntimeError(
            f"Expected {args.expected_samples} confirmation samples, found {len(sample_dirs)}"
        )
    required = ("measurement_b.npy", "gt_nodes.npy", "gt_voxels.npy", "tumor_params.json", "proj.npz")
    incomplete = {
        path.name: [name for name in required if not (path / name).is_file()]
        for path in sample_dirs
    }
    incomplete = {key: value for key, value in incomplete.items() if value}
    if incomplete:
        raise RuntimeError(f"Cannot seal incomplete confirmation samples: {incomplete}")

    metadata_paths = [path / "tumor_params.json" for path in sample_dirs]
    metadata_hash = _digest_json_files(metadata_paths)
    mcx_seeds = {
        path.name: int(
            json.loads((path / f"{path.name}.json").read_text())["Session"][
                "RNGSeed"
            ]
        )
        for path in sample_dirs
    }
    development_overlap = None
    if args.development_samples_dir:
        development_paths = sorted(args.development_samples_dir.glob("sample_*/tumor_params.json"))
        confirmation_records = {
            canonical_json_bytes(json.loads(path.read_text())) for path in metadata_paths
        }
        development_records = {
            canonical_json_bytes(json.loads(path.read_text())) for path in development_paths
        }
        development_overlap = len(confirmation_records & development_records)
        if development_overlap:
            raise RuntimeError("Exact tumor metadata overlap detected with development data")

    generator_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.generator_root, text=True
    ).strip()
    mesh_path = args.shared_dir / "mesh.npz"
    system_path = args.shared_dir / "system_matrix.A.npz"
    frame_path = args.shared_dir / "frame_manifest.json"
    dataset_manifest = args.dataset_root / "dataset_manifest.json"
    manifest = {
        "dataset_name": args.dataset_root.name,
        "generation_date": datetime.now(timezone.utc).isoformat(),
        "generator_commit_hash": generator_commit,
        "generator_worktree_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=args.generator_root, text=True
            ).strip()
        ),
        "generator_config_path": str(args.generator_config.resolve()),
        "generator_config_sha256": sha256_file(args.generator_config),
        "global_random_seed": args.global_seed,
        "sample_level_tumor_seeds": (
            "not emitted; deterministic streams derive from global seed"
        ),
        "sample_level_mcx_rng_seeds": mcx_seeds,
        "sample_ids": [path.name for path in sample_dirs],
        "number_of_samples": len(sample_dirs),
        "sample_index_sha256": hashlib.sha256(
            ("\n".join(path.name for path in sample_dirs) + "\n").encode("ascii")
        ).hexdigest(),
        "sample_metadata_sha256": metadata_hash,
        "exact_tumor_metadata_overlap_with_development": development_overlap,
        "samples_dir": str(samples_dir.resolve()),
        "mesh_version": {
            "path": str(mesh_path.resolve()),
            "sha256": sha256_file(mesh_path),
            "frame_manifest_sha256": sha256_file(frame_path),
        },
        "measurement_configuration": {
            "system_matrix_path": str(system_path.resolve()),
            "system_matrix_sha256": sha256_file(system_path),
            "du2vox_normalize_b": "per-sample positive maximum",
            "visible_mask": False,
            "projection_file": "proj.npz",
            "projection_normalization": "per_view_max",
        },
        "gt_semantics": {
            "source_type": "uniform",
            "training_target": "binary support via gt_voxels > 0.05",
            "voxel_frame": "mcx_trunk_local_mm",
        },
        "dataset_manifest_sha256": sha256_file(dataset_manifest),
        "seal_policy": {
            "threshold": 0.5,
            "confirmation_evaluation_requires_explicit_flag": True,
            "model_selection_allowed": False,
        },
    }
    digest = write_sealed_manifest(manifest, args.output)
    split_path = args.output.with_name("confirmation_ids.txt")
    split_path.write_text("\n".join(path.name for path in sample_dirs) + "\n")
    split_path.with_suffix(".sha256").write_text(
        f"{sha256_file(split_path)}  {split_path.name}\n"
    )
    print(f"Sealed {len(sample_dirs)} samples: {digest}")


if __name__ == "__main__":
    main()
