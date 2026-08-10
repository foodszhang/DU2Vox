#!/usr/bin/env python3
"""Build non-destructive physics-weight sidecars for existing CQR pools."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_split(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def tetrahedron_volumes(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    vertices = np.asarray(nodes, dtype=np.float64)[np.asarray(elements, dtype=np.int64)]
    jacobian = np.stack(
        [
            vertices[:, 1] - vertices[:, 0],
            vertices[:, 2] - vertices[:, 0],
            vertices[:, 3] - vertices[:, 0],
        ],
        axis=1,
    )
    return np.abs(np.linalg.det(jacobian)) / 6.0


def build_sidecar(
    cqr: dict[str, np.ndarray],
    tet_volumes: np.ndarray,
    roi_tet_indices: np.ndarray,
    *,
    sample_id: str,
    source_path: Path,
) -> dict[str, np.ndarray]:
    tet_ids = np.asarray(cqr["tet_ids"], dtype=np.int64)
    prior_8d = np.asarray(cqr["prior_8d"], dtype=np.float64)
    if len(tet_ids) != len(prior_8d):
        raise ValueError("tet_ids and prior_8d must share the candidate dimension")

    barycentric = prior_8d[:, 4:8]
    valid_tet = (tet_ids >= 0) & (tet_ids < len(tet_volumes))
    valid_barycentric = (
        np.isfinite(barycentric).all(axis=1)
        & (barycentric >= -1e-5).all(axis=1)
        & (barycentric <= 1.0 + 1e-5).all(axis=1)
        & np.isclose(barycentric.sum(axis=1), 1.0, atol=2e-4)
    )
    valid_physics = valid_tet & valid_barycentric

    # Never index tet arrays with -1: all gathering is guarded by valid_physics.
    candidate_volume = np.zeros(len(tet_ids), dtype=np.float64)
    candidate_volume[valid_physics] = tet_volumes[tet_ids[valid_physics]]
    counts_per_tet = np.bincount(tet_ids[valid_physics], minlength=len(tet_volumes))
    candidate_count = np.zeros(len(tet_ids), dtype=np.int64)
    candidate_count[valid_physics] = counts_per_tet[tet_ids[valid_physics]]
    candidate_weight = np.zeros(len(tet_ids), dtype=np.float64)
    candidate_weight[valid_physics] = (
        candidate_volume[valid_physics] / candidate_count[valid_physics]
    )

    represented_tets = np.unique(tet_ids[valid_physics])
    role = np.asarray(cqr.get("role", np.full(len(tet_ids), -1)), dtype=np.int64)
    proposal_tets = np.unique(tet_ids[valid_physics & (role == 4)])
    roi = np.unique(np.asarray(roi_tet_indices, dtype=np.int64))
    roi = roi[(roi >= 0) & (roi < len(tet_volumes))]
    target_tets = np.union1d(roi, proposal_tets)
    roi_volume = float(tet_volumes[roi].sum())
    target_volume = float(tet_volumes[target_tets].sum())
    represented_target_tets = np.intersect1d(represented_tets, target_tets, assume_unique=True)
    represented_volume = float(tet_volumes[represented_target_tets].sum())
    all_candidate_volume = float(tet_volumes[represented_tets].sum())
    coverage_ratio = represented_volume / max(target_volume, np.finfo(np.float64).tiny)
    metadata = {
        "schema_version": 1,
        "sample_id": sample_id,
        "source_cqr_npz": str(source_path),
        "weight_contract": "V_tet / n_tet over the complete valid 32768 candidate pool",
        "n_candidate_pool": int(len(tet_ids)),
        "negative_tet_count": int((tet_ids < 0).sum()),
        "target_physical_tet_count": int(len(target_tets)),
        "target_physical_tet_volume": target_volume,
        "all_candidate_represented_tet_volume": all_candidate_volume,
        "tet_coverage_ratio": coverage_ratio,
    }
    return {
        "candidate_tet_volume": candidate_volume.astype(np.float32),
        "candidate_tet_query_count": candidate_count.astype(np.int32),
        "candidate_cell_weight": candidate_weight.astype(np.float32),
        "candidate_valid_physics_mask": valid_physics.astype(np.bool_),
        "n_valid_candidate_pool": np.int64(valid_physics.sum()),
        "unique_tet_count": np.int64(len(represented_tets)),
        "represented_tet_volume": np.float64(represented_volume),
        "roi_total_tet_volume": np.float64(roi_volume),
        "proposal_tet_count": np.int64(len(proposal_tets)),
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    split_ids = load_split(cfg["data"][f"{args.split}_split"])
    if args.max_samples is not None:
        split_ids = split_ids[: args.max_samples]
    cqr_dir = REPO_ROOT / cfg["data"][f"precomputed_{args.split}_dir"]
    bridge_dir = REPO_ROOT / cfg["data"][f"{args.split}_bridge_dir"]
    output_root = Path(
        args.output_root
        or (cfg.get("transport", {}) or {}).get(
            "sidecar_root", "precomputed/cqr_v2_3k_rgl_main_physics"
        )
    )
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    output_dir = output_root / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = Path(cfg["data"]["shared_dir"]) / "mesh.npz"
    with np.load(mesh_path, allow_pickle=False) as mesh:
        nodes = mesh["nodes"]
        elements = mesh["elements"]
    volumes = tetrahedron_volumes(nodes, elements)

    processed = 0
    for sid in split_ids:
        cqr_path = cqr_dir / f"{sid}.npz"
        if not cqr_path.exists():
            print(f"[Sidecar][MISSING] {cqr_path}")
            continue
        output_path = output_dir / f"{sid}.npz"
        if output_path.exists() and not args.overwrite:
            print(f"[Sidecar] {sid}: exists, skip")
            continue
        roi_path = bridge_dir / sid / "roi_tet_indices.npy"
        if not roi_path.exists():
            print(f"[Sidecar][MISSING] {roi_path}")
            continue
        with np.load(cqr_path, allow_pickle=False) as loaded:
            cqr = {key: loaded[key] for key in loaded.files}
        sidecar = build_sidecar(
            cqr,
            volumes,
            np.load(roi_path),
            sample_id=sid,
            source_path=cqr_path,
        )
        np.savez_compressed(output_path, **sidecar)
        metadata = json.loads(str(sidecar["metadata"]))
        print(
            f"[Sidecar] {sid}: valid={int(sidecar['n_valid_candidate_pool'])}/{len(cqr['tet_ids'])}, "
            f"unique_tets={int(sidecar['unique_tet_count'])}, proposal_tets={int(sidecar['proposal_tet_count'])}, "
            f"represented_volume={float(sidecar['represented_tet_volume']):.6f}, "
            f"target_volume={metadata['target_physical_tet_volume']:.6f}, "
            f"coverage={metadata['tet_coverage_ratio']:.4f}"
        )
        processed += 1
    print(f"[Sidecar] processed={processed}/{len(split_ids)}, output={output_dir}")
    if processed == 0:
        raise RuntimeError("No sidecars were generated; existing CQR NPZ assets are required")


if __name__ == "__main__":
    main()
