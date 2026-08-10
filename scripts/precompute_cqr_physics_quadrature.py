#!/usr/bin/env python3
"""Precompute fixed, role-stratified physics quadrature for CQR samples."""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.bridge.coverage_field import QueryRole
from du2vox.bridge.coverage_field import correction_band_distance
from du2vox.bridge.fem_lift_indicators import compute_lifting_indicators
from du2vox.physics.tetra_quadrature import four_point_tetra_quadrature
from du2vox.physics.tetra_quadrature import tetrahedron_volumes


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_split(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def lifting_config(cqr_cfg: dict) -> tuple[float, float, dict]:
    lifting = cqr_cfg.get("lifting", {}) or {}
    return (
        float(lifting.get("tau_core", 0.65)),
        float(lifting.get("tau_halo", 0.18)),
        lifting.get("weights", {}) or {},
    )


def assign_target_roles(
    target_tets: np.ndarray,
    candidate_tets: np.ndarray,
    candidate_roles: np.ndarray,
    coarse_d: np.ndarray,
    elements: np.ndarray,
    tau_core: float,
    tau_halo: float,
) -> np.ndarray:
    tet_max = coarse_d[elements[target_tets]].max(axis=1)
    roles = np.full(len(target_tets), int(QueryRole.BG), dtype=np.int64)
    roles[tet_max >= tau_halo] = int(QueryRole.HALO)
    roles[tet_max >= tau_core] = int(QueryRole.CORE)
    lookup = {int(tet): index for index, tet in enumerate(target_tets)}
    counts = np.zeros((len(target_tets), 5), dtype=np.int64)
    valid = (candidate_tets >= 0) & (candidate_tets < len(elements))
    for tet, role in zip(candidate_tets[valid], candidate_roles[valid], strict=False):
        index = lookup.get(int(tet))
        if index is not None and 0 <= role < 5:
            counts[index, role] += 1
    has_candidate = counts.sum(axis=1) > 0
    counts[:, int(QueryRole.SENTINEL)] = 0
    roles[has_candidate] = counts[has_candidate].argmax(axis=1)
    # Preserve proposal as an explicit physical stratum whenever it contributed.
    roles[counts[:, int(QueryRole.PROPOSAL)] > 0] = int(QueryRole.PROPOSAL)
    return roles


def stratified_tet_sample(
    target_tets: np.ndarray,
    roles: np.ndarray,
    residual_indicator: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    if count >= len(target_tets):
        return np.sort(target_tets)
    residual = residual_indicator[target_tets]
    edges = np.quantile(residual, [0.25, 0.50, 0.75])
    quantile = np.digitize(residual, edges, right=True)
    rng = np.random.default_rng(seed)
    strata: list[np.ndarray] = []
    for role in (int(QueryRole.CORE), int(QueryRole.HALO), int(QueryRole.BG), int(QueryRole.PROPOSAL)):
        for qbin in range(4):
            values = target_tets[(roles == role) & (quantile == qbin)]
            if len(values):
                strata.append(rng.permutation(values))
    selected: list[int] = []
    cursor = np.zeros(len(strata), dtype=np.int64)
    while len(selected) < count:
        progressed = False
        for index, values in enumerate(strata):
            if cursor[index] < len(values):
                selected.append(int(values[cursor[index]]))
                cursor[index] += 1
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    return np.asarray(sorted(selected), dtype=np.int64)


def build_quadrature(
    *,
    sid: str,
    nodes: np.ndarray,
    elements: np.ndarray,
    coarse_d: np.ndarray,
    roi_tets: np.ndarray,
    cqr: dict[str, np.ndarray],
    cqr_cfg: dict,
    mode: str,
    max_tets: int,
) -> dict[str, np.ndarray]:
    candidate_tets = np.asarray(cqr["tet_ids"], dtype=np.int64)
    candidate_roles = np.asarray(cqr["role"], dtype=np.int64)
    proposal_tets = np.unique(
        candidate_tets[(candidate_roles == int(QueryRole.PROPOSAL)) & (candidate_tets >= 0)]
    )
    target_tets = np.union1d(np.asarray(roi_tets, dtype=np.int64), proposal_tets)
    target_tets = target_tets[(target_tets >= 0) & (target_tets < len(elements))]
    tau_core, tau_halo, weights_cfg = lifting_config(cqr_cfg)
    indicators = compute_lifting_indicators(
        nodes, elements, coarse_d, tau_core=tau_core, tau_halo=tau_halo, weights=weights_cfg
    )
    target_roles = assign_target_roles(
        target_tets,
        candidate_tets,
        candidate_roles,
        coarse_d,
        elements,
        tau_core,
        tau_halo,
    )
    if mode == "all":
        selected = target_tets
    elif mode == "stratified":
        selected = stratified_tet_sample(
            target_tets,
            target_roles,
            indicators["residual_indicator"],
            max_tets,
            int(zlib.crc32(sid.encode()) & 0xFFFFFFFF),
        )
    else:
        raise ValueError(f"Unsupported physics_tet_mode={mode!r}")

    target_role_by_tet = dict(zip(target_tets.tolist(), target_roles.tolist(), strict=True))
    selected_roles = np.asarray([target_role_by_tet[int(tet)] for tet in selected], dtype=np.int64)
    points, barycentric, quadrature_weight = four_point_tetra_quadrature(nodes, elements, selected)
    vertices = elements[selected]
    node_values = coarse_d[vertices]
    prior_8d = np.concatenate(
        [
            np.broadcast_to(node_values[:, None, :], barycentric.shape),
            barycentric,
        ],
        axis=2,
    )
    p1 = np.sum(node_values[:, None, :] * barycentric, axis=2)
    bands = np.broadcast_to(selected_roles[:, None], (len(selected), 4))
    lift_columns = [
        p1,
        np.broadcast_to(indicators["tet_grad_norm"][selected, None], p1.shape),
        np.broadcast_to(indicators["grad_jump_score"][selected, None], p1.shape),
        np.broadcast_to(indicators["recovery_error_score"][selected, None], p1.shape),
        np.broadcast_to(indicators["transition_score"][selected, None], p1.shape),
        np.broadcast_to(indicators["residual_indicator"][selected, None], p1.shape),
        correction_band_distance(bands),
    ]
    prior_lift = np.concatenate(
        [prior_8d, *[column[:, :, None] for column in lift_columns]], axis=2
    )
    bbox_min = np.asarray(cqr["bbox_min"], dtype=np.float64)
    bbox_max = np.asarray(cqr["bbox_max"], dtype=np.float64)
    coords_norm = 2.0 * (points - bbox_min) / (bbox_max - bbox_min + 1e-8) - 1.0
    volumes = tetrahedron_volumes(nodes, elements)
    metadata = {
        "schema_version": 1,
        "sample_id": sid,
        "physics_tet_mode": mode,
        "physics_tets_per_sample": max_tets,
        "target_tet_count": int(len(target_tets)),
        "selected_tet_count": int(len(selected)),
        "target_tet_volume": float(volumes[target_tets].sum()),
        "selected_tet_volume": float(volumes[selected].sum()),
        "quadrature": "four-point degree-2 tetrahedral",
    }
    return {
        "coords_world": points.reshape(-1, 3).astype(np.float32),
        "coords_norm": coords_norm.reshape(-1, 3).astype(np.float32),
        "tet_id": np.repeat(selected, 4).astype(np.int64),
        "tet_vertex_indices": np.repeat(vertices, 4, axis=0).astype(np.int64),
        "barycentric": barycentric.reshape(-1, 4).astype(np.float32),
        "quadrature_weight": quadrature_weight.reshape(-1).astype(np.float64),
        "prior_8d": prior_8d.reshape(-1, 8).astype(np.float32),
        "prior_lift": prior_lift.reshape(-1, 15).astype(np.float32),
        "correction_band": bands.reshape(-1).astype(np.int64),
        "role": bands.reshape(-1).astype(np.int64),
        "selected_tet_ids": selected.astype(np.int64),
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--physics_tet_mode", choices=("stratified", "all"), default=None)
    parser.add_argument("--physics_tets_per_sample", type=int, default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    transport = cfg.get("transport", {}) or {}
    mode = args.physics_tet_mode or transport.get("physics_tet_mode", "stratified")
    max_tets = int(args.physics_tets_per_sample or transport.get("physics_tets_per_sample", 1024))
    ids = load_split(cfg["data"][f"{args.split}_split"])
    if args.max_samples is not None:
        ids = ids[: args.max_samples]
    cqr_dir = REPO_ROOT / cfg["data"][f"precomputed_{args.split}_dir"]
    bridge_dir = REPO_ROOT / cfg["data"][f"{args.split}_bridge_dir"]
    root = Path(
        args.output_root
        or transport.get("quadrature_root", "precomputed/cqr_v2_3k_rgl_main_physics_quadrature")
    )
    if not root.is_absolute():
        root = REPO_ROOT / root
    output_dir = root / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(Path(cfg["data"]["shared_dir"]) / "mesh.npz", allow_pickle=False) as mesh:
        nodes = mesh["nodes"].astype(np.float64)
        elements = mesh["elements"].astype(np.int64)

    processed = 0
    for sid in ids:
        cqr_path = cqr_dir / f"{sid}.npz"
        bridge_sample = bridge_dir / sid
        required = [cqr_path, bridge_sample / "coarse_d.npy", bridge_sample / "roi_tet_indices.npy"]
        if not all(path.exists() for path in required):
            print(f"[Quadrature][MISSING] {sid}: {[str(path) for path in required if not path.exists()]}")
            continue
        output_path = output_dir / f"{sid}.npz"
        if output_path.exists() and not args.overwrite:
            print(f"[Quadrature] {sid}: exists, skip")
            continue
        with np.load(cqr_path, allow_pickle=False) as loaded:
            cqr = {key: loaded[key] for key in loaded.files}
        result = build_quadrature(
            sid=sid,
            nodes=nodes,
            elements=elements,
            coarse_d=np.load(bridge_sample / "coarse_d.npy").reshape(-1),
            roi_tets=np.load(bridge_sample / "roi_tet_indices.npy"),
            cqr=cqr,
            cqr_cfg=cfg.get("cqr", {}) or {},
            mode=mode,
            max_tets=max_tets,
        )
        np.savez_compressed(output_path, **result)
        metadata = json.loads(str(result["metadata"]))
        print(
            f"[Quadrature] {sid}: target_tets={metadata['target_tet_count']}, "
            f"selected_tets={metadata['selected_tet_count']}, points={len(result['tet_id'])}, "
            f"selected_volume_ratio={metadata['selected_tet_volume']/metadata['target_tet_volume']:.4f}"
        )
        processed += 1
    print(f"[Quadrature] processed={processed}/{len(ids)}, output={output_dir}")
    if processed == 0:
        raise RuntimeError("No quadrature files were generated")


if __name__ == "__main__":
    main()
