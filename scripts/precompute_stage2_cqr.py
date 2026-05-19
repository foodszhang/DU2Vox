#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).parent.parent))

from du2vox.bridge.cqr_query_builder import CQRQueryBuilder
from du2vox.bridge.fem_bridging import FEMBridge
from du2vox.bridge.view_evidence import compute_view_evidence, load_proj_npz
from du2vox.utils.frame import FrameManifest


def load_split(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def sample_seed(sid: str) -> int:
    return int(zlib.crc32(sid.encode("utf-8")) & 0xFFFFFFFF)


def normalize_coords(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = points.min(axis=0).astype(np.float32)
    hi = points.max(axis=0).astype(np.float32)
    coords_norm = (2.0 * (points - lo) / (hi - lo + 1e-8) - 1.0).astype(np.float32)
    return coords_norm, lo, hi


def gt_index_to_world(frame: FrameManifest, idx: np.ndarray) -> np.ndarray:
    return (
        np.asarray(idx, dtype=np.float64) * frame.gt_spacing_mm
        + frame.gt_offset_world_mm
        + frame.gt_spacing_mm / 2
    ).astype(np.float32)


def apply_oracle_sentinel(
    data: dict[str, np.ndarray],
    gt_voxels: np.ndarray,
    frame: FrameManifest,
    nodes: np.ndarray,
    elements: np.ndarray,
    coarse_d: np.ndarray,
    n_candidates: int,
    oracle_cfg: dict,
    seed: int,
) -> None:
    if not oracle_cfg.get("enabled", False):
        return
    ratio = float(oracle_cfg.get("oracle_sentinel_ratio", 0.05))
    n_oracle = int(round(len(data["grid_coords"]) * ratio))
    if n_oracle <= 0:
        return

    pos_idx = np.argwhere(gt_voxels >= float(oracle_cfg.get("gt_threshold", 0.5)))
    if len(pos_idx) == 0:
        return
    rng = np.random.default_rng(seed + 17)
    chosen = pos_idx[rng.choice(len(pos_idx), size=n_oracle, replace=len(pos_idx) < n_oracle)]
    points = gt_index_to_world(frame, chosen)

    replace = np.where(data["role"] == 3)[0]
    if len(replace) == 0:
        replace = np.arange(len(data["grid_coords"]))[-n_oracle:]
    if len(replace) > n_oracle:
        replace = replace[:n_oracle]
    points = points[: len(replace)]

    data["grid_coords"][replace] = points
    data["grid_coords_norm"], data["bbox_min"], data["bbox_max"] = normalize_coords(data["grid_coords"])
    bridge = FEMBridge(nodes, elements, n_candidates=n_candidates)
    prior_8d, valid = bridge.get_prior_features(points, coarse_d, K=n_candidates)
    data["prior_8d"][replace] = prior_8d
    if "prior_ext" in data:
        data["prior_ext"][replace, :8] = prior_8d
    data["gt_values"][replace] = 1.0
    data["valid_mask"][replace] = valid
    data["role"][replace] = 3


def precompute_one(
    sid: str,
    bridge_dir: Path,
    samples_dir: Path,
    nodes: np.ndarray,
    elements: np.ndarray,
    frame: FrameManifest,
    n_query_points: int,
    n_candidates: int,
    cqr_cfg: dict,
) -> dict[str, np.ndarray]:
    bd = bridge_dir / sid
    coarse_d = np.load(bd / "coarse_d.npy").astype(np.float32)
    roi_tet_indices = np.load(bd / "roi_tet_indices.npy").astype(np.int64)
    cqr_cfg = dict(cqr_cfg)
    if cqr_cfg.get("sentinel", {}).get("mode") == "view_guided":
        proj_path = samples_dir / sid / "proj.npz"
        if not proj_path.exists():
            raise FileNotFoundError(f"view_guided sentinel requires {proj_path}")
        centroids = nodes[elements].mean(axis=1).astype(np.float32)
        view_score, _ = compute_view_evidence(centroids, load_proj_npz(str(proj_path)))
        cqr_cfg["view_evidence"] = view_score

    builder = CQRQueryBuilder(
        nodes=nodes,
        elements=elements,
        config={
            **cqr_cfg,
            "n_query_points": int(n_query_points),
            "n_candidates": int(n_candidates),
        },
    )
    cqr = builder.build(
        coarse_d=coarse_d,
        roi_tet_indices=roi_tet_indices,
        n_query_points=n_query_points,
        seed=sample_seed(sid),
    )

    points = cqr["query_points"].astype(np.float32)
    coords_norm, bbox_min, bbox_max = normalize_coords(points)

    gt_voxels = np.load(samples_dir / sid / "gt_voxels.npy").astype(np.float32)
    idx_float = frame.world_to_gt_index(points)
    gt_values = map_coordinates(
        gt_voxels,
        idx_float.T,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype(np.float32)

    shape_arr = np.array(gt_voxels.shape)
    outside_gt = np.any((idx_float < 0) | (idx_float > shape_arr - 1), axis=1)
    valid = cqr["valid_mask"].astype(bool) & (~outside_gt)
    gt_values[outside_gt] = 0.0

    out = {
        "grid_coords": points,
        "grid_coords_norm": coords_norm,
        "prior_8d": cqr["prior_8d"].astype(np.float32),
        "prior_ext": cqr["prior_ext"].astype(np.float32),
        "gt_values": gt_values.astype(np.float32),
        "valid_mask": valid.astype(bool),
        "grid_shape": np.array([len(points), 1, 1], dtype=np.int32),
        "bbox_min": bbox_min.astype(np.float32),
        "bbox_max": bbox_max.astype(np.float32),
        "tet_ids": cqr["tet_ids"].astype(np.int64),
        "role": cqr["role"].astype(np.int64),
        "coverage_score": cqr["coverage_score"].astype(np.float32),
        "view_evidence_score": cqr["view_evidence_score"].astype(np.float32),
        "sentinel_score": cqr["sentinel_score"].astype(np.float32),
        "risk_components": cqr["risk_components"].astype(np.float32),
        "query_weight": cqr["query_weight"].astype(np.float32),
        "coverage_role_counts": cqr["role_counts"].astype(np.int64),
    }
    apply_oracle_sentinel(
        out,
        gt_voxels,
        frame,
        nodes,
        elements,
        coarse_d,
        n_candidates,
        cqr_cfg.get("oracle", {}),
        sample_seed(sid),
    )
    out["coverage_role_counts"] = np.bincount(out["role"], minlength=4).astype(np.int64)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute CQR Stage-2 query clouds")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_query_points", type=int, default=None)
    parser.add_argument("--n_candidates", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    split_file = cfg["data"][f"{args.split}_split"]
    sample_ids = load_split(split_file)
    if args.max_samples:
        sample_ids = sample_ids[: args.max_samples]

    bridge_dir = Path(cfg["data"].get(f"{args.split}_bridge_dir", cfg["data"].get("bridge_dir", "")))
    shared_dir = Path(cfg["data"]["shared_dir"])
    samples_dir = Path(cfg["data"]["samples_dir"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_query = args.n_query_points
    if n_query is None:
        n_query = int(
            cfg["data"].get(
                "precompute_query_points",
                max(32768, int(cfg["data"].get("n_query_points", 4096)) * 8),
            )
        )

    cqr_cfg = cfg.get("cqr", {})

    print(f"[CQR] config={cqr_cfg}")
    print(f"[CQR] Loading mesh: {shared_dir}")
    nodes, elements = FrameManifest.load_mesh_nodes(shared_dir)
    frame = FrameManifest.load(shared_dir)
    print(f"[CQR] nodes={len(nodes)}, tets={len(elements)}")
    print(f"[CQR] split={args.split}, samples={len(sample_ids)}, n_query={n_query}")
    print(f"[CQR] bridge_dir={bridge_dir}")
    print(f"[CQR] output_dir={output_dir}")
    print("-" * 80)

    total = 0.0
    done = 0
    for i, sid in enumerate(sample_ids, 1):
        out_path = output_dir / f"{sid}.npz"
        if out_path.exists() and not args.overwrite:
            print(f"[{i}/{len(sample_ids)}] {sid}: exists, skip")
            continue

        t0 = time.perf_counter()
        data = precompute_one(
            sid=sid,
            bridge_dir=bridge_dir,
            samples_dir=samples_dir,
            nodes=nodes,
            elements=elements,
            frame=frame,
            n_query_points=n_query,
            n_candidates=args.n_candidates,
            cqr_cfg=cqr_cfg,
        )
        np.savez_compressed(out_path, **data)
        elapsed = time.perf_counter() - t0
        total += elapsed
        done += 1

        valid = data["valid_mask"]
        counts = data["coverage_role_counts"].tolist()
        print(
            f"[{i}/{len(sample_ids)}] {sid}: "
            f"points={len(valid)}, valid={int(valid.sum())}/{len(valid)} "
            f"({100*valid.mean():.1f}%), roles(bg/core/halo/sentinel)={counts}, "
            f"t={elapsed:.1f}s"
        )

    print("-" * 80)
    print(f"[CQR] Done: processed={done}, total={total:.1f}s, avg={total/max(done,1):.2f}s/sample")


if __name__ == "__main__":
    main()
