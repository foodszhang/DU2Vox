#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import zlib
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from du2vox.bridge.coverage_field import QueryRole, compute_coverage_field
from du2vox.bridge.fem_bridging import FEMBridge
from du2vox.utils.frame import FrameManifest


ROLE_NAMES = {
    int(QueryRole.CORE): "core",
    int(QueryRole.HALO): "halo",
    int(QueryRole.SENTINEL): "sentinel",
    int(QueryRole.BG): "bg",
}


def load_split(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def sample_seed(sample_id: str) -> int:
    return int(zlib.crc32(sample_id.encode("utf-8")) & 0xFFFFFFFF)


def gt_index_to_world(frame: FrameManifest, idx: np.ndarray) -> np.ndarray:
    return (
        np.asarray(idx, dtype=np.float64) * frame.gt_spacing_mm
        + frame.gt_offset_world_mm
        + frame.gt_spacing_mm / 2
    ).astype(np.float32)


def summarize_one(
    sample_id: str,
    samples_dir: Path,
    bridge_dir: Path,
    nodes: np.ndarray,
    elements: np.ndarray,
    frame: FrameManifest,
    cqr_cfg: dict,
    n_gt_points: int,
) -> dict[str, float | int | str]:
    gt_voxels = np.load(samples_dir / sample_id / "gt_voxels.npy").astype(np.float32)
    pos_idx = np.argwhere(gt_voxels >= 0.5)
    if len(pos_idx) > n_gt_points:
        rng = np.random.default_rng(sample_seed(sample_id))
        pos_idx = pos_idx[rng.choice(len(pos_idx), size=n_gt_points, replace=False)]
    points = gt_index_to_world(frame, pos_idx)

    bd = bridge_dir / sample_id
    coarse_d = np.load(bd / "coarse_d.npy").astype(np.float32)
    roi_tet_indices = np.load(bd / "roi_tet_indices.npy").astype(np.int64)
    field = compute_coverage_field(
        coarse_d=coarse_d,
        elements=elements,
        roi_tet_indices=roi_tet_indices,
        cfg=cqr_cfg.get("coverage", {}),
    )
    bridge = FEMBridge(nodes, elements, n_candidates=32)
    tet_ids, _ = bridge.locate_points_batch(points)

    counts = {"core": 0, "halo": 0, "sentinel": 0, "bg": 0, "outside": 0}
    for tid in tet_ids:
        if tid < 0:
            counts["outside"] += 1
        else:
            counts[ROLE_NAMES.get(int(field["role"][tid]), "outside")] += 1
    total = max(len(tet_ids), 1)
    row: dict[str, float | int | str] = {"sample_id": sample_id, "n_gt_points": int(len(tet_ids))}
    for name in ["core", "halo", "sentinel", "bg", "outside"]:
        row[f"{name}_count"] = counts[name]
        row[f"{name}_ratio"] = counts[name] / total
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose which CQR roles contain GT-positive voxels")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val"])
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--n_gt_points", type=int, default=5000)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    sample_ids = load_split(cfg["data"][f"{args.split}_split"])[: args.max_samples]
    bridge_dir = Path(cfg["data"].get(f"{args.split}_bridge_dir", cfg["data"].get("bridge_dir", "")))
    samples_dir = Path(cfg["data"]["samples_dir"])
    shared_dir = Path(cfg["data"]["shared_dir"])
    nodes, elements = FrameManifest.load_mesh_nodes(shared_dir)
    frame = FrameManifest.load(shared_dir)

    rows = [
        summarize_one(sample_id, samples_dir, bridge_dir, nodes, elements, frame, cfg.get("cqr", {}), args.n_gt_points)
        for sample_id in sample_ids
    ]
    totals = {name: sum(int(row[f"{name}_count"]) for row in rows) for name in ["core", "halo", "sentinel", "bg", "outside"]}
    denom = max(sum(totals.values()), 1)
    overall = {f"overall_{name}_ratio": totals[name] / denom for name in totals}

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "n_gt_points",
        "core_ratio",
        "halo_ratio",
        "sentinel_ratio",
        "bg_ratio",
        "outside_ratio",
        "core_count",
        "halo_count",
        "sentinel_count",
        "bg_count",
        "outside_count",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CQR] wrote oracle coverage CSV: {out_csv}")
    print("[CQR] aggregate " + " ".join(f"{k}={v:.4f}" for k, v in overall.items()))


if __name__ == "__main__":
    main()
