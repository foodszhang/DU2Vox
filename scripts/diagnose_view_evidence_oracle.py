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

from du2vox.bridge.view_evidence import compute_view_evidence, load_proj_npz
from du2vox.utils.frame import FrameManifest


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


def mean_or_zero(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else 0.0


def summarize_scores(prefix: str, scores: np.ndarray, row: dict[str, float | int | str]) -> None:
    row[f"{prefix}_mean"] = mean_or_zero(scores)
    row[f"{prefix}_p90"] = float(np.percentile(scores, 90)) if len(scores) else 0.0
    row[f"{prefix}_p99"] = float(np.percentile(scores, 99)) if len(scores) else 0.0


def summarize_one(
    sample_id: str,
    samples_dir: Path,
    precomputed_dir: Path,
    frame: FrameManifest,
    n_points: int,
) -> dict[str, float | int | str]:
    rng = np.random.default_rng(sample_seed(sample_id))
    gt_voxels = np.load(samples_dir / sample_id / "gt_voxels.npy").astype(np.float32)
    proj_imgs = load_proj_npz(str(samples_dir / sample_id / "proj.npz"))

    pos_idx = np.argwhere(gt_voxels >= 0.5)
    bg_idx = np.argwhere(gt_voxels < 0.01)
    if len(pos_idx) > n_points:
        pos_idx = pos_idx[rng.choice(len(pos_idx), size=n_points, replace=False)]
    if len(bg_idx) > n_points:
        bg_idx = bg_idx[rng.choice(len(bg_idx), size=n_points, replace=False)]

    pos_score, _ = compute_view_evidence(gt_index_to_world(frame, pos_idx), proj_imgs)
    bg_score, _ = compute_view_evidence(gt_index_to_world(frame, bg_idx), proj_imgs)

    row: dict[str, float | int | str] = {
        "sample_id": sample_id,
        "n_pos": len(pos_idx),
        "n_bg": len(bg_idx),
    }
    summarize_scores("gt_pos", pos_score, row)
    summarize_scores("bg", bg_score, row)

    npz_path = precomputed_dir / f"{sample_id}.npz"
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as data:
            score = data["view_evidence_score"] if "view_evidence_score" in data.files else np.zeros(len(data["gt_values"]))
            role = data["role"] if "role" in data.files else np.zeros(len(score), dtype=np.int64)
            gt = data["gt_values"]
            valid = data["valid_mask"].astype(bool)
            for rid, name in [(1, "core_query"), (3, "sentinel_query")]:
                mask = valid & (role == rid)
                summarize_scores(name, score[mask], row)
            top_k = min(int((role == 3).sum()), len(score)) or min(1000, len(score))
            top_idx = np.argsort(score)[-top_k:]
            row["topk_view_gt_pos_05"] = mean_or_zero(gt[top_idx] >= 0.5)
            row["topk_view_k"] = int(top_k)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose whether projection view evidence separates GT-positive points")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--precomputed_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--n_points", type=int, default=5000)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    sample_ids = load_split(cfg["data"][f"{args.split}_split"])[: args.max_samples]
    samples_dir = Path(cfg["data"]["samples_dir"])
    frame = FrameManifest.load(cfg["data"]["shared_dir"])
    rows = [summarize_one(sid, samples_dir, Path(args.precomputed_dir), frame, args.n_points) for sid in sample_ids]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[ViewEvidence] wrote {out_csv}")
    for key in ["gt_pos_mean", "bg_mean", "sentinel_query_mean", "topk_view_gt_pos_05"]:
        vals = [float(row[key]) for row in rows if key in row]
        print(f"[ViewEvidence] {key}={mean_or_zero(np.asarray(vals)):.4f}")


if __name__ == "__main__":
    main()
