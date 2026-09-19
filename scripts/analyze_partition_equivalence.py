#!/usr/bin/env python3
"""Test whether the trained partition collapses to a plain residual."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.eval_stage2_unified import (  # noqa: E402
    build_model,
    load_checkpoint,
    load_split,
    run_model_on_sample,
)
from du2vox.utils.frame import FrameManifest  # noqa: E402


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0 else 0.0


def spatial_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def graph_roughness(values: np.ndarray, coords: np.ndarray, neighbors: int = 6) -> float:
    if len(values) < 2:
        return 0.0
    k = min(neighbors + 1, len(values))
    indices = cKDTree(coords).query(coords, k=k)[1]
    if indices.ndim == 1:
        return 0.0
    differences = values[:, None] - values[indices[:, 1:]]
    return float(np.mean(differences**2) / (np.mean(values**2) + 1e-12))


def configure_environment(cfg: dict) -> None:
    data = cfg.get("data", {})
    if data.get("shared_dir"):
        os.environ["DU2VOX_SHARED_DIR"] = str(data["shared_dir"])
    if data.get("allow_stale_frame_manifest", False):
        os.environ["DU2VOX_ALLOW_STALE_FRAME_MANIFEST"] = "1"
    if data.get("frame_manifest_sha256"):
        os.environ["DU2VOX_FRAME_MANIFEST_SHA256"] = str(data["frame_manifest_sha256"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plain_config", required=True)
    parser.add_argument("--plain_checkpoint", required=True)
    parser.add_argument("--partition_config", required=True)
    parser.add_argument("--partition_checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--correction_threshold", type=float, default=0.05)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.plain_config) as handle:
        plain_cfg = yaml.safe_load(handle)
    with open(args.partition_config) as handle:
        partition_cfg = yaml.safe_load(handle)
    configure_environment(partition_cfg)
    plain_ids = load_split(plain_cfg["data"][f"{args.split}_split"])
    partition_ids = load_split(partition_cfg["data"][f"{args.split}_split"])
    if plain_ids != partition_ids:
        raise ValueError("Plain and partition sample lists differ")
    sample_ids = plain_ids[: args.max_samples] if args.max_samples else plain_ids

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plain_model, plain_view = build_model(plain_cfg, device)
    partition_model, partition_view = build_model(partition_cfg, device)
    load_checkpoint(Path(args.plain_checkpoint), plain_model, plain_view, device)
    load_checkpoint(Path(args.partition_checkpoint), partition_model, partition_view, device)
    plain_model.eval()
    partition_model.eval()
    if plain_view is not None:
        plain_view.eval()
    if partition_view is not None:
        partition_view.eval()

    samples_dir = Path(partition_cfg["data"]["samples_dir"])
    precomputed_dir = Path(partition_cfg["data"][f"precomputed_{args.split}_dir"])
    frame = FrameManifest.load(partition_cfg["data"]["shared_dir"])
    rows = []
    with torch.no_grad():
        for sample_id in sample_ids:
            with np.load(precomputed_dir / f"{sample_id}.npz", allow_pickle=False) as archive:
                data = {key: archive[key] for key in archive.files}
            plain_result = run_model_on_sample(
                plain_model,
                plain_view,
                plain_cfg,
                data,
                samples_dir,
                frame,
                sample_id,
                32768,
                "all",
                device,
            )
            partition_result = run_model_on_sample(
                partition_model,
                partition_view,
                partition_cfg,
                data,
                samples_dir,
                frame,
                sample_id,
                32768,
                "all",
                device,
            )
            valid = plain_result[3]
            coords = data["grid_coords"].astype(np.float32)[valid]
            plain_diag = plain_result[4]
            partition_diag = partition_result[4]
            residual_b = np.asarray(plain_diag["correction"], dtype=np.float64)
            residual_c = np.asarray(partition_diag["correction"], dtype=np.float64)
            support_b = np.abs(residual_b) >= args.correction_threshold
            support_c = np.abs(residual_c) >= args.correction_threshold
            intersection = int((support_b & support_c).sum())
            union = int((support_b | support_c).sum())
            observable = np.asarray(partition_diag["observable_correction"], dtype=np.float64)
            ambiguous = np.asarray(partition_diag["ambiguous_correction"], dtype=np.float64)
            total_energy = float(np.sum(residual_c**2) + 1e-12)
            plain_roughness = graph_roughness(residual_b, coords)
            partition_roughness = graph_roughness(residual_c, coords)
            rows.append(
                {
                    "sample_id": sample_id,
                    "observable_energy_fraction": float(np.sum(observable**2) / total_energy),
                    "ambiguous_energy_fraction": float(np.sum(ambiguous**2) / total_energy),
                    "plain_partition_cosine": cosine(residual_b, residual_c),
                    "correction_support_iou": intersection / max(union, 1),
                    "spatial_correlation": spatial_correlation(residual_b, residual_c),
                    "plain_graph_spectral_roughness": plain_roughness,
                    "partition_graph_spectral_roughness": partition_roughness,
                    "spectral_roughness_ratio_c_over_b": partition_roughness
                    / (plain_roughness + 1e-12),
                }
            )
            print(
                f"[Equivalence] {sample_id}: cos={rows[-1]['plain_partition_cosine']:.4f} "
                f"support_iou={rows[-1]['correction_support_iou']:.4f}"
            )

    keys = [key for key in rows[0] if key != "sample_id"] if rows else []
    summary = {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "std": float(np.std([row[key] for row in rows])),
        }
        for key in keys
    }
    result = {
        "plain_config": args.plain_config,
        "partition_config": args.partition_config,
        "split": args.split,
        "correction_threshold": args.correction_threshold,
        "n_samples": len(rows),
        "summary": summary,
        "per_sample": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"[Equivalence] wrote {output}")


if __name__ == "__main__":
    main()
