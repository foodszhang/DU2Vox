#!/usr/bin/env python3
"""Sanity-check CQR observability with all surface measurement directions."""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
import yaml
from scipy.sparse.linalg import splu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.physics.compressed_green_operator import load_surface_index


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_split(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def prepare_samples(cfg: dict[str, Any], split: str, max_samples: int) -> list[dict[str, Any]]:
    ids = load_split(cfg["data"][f"{split}_split"])[:max_samples]
    cqr_dir = REPO_ROOT / cfg["data"][f"precomputed_{split}_dir"]
    sidecar_dir = REPO_ROOT / cfg["transport"]["sidecar_root"] / split
    with np.load(Path(cfg["data"]["shared_dir"]) / "mesh.npz", allow_pickle=False) as mesh:
        elements = np.asarray(mesh["elements"], dtype=np.int64)
    records = []
    for sid in ids:
        with np.load(cqr_dir / f"{sid}.npz", allow_pickle=False) as cqr, np.load(
            sidecar_dir / f"{sid}.npz", allow_pickle=False
        ) as sidecar:
            tet_ids = np.asarray(cqr["tet_ids"], dtype=np.int64)
            prior_8d = np.asarray(cqr["prior_8d"], dtype=np.float32)
            gt = np.asarray(cqr["gt_values"], dtype=np.float32)
            valid = np.asarray(sidecar["candidate_valid_physics_mask"], dtype=bool)
            cell_weight = np.asarray(sidecar["candidate_cell_weight"], dtype=np.float32)
            n_valid = int(sidecar["n_valid_candidate_pool"])
        valid_indices = np.flatnonzero(valid)
        rng = np.random.default_rng(zlib.crc32(sid.encode()) & 0xFFFFFFFF)
        chosen = rng.choice(
            valid_indices, size=min(2048, len(valid_indices)), replace=False
        )
        vertices = elements[tet_ids[chosen]]
        barycentric = prior_8d[chosen, 4:8]
        correction = gt[chosen] - np.sum(
            prior_8d[chosen, :4] * barycentric, axis=1
        )
        sampled_weight = cell_weight[chosen] * n_valid / len(chosen)
        records.append(
            {
                "sample_id": sid,
                "vertices": vertices,
                "barycentric": barycentric,
                "correction": correction.astype(np.float32),
                "sampled_weight": sampled_weight.astype(np.float32),
            }
        )
    return records


def build_selected_full_green(
    shared_dir: Path,
    selected_nodes: np.ndarray,
    cache_prefix: Path,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    nodes_path = cache_prefix.with_suffix(".nodes.npy")
    green_path = cache_prefix.with_suffix(".green.npy")
    if nodes_path.exists() and green_path.exists():
        cached_nodes = np.load(nodes_path)
        if np.array_equal(cached_nodes, selected_nodes):
            print(f"[FullGreen] using cache {green_path}")
            return cached_nodes, np.load(green_path, mmap_mode="r")

    mass = sp.load_npz(shared_dir / "system_matrix.M.npz").tocsc().astype(np.float64)
    surface_index = load_surface_index(shared_dir)
    cache_prefix.parent.mkdir(parents=True, exist_ok=True)
    green = np.lib.format.open_memmap(
        green_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(surface_index), len(selected_nodes)),
    )
    print(f"[FullGreen] factorizing M={mass.shape}, selected_nodes={len(selected_nodes)}")
    factor = splu(mass)
    for start in range(0, len(selected_nodes), chunk_size):
        stop = min(start + chunk_size, len(selected_nodes))
        rhs = np.zeros((mass.shape[0], stop - start), dtype=np.float64)
        rhs[selected_nodes[start:stop], np.arange(stop - start)] = 1.0
        solution = factor.solve(rhs)
        green[:, start:stop] = solution[surface_index].astype(np.float32)
        green.flush()
        print(f"[FullGreen] solved {stop}/{len(selected_nodes)} node columns")
    np.save(nodes_path, selected_nodes)
    return selected_nodes, green


def query_operator(
    green: np.ndarray,
    vertex_positions: np.ndarray,
    barycentric: np.ndarray,
    sampled_weight: np.ndarray,
) -> np.ndarray:
    output = np.zeros((green.shape[0], len(vertex_positions)), dtype=np.float32)
    for local_vertex in range(4):
        output += (
            np.asarray(green[:, vertex_positions[:, local_vertex]])
            * barycentric[None, :, local_vertex]
        )
    return output * sampled_weight[None, :]


def spectral_projection_metrics(
    a_query: torch.Tensor,
    correction: torch.Tensor,
    mu_values: list[float],
    measurement_dim: int,
    jitter: float,
) -> dict[str, dict[str, float]]:
    gram = a_query.transpose(0, 1) @ a_query
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues.clamp_min(0.0)
    coordinates = eigenvectors.transpose(0, 1) @ correction
    correction_energy = correction.square().sum() + 1e-12
    trace = eigenvalues.sum()
    result = {}
    for mu_relative in mu_values:
        mu = mu_relative * trace / max(measurement_dim, 1) + jitter
        spectral_filter = eigenvalues / (eigenvalues + mu)
        observable = eigenvectors @ (spectral_filter * coordinates)
        result[f"{mu_relative:.12g}"] = {
            "mu_relative": mu_relative,
            "mu_absolute": float(mu),
            "effective_observable_dof": float(spectral_filter.sum()),
            "observable_correction_norm_ratio": float(
                torch.linalg.vector_norm(observable)
                / (torch.linalg.vector_norm(correction) + 1e-12)
            ),
            "observable_correction_energy_ratio": float(
                observable.square().sum() / correction_energy
            ),
        }
    return result


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Full-measurement CQR observability sanity check",
        "",
        f"- Samples: {report['sample_count']}",
        f"- Surface measurements: {report['measurement_dimension']}",
        f"- Query count per sample: {report['query_count']}",
        "- Scope: all surface measurement directions on the sampled CQR correction basis.",
        "",
        "| mu_rel | effective dof | observable GT energy | observable GT norm |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for mu_key, row in report["summary_by_mu"].items():
        lines.append(
            f"| {mu_key} | {row['effective_observable_dof_mean']:.2f} | "
            f"{row['observable_correction_energy_ratio_mean']:.4f} | "
            f"{row['observable_correction_norm_ratio_mean']:.4f} |"
        )
    lines.extend(["", "## Per sample", ""])
    for row in report["samples"]:
        values = row["by_mu"][report["mu_values"][0]]
        lines.append(
            f"- {row['sample_id']}: energy={values['observable_correction_energy_ratio']:.4f}, "
            f"d_eff={values['effective_observable_dof']:.2f} at mu_rel={report['mu_values'][0]}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--mu_relative", action="append", type=float, default=None)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument(
        "--green_cache_prefix",
        default="physics_cache/v2_3k_20k_full_green_val5",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--out_json", default="diagnosis/cqr_full_measurement_observability.json"
    )
    parser.add_argument(
        "--out_md", default="diagnosis/cqr_full_measurement_observability.md"
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    mu_values = args.mu_relative or [float(cfg["observability"]["mu_relative"])]
    samples = prepare_samples(cfg, args.split, args.max_samples)
    selected_nodes = np.unique(
        np.concatenate([row["vertices"].reshape(-1) for row in samples])
    )
    cache_prefix = Path(args.green_cache_prefix)
    if not cache_prefix.is_absolute():
        cache_prefix = REPO_ROOT / cache_prefix
    selected_nodes, green = build_selected_full_green(
        Path(cfg["data"]["shared_dir"]), selected_nodes, cache_prefix, args.chunk_size
    )
    node_to_position = np.full(19990, -1, dtype=np.int64)
    node_to_position[selected_nodes] = np.arange(len(selected_nodes))
    device = torch.device(args.device)
    sample_rows = []
    for index, row in enumerate(samples, start=1):
        positions = node_to_position[row["vertices"]]
        a_numpy = query_operator(
            green,
            positions,
            row["barycentric"],
            row["sampled_weight"],
        )
        a_query = torch.from_numpy(a_numpy).to(device)
        correction = torch.from_numpy(row["correction"]).to(device)
        by_mu = spectral_projection_metrics(
            a_query,
            correction,
            mu_values,
            measurement_dim=a_query.shape[0],
            jitter=float(cfg["observability"]["jitter"]),
        )
        sample_rows.append({"sample_id": row["sample_id"], "by_mu": by_mu})
        print(f"[FullProjector] {index}/{len(samples)} {row['sample_id']} complete")

    summary = {}
    for mu_relative in mu_values:
        key = f"{mu_relative:.12g}"
        rows = [row["by_mu"][key] for row in sample_rows]
        summary[key] = {
            "effective_observable_dof_mean": float(
                np.mean([row["effective_observable_dof"] for row in rows])
            ),
            "observable_correction_norm_ratio_mean": float(
                np.mean([row["observable_correction_norm_ratio"] for row in rows])
            ),
            "observable_correction_energy_ratio_mean": float(
                np.mean([row["observable_correction_energy_ratio"] for row in rows])
            ),
            "observable_correction_energy_ratio_max": float(
                np.max([row["observable_correction_energy_ratio"] for row in rows])
            ),
        }
    report = {
        "schema_version": 1,
        "split": args.split,
        "sample_count": len(sample_rows),
        "measurement_dimension": int(green.shape[0]),
        "query_count": int(len(samples[0]["correction"])) if samples else 0,
        "selected_green_node_count": int(len(selected_nodes)),
        "mu_values": [f"{value:.12g}" for value in mu_values],
        "summary_by_mu": summary,
        "samples": sample_rows,
        "operator_scope": "Full surface measurements on the sampled CQR correction basis",
    }
    out_json = REPO_ROOT / args.out_json
    out_md = REPO_ROOT / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    out_md.write_text(markdown(report))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
