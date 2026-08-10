#!/usr/bin/env python3
"""Verify P1 source-mass and compressed-measurement quadrature consistency."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.physics.compressed_green_operator import CompressedGreenOperator
from du2vox.physics.tetra_quadrature import local_p1_source_mass


REPO_ROOT = Path(__file__).resolve().parents[1]
ROLE_NAMES = {0: "bg", 1: "core", 2: "halo", 3: "sentinel", 4: "proposal"}


def load_split(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def relative_error(actual: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(actual - reference)
        / max(np.linalg.norm(reference), np.finfo(np.float64).tiny)
    )


def scatter_local_loads(
    vertex_indices: np.ndarray, local_loads: np.ndarray, n_nodes: int
) -> np.ndarray:
    output = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(output, vertex_indices.reshape(-1), local_loads.reshape(-1))
    return output


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"count": 0}
    return {
        "count": len(records),
        "source_load_relative_error_mean": float(
            np.mean([row["source_load_relative_error"] for row in records])
        ),
        "source_load_relative_error_max": float(
            np.max([row["source_load_relative_error"] for row in records])
        ),
        "compressed_measurement_relative_error_mean": float(
            np.mean([row["compressed_measurement_relative_error"] for row in records])
        ),
        "compressed_measurement_relative_error_max": float(
            np.max([row["compressed_measurement_relative_error"] for row in records])
        ),
    }


def analyze_sample(
    sid: str,
    quadrature_path: Path,
    coarse_path: Path,
    green_modes: np.ndarray,
    n_nodes: int,
    manifest_record: dict[str, Any],
) -> dict[str, Any]:
    with np.load(quadrature_path, allow_pickle=False) as data:
        tet_ids = np.asarray(data["tet_id"], dtype=np.int64)
        vertices_q = np.asarray(data["tet_vertex_indices"], dtype=np.int64)
        barycentric_q = np.asarray(data["barycentric"], dtype=np.float64)
        weights_q = np.asarray(data["quadrature_weight"], dtype=np.float64)
        roles_q = np.asarray(data["role"], dtype=np.int64)
    if len(tet_ids) % 4:
        raise ValueError(f"{quadrature_path}: expected four points per tet")
    n_tets = len(tet_ids) // 4
    tet_ids = tet_ids.reshape(n_tets, 4)
    if not np.all(tet_ids == tet_ids[:, :1]):
        raise ValueError(f"{quadrature_path}: quadrature points are not grouped by tet")
    vertices = vertices_q.reshape(n_tets, 4, 4)[:, 0]
    if not np.all(vertices_q.reshape(n_tets, 4, 4) == vertices[:, None, :]):
        raise ValueError(f"{quadrature_path}: vertex indices differ within a tet")
    barycentric = barycentric_q.reshape(n_tets, 4, 4)
    weights = weights_q.reshape(n_tets, 4)
    roles = roles_q.reshape(n_tets, 4)[:, 0]
    coarse = np.load(coarse_path).reshape(-1).astype(np.float64)
    node_values = coarse[vertices]
    volumes = weights.sum(axis=1)

    exact_local = np.einsum("tij,tj->ti", local_p1_source_mass(volumes), node_values)
    p1_values = np.einsum("tqk,tk->tq", barycentric, node_values)
    quadrature_local = np.einsum("tq,tqk,tq->tk", weights, barycentric, p1_values)
    exact_global = scatter_local_loads(vertices, exact_local, n_nodes)
    quadrature_global = scatter_local_loads(vertices, quadrature_local, n_nodes)
    exact_modes = green_modes @ exact_global
    quadrature_modes = green_modes @ quadrature_global

    per_role: dict[str, Any] = {}
    for role_id, role_name in ROLE_NAMES.items():
        mask = roles == role_id
        if not np.any(mask):
            continue
        exact_role = scatter_local_loads(vertices[mask], exact_local[mask], n_nodes)
        quad_role = scatter_local_loads(vertices[mask], quadrature_local[mask], n_nodes)
        per_role[role_name] = {
            "tet_count": int(mask.sum()),
            "source_load_relative_error": relative_error(
                quadrature_local[mask], exact_local[mask]
            ),
            "compressed_measurement_relative_error": relative_error(
                green_modes @ quad_role, green_modes @ exact_role
            ),
        }
    return {
        "sample_id": sid,
        "selected_tet_count": n_tets,
        "source_load_relative_error": relative_error(quadrature_local, exact_local),
        "compressed_measurement_relative_error": relative_error(
            quadrature_modes, exact_modes
        ),
        "source_load_max_absolute_error": float(np.max(np.abs(quadrature_local - exact_local))),
        "compressed_measurement_max_absolute_error": float(
            np.max(np.abs(quadrature_modes - exact_modes))
        ),
        "per_role": per_role,
        "depth_tier": manifest_record.get("depth_tier", "unknown"),
        "depth_mm": manifest_record.get("depth_mm"),
        "num_foci": manifest_record.get("num_foci", "unknown"),
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CQR P1 Transport Consistency",
        "",
        "The four-point degree-2 rule integrates the product `N_i * rho_P1` exactly. "
        "The reference local matrix is the generator's source mass, "
        "`V/20 * (ones(4,4) + eye(4))`.",
        "",
        f"- Samples: {report['summary']['count']}",
        f"- Source-load max relative error: `{report['summary'].get('source_load_relative_error_max'):.6e}`",
        f"- Compressed-measurement max relative error: "
        f"`{report['summary'].get('compressed_measurement_relative_error_max'):.6e}`",
        f"- P1/local-mass gate passed: **{report['p1_consistency_passed']}**",
        "",
        "## Per sample",
        "",
        "| sample | tets | depth | foci | source error | compressed error |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in report["samples"]:
        lines.append(
            f"| {row['sample_id']} | {row['selected_tet_count']} | {row['depth_tier']} | "
            f"{row['num_foci']} | {row['source_load_relative_error']:.3e} | "
            f"{row['compressed_measurement_relative_error']:.3e} |"
        )
    lines.extend(
        [
            "",
            "Conclusion: P1 already preserves the FEM transport action to numerical precision. "
            "The learned lifter must therefore be described as morphology-sensitive support "
            "redistribution under a FEM transport-action constraint.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--max_samples", type=int, default=4)
    parser.add_argument("--quadrature_root", default=None)
    parser.add_argument("--operator_cache", default=None)
    parser.add_argument("--out_json", default="diagnosis/cqr_p1_transport_consistency.json")
    parser.add_argument("--out_md", default="diagnosis/cqr_p1_transport_consistency.md")
    parser.add_argument("--max_relative_error", type=float, default=1e-6)
    args = parser.parse_args()

    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    transport = cfg.get("transport", {}) or {}
    ids = load_split(cfg["data"][f"{args.split}_split"])
    if args.max_samples is not None:
        ids = ids[: args.max_samples]
    quadrature_root = Path(
        args.quadrature_root
        or transport.get("quadrature_root", "precomputed/cqr_v2_3k_rgl_main_physics_quadrature")
    )
    operator_path = Path(
        args.operator_cache
        or transport.get("operator_cache", "physics_cache/v2_3k_20k_transport_rank64.npz")
    )
    if not quadrature_root.is_absolute():
        quadrature_root = REPO_ROOT / quadrature_root
    if not operator_path.is_absolute():
        operator_path = REPO_ROOT / operator_path
    operator = CompressedGreenOperator.load(operator_path)
    green_modes = np.asarray(operator.green_node_modes, dtype=np.float64)

    manifest_path = Path(cfg["data"]["dataset_root"]) / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_by_id = {row["id"]: row for row in manifest.get("samples", [])}
    bridge_dir = REPO_ROOT / cfg["data"][f"{args.split}_bridge_dir"]
    records = []
    missing = []
    for sid in ids:
        quadrature_path = quadrature_root / args.split / f"{sid}.npz"
        coarse_path = bridge_dir / sid / "coarse_d.npy"
        if not quadrature_path.exists() or not coarse_path.exists():
            missing.append(sid)
            continue
        records.append(
            analyze_sample(
                sid,
                quadrature_path,
                coarse_path,
                green_modes,
                green_modes.shape[1],
                manifest_by_id.get(sid, {}),
            )
        )

    grouped_depth: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_foci: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped_depth[str(row["depth_tier"])].append(row)
        grouped_foci[str(row["num_foci"])].append(row)
    summary = summarize_records(records)
    passed = bool(
        records
        and summary["source_load_relative_error_max"] <= args.max_relative_error
        and summary["compressed_measurement_relative_error_max"] <= args.max_relative_error
    )
    report = {
        "schema_version": 1,
        "operator_cache": str(operator_path),
        "split": args.split,
        "summary": summary,
        "per_depth": {key: summarize_records(value) for key, value in grouped_depth.items()},
        "per_foci": {key: summarize_records(value) for key, value in grouped_foci.items()},
        "samples": records,
        "missing_samples": missing,
        "max_relative_error_gate": args.max_relative_error,
        "p1_consistency_passed": passed,
    }
    out_json = REPO_ROOT / args.out_json
    out_md = REPO_ROOT / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    out_md.write_text(markdown(report))
    print(json.dumps(summary, indent=2))
    print(f"P1 consistency passed: {passed}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    if not passed:
        raise RuntimeError("P1/local source-mass consistency gate failed")


if __name__ == "__main__":
    main()
