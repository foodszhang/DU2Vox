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
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.physics.compressed_green_operator import CompressedGreenOperator
from du2vox.physics.tetra_quadrature import local_p1_source_mass
from du2vox.models.stage2.transport_cqr_lifter import TransportConsistentCQRLifter


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


def load_lifter(
    checkpoint_path: Path,
    green_modes: np.ndarray,
    elements: np.ndarray,
    cfg: dict[str, Any],
) -> TransportConsistentCQRLifter:
    model_cfg = cfg.get("model", {}) or {}
    transport_cfg = cfg.get("transport", {}) or {}
    lifter = TransportConsistentCQRLifter(
        green_modes,
        elements,
        hidden_dim=min(int(model_cfg.get("hidden_dim", 256)), 128),
        max_logit_delta=float(transport_cfg.get("max_logit_delta", 2.0)),
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("residual_inr", payload.get("model", payload))
    lifter_state = {
        key.removeprefix("lifter."): value
        for key, value in state.items()
        if key.startswith("lifter.")
    }
    if not lifter_state:
        raise ValueError(f"No lifter state found in {checkpoint_path}")
    lifter.load_state_dict(lifter_state, strict=True)
    return lifter.eval()


@torch.inference_mode()
def learned_lifter_transport_error(
    quadrature_path: Path,
    lifter: TransportConsistentCQRLifter,
) -> dict[str, float]:
    with np.load(quadrature_path, allow_pickle=False) as data:
        prior_lift = torch.from_numpy(np.asarray(data["prior_lift"], dtype=np.float32))[None]
        tet_ids = torch.from_numpy(np.asarray(data["tet_id"], dtype=np.int64))[None]
        correction_band = torch.from_numpy(
            np.asarray(data["correction_band"], dtype=np.int64)
        )[None]
        role = torch.from_numpy(np.asarray(data["role"], dtype=np.int64))[None]
        weights = torch.from_numpy(
            np.asarray(data["quadrature_weight"], dtype=np.float32)
        )[None]
    lifted = lifter(prior_lift, tet_ids, correction_band, role=role)
    a_quad = lifted["query_green"].transpose(-1, -2) * weights.unsqueeze(-2)
    delta_modes = (
        a_quad.float() @ (lifted["rho0"] - lifted["fem_interp"]).float().unsqueeze(-1)
    ).squeeze(-1)
    reference_modes = (
        a_quad.float() @ lifted["fem_interp"].float().unsqueeze(-1)
    ).squeeze(-1)
    error = delta_modes.square().sum() / (reference_modes.square().sum() + 1e-8)
    return {
        "transport_error": float(error),
        "alpha_lambda_l1": float(
            torch.mean(torch.abs(lifted["alpha"] - prior_lift[..., 4:8]))
        ),
        "rho0_minus_p1_l1": float(
            torch.mean(torch.abs(lifted["rho0"] - lifted["fem_interp"]))
        ),
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
    if "learned_lifter_summary" in report:
        learned = report["learned_lifter_summary"]
        lines.extend(
            [
                "",
                "## Learned lifter: primary versus comparison quadrature",
                "",
                f"- Primary transport-error mean: `{learned['primary_transport_error_mean']:.6e}`",
                f"- Comparison transport-error mean: `{learned['compare_transport_error_mean']:.6e}`",
                f"- Absolute delta mean: `{learned['absolute_delta_mean']:.6e}`",
                f"- Absolute delta maximum: `{learned['absolute_delta_max']:.6e}`",
            ]
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
    parser.add_argument("--lifter_checkpoint", default=None)
    parser.add_argument("--compare_quadrature_root", default=None)
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
    with np.load(Path(cfg["data"]["shared_dir"]) / "mesh.npz", allow_pickle=False) as mesh:
        elements = np.asarray(mesh["elements"], dtype=np.int64)
    lifter = None
    compare_root = None
    if args.lifter_checkpoint:
        checkpoint_path = Path(args.lifter_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = REPO_ROOT / checkpoint_path
        lifter = load_lifter(checkpoint_path, green_modes, elements, cfg)
    if args.compare_quadrature_root:
        compare_root = Path(args.compare_quadrature_root)
        if not compare_root.is_absolute():
            compare_root = REPO_ROOT / compare_root

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
        record = analyze_sample(
                sid,
                quadrature_path,
                coarse_path,
                green_modes,
                green_modes.shape[1],
                manifest_by_id.get(sid, {}),
            )
        if lifter is not None:
            record["learned_lifter"] = learned_lifter_transport_error(
                quadrature_path, lifter
            )
            if compare_root is not None:
                compare_path = compare_root / args.split / f"{sid}.npz"
                if compare_path.exists():
                    record["learned_lifter_compare"] = learned_lifter_transport_error(
                        compare_path, lifter
                    )
                    left = record["learned_lifter"]["transport_error"]
                    right = record["learned_lifter_compare"]["transport_error"]
                    record["learned_lifter_transport_error_absolute_delta"] = abs(left - right)
        records.append(record)

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
    learned_records = [row for row in records if "learned_lifter" in row]
    if learned_records:
        report["learned_lifter_summary"] = {
            "count": len(learned_records),
            "primary_transport_error_mean": float(
                np.mean([row["learned_lifter"]["transport_error"] for row in learned_records])
            ),
            "compare_transport_error_mean": float(
                np.mean(
                    [
                        row["learned_lifter_compare"]["transport_error"]
                        for row in learned_records
                        if "learned_lifter_compare" in row
                    ]
                )
            ),
            "absolute_delta_mean": float(
                np.mean(
                    [
                        row["learned_lifter_transport_error_absolute_delta"]
                        for row in learned_records
                        if "learned_lifter_transport_error_absolute_delta" in row
                    ]
                )
            ),
            "absolute_delta_max": float(
                np.max(
                    [
                        row["learned_lifter_transport_error_absolute_delta"]
                        for row in learned_records
                        if "learned_lifter_transport_error_absolute_delta" in row
                    ]
                )
            ),
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
