#!/usr/bin/env python3
"""Analyze CQR candidate coverage, residual evidence, and projector numerics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.models.stage2.cqr_observability_projector import CQRObservabilityProjector
from du2vox.physics.compressed_green_operator import CompressedGreenOperator


REPO_ROOT = Path(__file__).resolve().parents[1]
ROLE_NAMES = {0: "bg", 1: "core", 2: "halo", 3: "sentinel", 4: "proposal"}


def load_split(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def residual_modes(y: np.ndarray, y_h: np.ndarray, mode: str) -> tuple[np.ndarray, float]:
    eps = 1e-8
    if mode == "least_squares_stage1_scale":
        scale = max(0.0, float(np.dot(y, y_h) / (np.dot(y_h, y_h) + eps)))
        return y - scale * y_h, scale
    if mode == "l2":
        return y / (np.linalg.norm(y) + eps) - y_h / (np.linalg.norm(y_h) + eps), 1.0
    if mode == "max":
        return y / (np.max(np.abs(y)) + eps) - y_h / (np.max(np.abs(y_h)) + eps), 1.0
    raise ValueError(mode)


def cosine_similarity(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-12) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) <= eps:
        return 0.0
    return float(torch.dot(left, right) / denominator)


def correction_information_metrics(
    operator: CompressedGreenOperator,
    projector: CQRObservabilityProjector,
    elements: np.ndarray,
    tet_ids: np.ndarray,
    prior_8d: np.ndarray,
    gt: np.ndarray,
    chosen: np.ndarray,
    cell_weight: np.ndarray,
    n_valid: int,
    measurement: np.ndarray,
    coarse: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    vertices = elements[tet_ids[chosen]]
    node_modes = operator.green_node_modes[:, vertices]
    query_green = np.einsum(
        "nk,rnk->nr", prior_8d[chosen, 4:8].astype(np.float64), node_modes
    ).astype(np.float32)
    sampled_weight = cell_weight[chosen] * n_valid / max(len(chosen), 1)
    a_query = torch.from_numpy(query_green.T * sampled_weight[None, :]).to(device)
    correction = torch.from_numpy(
        (
            gt[chosen]
            - np.sum(prior_8d[chosen, :4] * prior_8d[chosen, 4:8], axis=1)
        ).astype(np.float32)
    ).to(device)
    measured_modes = operator.measurement_basis.astype(np.float64).T @ measurement
    stage1_modes = operator.projected_forward_modes.astype(np.float64) @ coarse
    residual, scale = residual_modes(
        measured_modes, stage1_modes, "least_squares_stage1_scale"
    )
    residual_t = torch.from_numpy(residual.astype(np.float32)).to(device)
    with torch.no_grad():
        observable = projector.project_observable(correction, a_query)
        adjoint = projector.adjoint_evidence(residual_t, a_query)
    correction_norm = torch.linalg.vector_norm(correction)
    observable_norm = torch.linalg.vector_norm(observable)
    gt_norm = torch.linalg.vector_norm(torch.from_numpy(gt[chosen]).to(device))
    residual_ratio = np.linalg.norm(residual) / max(
        np.linalg.norm(measured_modes), np.finfo(np.float64).tiny
    )
    return {
        "measurement_scale": float(scale),
        "measurement_residual_ratio": float(residual_ratio),
        "gt_correction_relative_norm": float(correction_norm / (gt_norm + 1e-12)),
        "observable_correction_norm_ratio": float(
            observable_norm / (correction_norm + 1e-12)
        ),
        "observable_correction_energy_ratio": float(
            observable_norm.square() / (correction_norm.square() + 1e-12)
        ),
        "adjoint_observable_cosine": cosine_similarity(adjoint, observable),
        "gt_correction_l2": float(correction_norm),
        "observable_target_l2": float(observable_norm),
        "adjoint_evidence_l2": float(torch.linalg.vector_norm(adjoint)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--max_samples", type=int, default=4)
    parser.add_argument("--subsample_repeats", type=int, default=20)
    parser.add_argument(
        "--operator_cache",
        action="append",
        default=None,
        help="Operator cache to audit; repeat for a rank sweep",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_json", default="diagnosis/cqr_observability.json")
    parser.add_argument("--out_md", default="diagnosis/cqr_observability.md")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    transport = cfg.get("transport", {}) or {}
    operator_paths = args.operator_cache or [transport["operator_cache"]]
    operators = []
    for raw_path in operator_paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        operators.append((path, CompressedGreenOperator.load(path)))
    operator = operators[0][1]
    device = torch.device(args.device)
    with np.load(Path(cfg["data"]["shared_dir"]) / "mesh.npz", allow_pickle=False) as mesh:
        elements = mesh["elements"].astype(np.int64)
    ids = load_split(cfg["data"][f"{args.split}_split"])[: args.max_samples]
    cqr_dir = REPO_ROOT / cfg["data"][f"precomputed_{args.split}_dir"]
    sidecar_dir = REPO_ROOT / transport["sidecar_root"] / args.split
    bridge_dir = REPO_ROOT / cfg["data"][f"{args.split}_bridge_dir"]
    samples_dir = Path(cfg["data"]["samples_dir"])
    rng = np.random.default_rng(20260722)
    records = []
    scales = []
    for sid in ids:
        cqr_path = cqr_dir / f"{sid}.npz"
        sidecar_path = sidecar_dir / f"{sid}.npz"
        if not cqr_path.exists() or not sidecar_path.exists():
            continue
        with np.load(cqr_path, allow_pickle=False) as cqr, np.load(
            sidecar_path, allow_pickle=False
        ) as sidecar:
            tet_ids = cqr["tet_ids"].astype(np.int64)
            role = cqr["role"].astype(np.int64)
            prior_8d = cqr["prior_8d"].astype(np.float32)
            barycentric = prior_8d[:, 4:8].astype(np.float64)
            gt = cqr["gt_values"].astype(np.float32)
            valid = sidecar["candidate_valid_physics_mask"].astype(bool)
            cell_weight = sidecar["candidate_cell_weight"].astype(np.float64)
            n_valid = int(sidecar["n_valid_candidate_pool"])
            metadata = json.loads(str(sidecar["metadata"]))
            candidate_volume = sidecar["candidate_tet_volume"].astype(np.float64)
        query_green = np.zeros((len(tet_ids), operator.rank), dtype=np.float64)
        vertices = elements[tet_ids[valid]]
        node_modes = operator.green_node_modes[:, vertices]
        query_green[valid] = np.einsum("nk,rnk->nr", barycentric[valid], node_modes)
        coarse = np.load(bridge_dir / sid / "coarse_d.npy").reshape(-1).astype(np.float64)
        measurement = np.load(samples_dir / sid / "measurement_b.npy").reshape(-1).astype(np.float64)
        y = operator.measurement_basis.astype(np.float64).T @ measurement
        y_h = operator.projected_forward_modes.astype(np.float64) @ coarse

        evidence_by_mode = {}
        evidence_vectors = {}
        full_a = query_green.T * cell_weight[None, :]
        for mode in ("l2", "max", "least_squares_stage1_scale"):
            residual, scale = residual_modes(y, y_h, mode)
            evidence = full_a.T @ residual
            evidence_vectors[mode] = evidence
            evidence_by_mode[mode] = {
                "scale": scale,
                "residual_norm": float(np.linalg.norm(residual)),
                "evidence_l2": float(np.linalg.norm(evidence)),
                "evidence_abs_mean": float(np.mean(np.abs(evidence))),
                "evidence_abs_max": float(np.max(np.abs(evidence))),
            }
            if mode == "least_squares_stage1_scale":
                scales.append(scale)
        correlations = {}
        modes = list(evidence_vectors)
        for i, left in enumerate(modes):
            for right in modes[i + 1 :]:
                correlations[f"{left}__{right}"] = float(
                    np.corrcoef(evidence_vectors[left], evidence_vectors[right])[0, 1]
                )

        coverage_repeats = []
        valid_indices = np.flatnonzero(valid)
        for _ in range(args.subsample_repeats):
            chosen = rng.choice(valid_indices, size=2048, replace=len(valid_indices) < 2048)
            unique_tets, first = np.unique(tet_ids[chosen], return_index=True)
            del unique_tets
            represented_volume = float(candidate_volume[chosen[first]].sum())
            coverage_repeats.append(
                represented_volume / max(metadata["target_physical_tet_volume"], 1e-12)
            )
        role_stats = {}
        for role_id, name in ROLE_NAMES.items():
            mask = valid & (role == role_id)
            role_stats[name] = {
                "query_count": int(mask.sum()),
                "unique_tet_count": int(len(np.unique(tet_ids[mask]))),
            }
        proposal_fail = int(((role == 4) & (tet_ids < 0)).sum())

        chosen = rng.choice(valid_indices, size=min(2048, len(valid_indices)), replace=False)
        a_query = torch.from_numpy(
            query_green[chosen].T
            * (cell_weight[chosen] * n_valid / len(chosen))[None, :]
        ).float()
        projector = CQRObservabilityProjector(
            **{key: cfg["observability"][key] for key in ("mu_relative", "jitter")}
        )
        information_by_rank = {}
        for operator_path, rank_operator in operators:
            information_by_rank[str(rank_operator.rank)] = {
                "operator_cache": str(operator_path),
                **correction_information_metrics(
                    rank_operator,
                    projector,
                    elements,
                    tet_ids,
                    prior_8d,
                    gt,
                    chosen,
                    cell_weight,
                    n_valid,
                    measurement,
                    coarse,
                    device,
                ),
            }
        raw = torch.from_numpy(rng.standard_normal(len(chosen))).float().requires_grad_(True)
        observable = projector.project_observable(raw, a_query)
        ambiguous = projector.project_ambiguous(raw, a_query)
        decomposition_error = float((observable + ambiguous - raw).abs().max().detach())
        ambiguous_leakage = (
            torch.linalg.vector_norm(a_query @ ambiguous)
            / (torch.linalg.vector_norm(ambiguous) + 1e-8)
        ).detach().item()
        (observable.square().mean() + ambiguous.square().mean()).backward()
        records.append(
            {
                "sample_id": sid,
                "candidate_pool_size": len(tet_ids),
                "n_valid_candidate_pool": n_valid,
                "unique_tet_count": int(len(np.unique(tet_ids[valid]))),
                "tet_coverage_ratio": metadata["tet_coverage_ratio"],
                "proposal_locate_fail": proposal_fail,
                "role_stats": role_stats,
                "subsample_2048_coverage_mean": float(np.mean(coverage_repeats)),
                "subsample_2048_coverage_std": float(np.std(coverage_repeats)),
                "evidence_by_normalization": evidence_by_mode,
                "evidence_correlations": correlations,
                "projector_decomposition_max_error": decomposition_error,
                "ambiguous_measurement_leakage_norm_ratio": ambiguous_leakage,
                "projector_backward_finite": bool(torch.isfinite(raw.grad).all()),
                "correction_information_by_rank": information_by_rank,
            }
        )
    metric_names = (
        "measurement_residual_ratio",
        "gt_correction_relative_norm",
        "observable_correction_norm_ratio",
        "observable_correction_energy_ratio",
        "adjoint_observable_cosine",
    )
    information_summary = {}
    for _, rank_operator in operators:
        rank_key = str(rank_operator.rank)
        rank_rows = [
            row["correction_information_by_rank"][rank_key] for row in records
        ]
        information_summary[rank_key] = {
            name: {
                "mean": float(np.mean([row[name] for row in rank_rows])),
                "median": float(np.median([row[name] for row in rank_rows])),
                "min": float(np.min([row[name] for row in rank_rows])),
                "max": float(np.max([row[name] for row in rank_rows])),
            }
            for name in metric_names
        }
    report = {
        "schema_version": 1,
        "split": args.split,
        "sample_count": len(records),
        "scale_distribution": {
            "min": float(np.min(scales)) if scales else None,
            "mean": float(np.mean(scales)) if scales else None,
            "max": float(np.max(scales)) if scales else None,
        },
        "samples": records,
        "correction_information_summary_by_rank": information_summary,
        "operator_scope": "CQR candidate-supported correction operator; not full-volume quadrature",
    }
    out_json = REPO_ROOT / args.out_json
    out_md = REPO_ROOT / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# CQR Observability Analysis",
        "",
        f"- Samples: {len(records)}",
        f"- Least-squares Stage 1 scale distribution: `{report['scale_distribution']}`",
        "- Scope: candidate-supported correction operator; strict forward checks use fixed quadrature.",
        "",
        "## Residual and oracle correction information",
        "",
        "| rank | residual / y | GT correction / GT | observable norm fraction | observable energy | adjoint cosine |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, values in information_summary.items():
        lines.append(
            f"| {rank} | {values['measurement_residual_ratio']['mean']:.4f} | "
            f"{values['gt_correction_relative_norm']['mean']:.4f} | "
            f"{values['observable_correction_norm_ratio']['mean']:.4f} | "
            f"{values['observable_correction_energy_ratio']['mean']:.4f} | "
            f"{values['adjoint_observable_cosine']['mean']:.4f} |"
        )
    lines.extend(["", "## Per sample", ""])
    for row in records:
        lines.append(
            f"- {row['sample_id']}: pool={row['candidate_pool_size']}, "
            f"unique_tets={row['unique_tet_count']}, coverage={row['tet_coverage_ratio']:.4f}, "
            f"2048 coverage std={row['subsample_2048_coverage_std']:.4f}, "
            f"proposal_fail={row['proposal_locate_fail']}, "
            f"projector_error={row['projector_decomposition_max_error']:.3e}"
        )
    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
