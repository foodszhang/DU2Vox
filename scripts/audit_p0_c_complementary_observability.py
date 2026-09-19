#!/usr/bin/env python3
"""P0-C complementary-observability audit (no training, no confirmation data).

The four subcommands keep the three evidence lines separate. ``d0`` audits the
existing generator contract. ``fine-fem`` constructs/runs the matched diagnostic
operator. ``mcx`` prepares nonnegative positive/negative sources and, optionally,
summarizes externally produced replicate measurements. ``summarize`` applies the
preregistered gates without upgrading one evidence line into another.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.bridge.canonical_cross_discretization import CanonicalCrossDiscretization
from du2vox.physics.voxel_resolved_forward import (
    CoarseExtensionOperator,
    SolverContract,
    VoxelResolvedLinearOperator,
    assemble_diffusion_system,
    build_refined_sampling_matrix,
    estimate_complement_diagonal,
    infer_piecewise_optical_coefficients,
    load_refined_mesh,
    refine_tetra_mesh,
    randomized_complement_spectrum,
    save_refined_mesh,
    sha256_file,
    write_metadata,
)
from du2vox.utils.confirmation import require_confirmation_permission


SEED = 20260907
DEPTHS = ("shallow", "medium", "deep")


def _load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    data = cfg["data"]
    require_confirmation_permission(
        samples_dir=data["samples_dir"],
        allow_confirmation_eval=False,
        manifest_path=Path("data/confirmation_manifest.json"),
    )
    os.environ["DU2VOX_SHARED_DIR"] = str(data["shared_dir"])
    if data.get("allow_stale_frame_manifest", False):
        os.environ["DU2VOX_ALLOW_STALE_FRAME_MANIFEST"] = "1"
    if data.get("frame_manifest_sha256"):
        os.environ["DU2VOX_FRAME_MANIFEST_SHA256"] = str(data["frame_manifest_sha256"])
    return cfg


def _ids(path: str | Path) -> list[str]:
    return sorted(line.strip() for line in Path(path).read_text().splitlines() if line.strip())


def _load_a(path: Path, visible_mask: np.ndarray | None = None) -> np.ndarray:
    try:
        matrix = sp.load_npz(path).toarray()
    except (ValueError, KeyError):
        with np.load(path, allow_pickle=True) as archive:
            key = "forward_matrix" if "forward_matrix" in archive else "arr_0"
            matrix = archive[key]
    matrix = np.asarray(matrix, dtype=np.float64)
    return matrix[visible_mask] if visible_mask is not None else matrix


def _q(canonical: CanonicalCrossDiscretization, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    return values - canonical.prolong(canonical.project_coefficients(values))


def _safe_correlation(first: np.ndarray, second: np.ndarray) -> dict[str, float | None]:
    from scipy.stats import pearsonr, spearmanr

    first = np.asarray(first, dtype=np.float64).ravel()
    second = np.asarray(second, dtype=np.float64).ravel()
    if first.size < 2 or np.ptp(first) == 0 or np.ptp(second) == 0:
        return {"pearson": None, "spearman": None, "reason": "constant_or_insufficient"}
    return {
        "pearson": float(pearsonr(first, second).statistic),
        "spearman": float(spearmanr(first, second).statistic),
    }


def _gt(canonical: CanonicalCrossDiscretization, sample_dir: Path, *, binary: bool) -> np.ndarray:
    volume = np.load(sample_dir / "gt_voxels.npy", mmap_mode="r")
    values = np.asarray(volume).ravel()[canonical.operator.valid_flat_indices]
    return (values > 0.05).astype(np.float64) if binary else values.astype(np.float64)


def _prediction_nodes(directory: Path, sid: str) -> np.ndarray:
    path = directory / f"{sid}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Frozen V4 cached state is missing: {path}")
    with np.load(path) as data:
        return data["step3_fem_nodes"].astype(np.float64)


def run_d0(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    data = cfg["data"]
    shared = Path(data["shared_dir"])
    samples = Path(data["samples_dir"])
    canonical = CanonicalCrossDiscretization(
        data["operator_cache"], shared_dir=shared, factorize=True
    )
    visible = None
    if data.get("use_visible_mask", False):
        visible = np.load(shared / "visible_mask.npy").astype(bool)
    a = _load_a(shared / "system_matrix.A.npz", visible)
    b0 = CoarseExtensionOperator(a, canonical)
    sample_ids = _ids(data["test_split"])
    if args.max_samples is not None:
        sample_ids = sample_ids[: args.max_samples]
    rng = np.random.default_rng(SEED)
    action_errors = [b0.coarse_action_error(rng.standard_normal(a.shape[1])) for _ in range(3)]
    bq_errors = []
    rows = []
    for index, sid in enumerate(sample_ids):
        sample_dir = samples / sid
        rho = _gt(canonical, sample_dir, binary=True)
        d_star = _q(canonical, rho)
        d_nodes = np.load(sample_dir / "gt_nodes.npy").astype(np.float64).ravel()
        y_saved = np.load(sample_dir / "measurement_b.npy").astype(np.float64).ravel()
        if visible is not None:
            y_saved = y_saved[visible]
        y_generated = a @ d_nodes
        generator_error = np.linalg.norm(y_saved - y_generated) / max(
            np.linalg.norm(y_saved), np.finfo(float).tiny
        )
        generator_clipped_error = np.linalg.norm(y_saved - np.maximum(y_generated, 0.0)) / max(
            np.linalg.norm(y_saved), np.finfo(float).tiny
        )
        pi_rho = canonical.project_coefficients(rho)
        delta_h = d_nodes - pi_rho
        delta_response = a @ delta_h
        x_h = _prediction_nodes(args.v4_predictions, sid)
        # V4 consumed per-sample max-normalized measurements.
        y_model = y_saved / max(float(y_saved.max(initial=0)), 1e-8)
        residual = y_model - a @ x_h
        coarse_adjoint = a.T @ residual
        p1_evidence = canonical.prolong(coarse_adjoint)
        q_evidence = _q(canonical, p1_evidence)
        bq = b0.complement_forward(rho)
        bq_errors.append(np.linalg.norm(bq) / max(np.linalg.norm(y_saved), np.finfo(float).tiny))
        q_evidence_degenerate = np.linalg.norm(q_evidence) <= (
            1e-10 * max(np.linalg.norm(p1_evidence), np.finfo(float).tiny)
        )
        rows.append(
            {
                "sample_id": sid,
                "generator_relative_error": float(generator_error),
                "generator_clipped_relative_error": float(generator_clipped_error),
                "generator_negative_entries_before_clip": int(np.count_nonzero(y_generated < 0)),
                "delta_h_relative_norm": float(np.linalg.norm(delta_h) / max(np.linalg.norm(d_nodes), np.finfo(float).tiny)),
                "delta_response_relative_norm": float(np.linalg.norm(delta_response) / max(np.linalg.norm(y_saved), np.finfo(float).tiny)),
                "detail_relative_norm": float(np.linalg.norm(d_star) / max(np.linalg.norm(rho), np.finfo(float).tiny)),
                "b0_q_relative_response": float(bq_errors[-1]),
                "p1_evidence_vs_detail": _safe_correlation(p1_evidence, d_star),
                "q_evidence_vs_detail": (
                    {"pearson": None, "spearman": None, "reason": "structurally_zero_after_Q"}
                    if q_evidence_degenerate
                    else _safe_correlation(q_evidence, d_star)
                ),
                "q_evidence_norm": float(np.linalg.norm(q_evidence)),
                "model_residual_norm": float(np.linalg.norm(residual)),
            }
        )
        print(f"[d0 {index + 1}/{len(sample_ids)}] {sid}", flush=True)
    numeric_fields = (
        "generator_relative_error",
        "generator_clipped_relative_error",
        "delta_h_relative_norm",
        "delta_response_relative_norm",
        "detail_relative_norm",
        "b0_q_relative_response",
        "q_evidence_norm",
        "model_residual_norm",
    )
    summary = {
        field: {
            "mean": float(np.mean([row[field] for row in rows])),
            "median": float(np.median([row[field] for row in rows])),
            "max": float(np.max([row[field] for row in rows])),
        }
        for field in numeric_fields
    }
    summary["q_evidence_correlation_undefined_cases"] = sum(
        row["q_evidence_vs_detail"]["pearson"] is None for row in rows
    )
    result = {
        "audit": "P0-C D0 generator contract",
        "generator_contract": (
            "measurement_b = maximum(A @ gt_nodes, 0); gt_nodes and gt_voxels are "
            "independent samples of the same analytic source"
        ),
        "evidence_scope": (
            "D0 residual may have shared-source statistical association with voxel detail; "
            "it does not establish direct measurement support for Q rho*."
        ),
        "n_cases": len(rows),
        "summary": summary,
        "operator": {
            "definition": "B0 = A Pi_h",
            "mesh_sha256": sha256_file(shared / "mesh.npz"),
            "frame_manifest_sha256": sha256_file(shared / "frame_manifest.json"),
            "a_sha256": sha256_file(shared / "system_matrix.A.npz"),
            "canonical_cache_hashes": canonical.cache_hashes,
            "b0_i_h_action_error_max": float(max(action_errors)),
            "b0_q_relative_response_max": float(max(bq_errors, default=0.0)),
        },
        "per_case": rows,
    }
    write_metadata(result, args.output)


def select_mechanism_cases(sample_ids: list[str], samples_dir: Path, n: int = 54) -> list[str]:
    """Deterministically sample 6 per foci/depth cell, then fill sparse cells."""

    rng = np.random.default_rng(SEED)
    cells: dict[tuple[int, str], list[str]] = defaultdict(list)
    for sid in sorted(sample_ids):
        params = json.loads((samples_dir / sid / "tumor_params.json").read_text())
        cells[(len(params.get("foci", [])), params.get("depth_tier", "unknown"))].append(sid)
    chosen: list[str] = []
    leftovers: dict[tuple[int, str], list[str]] = {}
    for foci in (1, 2, 3):
        for depth in DEPTHS:
            candidates = cells[(foci, depth)]
            order = rng.permutation(len(candidates))
            chosen.extend(candidates[i] for i in order[:6])
            leftovers[(foci, depth)] = [candidates[i] for i in order[6:]]
    while len(chosen) < n:
        available = [(len(values), key) for key, values in leftovers.items() if values]
        if not available:
            raise RuntimeError(f"Only {len(chosen)} eligible validation cases are available")
        _, key = min(available, key=lambda item: (item[0], item[1]))
        chosen.append(leftovers[key].pop(0))
    return chosen[:n]


def _optical_contract(
    dataset_root: Path, shared_dir: Path, mesh: Any
) -> tuple[dict[int, dict[str, float]], float, dict[str, float]]:
    manifest = json.loads((dataset_root / "dataset_manifest.json").read_text())
    physics = manifest["config"]["physics"]
    inferred, diagnostics = infer_piecewise_optical_coefficients(
        mesh["nodes"], mesh["elements"], mesh["tissue_labels"],
        sp.load_npz(shared_dir / "system_matrix.K.npz"),
        sp.load_npz(shared_dir / "system_matrix.C.npz"),
    )
    return inferred, float(physics.get("n", 1.37)), diagnostics


def _prepare_fine_operator(
    cfg: dict[str, Any], artifact_dir: Path, *, force: bool
) -> tuple[VoxelResolvedLinearOperator, CanonicalCrossDiscretization]:
    data = cfg["data"]
    shared = Path(data["shared_dir"])
    canonical = CanonicalCrossDiscretization(
        data["operator_cache"], shared_dir=shared, factorize=True
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = artifact_dir / "fine_mesh.npz"
    matrix_path = artifact_dir / "M_H.npz"
    sampling_path = artifact_dir / "P_H.npz"
    coarse = np.load(shared / "mesh.npz")
    if force or not mesh_path.exists():
        fine = refine_tetra_mesh(
            coarse["nodes"], coarse["elements"], coarse["tissue_labels"],
            coarse["surface_faces"], coarse["surface_node_indices"],
        )
        if len(coarse["nodes"]) == 19_990 and (
            len(fine.nodes) != 146_599 or len(fine.elements) != 793_672
        ):
            raise RuntimeError(
                f"Unexpected production refinement size: {len(fine.nodes)} nodes, "
                f"{len(fine.elements)} elements"
            )
        save_refined_mesh(fine, mesh_path)
    else:
        fine = load_refined_mesh(mesh_path)
    if force or not matrix_path.exists():
        optical, refractive_index, fit_diagnostics = _optical_contract(
            Path(data["dataset_root"]), shared, coarse
        )
        if max(fit_diagnostics.values()) > 1e-10:
            raise RuntimeError(f"Could not recover saved coarse physics: {fit_diagnostics}")
        matrix = assemble_diffusion_system(fine, optical, refractive_index=refractive_index)
        sp.save_npz(matrix_path, matrix)
    else:
        matrix = sp.load_npz(matrix_path)
    if force or not sampling_path.exists():
        sampling = build_refined_sampling_matrix(
            canonical.p, coarse["elements"], canonical.operator.coords_world, fine
        )
        sp.save_npz(sampling_path, sampling)
    else:
        sampling = sp.load_npz(sampling_path)
    def complement(value: np.ndarray) -> np.ndarray:
        return _q(canonical, value)
    operator = VoxelResolvedLinearOperator(
        matrix,
        sampling,
        fine.detector_node_indices,
        voxel_weight=canonical.quadrature_weight,
        complement=complement,
        solver=SolverContract(),
        provenance={
            "coarse_mesh_sha256": sha256_file(shared / "mesh.npz"),
            "frame_manifest_sha256": sha256_file(shared / "frame_manifest.json"),
            "fine_mesh_sha256": sha256_file(mesh_path),
            "M_H_file_sha256": sha256_file(matrix_path),
            "P_H_file_sha256": sha256_file(sampling_path),
            "detectors": "unchanged coarse surface_node_indices",
        },
    )
    return operator, canonical


def _noise(values: np.ndarray, sid: str, snr_db: float = 30.0) -> tuple[np.ndarray, int, float]:
    seed = SEED + int(sid.rsplit("_", 1)[-1])
    rng = np.random.default_rng(seed)
    rms = float(np.linalg.norm(values) / np.sqrt(values.size))
    sigma = rms / (10.0 ** (snr_db / 20.0))
    return values + rng.normal(0.0, sigma, values.shape), seed, sigma


def run_fine_fem(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    data = cfg["data"]
    samples = Path(data["samples_dir"])
    selected = select_mechanism_cases(_ids(data["val_split"]), samples)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    (args.artifact_dir / "mechanism_cases.txt").write_text("\n".join(selected) + "\n")
    operator, canonical = _prepare_fine_operator(cfg, args.artifact_dir, force=args.force)
    dot_error = operator.dot_product_error()
    if dot_error > 1e-7:
        raise RuntimeError(f"Forward/adjoint dot-product gate failed: {dot_error:.3e}")
    visible = None
    if data.get("use_visible_mask", False):
        visible = np.load(Path(data["shared_dir"]) / "visible_mask.npy").astype(bool)
    coarse_a = _load_a(Path(data["shared_dir"]) / "system_matrix.A.npz", visible)
    rng = np.random.default_rng(SEED)
    nesting_errors = []
    for _ in range(args.nesting_actions):
        coefficients = rng.standard_normal(coarse_a.shape[1])
        expected = coarse_a @ coefficients
        actual = operator.forward(canonical.prolong(coefficients))
        nesting_errors.append(
            float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), np.finfo(float).tiny))
        )
    nesting_mean = float(np.mean(nesting_errors))
    nesting_p95 = float(np.percentile(nesting_errors, 95))
    if nesting_mean > 0.05 or nesting_p95 > 0.10:
        result = {
            "audit": "P0-C matched fine-FEM mechanism study",
            "status": "feasibility_failure",
            "failure_gate": "B_H I_h versus A random-action consistency",
            "failure_reason": (
                f"mean={nesting_mean:.3%} exceeds 5% or p95={nesting_p95:.3%} "
                "exceeds 10%; full 54-case audit was not started"
            ),
            "selected_cases": selected,
            "processed_cases": 0,
            "operator": operator.audit_metadata(),
            "forward_adjoint_relative_error": dot_error,
            "b_h_i_h_vs_a": {
                "n_actions": len(nesting_errors),
                "mean_relative_error": nesting_mean,
                "p95_relative_error": nesting_p95,
                "per_action_relative_error": nesting_errors,
            },
            "scientific_verdict": (
                "NO-GO for residual interrogation under this preregistered matched "
                "fine-FEM operator; retain frozen V4 plus hard-Q state-conditioned prior."
            ),
        }
        write_metadata(result, args.output)
        print(result["failure_reason"], flush=True)
        return
    # Five-case smoke is mandatory. A smaller explicit max is allowed for CI only.
    limit = len(selected) if args.max_samples is None else min(args.max_samples, len(selected))
    rows: list[dict[str, Any]] = []
    for index, sid in enumerate(selected[:limit]):
        rho = _gt(canonical, samples / sid, binary=True)
        coarse_rho = canonical.prolong(canonical.project_coefficients(rho))
        detail = rho - coarse_rho
        y = operator.forward(rho)
        yc = operator.forward(coarse_rho)
        yq = operator.forward(detail)
        closure = np.linalg.norm(y - yc - yq) / max(np.linalg.norm(y), np.finfo(float).tiny)
        noisy, noise_seed, sigma = _noise(y, sid)
        detail_snr = np.linalg.norm(yq) / max(sigma * np.sqrt(len(yq)), np.finfo(float).tiny)
        if closure > 1e-6:
            raise RuntimeError(f"{sid}: closure gate failed ({closure:.3e})")
        np.savez_compressed(
            args.artifact_dir / f"{sid}_measurements.npz",
            y_h=y, y_h_coarse=yc, y_h_q=yq, y_h_30db=noisy,
            noise_seed=np.asarray(noise_seed), noise_sigma=np.asarray(sigma),
        )
        peak_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0**2)
        rows.append(
            {
                "sample_id": sid,
                "closure_relative_error": float(closure),
                "detail_response_snr": float(detail_snr),
                "detail_response_relative_norm": float(np.linalg.norm(yq) / max(np.linalg.norm(y), np.finfo(float).tiny)),
                "noise_seed": noise_seed,
                "noise_sigma": sigma,
                "peak_ram_gib": peak_gib,
            }
        )
        if index == 4 and peak_gib > 16.0:
            raise RuntimeError(f"Five-case resource gate failed: peak RAM {peak_gib:.2f} GiB")
        print(f"[fine-fem {index + 1}/{limit}] {sid}", flush=True)
    probe_summary = None
    probe_stability = None
    if args.probes > 0:
        diagonal, probe_summary = estimate_complement_diagonal(
            operator, n_probes=args.probes, seed=SEED
        )
        np.save(args.artifact_dir / f"complement_diagonal_{args.probes}.npy", diagonal)
        if args.probes == 64:
            diagonal32, summary32 = estimate_complement_diagonal(
                operator, n_probes=32, seed=SEED
            )
            change = np.linalg.norm(diagonal - diagonal32) / max(
                np.linalg.norm(diagonal), np.finfo(float).tiny
            )
            ratio_change = abs(
                probe_summary["frobenius_ratio_estimate"]
                - summary32["frobenius_ratio_estimate"]
            ) / max(abs(probe_summary["frobenius_ratio_estimate"]), np.finfo(float).tiny)
            probe_stability = {
                "diagonal_relative_change_32_to_64": float(change),
                "frobenius_ratio_relative_change_32_to_64": float(ratio_change),
                "passes_5_percent": bool(max(change, ratio_change) <= 0.05),
            }
    spectrum = None
    if args.spectrum:
        spectrum = randomized_complement_spectrum(operator)
    result = {
        "audit": "P0-C matched fine-FEM mechanism study",
        "status": "measurements_complete; frozen-reconstruction interrogation pending"
        if limit else "operator_only",
        "selected_cases": selected,
        "processed_cases": limit,
        "operator": operator.audit_metadata(),
        "forward_adjoint_relative_error": dot_error,
        "b_h_i_h_vs_a": {
            "n_actions": len(nesting_errors),
            "mean_relative_error": nesting_mean,
            "p95_relative_error": nesting_p95,
        },
        "probe_estimate": probe_summary,
        "probe_stability": probe_stability,
        "randomized_spectrum": spectrum,
        "per_case": rows,
        "important_scope": (
            "These files establish matched fine-diffusion responses only. They are not "
            "D0 evidence and are not MCX/transport claims."
        ),
    }
    write_metadata(result, args.output)


def _select_mcx_cases(mechanism_ids: list[str], samples_dir: Path) -> list[str]:
    cells: dict[tuple[int, str], list[str]] = defaultdict(list)
    for sid in mechanism_ids:
        params = json.loads((samples_dir / sid / "tumor_params.json").read_text())
        cells[(len(params.get("foci", [])), params.get("depth_tier", "unknown"))].append(sid)
    chosen = []
    for key in sorted(cells):
        chosen.extend(sorted(cells[key])[:2])
    if len(chosen) < 12:
        chosen.extend(sid for sid in mechanism_ids if sid not in chosen)
    return chosen[:12]


def run_mcx(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    data = cfg["data"]
    samples = Path(data["samples_dir"])
    canonical = CanonicalCrossDiscretization(
        data["operator_cache"], shared_dir=data["shared_dir"], factorize=True
    )
    mechanism_ids = _ids(args.mechanism_cases)
    selected = _select_mcx_cases(mechanism_ids, samples)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for sid in selected:
        detail = _q(canonical, _gt(canonical, samples / sid, binary=True))
        positive = np.maximum(detail, 0.0)
        negative = np.maximum(-detail, 0.0)
        positive_mass = canonical.quadrature_weight * float(positive.sum())
        negative_mass = canonical.quadrature_weight * float(negative.sum())
        if positive_mass <= 0 or negative_mass <= 0:
            raise RuntimeError(f"{sid}: complementary source lacks a positive or negative part")
        full_positive = np.zeros(np.prod(canonical.operator.grid_shape), dtype=np.float32)
        full_negative = np.zeros_like(full_positive)
        full_positive[canonical.operator.valid_flat_indices] = positive / positive_mass
        full_negative[canonical.operator.valid_flat_indices] = negative / negative_mass
        source_path = args.artifact_dir / f"{sid}_mcx_sources.npz"
        np.savez_compressed(
            source_path,
            positive_pattern3d=full_positive.reshape(canonical.operator.grid_shape),
            negative_pattern3d=full_negative.reshape(canonical.operator.grid_shape),
            positive_mass=np.asarray(positive_mass), negative_mass=np.asarray(negative_mass),
        )
        records.append(
            {
                "sample_id": sid,
                "source_file": str(source_path),
                "positive_mass": positive_mass,
                "negative_mass": negative_mass,
                "photons_per_sign_per_seed": 100_000_000,
                "seeds": [SEED + 1000 * i + int(sid.rsplit('_', 1)[-1]) for i in range(3)],
            }
        )
    result = {
        "audit": "P0-C MCX secondary sensitivity preparation",
        "status": "sources_prepared; external MCX responses pending",
        "n_cases": len(records),
        "cases": records,
        "claim_limit": (
            "MCX may test sensitivity to Q rho* only. No V4 remaining residual is "
            "computed or claimed because V4 was not reconstructed on MCX measurements."
        ),
    }
    write_metadata(result, args.output)


def run_summarize(args: argparse.Namespace) -> None:
    d0 = json.loads(args.d0.read_text())
    fine = json.loads(args.fine_fem.read_text()) if args.fine_fem.exists() else None
    mcx = json.loads(args.mcx.read_text()) if args.mcx.exists() else None
    verdict = "NO-GO: matched fine-FEM evidence is incomplete"
    gates: dict[str, Any] = {"d0_b0_q_zero": d0["operator"]["b0_q_relative_response_max"] <= 1e-10}
    if fine is not None:
        rows = fine.get("per_case", [])
        gates.update(
            {
                "dot_product": fine.get("forward_adjoint_relative_error", np.inf) <= 1e-7,
                "closure": bool(rows) and max(row["closure_relative_error"] for row in rows) <= 1e-6,
                "median_detail_response_snr": float(np.median([row["detail_response_snr"] for row in rows])) if rows else None,
                "all_54_cases": fine.get("processed_cases") == 54,
                "residual_interrogation_complete": fine.get("status") == "complete",
            }
        )
        if all(gates.get(key) is True for key in ("dot_product", "closure", "all_54_cases", "residual_interrogation_complete")):
            verdict = "Gate evaluation requires the registered spatial/statistical fields"
    report = {
        "audit": "P0-C complementary observability summary",
        "gates": gates,
        "verdict": verdict,
        "d0_conclusion": (
            "B0 Q=0 is a structural result. D0 permits only shared-source statistical "
            "association language, never direct measurement-supported complementary recovery."
        ),
        "mcx_status": None if mcx is None else mcx.get("status"),
        "sealed_confirmation_opened": False,
        "network_training_performed": False,
    }
    write_metadata(report, args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(func=None)
    sub = parser.add_subparsers(dest="command", required=True)
    common_config = Path("configs/stage2/unified_dual_evidence_fem_v4_2400.yaml")
    d0 = sub.add_parser("d0")
    d0.add_argument("--config", type=Path, default=common_config)
    d0.add_argument("--v4-predictions", type=Path, default=Path("runs/unified_dual_evidence_fem_v4_2400/test_predictions"))
    d0.add_argument("--output", type=Path, default=Path("diagnosis/p0_c_d0_audit.json"))
    d0.add_argument("--max-samples", type=int)
    d0.set_defaults(func=run_d0)

    fine = sub.add_parser("fine-fem")
    fine.add_argument("--config", type=Path, default=common_config)
    fine.add_argument("--artifact-dir", type=Path, default=Path("diagnosis/p0_c_fine_fem_artifacts"))
    fine.add_argument("--output", type=Path, default=Path("diagnosis/p0_c_fine_fem_audit.json"))
    fine.add_argument("--max-samples", type=int)
    fine.add_argument("--probes", type=int, default=0, help="Use 64 for the preregistered full audit")
    fine.add_argument("--nesting-actions", type=int, default=20)
    fine.add_argument("--spectrum", action="store_true")
    fine.add_argument("--force", action="store_true")
    fine.set_defaults(func=run_fine_fem)

    mcx = sub.add_parser("mcx")
    mcx.add_argument("--config", type=Path, default=common_config)
    mcx.add_argument("--mechanism-cases", type=Path, default=Path("diagnosis/p0_c_fine_fem_artifacts/mechanism_cases.txt"))
    mcx.add_argument("--artifact-dir", type=Path, default=Path("diagnosis/p0_c_mcx_artifacts"))
    mcx.add_argument("--output", type=Path, default=Path("diagnosis/p0_c_mcx_audit.json"))
    mcx.set_defaults(func=run_mcx)

    summary = sub.add_parser("summarize")
    summary.add_argument("--d0", type=Path, default=Path("diagnosis/p0_c_d0_audit.json"))
    summary.add_argument("--fine-fem", type=Path, default=Path("diagnosis/p0_c_fine_fem_audit.json"))
    summary.add_argument("--mcx", type=Path, default=Path("diagnosis/p0_c_mcx_audit.json"))
    summary.add_argument("--output", type=Path, default=Path("diagnosis/p0_c_complementary_observability_summary.json"))
    summary.set_defaults(func=run_summarize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
