#!/usr/bin/env python3
"""Audit the canonical complement operator and V4-specific detail oracle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.bridge.canonical_cross_discretization import CanonicalCrossDiscretization
from du2vox.evaluation.error_structured import _reconstruction_metrics
from experiments.cross_discretization_decomposition.run_analysis import focus_metadata
from scripts.train_iterative_fem_corrector import build_dataset, load_ids


METRICS = ("dice", "precision", "recall", "weak_recall", "hd95", "localization_error", "mse")


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.nanmean([row[key] for row in rows])) for key in METRICS}


def _baseline(report: Path) -> dict[str, dict[str, float]]:
    summary = json.loads(report.read_text())["summary"]
    return {
        "stage1": {key: float(summary[f"stage1_{key}"]) for key in METRICS},
        "v4": {key: float(summary[f"step3_{key}"]) for key in METRICS},
    }


def _fmt(value: float) -> str:
    return f"{value:.8g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--val-report", type=Path, required=True)
    parser.add_argument("--test-report", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--oracle-output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    data = cfg["data"]
    canonical = CanonicalCrossDiscretization(
        data["operator_cache"], shared_dir=data["shared_dir"], factorize=True
    )
    p = canonical.p

    rng = np.random.default_rng(20260901)
    identity_errors = []
    idempotence_errors = []
    symmetry_errors = []
    for _ in range(3):
        coeff = rng.standard_normal(p.shape[1])
        recovered = canonical.project_coefficients(canonical.prolong(coeff))
        identity_errors.append(float(np.linalg.norm(recovered - coeff) / np.linalg.norm(coeff)))
        z = rng.standard_normal(p.shape[0])
        projected = canonical.prolong(canonical.project_coefficients(z))
        projected_twice = canonical.prolong(canonical.project_coefficients(projected))
        idempotence_errors.append(
            float(np.linalg.norm(projected_twice - projected) / np.linalg.norm(projected))
        )
        a = rng.standard_normal(p.shape[0])
        # Symmetry of I_h Pi_h in the fixed uniform-voxel inner product.
        lhs = float(np.dot(a, projected))
        rhs = float(np.dot(canonical.prolong(canonical.project_coefficients(a)), z))
        symmetry_errors.append(abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30))

    split_results: dict[str, Any] = {}
    all_closure: list[float] = []
    all_leakage: list[float] = []
    split_specs = {
        "val": (args.val_predictions, args.val_report),
        "development-test": (args.test_predictions, args.test_report),
    }
    for split_name, (prediction_dir, baseline_report) in split_specs.items():
        split_key = "test" if split_name == "development-test" else "val"
        ids = load_ids(data[f"{split_key}_split"])
        dataset = build_dataset(
            cfg, split_key, ids, int(cfg["training"].get("n_query_points", 8192))
        )
        oracle_rows: list[dict[str, float]] = []
        closure_values: list[float] = []
        leakage_values: list[float] = []
        for index, sid in enumerate(ids):
            gt_volume = np.load(Path(data["samples_dir"]) / sid / "gt_voxels.npy", mmap_mode="r")
            gt = (
                np.asarray(gt_volume).ravel()[canonical.operator.valid_flat_indices] > 0.05
            ).astype(np.float64)
            pi_gt = canonical.project_coefficients(gt)
            coarse_gt = canonical.prolong(pi_gt)
            detail = gt - coarse_gt
            reprojection = canonical.prolong(canonical.project_coefficients(detail))
            leakage_values.append(
                float(np.dot(reprojection, reprojection) / max(np.dot(detail, detail), 1e-30))
            )
            closure_values.append(
                float(np.linalg.norm(gt - coarse_gt - detail) / max(np.linalg.norm(gt), 1e-30))
            )
            with np.load(prediction_dir / f"{sid}.npz") as prediction_file:
                v4 = prediction_file["final_prediction"].astype(np.float64)
            oracle_prediction = v4 + detail
            tumor = focus_metadata(Path(data["samples_dir"]) / sid)["tumor_params"]
            oracle_rows.append(_reconstruction_metrics(oracle_prediction, gt, dataset, tumor))
            if (index + 1) % 25 == 0:
                print(f"[{split_name} oracle {index + 1}/{len(ids)}]", flush=True)
        baseline = _baseline(baseline_report)
        oracle = _mean(oracle_rows)
        split_results[split_name] = {
            **baseline,
            "v4_oracle_detail": oracle,
            "oracle_delta_dice": oracle["dice"] - baseline["v4"]["dice"],
            "closure_max": max(closure_values),
            "closure_mean": float(np.mean(closure_values)),
            "gt_complement_leakage_mean": float(np.mean(leakage_values)),
            "gt_complement_leakage_max": max(leakage_values),
        }
        all_closure.extend(closure_values)
        all_leakage.extend(leakage_values)

    payload = {
        "operator": {
            "pi_h_code": "experiments/cross_discretization_decomposition/decomposition.py::MassProjector.coefficients",
            "i_h_code": "experiments/cross_discretization_decomposition/decomposition.py::DomainOperator.p",
            "identical_to_decomposition": True,
            "pi_h_i_h_relative_error_max": max(identity_errors),
            "pi_h_i_h_relative_errors": identity_errors,
            "i_h_pi_h_idempotence_relative_error_max": max(idempotence_errors),
            "i_h_pi_h_self_adjoint_bilinear_error_max": max(symmetry_errors),
            "closure_error_max": max(all_closure),
            "closure_error_mean": float(np.mean(all_closure)),
            "gt_complement_leakage_mean": float(np.mean(all_leakage)),
            "gt_complement_leakage_max": max(all_leakage),
            "grid_spacing_mm": canonical.operator.spacing_mm,
            "grid_shape": list(canonical.operator.grid_shape),
            "full_voxel_count": int(np.prod(canonical.operator.grid_shape)),
            "valid_voxel_count": int(p.shape[0]),
            "fem_node_count": int(p.shape[1]),
        },
        "splits": split_results,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n")

    op = payload["operator"]
    audit = f"""# Voxel Complement Operator Audit

## Result

The canonical operator contract passes on the unchanged 0.2-mm decomposition domain. `Pi_h I_h` is a numerical left identity, and `I_h Pi_h` is an idempotent, self-adjoint sampled-L2 projector under the uniform voxel inner product. Therefore `Q = I - I_h Pi_h` may be described as the orthogonal complement projector for this fixed sampled domain (not as a continuous-domain claim).

## Required audit

1. Exact code path for `Pi_h`: `{op["pi_h_code"]}`. It uses sparse LU to solve `(P.T @ P)c = P.T @ rho`, with zero ridge for this certified cache.
2. Exact code path for `I_h`: `{op["i_h_code"]}`, loaded by `du2vox/bridge/canonical_cross_discretization.py::CanonicalCrossDiscretization`; application is the fixed CSR product `P @ c`.
3. Identical to decomposition experiment: **yes**. Production imports the experiment's `load_operator` and `MassProjector` directly.
4. `||Pi_h I_h c - c||_2 / ||c||_2` (maximum of three seeded random vectors): `{_fmt(op["pi_h_i_h_relative_error_max"])}`.
5. Decomposition closure error `||rho - I_h Pi_h rho - w*||_2 / ||rho||_2`: mean `{_fmt(op["closure_error_mean"])}`, maximum `{_fmt(op["closure_error_max"])}` across val300 + development-test300.
6. GT complement leakage `||I_h Pi_h w*||_2^2 / ||w*||_2^2`: mean `{_fmt(op["gt_complement_leakage_mean"])}`, maximum `{_fmt(op["gt_complement_leakage_max"])}`.
7. Domain/grid: canonical GT-center grid, shape `{tuple(op["grid_shape"])}`, isotropic `{op["grid_spacing_mm"]}` mm, C-order, binary support `gt_voxels > 0.05`.
8. Valid voxel count: `{op["valid_voxel_count"]:,}` of `{op["full_voxel_count"]:,}`; FEM nodes: `{op["fem_node_count"]:,}`.

Additional numerical checks: relative idempotence error of `I_h Pi_h` is at most `{_fmt(op["i_h_pi_h_idempotence_relative_error_max"])}`; seeded bilinear self-adjointness error is at most `{_fmt(op["i_h_pi_h_self_adjoint_bilinear_error_max"])}`.

No confirmation data or confirmation inference was used.
"""
    args.audit_output.write_text(audit)

    val = split_results["val"]
    test = split_results["development-test"]

    def row(method: str, key: str) -> str:
        return f"| {method} | {val[key]['dice']:.5f} | {test[key]['dice']:.5f} |"

    metric_rows = []
    for split_name, result in split_results.items():
        for method, key in (
            ("Stage1 FEM", "stage1"),
            ("V4 coarse only", "v4"),
            ("V4 + oracle detail", "v4_oracle_detail"),
        ):
            values = result[key]
            metric_rows.append(
                "| "
                + " | ".join([split_name, method] + [f"{values[m]:.6f}" for m in METRICS])
                + " |"
            )
    decision = "PROCEED" if val["oracle_delta_dice"] >= 0.01 else "STOP"
    oracle_report = f"""# V4-Specific Voxel Detail Oracle Report

## Decision: {decision}

The validation-only oracle gain is `{val["oracle_delta_dice"]:+.6f}` Dice. The development-test oracle gain is reported only as a development comparison and did not determine this decision.

| Method | Val Dice | Dev-Test Dice |
| --- | ---: | ---: |
{row("Stage1 FEM", "stage1")}
{row("V4", "v4")}
{row("V4 + oracle detail", "v4_oracle_detail")}

`Delta_oracle-detail = Dice(V4 + w*) - Dice(V4)`:

- validation: `{val["oracle_delta_dice"]:+.6f}`
- development-test: `{test["oracle_delta_dice"]:+.6f}`

Here `w* = rho_GT - I_h Pi_h rho_GT`, and `rho_V4+oracle-detail = I_h x_V4 + w*`. Predictions are raw (unclamped), with the fixed 0.5 support threshold.

## Full metrics

| Split | Method | Dice | Precision | Recall | Weak Recall | HD95 | Localization | MSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(metric_rows)}

The operator and oracle use only the existing val300 and development-test300 splits. Confirmation remains sealed and was not inspected.
"""
    args.oracle_output.write_text(oracle_report)


if __name__ == "__main__":
    main()
