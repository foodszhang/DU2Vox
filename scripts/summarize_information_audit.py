#!/usr/bin/env python3
"""Create paired A0--A3 tables and the preregistered go/no-go decision report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ARMS = ("A0", "A1", "A2", "A3", "VA0", "VA1", "VA2", "VA3")
METRICS = (
    "final_dice",
    "final_hd95",
    "final_localization_error",
    "final_mse",
    "final_relative_l2",
    "stage2_marginal_dice",
    "detail_cosine",
    "detail_relative_l1",
    "detail_relative_l2",
    "boundary_energy_le_0.2mm",
    "boundary_energy_le_0.6mm",
    "coarse_leakage",
    "coarse_preservation_relative_l2",
)


def load_payload(path: Path, expected_split: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("confirmation_data_used", True):
        raise RuntimeError(f"Confirmation data marker is not false in {path}")
    if payload.get("split") != expected_split:
        raise RuntimeError(
            f"Expected {expected_split} payload at {path}, got {payload.get('split')}"
        )
    return payload


def row_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["sample_id"]: row for row in payload["per_sample"]}


def paired(
    left: dict[str, Any], right: dict[str, Any], key: str, seed: int
) -> dict[str, Any]:
    left_rows, right_rows = row_map(left), row_map(right)
    ids = sorted(left_rows)
    if ids != sorted(right_rows):
        raise RuntimeError("Paired audit files have different sample identities")
    delta = np.asarray(
        [right_rows[sid][key] - left_rows[sid][key] for sid in ids], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(delta), size=(10000, len(delta)))
    means = delta[draws].mean(axis=1)
    return {
        "left": left["audit_arm"],
        "right": right["audit_arm"],
        "metric": key,
        "n": len(delta),
        "mean_delta": float(delta.mean()),
        "paired_bootstrap_95_ci": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "bootstrap_draws": 10000,
    }


def bootstrap_mean(values: np.ndarray, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(10000, len(values)))
    means = values[draws].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "paired_bootstrap_95_ci": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "bootstrap_draws": 10000,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--development-test-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    if args.seed != 20260901:
        raise RuntimeError("Canonical audit bootstrap seed is fixed at 20260901")
    validation = {
        arm: load_payload(args.val_dir / f"{arm.lower()}.json", "validation")
        for arm in ARMS
    }
    for arm, payload in validation.items():
        if payload.get("audit_arm") != arm:
            raise RuntimeError(f"Validation file identity mismatch for {arm}")
        if payload.get("n_samples") != 300 or len(row_map(payload)) != 300:
            raise RuntimeError(f"Validation arm {arm} is not the frozen val300")
    contract_keys = (
        "input_dim",
        "parameter_count",
        "initialization_sha256",
        "train_split_order_sha256",
    )
    matched_contract = {key: validation["A0"][key] for key in contract_keys}
    for arm in ARMS[1:]:
        for key, expected in matched_contract.items():
            if validation[arm].get(key) != expected:
                raise RuntimeError(f"Matched-arm contract differs for {arm}: {key}")
    comparisons: dict[str, dict[str, Any]] = {}
    pairs = (("A0", "A1"), ("A1", "A2"), ("A1", "A3")) + tuple(
        (arm, f"V{arm}") for arm in ("A0", "A1", "A2", "A3")
    )
    for left, right in pairs:
        for metric in METRICS:
            key = f"{right}-{left}:{metric}"
            comparisons[key] = paired(
                validation[left], validation[right], metric, args.seed
            )
    marginal_gains = {
        arm: bootstrap_mean(
            np.asarray(
                [row["stage2_marginal_dice"] for row in validation[arm]["per_sample"]],
                dtype=np.float64,
            ),
            args.seed,
        )
        for arm in ARMS
    }

    def passes(key: str, minimum: float) -> bool:
        item = comparisons[key]
        return bool(
            item["mean_delta"] >= minimum
            and item["paired_bootstrap_95_ci"][0] > 0
        )

    a1 = passes("A1-A0:final_dice", 0.003)
    a2 = passes("A2-A1:final_dice", 0.002)
    a3_vs_a1 = passes("A3-A1:final_dice", 0.005)
    a3_summary = validation["A3"]["summary"]
    a3_detail = bool(
        a3_summary["stage2_marginal_dice"] >= 0.005
        and a3_summary["detail_cosine"] >= 0.10
    )
    # The marginal gain is paired final-vs-own-coarse within each case.
    marginal_ci = marginal_gains["A3"]["paired_bootstrap_95_ci"]
    a3 = a3_detail and marginal_ci[0] > 0
    view_gates = {}
    for strict in ("A0", "A1", "A2", "A3"):
        direct = f"V{strict}"
        dice = comparisons[f"{direct}-{strict}:final_dice"]
        view_gates[strict] = bool(
            dice["mean_delta"] >= 0.002
            and dice["paired_bootstrap_95_ci"][0] > 0
            and validation[direct]["summary"]["detail_cosine"]
            >= validation[strict]["summary"]["detail_cosine"]
        )
    decisions = {
        "A1_coarse_state_information": a1,
        "A2_terminal_latent_information": a2,
        "A3_stage1_bottleneck_vs_A1": a3_vs_a1,
        "A3_stage2_viability": a3,
        "A3_marginal_bootstrap_95_ci": marginal_ci,
        "retain_direct_views_by_arm": view_gates,
        "approve_V1_local_state_decoder": a1 and a3,
        "approve_latent_for_V2_or_later": a1 and a2 and a3,
        "stop_V1_to_V5": not a3,
        "prioritize_stage1_quality": a3_vs_a1,
    }
    result: dict[str, Any] = {
        "protocol": "A0-A3 matched information-source audit",
        "validation_only_decisions": True,
        "matched_arm_contract": matched_contract,
        "validation_summaries": {
            arm: validation[arm]["summary"] for arm in ARMS
        },
        "frozen_checkpoints": {
            arm: validation[arm]["checkpoint"] for arm in ARMS
        },
        "paired_validation_comparisons": comparisons,
        "within_arm_stage2_marginal_gains": marginal_gains,
        "decisions": decisions,
        "frozen_v4_retrained": False,
        "sealed_confirmation_accessed": False,
        "D1_D3_generated": False,
        "V1_V5_trained": False,
        "residual_interrogation_continued": False,
    }
    if args.development_test_dir is not None:
        development = {
            arm: load_payload(
                args.development_test_dir / f"{arm.lower()}.json", "development_test"
            )
            for arm in ARMS
        }
        for arm in ARMS:
            if development[arm].get("audit_arm") != arm:
                raise RuntimeError(f"Development-test file identity mismatch for {arm}")
            if development[arm].get("n_samples") != 300 or len(row_map(development[arm])) != 300:
                raise RuntimeError(
                    f"Development-test arm {arm} is not the frozen development-test300"
                )
            if development[arm]["checkpoint"] != validation[arm]["checkpoint"]:
                raise RuntimeError(f"Development-test checkpoint changed for {arm}")
        result["development_test_summaries"] = {
            arm: development[arm]["summary"] for arm in ARMS
        }
        result["development_test_paired_comparisons"] = {
            f"{right}-{left}:{metric}": paired(
                development[left], development[right], metric, args.seed
            )
            for left, right in pairs
            for metric in METRICS
        }
        result["development_test_within_arm_stage2_marginal_gains"] = {
            arm: bootstrap_mean(
                np.asarray(
                    [
                        row["stage2_marginal_dice"]
                        for row in development[arm]["per_sample"]
                    ],
                    dtype=np.float64,
                ),
                args.seed,
            )
            for arm in ARMS
        }
        result["development_test_evaluated_after_validation_freeze"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    report = [
        "# A0-A3 information-source audit decision",
        "",
        "Architecture decisions below are derived from validation only.",
        "",
        "| Gate | Pass |",
        "| --- | --- |",
    ]
    report.extend(
        f"| {key} | {'GO' if value else 'NO-GO'} |"
        for key, value in decisions.items()
        if isinstance(value, bool)
    )
    for title, names in (
        ("Strict geometry", ("A0", "A1", "A2", "A3")),
        ("Direct views-on", ("VA0", "VA1", "VA2", "VA3")),
    ):
        report.extend(
            [
                "",
                f"## {title}",
                "",
                "| Arm | Final Dice | Stage-II marginal | Detail cosine |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for arm in names:
            summary = validation[arm]["summary"]
            report.append(
                f"| {arm} | {summary['final_dice']:.6f} | "
                f"{summary['stage2_marginal_dice']:+.6f} | "
                f"{summary['detail_cosine']:.6f} |"
            )
    report.extend(
        [
            "",
            "## Preregistered validation contrasts",
            "",
            "| Contrast | Dice delta | Paired 95% CI |",
            "| --- | ---: | ---: |",
        ]
    )
    for left, right in pairs:
        item = comparisons[f"{right}-{left}:final_dice"]
        low, high = item["paired_bootstrap_95_ci"]
        report.append(
            f"| {right} - {left} | {item['mean_delta']:+.6f} | "
            f"[{low:+.6f}, {high:+.6f}] |"
        )
    if args.development_test_dir is not None:
        report.extend(
            [
                "",
                "## Development-test results",
                "",
                "These results were evaluated only after the validation checkpoint "
                "freeze and did not affect any gate decision.",
            ]
        )
        development_summaries = result["development_test_summaries"]
        for title, names in (
            ("Strict geometry", ("A0", "A1", "A2", "A3")),
            ("Direct views-on", ("VA0", "VA1", "VA2", "VA3")),
        ):
            report.extend(
                [
                    "",
                    f"### {title}",
                    "",
                    "| Arm | Final Dice | HD95 | Localization | MSE | rL2 | "
                    "Stage-II marginal | Detail cosine |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for arm in names:
                summary = development_summaries[arm]
                report.append(
                    f"| {arm} | {summary['final_dice']:.6f} | "
                    f"{summary['final_hd95']:.6f} | "
                    f"{summary['final_localization_error']:.6f} | "
                    f"{summary['final_mse']:.6g} | "
                    f"{summary['final_relative_l2']:.6f} | "
                    f"{summary['stage2_marginal_dice']:+.6f} | "
                    f"{summary['detail_cosine']:.6f} |"
                )
        report.extend(
            [
                "",
                "### Paired development-test contrasts",
                "",
                "| Contrast | Dice delta | Paired 95% CI |",
                "| --- | ---: | ---: |",
            ]
        )
        development_comparisons = result["development_test_paired_comparisons"]
        for left, right in pairs:
            item = development_comparisons[f"{right}-{left}:final_dice"]
            low, high = item["paired_bootstrap_95_ci"]
            report.append(
                f"| {right} - {left} | {item['mean_delta']:+.6f} | "
                f"[{low:+.6f}, {high:+.6f}] |"
            )
    report.extend(
        [
            "",
            "Strict A0-A3 and paired views-on VA0-VA3 are reported separately in "
            f"`{args.output.name}`.",
            "",
            "Frozen V4 was not trained; sealed confirmation was not accessed; "
            "D1-D3 and V1-V5 were not run.",
            "",
        ]
    )
    args.output.with_suffix(".md").write_text("\n".join(report))


if __name__ == "__main__":
    main()
