#!/usr/bin/env python3
"""Freeze the validation-selected CST model before development-test access.

Candidate roles are derived from each candidate's own resolved config instead of
hardcoded run names:

* ``plain_delta``  -- ``innovation: delta`` with no optional module (the reference pool)
* ``none``         -- Hc-only (``innovation: none``)
* ``hc_delta``     -- the ``[Hc, dH]`` innovation adapter
* ``one_ring``     -- ``one_ring_context_dim > 0``
* ``direct_views`` -- ``use_views: true``

Retention gates follow the preregistered rules: one-ring requires a paired
bootstrap CI lower bound above zero; direct views require a mean gain of at least
``+0.002`` together with a CI lower bound above zero. The best eligible CST
candidate is then compared against the strong sequential-concat baseline with a
10,000-draw paired bootstrap, and the claim is only supported when that CI lower
bound exceeds zero.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_resolved_config(path: Path) -> dict[str, Any]:
    """Merge a CST config with its ``base_config`` exactly as training does."""
    config = yaml.safe_load(path.read_text())
    base_path = config.pop("base_config", None)
    if base_path is None:
        return config
    base = yaml.safe_load((Path.cwd() / base_path).read_text())

    def merge(target: dict[str, Any], source: dict[str, Any]) -> None:
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = value

    merge(base, config)
    return base


def classify(config_path: Path) -> dict[str, Any]:
    model = load_resolved_config(config_path)["model"]
    innovation = str(model["innovation"])
    one_ring = int(model.get("one_ring_context_dim", 0)) > 0
    views = bool(model.get("use_views", False))
    return {
        "innovation": innovation,
        "one_ring": one_ring,
        "views": views,
        "plain_delta": innovation == "delta" and not one_ring and not views,
    }


def paired_dice(left: dict, right: dict, seed: int = 20260901) -> dict:
    left_rows = {row["sample_id"]: row for row in left["per_sample"]}
    right_rows = {row["sample_id"]: row for row in right["per_sample"]}
    ids = sorted(left_rows)
    if ids != sorted(right_rows) or len(ids) != 300:
        raise RuntimeError("Paired comparison inputs must contain identical 300 IDs")
    delta = np.asarray(
        [right_rows[sid]["final_dice"] - left_rows[sid]["final_dice"] for sid in ids]
    )
    rng = np.random.default_rng(seed)
    samples = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(10000, len(delta)))
    null = (signs * delta).mean(axis=1)
    return {
        "mean_difference": float(delta.mean()),
        "paired_bootstrap_95_ci": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "two_sided_sign_flip_p_value": float(
            (np.count_nonzero(np.abs(null) >= abs(delta.mean())) + 1) / 10001
        ),
        "effect_consistency_fraction_positive": float(np.mean(delta > 0)),
        "draws": 10000,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-candidates", type=int, required=True)
    parser.add_argument("--baseline-validation", type=Path, required=True)
    args = parser.parse_args()

    payloads = [
        json.loads(path.read_text()) for path in sorted(args.validation_dir.glob("*.json"))
    ]
    if len(payloads) != args.expected_candidates:
        raise RuntimeError(
            f"Found {len(payloads)} validation candidates, expected {args.expected_candidates}"
        )
    baseline = json.loads(args.baseline_validation.read_text())
    if baseline.get("n_samples") != 300 or baseline.get("split") != "validation":
        raise RuntimeError("Strong concat baseline must be a complete val300 artifact")
    if baseline.get("confirmation_data_used", True) or not baseline.get("hard_q", False):
        raise RuntimeError("Strong concat baseline violates governance")

    rows: list[dict[str, Any]] = []
    payload_by_name: dict[str, dict] = {}
    role_by_name: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        if payload.get("split") != "validation" or payload.get("n_samples") != 300:
            raise RuntimeError("Every selection artifact must be a complete val300 result")
        if payload.get("confirmation_data_used", True) or not payload.get("hard_q", False):
            raise RuntimeError("Candidate violates confirmation/hard-Q governance")
        checkpoint = Path(payload["checkpoint"])
        config = Path(payload["config"])
        name = config.stem
        role = classify(config)
        payload_by_name[name] = payload
        role_by_name[name] = role
        if role["plain_delta"]:
            role_label = "plain_delta"
        elif role["one_ring"]:
            role_label = "one_ring"
        elif role["views"]:
            role_label = "direct_views"
        else:
            role_label = role["innovation"]
        rows.append(
            {
                "name": name,
                "role": role_label,
                "innovation": role["innovation"],
                "config": str(config.resolve()),
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_epoch": payload["checkpoint_epoch"],
                "val_final_dice": payload["summary"]["final_dice"],
                "val_detail_cosine": payload["summary"]["detail_cosine"],
                "parameter_count": payload["parameter_count"],
                "config_sha256": sha256(config),
                "checkpoint_sha256": sha256(checkpoint),
            }
        )

    def best_of(predicate) -> str | None:
        pool = [name for name, role in role_by_name.items() if predicate(role)]
        if not pool:
            return None
        return max(pool, key=lambda name: payload_by_name[name]["summary"]["final_dice"])

    reference_name = best_of(lambda role: role["plain_delta"])
    if reference_name is None:
        raise RuntimeError("No plain delta candidate is available as the CST reference")
    reference = payload_by_name[reference_name]

    # Preregistered optional-module gates, measured against the plain-delta reference.
    module_rules = {
        "one_ring": "ci_positive",
        "direct_views": "gain_0.002_and_ci_positive",
    }
    module_gates: dict[str, dict[str, Any]] = {}
    for name, role in role_by_name.items():
        if role["plain_delta"]:
            continue
        rule = None
        if role["one_ring"]:
            rule = module_rules["one_ring"]
        elif role["views"]:
            rule = module_rules["direct_views"]
        if rule is None:
            continue
        stats = paired_dice(reference, payload_by_name[name])
        retained = stats["paired_bootstrap_95_ci"][0] > 0
        if rule == "gain_0.002_and_ci_positive":
            retained = retained and stats["mean_difference"] >= 0.002
        module_gates[name] = {"rule": rule, "retained": bool(retained), **stats}

    # Latent-transition representation ablation: Hc + dH vs Hc-only, and the
    # [Hc, dH] adapter vs the plain dH innovation.
    hc_only_name = best_of(lambda role: role["innovation"] == "none")
    hc_delta_name = best_of(lambda role: role["innovation"] == "hc_delta")
    if hc_only_name is None or hc_delta_name is None:
        raise RuntimeError(
            "Formal CST freeze requires both the Hc-only and the [Hc, dH] candidates"
        )
    representation_comparisons = {
        "reference_name": reference_name,
        "hc_only_name": hc_only_name,
        "hc_delta_name": hc_delta_name,
        "hc_plus_delta_h_vs_hc_only": paired_dice(
            payload_by_name[hc_only_name], reference
        ),
        "hc_delta_adapter_vs_delta_h": paired_dice(
            reference, payload_by_name[hc_delta_name]
        ),
    }

    for row in rows:
        gate = module_gates.get(row["name"])
        row["eligible"] = gate is None or gate["retained"]
    eligible = [row for row in rows if row["eligible"]]
    eligible.sort(key=lambda row: (-row["val_final_dice"], row["name"]))
    selected = eligible[0]
    selected_payload = payload_by_name[selected["name"]]

    cst_vs_b4 = paired_dice(baseline, selected_payload)
    cst_supported = (
        cst_vs_b4["mean_difference"] > 0
        and cst_vs_b4["paired_bootstrap_95_ci"][0] > 0
    )
    baseline_checkpoint = Path(baseline["checkpoint"])
    final_model = {
        "method": "CST" if cst_supported else "strong_sequential_concat_B4",
        "config": selected["config"] if cst_supported else None,
        "checkpoint": (
            selected["checkpoint"] if cst_supported else str(baseline_checkpoint.resolve())
        ),
        "checkpoint_sha256": (
            selected["checkpoint_sha256"] if cst_supported else sha256(baseline_checkpoint)
        ),
        "validation_dice": (
            selected["val_final_dice"]
            if cst_supported
            else baseline["summary"]["final_dice"]
        ),
        "selection_reason": (
            "CST paired bootstrap CI lower bound exceeds zero"
            if cst_supported
            else "CST did not significantly exceed the strong concat baseline on validation"
        ),
    }
    rows.sort(key=lambda row: (-row["val_final_dice"], row["name"]))
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    receipt = {
        "frozen_on_validation": True,
        "selection_metric": "mean_val300_final_dice_at_fixed_threshold_0.5",
        "threshold": 0.5,
        "selected_config": selected["config"],
        "selected_config_sha256": selected["config_sha256"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "selected_epoch": selected["checkpoint_epoch"],
        "selected_val_dice": selected["val_final_dice"],
        "selected_cst_candidate": selected["name"],
        "cst_reference_candidate": reference_name,
        "optional_module_gates": module_gates,
        "latent_transition_ablation": representation_comparisons,
        "selected_cst_vs_strong_concat": cst_vs_b4,
        "cst_claim_supported_on_validation": bool(cst_supported),
        "final_model_selected_on_validation": final_model,
        "strong_concat_baseline": {
            "artifact": str(args.baseline_validation.resolve()),
            "checkpoint": str(baseline_checkpoint.resolve()),
            "checkpoint_sha256": sha256(baseline_checkpoint),
            "validation_dice": baseline["summary"]["final_dice"],
        },
        "candidates": rows,
        "git_head_before_development_test": git_head,
        "development_test_accessed_before_freeze": False,
        "confirmation_data_used": False,
        "sealed_confirmation_accessed": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "VALIDATION_FREEZE.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    with (args.output_dir / "validation_model_selection.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
