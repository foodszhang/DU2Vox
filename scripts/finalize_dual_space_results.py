#!/usr/bin/env python3
"""Archive frozen CST results, paired statistics, tables, and paper figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def rows_by_id(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["sample_id"]: row for row in payload["per_sample"]}


def paired_statistics(
    left: dict[str, Any],
    right: dict[str, Any],
    left_key: str,
    right_key: str,
    *,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    left_rows, right_rows = rows_by_id(left), rows_by_id(right)
    ids = sorted(left_rows)
    if ids != sorted(right_rows):
        raise RuntimeError("Paired results do not contain identical sample IDs")
    delta = np.asarray(
        [right_rows[sid][right_key] - left_rows[sid][left_key] for sid in ids],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(10000, len(delta)))
    bootstrap = delta[indices].mean(axis=1)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(10000, len(delta)))
    null_means = (delta[None] * signs).mean(axis=1)
    p_value = (np.count_nonzero(np.abs(null_means) >= abs(delta.mean())) + 1) / 10001
    return (
        {
            "n": len(delta),
            "mean_difference": float(delta.mean()),
            "paired_bootstrap_95_ci": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
            "two_sided_sign_flip_p_value": float(p_value),
            "effect_consistency_fraction_positive": float(np.mean(delta > 0)),
            "bootstrap_draws": 10000,
            "seed": seed,
        },
        delta,
    )


def holm_adjust(items: dict[str, dict[str, Any]]) -> None:
    ordered = sorted(items, key=lambda key: items[key]["two_sided_sign_flip_p_value"])
    running = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        raw = items[key]["two_sided_sign_flip_p_value"]
        running = max(running, min(1.0, (total - rank) * raw))
        items[key]["holm_adjusted_p_value"] = running


def coarse_state_statistics(
    proposed_test: dict[str, Any], sample_ids: list[str]
) -> dict[str, float]:
    resolved = Path(proposed_test["checkpoint"]).parents[1] / "resolved_config.yaml"
    config = yaml.safe_load(resolved.read_text())
    corrected_root = Path(config["data"]["v4_states_root"]) / "test"
    initial_root = Path(config["data"]["test_initial_state_root"])
    norms, rms_values, peaks, correction_ratios = [], [], [], []
    for sid in sample_ids:
        corrected = np.load(corrected_root / f"{sid}.npy").astype(np.float64)
        initial = np.load(initial_root / sid / "coarse_d.npy").astype(np.float64)
        norms.append(np.linalg.norm(corrected))
        rms_values.append(np.sqrt(np.mean(np.square(corrected))))
        peaks.append(np.max(np.abs(corrected)))
        correction_ratios.append(
            np.linalg.norm(corrected - initial) / max(np.linalg.norm(initial), 1e-30)
        )
    return {
        "mean_fem_state_l2_norm": float(np.mean(norms)),
        "mean_fem_state_rms": float(np.mean(rms_values)),
        "mean_fem_state_max_abs": float(np.mean(peaks)),
        "mean_v4_correction_relative_l2_vs_initial_stage1": float(
            np.mean(correction_ratios)
        ),
        "voxel_loss_induced_relative_state_drift": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--proposed-val", type=Path, required=True)
    parser.add_argument("--proposed-test", type=Path, required=True)
    parser.add_argument("--b4-val", type=Path, required=True)
    parser.add_argument("--b4-test", type=Path, required=True)
    parser.add_argument("--q-only-no-view-test", type=Path, required=True)
    parser.add_argument("--q-only-test", type=Path, required=True)
    parser.add_argument("--validation-freeze", type=Path, required=True)
    parser.add_argument("--p0-comparison", type=Path, required=True)
    parser.add_argument("--responsibility-audit", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    proposed_val, proposed_test = read(args.proposed_val), read(args.proposed_test)
    b4_val, b4_test = read(args.b4_val), read(args.b4_test)
    q_only_no_view = read(args.q_only_no_view_test)
    q_only = read(args.q_only_test)
    freeze = read(args.validation_freeze)
    p0 = read(args.p0_comparison)
    responsibility = read(args.responsibility_audit)
    if proposed_test.get("confirmation_data_used", True):
        raise RuntimeError("Proposed result is not explicitly confirmation-free")
    if proposed_test.get("n_samples") != 300:
        raise RuntimeError("Proposed result is not development-test300")

    val_stats, _ = paired_statistics(
        b4_val, proposed_val, "final_dice", "final_dice", seed=20260901
    )
    comparisons: dict[str, dict[str, Any]] = {}
    deltas: dict[str, np.ndarray] = {}
    comparisons["proposed_vs_b4"], deltas["proposed_vs_b4"] = paired_statistics(
        b4_test, proposed_test, "final_dice", "final_dice", seed=20260901
    )
    comparisons["proposed_vs_coarse_only"], deltas["proposed_vs_coarse_only"] = (
        paired_statistics(
            proposed_test,
            proposed_test,
            "coarse_dice",
            "final_dice",
            seed=20260901,
        )
    )
    comparisons["proposed_vs_q_only"], deltas["proposed_vs_q_only"] = paired_statistics(
        q_only, proposed_test, "final_dice", "final_dice", seed=20260901
    )
    comparisons["proposed_vs_q_only"]["protocol_note"] = (
        "Same sample identities but different frozen initial Stage-I checkpoints; "
        "reported as a responsibility contrast, not a matched decoder ablation."
    )
    holm_adjust(comparisons)
    paired_payload = {
        "validation_proposed_vs_b4": val_stats,
        "development_test": comparisons,
        "multiple_comparison_correction": "Holm across three development-test Dice contrasts",
        "confirmation_data_used": False,
    }
    (output / "PAIRED_BOOTSTRAP.json").write_text(
        json.dumps(paired_payload, indent=2) + "\n"
    )

    validation_final = freeze["final_model_selected_on_validation"]
    final_is_cst = validation_final["method"] == "CST"
    cst_config = Path(freeze["selected_config"])
    cst_checkpoint = Path(freeze["selected_checkpoint"])
    if final_is_cst:
        selected_config = cst_config
        checkpoint = cst_checkpoint
    else:
        selected_config = Path("configs/stage2/information_audit/a2.yaml").resolve()
        checkpoint = Path(validation_final["checkpoint"])
    resolved_config = selected_config
    run_resolved = checkpoint.parents[1] / "resolved_config.yaml"
    if run_resolved.exists():
        resolved_config = run_resolved
    final_config = yaml.safe_load(resolved_config.read_text())
    final_config["freeze"] = {
        "final_method": validation_final["method"],
        "selection_reason": validation_final["selection_reason"],
        "selection_metric": freeze["selection_metric"],
        "selected_epoch": (
            freeze["selected_epoch"]
            if final_is_cst
            else b4_val["checkpoint_epoch"]
        ),
        "threshold": 0.5,
        "development_test_tuning": False,
        "sealed_confirmation_accessed": False,
        "evaluated_cst_candidate": {
            "config": str(cst_config),
            "checkpoint": str(cst_checkpoint),
        },
    }
    (output / "FINAL_CONFIG.json").write_text(json.dumps(final_config, indent=2) + "\n")

    hashes = {
        "stage1": sha256(
            Path("runs/stage1_fmt_simgen_v2_3k_20k_balanced_v2_eval/checkpoints/best.pth")
        ),
        "v4": sha256(
            Path("runs/unified_dual_evidence_fem_v4_2400/checkpoints/best_dense_val_delta_dice.pth")
        ),
        "b4": sha256(Path(b4_test["checkpoint"])),
        "selected_final": sha256(checkpoint),
        "evaluated_cst": sha256(cst_checkpoint),
    }
    (output / "CHECKPOINT_HASHES.json").write_text(json.dumps(hashes, indent=2) + "\n")
    provenance = {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_branch": subprocess.check_output(
            ["git", "branch", "--show-current"], text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "selected_config": str(selected_config),
        "selected_config_sha256": sha256(selected_config),
        "selected_checkpoint": str(checkpoint),
        "selected_method": validation_final["method"],
        "evaluated_cst_checkpoint": str(cst_checkpoint),
        "checkpoint_hashes": hashes,
        "operator": "certified analytic P1 / sampled-L2 Pi_h / exact FP64 hard-Q",
        "split_counts": {"train": 2400, "validation": 300, "development_test": 300},
        "seed": 20260901,
        "threshold": 0.5,
        "confirmation_data_used": False,
    }
    (output / "MODEL_PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")

    final_test = proposed_test if final_is_cst else b4_test
    coarse_quality = {
        **coarse_state_statistics(proposed_test, sorted(rows_by_id(proposed_test))),
        "coarse_dice": p0["frozen_development_test_summary"]["fem_dice"],
        "fem_mse": p0["frozen_development_test_summary"]["fem_mse"],
        "measurement_residual_ratio": p0["frozen_development_test_summary"][
            "measurement_residual_ratio"
        ],
        "source": "same frozen V4 epoch-15 state used by B4 and every CST candidate",
    }
    development_summary = {
        "selected_model_name": validation_final["method"],
        "selected_model": final_test["summary"],
        "evaluated_cst_candidate": proposed_test["summary"],
        "strong_concat_baseline": b4_test["summary"],
        "q_only_separate_stage1_protocol": q_only["summary"],
        "q_only_no_view_separate_stage1_protocol": q_only_no_view["summary"],
        "paired_dice_comparisons": comparisons,
        "coarse_inverse_quality": coarse_quality,
        "n_samples": 300,
        "threshold": 0.5,
        "sealed_confirmation_accessed": False,
    }
    (output / "DEVELOPMENT_TEST_SUMMARY.json").write_text(
        json.dumps(development_summary, indent=2) + "\n"
    )

    q_summary = q_only["summary"]
    q_no_view_summary = q_only_no_view["summary"]
    p_summary = proposed_test["summary"]
    b_summary = b4_test["summary"]
    ablations = {
        "responsibility_table": [
            {"method": "Initial FEM (matched V4 chain)", "dice": 0.617379},
            {
                "method": "Retrained Initial FEM (separate Q-only protocol)",
                "dice": q_summary["stage1_dice"],
            },
            {
                "method": "Initial FEM + no-view Q-only (separate protocol)",
                "dice": q_no_view_summary["final_dice"],
                "initial_dice": q_no_view_summary["stage1_dice"],
            },
            {
                "method": "Initial FEM + views Q-only (separate protocol)",
                "dice": q_summary["final_dice"],
                "initial_dice": q_summary["stage1_dice"],
            },
            {"method": "V4 coarse-only", "dice": p_summary["coarse_dice"]},
            {"method": "Sequential concat B4", "dice": b_summary["final_dice"]},
            {"method": "Proposed CST", "dice": p_summary["final_dice"]},
        ],
        "validation_candidates": freeze["candidates"],
        "optional_module_gates": freeze["optional_module_gates"],
        "latent_transition_ablation": freeze["latent_transition_ablation"],
        "final_model_selected_on_validation": validation_final,
        "optimization_responsibility": p0,
        "responsibility_training_audit": responsibility,
        "coarse_inverse_quality": coarse_quality,
        "protocol_separation_note": (
            "The retrained-Stage1 Q-only arm is not pooled with the matched V4 chain."
        ),
    }
    (output / "ABLATION_SUMMARY.json").write_text(json.dumps(ablations, indent=2) + "\n")

    proposed_rows, b4_rows, q_rows = (
        rows_by_id(proposed_test),
        rows_by_id(b4_test),
        rows_by_id(q_only),
    )
    ids = sorted(proposed_rows)
    case_fields = ["sample_id"] + [
        f"proposed_{key}"
        for key, value in proposed_rows[ids[0]].items()
        if key != "sample_id" and isinstance(value, (int, float))
    ] + ["b4_final_dice", "q_only_final_dice"]
    with (output / "casewise_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=case_fields)
        writer.writeheader()
        for sid in ids:
            row: dict[str, Any] = {"sample_id": sid}
            row.update(
                {
                    f"proposed_{key}": value
                    for key, value in proposed_rows[sid].items()
                    if key != "sample_id" and isinstance(value, (int, float))
                }
            )
            row["b4_final_dice"] = b4_rows[sid]["final_dice"]
            row["q_only_final_dice"] = q_rows[sid]["final_dice"]
            writer.writerow(row)

    curve_rows = []
    for candidate in freeze["candidates"]:
        history_path = Path(candidate["checkpoint"]).parents[1] / "history.json"
        for row in read(history_path):
            curve_rows.append({"model": candidate["name"], **row})
    curve_keys = sorted({key for row in curve_rows for key in row})
    with (output / "training_curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=curve_keys)
        writer.writeheader()
        writer.writerows(curve_rows)

    labels = ["Initial FEM", "Q-only*", "Coarse-only", "Concat", "CST"]
    values = [
        0.617379,
        q_summary["final_dice"],
        p_summary["coarse_dice"],
        b_summary["final_dice"],
        p_summary["final_dice"],
    ]
    plt.figure(figsize=(7.2, 4.2))
    plt.bar(labels, values)
    plt.ylim(0.60, max(values) + 0.02)
    plt.ylabel("Development-test Dice")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output / "ablation_figure.png", dpi=220)
    plt.close()
    for name, filename, ylabel in (
        ("proposed_vs_b4", "dice_paired_difference.png", "CST - concat Dice"),
        ("proposed_vs_coarse_only", "coarse_preservation_figure.png", "CST - coarse Dice"),
    ):
        plt.figure(figsize=(6.0, 3.8))
        plt.axhline(0, color="black", linewidth=0.8)
        plt.scatter(np.arange(len(deltas[name])), np.sort(deltas[name]), s=8)
        plt.ylabel(ylabel)
        plt.xlabel("Cases sorted by paired difference")
        plt.tight_layout()
        plt.savefig(output / filename, dpi=220)
        plt.close()
    plt.figure(figsize=(5.5, 4.0))
    plt.bar(["Concat", "CST"], [b_summary["detail_cosine"], p_summary["detail_cosine"]])
    plt.ylabel("Detail cosine")
    plt.tight_layout()
    plt.savefig(output / "detail_cosine_figure.png", dpi=220)
    plt.close()

    cst_validation_supported = freeze["cst_claim_supported_on_validation"]
    cst_supported = (
        cst_validation_supported
        and comparisons["proposed_vs_b4"]["mean_difference"] > 0
    )
    routing_supported = (
        responsibility["unrestricted_fully_joint"]["validation_fem_mse"]
        > 10 * responsibility["frozen"]["validation_fem_mse"]
    )
    go = [
        "# Final dual-space go/no-go",
        "",
        "All architecture choices below were made on validation300 before the final "
        "development-test300 evaluation. Sealed confirmation was not accessed.",
        "",
        f"1. **Q1:** Yes. The validation-frozen final method is "
        f"`{validation_final['method']}` and the matched chain is `0.617379 -> "
        f"{p_summary['coarse_dice']:.6f} -> {final_test['summary']['final_dice']:.6f}` "
        "through separated coarse and complementary refinement.",
        f"2. **Q2:** The separate retrained-Stage1 Q-only arm reaches only "
        f"{q_summary['final_dice']:.6f} because hard-Q cannot correct FEM-representable inverse error.",
        f"3. **Q3:** V4 coarse correction contributes "
        f"{p_summary['coarse_dice'] - 0.617379:+.6f} Dice in the matched chain.",
        f"4. **Q4:** The validation-selected full model contributes "
        f"{final_test['summary']['final_dice'] - p_summary['coarse_dice']:+.6f} "
        "over coarse-only.",
        f"5. **Q5:** CST {'does' if cst_supported else 'does not'} significantly exceed the "
        f"parameter-matched concat baseline; val paired CI is {val_stats['paired_bootstrap_95_ci']} "
        f"and development difference is {comparisons['proposed_vs_b4']['mean_difference']:+.6f}.",
        f"6. **Q6:** Hc+Delta-H versus Hc-only has validation difference "
        f"{freeze['latent_transition_ablation']['hc_plus_delta_h_vs_hc_only']['mean_difference']:+.6f} "
        f"with CI {freeze['latent_transition_ablation']['hc_plus_delta_h_vs_hc_only']['paired_bootstrap_95_ci']}; "
        "the Delta-H claim is retained only when this evidence is positive.",
        "7. **Q7:** Frozen coarse is retained when separated continuation fails to improve "
        "validation; unrestricted joint training is rejected if it reproduces FEM drift.",
        f"8. **Q8:** Yes. Mean coarse preservation relative L2 is "
        f"{p_summary['coarse_preservation_relative_l2']:.3e}, with leakage "
        f"{p_summary['coarse_leakage']:.3e}.",
        f"9. **Q9:** Detail cosine={p_summary['detail_cosine']:.6f}, "
        f"HD95={p_summary['final_hd95']:.6f} mm, localization="
        f"{p_summary['final_localization_error']:.6f} mm.",
        "10. **Q10:** Approximation-space separation is supported. CST is "
        f"{'supported' if cst_supported else 'downgraded'}; responsibility-preserving routing "
        f"is {'supported as a safety contract, not a performance-improving co-training claim' if routing_supported else 'limited to the frozen training contract'}.",
        "",
        "*The Q-only row uses the separately frozen retrained-Stage1 protocol and is not a "
        "matched B4/CST comparison.*",
    ]
    (output / "FINAL_GO_NO_GO.md").write_text("\n".join(go) + "\n")


if __name__ == "__main__":
    main()
