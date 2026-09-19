#!/usr/bin/env python3
"""Summarize how the train2400 Stage-1 retrain changes frozen conclusions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.bridge.canonical_cross_discretization import CanonicalCrossDiscretization
from experiments.cross_discretization_decomposition.decomposition import binary_metrics


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def case_map(path: Path, key: str) -> dict[str, float]:
    return {row["sample_id"]: float(row[key]) for row in load(path)["per_sample"]}


def paired_bootstrap(
    first: dict[str, float],
    second: dict[str, float],
    *,
    seed: int,
    draws: int,
) -> dict[str, float | int]:
    ids = sorted(set(first) & set(second))
    if set(ids) != set(first) or set(ids) != set(second):
        raise RuntimeError("Paired artifacts do not contain identical sample IDs")
    delta = np.asarray([first[sid] - second[sid] for sid in ids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = delta[rng.integers(0, len(delta), size=(draws, len(delta)))]
    means = sampled.mean(axis=1)
    return {
        "n": len(delta),
        "mean_delta": float(delta.mean()),
        "ci95_lower": float(np.quantile(means, 0.025)),
        "ci95_upper": float(np.quantile(means, 0.975)),
        "fraction_improved": float(np.mean(delta > 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    diagnosis = root / "diagnosis"
    old_v4 = {
        "val": diagnosis / "unified_dual_evidence_fem_v4_val300_voxel_oracle_input.json",
        "test": diagnosis / "unified_dual_evidence_fem_v4_test300.json",
    }
    new_stage1 = {
        "val": diagnosis / "stage1_retrain_2400_val300_canonical.json",
        "test": diagnosis / "stage1_retrain_2400_development_test300_canonical.json",
    }
    new_v4 = {
        "val": diagnosis / "unified_dual_evidence_fem_v4_on_retrained_stage1_val300.json",
        "test": diagnosis
        / "unified_dual_evidence_fem_v4_on_retrained_stage1_development_test300.json",
    }
    old_final = {
        "val": diagnosis / "complement_voxel_detail_constrained_val300.json",
        "test": diagnosis
        / "complement_voxel_detail_constrained_development_test300.json",
    }
    new_final = {
        "val": diagnosis / "complement_voxel_detail_on_retrained_stage1_val300.json",
        "test": diagnosis
        / "complement_voxel_detail_on_retrained_stage1_development_test300.json",
    }

    comparisons: dict[str, dict[str, dict[str, float | int]]] = {}
    summaries: dict[str, dict[str, float]] = {}
    for split in ("val", "test"):
        old_stage1_cases = case_map(old_v4[split], "stage1_dice")
        old_v4_cases = case_map(old_v4[split], "step3_dice")
        new_stage1_cases = case_map(new_stage1[split], "dice")
        new_v4_cases = case_map(new_v4[split], "step3_dice")
        old_final_cases = case_map(old_final[split], "final_dice")
        new_final_cases = case_map(new_final[split], "final_dice")
        series = {
            "old_stage1": old_stage1_cases,
            "new_stage1": new_stage1_cases,
            "old_v4": old_v4_cases,
            "new_v4": new_v4_cases,
            "old_final": old_final_cases,
            "new_final": new_final_cases,
        }
        summaries[split] = {
            name: float(np.mean(list(values.values()))) for name, values in series.items()
        }
        pairs = {
            "new_stage1_minus_old_stage1": (new_stage1_cases, old_stage1_cases),
            "new_v4_minus_old_v4": (new_v4_cases, old_v4_cases),
            "new_final_minus_old_final": (new_final_cases, old_final_cases),
            "new_v4_minus_new_stage1": (new_v4_cases, new_stage1_cases),
            "new_final_minus_new_v4": (new_final_cases, new_v4_cases),
        }
        comparisons[split] = {
            name: paired_bootstrap(
                first,
                second,
                seed=args.seed + index,
                draws=args.bootstrap_draws,
            )
            for index, (name, (first, second)) in enumerate(pairs.items())
        }

    dataset = Path("/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k")
    samples_dir = dataset / "samples"
    projection_dir = root / "precomputed/error_structured_projection_targets"
    bridge_dirs = {
        "train": root / "output/bridge_stage1_retrain_2400_train",
        "val": root / "output/bridge_stage1_retrain_2400_val",
        "test": root / "output/bridge_stage1_retrain_2400_test",
    }
    canonical = CanonicalCrossDiscretization(
        root / "experiments/cross_discretization_decomposition/artifacts/operator_cache",
        shared_dir="/home/foods/pro/FMT-SimGen/output/shared_mesh_20k",
        factorize=False,
    )
    energy_rows: list[dict[str, float | str]] = []
    oracle_by_split: dict[str, list[float]] = {"train": [], "val": [], "test": []}
    for split in ("train", "val", "test"):
        ids = [
            line.strip()
            for line in (dataset / "splits" / f"{split}.txt").read_text().splitlines()
            if line.strip()
        ]
        for index, sid in enumerate(ids, start=1):
            x_h = np.load(bridge_dirs[split] / sid / "coarse_d.npy").astype(np.float64)
            pi_gt = np.load(projection_dir / f"{sid}.npy").astype(np.float64)
            gt_volume = np.load(samples_dir / sid / "gt_voxels.npy", mmap_mode="r")
            gt = (
                np.asarray(gt_volume).ravel()[canonical.operator.valid_flat_indices]
                > 0.05
            ).astype(np.float64)
            coarse = canonical.prolong(x_h)
            projected_gt = canonical.prolong(pi_gt)
            inverse = projected_gt - coarse
            detail = gt - projected_gt
            total = gt - coarse
            total_energy = canonical.weighted_energy(total)
            inverse_energy = canonical.weighted_energy(inverse)
            detail_energy = canonical.weighted_energy(detail)
            energy_rows.append(
                {
                    "sample_id": sid,
                    "split": split,
                    "ratio_inverse": inverse_energy / total_energy,
                    "ratio_representation": detail_energy / total_energy,
                }
            )
            oracle = coarse + detail
            oracle_by_split[split].append(binary_metrics(oracle, gt)["dice"])
            if index % 300 == 0:
                print(f"[decomposition {split} {index}/{len(ids)}]", flush=True)

    energy_summary = {
        "n": len(energy_rows),
        "mean_inverse_fraction": float(
            np.mean([float(row["ratio_inverse"]) for row in energy_rows])
        ),
        "mean_representation_fraction": float(
            np.mean([float(row["ratio_representation"]) for row in energy_rows])
        ),
    }
    oracle_summary = {
        split: float(np.mean(values)) for split, values in oracle_by_split.items()
    }

    test_total_gain = summaries["test"]["new_final"] - summaries["test"]["new_stage1"]
    allocation = {
        "v4_fraction_of_new_pipeline_gain": (
            summaries["test"]["new_v4"] - summaries["test"]["new_stage1"]
        )
        / test_total_gain,
        "hard_q_fraction_of_new_pipeline_gain": (
            summaries["test"]["new_final"] - summaries["test"]["new_v4"]
        )
        / test_total_gain,
    }
    result = {
        "bootstrap_draws": args.bootstrap_draws,
        "seed": args.seed,
        "summaries": summaries,
        "paired_comparisons": comparisons,
        "new_stage1_error_energy": energy_summary,
        "new_stage1_plus_oracle_q_detail_dice": oracle_summary,
        "development_test_gain_allocation": allocation,
        "sealed_confirmation_accessed": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")

    def ci(split: str, name: str) -> str:
        row = comparisons[split][name]
        return (
            f"{row['mean_delta']:+.6f} "
            f"[{row['ci95_lower']:+.6f}, {row['ci95_upper']:+.6f}]"
        )

    lines = [
        "# Stage-1 Train2400 Retraining Audit",
        "",
        "The checkpoint was selected on val300 before development-test evaluation. "
        "Sealed confirmation was not accessed.",
        "",
        "## Canonical Dice",
        "",
        "| Pipeline state | Val300 | Development-test300 |",
        "| --- | ---: | ---: |",
    ]
    labels = {
        "old_stage1": "Historical Stage 1",
        "new_stage1": "Retrained Stage 1",
        "old_v4": "Historical Stage 1 + frozen V4",
        "new_v4": "Retrained Stage 1 + frozen V4",
        "old_final": "Historical final hard-Q candidate",
        "new_final": "Retrained Stage 1 + frozen V4 + frozen hard-Q",
    }
    for key, label in labels.items():
        lines.append(
            f"| {label} | {summaries['val'][key]:.6f} | "
            f"{summaries['test'][key]:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Paired Dice differences (10,000-case bootstrap 95% CI)",
            "",
            "| Contrast | Val300 | Development-test300 |",
            "| --- | ---: | ---: |",
        ]
    )
    for key in comparisons["val"]:
        lines.append(f"| {key} | {ci('val', key)} | {ci('test', key)} |")
    lines.extend(
        [
            "",
            "## Recomputed decomposition and oracle",
            "",
            f"- Mean inverse-discrepancy energy fraction: "
            f"{energy_summary['mean_inverse_fraction']:.6f}.",
            f"- Mean representation energy fraction: "
            f"{energy_summary['mean_representation_fraction']:.6f}.",
            f"- Retrained Stage 1 + oracle hard-Q detail Dice: val "
            f"{oracle_summary['val']:.6f}, development-test "
            f"{oracle_summary['test']:.6f}.",
            f"- In the observed new development pipeline, V4 accounts for "
            f"{allocation['v4_fraction_of_new_pipeline_gain']:.1%} of the gain and "
            f"hard-Q detail for {allocation['hard_q_fraction_of_new_pipeline_gain']:.1%}.",
            "",
            "## Decision audit",
            "",
            "- Confirmed: the historical Stage 1 was undertrained for the current cohort; "
            "exact-architecture train2400 retraining improves its canonical Dice.",
            "- Not overturned: a learned hard-Q-only model has not demonstrated 0.73 without "
            "FEM correction. The oracle feasibility statement is updated separately above.",
            "- Not overturned: frozen V4 still supplies the dominant observed gain, although "
            "its role is scientifically better described as continuation of Stage 1 FEM inversion.",
            "- Not overturned: hard-Q coarse preservation and zero-leakage contracts are "
            "independent of the upstream Stage-1 checkpoint.",
            "- Not superseded: the retrained upstream gives no robust final-pipeline advantage "
            "over the historical freeze candidate.",
            "- Scoped: A0-A3 remain valid for the historical frozen-V4 backbone. Their exact "
            "numeric gates are not automatically transferable to a newly retrained backbone.",
        ]
    )
    args.output_md.write_text("\n".join(lines) + "\n")
    print(f"[audit] wrote {args.output_json}")
    print(f"[audit] wrote {args.output_md}")


if __name__ == "__main__":
    main()
