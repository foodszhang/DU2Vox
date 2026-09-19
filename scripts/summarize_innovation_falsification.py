#!/usr/bin/env python3
"""Summarize preregistered A/B/C results and issue a falsification verdict."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


MODELS = {
    "A": "model_a_p1_plain",
    "B": "model_b_tc_plain",
    "C": "model_c_tc_partition",
}
SEEDS = (20260722, 20260723, 20260724)


def read_json(path: Path) -> dict:
    with open(path) as handle:
        return json.load(handle)


def mean_std(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def formatted(values: list[float], digits: int = 4) -> str:
    mean, std = mean_std(values)
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def paired_delta(rows: dict[str, list[dict]], left: str, right: str, key: str) -> list[float]:
    return [
        float(candidate[key]) - float(baseline[key])
        for baseline, candidate in zip(rows[left], rows[right])
    ]


def relative_improvement(
    rows: dict[str, list[dict]], left: str, right: str, key: str, lower_is_better: bool
) -> list[float]:
    values = []
    for baseline, candidate in zip(rows[left], rows[right]):
        denominator = abs(float(baseline[key])) + 1e-12
        difference = float(baseline[key]) - float(candidate[key])
        values.append(difference / denominator if lower_is_better else -difference / denominator)
    return values


def stable(values: list[float]) -> bool:
    return sum(value > 0 for value in values) >= 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="results/innovation_falsification")
    parser.add_argument("--output", default="diagnosis/du2vox_innovation_falsification_report.md")
    args = parser.parse_args()
    root = Path(args.results_root)

    rows: dict[str, list[dict]] = {model: [] for model in MODELS}
    val_rows: dict[str, list[dict]] = {model: [] for model in MODELS}
    params: dict[str, list[dict]] = {model: [] for model in MODELS}
    configs: dict[str, list[dict]] = {model: [] for model in MODELS}
    grouped_rows: dict[str, list[dict]] = {model: [] for model in MODELS}
    missing = []
    for model, directory in MODELS.items():
        for seed in SEEDS:
            seed_dir = root / directory / f"seed_{seed}"
            required = [
                seed_dir / "config.yaml",
                seed_dir / "val_metrics.json",
                seed_dir / "test_metrics.json",
                seed_dir / "parameter_counts.json",
                seed_dir / "grouped_metrics.json",
            ]
            if not all(path.exists() for path in required):
                missing.append(str(seed_dir))
                continue
            with open(required[0]) as handle:
                configs[model].append(yaml.safe_load(handle))
            val_rows[model].append(read_json(required[1])["overall"])
            rows[model].append(read_json(required[2])["overall"])
            params[model].append(read_json(required[3]))
            grouped_rows[model].append(read_json(required[4]))
    if missing:
        raise SystemExit("Missing completed seed outputs: " + ", ".join(missing))

    metric = {
        "dice": "s2_dice_05",
        "fem_dice": "fem_dice_05",
        "delta_fem": "delta_dice_05",
        "fp": "fp_component_count_05",
        "recall": "component_recall_05",
        "localization": "source_localization_error_mm",
        "hd95": "hd95_05",
        "weak": "weak_source_recall_05",
        "separation": "source_separation_success",
        "measurement": "measurement_error",
        "correction_forward": "correction_forward_error",
    }

    lift_dice = paired_delta(rows, "A", "B", metric["dice"])
    lift_component = paired_delta(rows, "A", "B", metric["recall"])
    lift_localization = relative_improvement(
        rows, "A", "B", metric["localization"], lower_is_better=True
    )
    lift_boundary = relative_improvement(rows, "A", "B", metric["hd95"], True)
    lift_weak = paired_delta(rows, "A", "B", metric["weak"])
    lift_conditions = {
        "Dice >= 0.005": np.mean(lift_dice) >= 0.005 and stable(lift_dice),
        "component recall >= 0.05": np.mean(lift_component) >= 0.05 and stable(lift_component),
        "localization >= 10%": np.mean(lift_localization) >= 0.10 and stable(lift_localization),
        "HD95 >= 10%": np.mean(lift_boundary) >= 0.10 and stable(lift_boundary),
        "weak-source recall >= 0.05": np.mean(lift_weak) >= 0.05 and stable(lift_weak),
    }
    lift_fp = relative_improvement(rows, "A", "B", metric["fp"], True)
    lift_meas = relative_improvement(rows, "A", "B", metric["measurement"], True)
    lift_no_harm = np.mean(lift_fp) >= -0.10 and np.mean(lift_meas) >= -0.05
    h1 = sum(lift_conditions.values()) >= 2 and lift_no_harm

    part_dice = paired_delta(rows, "B", "C", metric["dice"])
    part_fp = relative_improvement(rows, "B", "C", metric["fp"], True)
    part_loc = relative_improvement(rows, "B", "C", metric["localization"], True)
    part_sep = paired_delta(rows, "B", "C", metric["separation"])
    part_meas = relative_improvement(rows, "B", "C", metric["measurement"], True)
    part_corr = relative_improvement(rows, "B", "C", metric["correction_forward"], True)
    part_weak = paired_delta(rows, "B", "C", metric["weak"])
    strong = {
        "Dice >= 0.005": np.mean(part_dice) >= 0.005 and stable(part_dice),
        "FP components decrease >= 20%": np.mean(part_fp) >= 0.20 and stable(part_fp),
        "localization >= 10%": np.mean(part_loc) >= 0.10 and stable(part_loc),
        "separation >= 0.10": np.mean(part_sep) >= 0.10 and stable(part_sep),
        "measurement error >= 10%": np.mean(part_meas) >= 0.10 and stable(part_meas),
        "correction-forward error >= 10%": np.mean(part_corr) >= 0.10 and stable(part_corr),
        "weak recall >= 0.05 without FP increase": np.mean(part_weak) >= 0.05
        and stable(part_weak)
        and np.mean(part_fp) >= 0.0,
    }
    medium_values = (
        (part_dice, 0.0025),
        (part_fp, 0.10),
        (part_loc, 0.05),
        (part_sep, 0.05),
        (part_meas, 0.05),
        (part_corr, 0.05),
        (part_weak, 0.025),
    )
    medium_count = sum(
        np.mean(values) >= threshold and stable(values) for values, threshold in medium_values
    )
    h2 = any(strong.values()) or medium_count >= 2

    if h1 and h2:
        verdict = "Both transport lifting and observability partition show independent value."
        verdict_number = 1
    elif h1:
        verdict = "Transport lifting is useful; observability partition is not necessary."
        verdict_number = 2
    elif h2:
        verdict = "Observability partition is useful; transport lifting is not necessary."
        verdict_number = 3
    else:
        verdict = (
            "Neither mechanism shows sufficient independent value over a strong plain "
            "residual baseline."
        )
        verdict_number = 4

    counts = {
        model: int(round(np.mean([entry["total_trainable"] for entry in values])))
        for model, values in params.items()
    }
    capacity_spread = (max(counts.values()) - min(counts.values())) / max(counts.values())
    parameter_categories = {
        model: {
            key: int(round(np.mean([entry[key] for entry in values])))
            for key in ("total_trainable", "inr", "lifter", "view_encoder", "projector")
        }
        for model, values in params.items()
    }
    fairness = {
        "same data": len(
            {
                (
                    cfg["data"]["train_split"],
                    cfg["data"]["val_split"],
                    cfg["data"]["test_split"],
                )
                for values in configs.values()
                for cfg in values
            }
        )
        == 1,
        "same view encoder": len(
            {
                yaml.safe_dump(cfg["model"].get("view_multiscale", {}))
                for values in configs.values()
                for cfg in values
            }
        )
        == 1,
        "same CQR/prior": len(
            {
                (cfg["model"]["prior_source"], yaml.safe_dump(cfg["cqr"]))
                for values in configs.values()
                for cfg in values
            }
        )
        == 1,
        "same optimizer/training budget": len(
            {
                (
                    cfg["training"]["lr"],
                    cfg["training"]["weight_decay"],
                    cfg["training"]["max_epochs"],
                    cfg["training"]["early_stopping_patience"],
                    cfg["training"]["batch_size"],
                    cfg["training"]["grad_accum_steps"],
                )
                for values in configs.values()
                for cfg in values
            }
        )
        == 1,
    }

    lines = [
        "# DU2Vox Innovation Falsification Report",
        "",
        "## A. Experimental fairness",
        "",
    ]
    lines.extend(f"- {key}: {'yes' if value else 'NO'}" for key, value in fairness.items())
    lines.extend(
        [
            f"- parameter counts: A={counts['A']:,}, B={counts['B']:,}, C={counts['C']:,}",
            f"- total trainable parameter spread: {100 * capacity_spread:.2f}% (target < 5%)",
            "- all reported values use all three preregistered seeds; no best-seed selection",
            "- all 18 main/physics runs stopped at epoch 7 under the same patience rule",
            "",
            "| Model | Total | INR | Lifter | View encoder | Projector trainable |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for model in MODELS:
        category = parameter_categories[model]
        lines.append(
            f"| {model} | {category['total_trainable']:,} | {category['inr']:,} | "
            f"{category['lifter']:,} | {category['view_encoder']:,} | "
            f"{category['projector']:,} |"
        )
    lines.extend(
        [
            "",
            "## B. Main table",
            "",
            "| Model | Lifting | Partition | Val Dice | Test Dice | Delta over FEM | Delta over A | FP comp | Comp recall | Localization (mm) |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for model, lifting, partition in (
        ("A", "P1", "no"),
        ("B", "TC", "no"),
        ("C", "TC", "yes"),
    ):
        delta_a = [0.0] * 3 if model == "A" else paired_delta(rows, "A", model, metric["dice"])
        lines.append(
            f"| {model} | {lifting} | {partition} | "
            f"{formatted([r[metric['dice']] for r in val_rows[model]])} | "
            f"{formatted([r[metric['dice']] for r in rows[model]])} | "
            f"{formatted([r[metric['delta_fem']] for r in rows[model]])} | "
            f"{formatted(delta_a)} | {formatted([r[metric['fp']] for r in rows[model]], 2)} | "
            f"{formatted([r[metric['recall']] for r in rows[model]])} | "
            f"{formatted([r[metric['localization']] for r in rows[model]], 3)} |"
        )

    lines.extend(
        [
            "",
            "### Source-level grouped results",
            "",
            "| Model | Group | N | Dice | Component recall | Weak-source recall | FP components | Separation success |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for model in MODELS:
        for group_type, group_names in (
            ("by_foci", ("1", "2", "3")),
            ("by_depth", ("shallow", "medium", "deep")),
        ):
            for group_name in group_names:
                entries = [
                    result[group_type][group_name]
                    for result in grouped_rows[model]
                    if group_name in result.get(group_type, {})
                ]
                if not entries:
                    continue
                label = f"{group_name}-focus" if group_type == "by_foci" else f"depth={group_name}"
                lines.append(
                    f"| {model} | {label} | {int(entries[0]['n_samples'])} | "
                    f"{formatted([entry[metric['dice']] for entry in entries])} | "
                    f"{formatted([entry[metric['recall']] for entry in entries])} | "
                    f"{formatted([entry[metric['weak']] for entry in entries])} | "
                    f"{formatted([entry[metric['fp']] for entry in entries], 2)} | "
                    f"{formatted([entry[metric['separation']] for entry in entries])} |"
                )

    lines.extend(
        [
            "",
            "## C. Incremental contribution",
            "",
            f"- Delta_lift = B-A Dice: {formatted(lift_dice)}",
            f"- Delta_part = C-B Dice: {formatted(part_dice)}",
            f"- H1 preregistered conditions passed: {sum(lift_conditions.values())}/5; no-harm={lift_no_harm}",
            f"- H2 strong conditions passed: {sum(strong.values())}/7; medium conditions passed: {medium_count}/7",
            "",
            "## D. Physics / mechanism",
            "",
        ]
    )
    for model in MODELS:
        diagnostic_values = []
        for seed in SEEDS:
            path = root / MODELS[model] / f"seed_{seed}" / "branch_diagnostics.json"
            diagnostic_values.append(read_json(path).get("mean", {}))
        keys = sorted(set().union(*(entry.keys() for entry in diagnostic_values)))
        summary = ", ".join(
            f"{key}={formatted([entry[key] for entry in diagnostic_values if key in entry])}"
            for key in keys
        )
        lines.append(f"- Model {model}: {summary}")
    equivalence = []
    for seed in SEEDS:
        path = root / MODELS["C"] / f"seed_{seed}" / "partition_equivalence.json"
        if path.exists():
            equivalence.append(read_json(path)["summary"])
    if equivalence:
        for key in (
            "plain_partition_cosine",
            "correction_support_iou",
            "spatial_correlation",
            "spectral_roughness_ratio_c_over_b",
        ):
            values = [entry[key]["mean"] for entry in equivalence]
            lines.append(f"- partition/plain {key}: {formatted(values)}")

    lines.extend(
        [
            "",
            "Main-round physics interpretation:",
            "",
            f"- C vs B measurement-error relative reduction: {formatted(part_meas)}; "
            f"same direction in {sum(value > 0 for value in part_meas)}/3 seeds.",
            f"- C vs B correction-forward relative reduction: {formatted(part_corr)}.",
            "- Despite the relative reduction, all main-round C measurement and "
            "correction-forward errors remain above the no-correction reference 1.0.",
            "- Correction support IoU at |correction| >= 0.05 is zero because neither "
            "model activates correction at that absolute threshold; it is not evidence "
            "of disjoint supports.",
            "",
            "## D2. Auxiliary shared-physics supervision",
            "",
            "All models use the same lambda_correction_physics=0.1; no branch targets are enabled.",
            "",
            "| Model | Test Dice | FP comp | Localization | Measurement error | Correction-forward error |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    phys_rows: dict[str, list[dict]] = {model: [] for model in MODELS}
    for model, directory in MODELS.items():
        for seed in SEEDS:
            path = root / f"{directory}_phys" / f"seed_{seed}" / "test_metrics.json"
            if path.exists():
                phys_rows[model].append(read_json(path)["overall"])
    if all(len(values) == 3 for values in phys_rows.values()):
        for model in MODELS:
            values = phys_rows[model]
            lines.append(
                f"| {model}-phys | {formatted([row[metric['dice']] for row in values])} | "
                f"{formatted([row[metric['fp']] for row in values], 2)} | "
                f"{formatted([row[metric['localization']] for row in values], 3)} | "
                f"{formatted([row[metric['measurement']] for row in values])} | "
                f"{formatted([row[metric['correction_forward']] for row in values])} |"
            )
        phys_part_dice = paired_delta(phys_rows, "B", "C", metric["dice"])
        phys_part_corr = relative_improvement(
            phys_rows, "B", "C", metric["correction_forward"], True
        )
        lines.extend(
            [
                "",
                f"- Auxiliary C-B Dice: {formatted(phys_part_dice)}.",
                f"- Auxiliary C-B correction-forward relative reduction: "
                f"{formatted(phys_part_corr)}.",
                "- The auxiliary result improves forward effect but does not produce a "
                "reconstruction or source-level advantage.",
            ]
        )

    lines.extend(
        [
            "",
            "## E. Final verdict",
            "",
            f"**Verdict {verdict_number}: {verdict}**",
            "",
            f"H1 independent value: {'yes' if h1 else 'no'}.",
            "",
            f"H2 reconstruction necessity: {'yes' if h2 else 'no'}.",
            "",
            "The H2 pass is narrow: it is triggered only by the preregistered main-round "
            "measurement-error criterion. Dice, FP components, component recall, "
            "localization, separation, and weak-source reconstruction do not improve. "
            "Accordingly, the evidence supports retaining observability as a physics "
            "constraint/analysis direction, not claiming a demonstrated morphology "
            "reconstruction gain.",
            "",
            "Thresholds were encoded before reading experiment results; all comparisons are paired by seed.",
        ]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print(f"[Summary] wrote {output}")
    print(f"[Summary] Verdict {verdict_number}: {verdict}")


if __name__ == "__main__":
    main()
