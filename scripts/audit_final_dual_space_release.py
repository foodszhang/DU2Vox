#!/usr/bin/env python3
"""Fail closed unless the final dual-space development release is complete."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir
    required = [
        "METHOD_SPEC.md",
        "FINAL_CONFIG.json",
        "VALIDATION_FREEZE.json",
        "DEVELOPMENT_TEST_SUMMARY.json",
        "ABLATION_SUMMARY.json",
        "PAIRED_BOOTSTRAP.json",
        "MODEL_PROVENANCE.json",
        "CHECKPOINT_HASHES.json",
        "FINAL_GO_NO_GO.md",
        "casewise_metrics.csv",
        "training_curves.csv",
        "validation_model_selection.csv",
        "ablation_figure.png",
        "dice_paired_difference.png",
        "detail_cosine_figure.png",
        "coarse_preservation_figure.png",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    require(not missing, f"Missing final artifacts: {missing}")

    freeze = read_json(root / "VALIDATION_FREEZE.json")
    require(freeze.get("frozen_on_validation") is True, "Validation freeze is absent")
    require(
        freeze.get("development_test_accessed_before_freeze") is False,
        "Development-test governance violation",
    )
    require(freeze.get("sealed_confirmation_accessed") is False, "Sealed data accessed")
    require(len(freeze.get("candidates", [])) == 8, "Expected eight formal candidates")
    require(
        freeze["final_model_selected_on_validation"]["method"]
        in {"CST", "strong_sequential_concat_B4"},
        "Unknown validation-selected final method",
    )

    cst_test = read_json(root / "development_test/cst_selected.json")
    require(cst_test.get("n_samples") == 300, "CST evaluation is not test300")
    require(cst_test.get("split") == "development_test", "Wrong evaluation split")
    require(cst_test.get("hard_q") is True, "CST result is not exact hard-Q")
    require(cst_test.get("confirmation_data_used") is False, "Sealed data accessed")
    require(
        cst_test["summary"]["coarse_leakage"] <= 1e-15,
        "Complementary output leaks into the FEM space",
    )
    require(
        cst_test["summary"]["coarse_preservation_relative_l2"] <= 1e-7,
        "Final output does not preserve the frozen coarse state",
    )

    development = read_json(root / "DEVELOPMENT_TEST_SUMMARY.json")
    require(development.get("n_samples") == 300, "Summary is not development-test300")
    require(development.get("sealed_confirmation_accessed") is False, "Sealed data accessed")
    paired = read_json(root / "PAIRED_BOOTSTRAP.json")
    require(
        paired.get("confirmation_data_used") is False,
        "Paired statistics used confirmation data",
    )
    comparisons = paired.get("development_test", {})
    require(len(comparisons) == 3, "Expected three development-test comparisons")
    for name, result in comparisons.items():
        require(result.get("n") == 300, f"{name} is not paired over 300 cases")
        require(result.get("bootstrap_draws") == 10000, f"{name} is not 10k bootstrap")
        require("holm_adjusted_p_value" in result, f"{name} lacks Holm correction")

    provenance = read_json(root / "MODEL_PROVENANCE.json")
    require(
        provenance.get("split_counts")
        == {"train": 2400, "validation": 300, "development_test": 300},
        "Split contract mismatch",
    )
    require(provenance.get("threshold") == 0.5, "Threshold contract mismatch")
    require(provenance.get("confirmation_data_used") is False, "Sealed data accessed")
    selected_checkpoint = Path(provenance["selected_checkpoint"])
    require(selected_checkpoint.is_file(), "Selected checkpoint is missing")
    hashes = read_json(root / "CHECKPOINT_HASHES.json")
    require(
        hashes["selected_final"] == sha256(selected_checkpoint),
        "Selected checkpoint hash mismatch",
    )

    with (root / "casewise_metrics.csv").open(newline="") as stream:
        case_rows = list(csv.DictReader(stream))
    require(len(case_rows) == 300, "casewise_metrics.csv is not 300 cases")
    require(
        len({row["sample_id"] for row in case_rows}) == 300,
        "casewise_metrics.csv contains duplicate cases",
    )
    with (root / "validation_model_selection.csv").open(newline="") as stream:
        selection_rows = list(csv.DictReader(stream))
    require(len(selection_rows) == 8, "Validation model-selection table is incomplete")
    with (root / "training_curves.csv").open(newline="") as stream:
        curve_rows = list(csv.DictReader(stream))
    curves: dict[str, set[int]] = {}
    for row in curve_rows:
        curves.setdefault(row["model"], set()).add(int(row["epoch"]))
    require(len(curves) == 8, "Training curves omit formal candidates")
    require(
        all(epochs == {1, 2, 3, 4, 5} for epochs in curves.values()),
        "At least one formal candidate lacks five complete epochs",
    )

    report = {
        "complete": True,
        "required_artifacts": required,
        "formal_candidates": 8,
        "train_validation_development_test": [2400, 300, 300],
        "paired_bootstrap_draws": 10000,
        "hard_q": True,
        "coarse_leakage": cst_test["summary"]["coarse_leakage"],
        "coarse_preservation_relative_l2": cst_test["summary"][
            "coarse_preservation_relative_l2"
        ],
        "selected_method": provenance["selected_method"],
        "sealed_confirmation_accessed": False,
    }
    (root / "COMPLETION_AUDIT.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
