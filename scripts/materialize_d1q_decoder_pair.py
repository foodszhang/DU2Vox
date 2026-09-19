#!/usr/bin/env python3
"""Materialize a matched hard-Q/unconstrained decoder config pair.

This is a configuration-only gate. It does not train or evaluate a model.

The gate is cohort-agnostic: experiment names, output config names and the
receipt name come from ``--name-prefix``, and the expected projection-target
contract is derived from the selected V4 config instead of being hardcoded.
That matters because ``gt_mode``/``normalize_gt`` differ between cohorts (for
example ``per_sample_peak`` + ``gt_scale.npy`` for the d1q canonical cohort
versus ``normalize_gt: none`` for a raw-amplitude cohort).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.utils.confirmation import sha256_file


ALLOWED_PAIR_DIFFERENCES = {"experiment.name", "model.mode"}


def flattened(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    output: dict[str, Any] = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        output.update(flattened(child, path))
    return output


def assert_matched_pair(hard: dict[str, Any], unconstrained: dict[str, Any]) -> None:
    hard_flat = flattened(hard)
    free_flat = flattened(unconstrained)
    differing = {
        key
        for key in hard_flat.keys() | free_flat.keys()
        if hard_flat.get(key) != free_flat.get(key)
    }
    unexpected = differing - ALLOWED_PAIR_DIFFERENCES
    if unexpected:
        raise RuntimeError(f"Decoder pair has unmatched fields: {sorted(unexpected)}")
    if hard.get("model", {}).get("mode") != "hard":
        raise RuntimeError("Hard-Q arm must declare model.mode=hard")
    if unconstrained.get("model", {}).get("mode") != "unconstrained":
        raise RuntimeError("Control arm must declare model.mode=unconstrained")


def config_hash(value: dict[str, Any]) -> str:
    payload = yaml.safe_dump(value, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-config", type=Path, required=True)
    parser.add_argument("--v4-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-epoch", type=int, required=True)
    parser.add_argument("--v4-states-root", type=Path, required=True)
    parser.add_argument("--terminal-hidden-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--name-prefix",
        default="d1q_mcx_canonical_decoder",
        help=(
            "Prefix for the two experiment names, the two output config filenames "
            "and the contract receipt filename"
        ),
    )
    args = parser.parse_args()

    v4 = yaml.safe_load(args.v4_config.read_text())
    checkpoint = torch.load(args.v4_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != "unified_dual_evidence_fem_v4":
        raise RuntimeError("Selected checkpoint is not a unified V4 corrector")
    if int(checkpoint.get("epoch", -1)) != args.expected_epoch:
        raise RuntimeError("Selected V4 checkpoint epoch does not match the freeze")
    if checkpoint.get("config") != v4:
        raise RuntimeError("Selected V4 checkpoint embeds a different training config")

    target_metadata_path = Path(v4["data"]["projection_targets_dir"]) / "metadata.json"
    if not target_metadata_path.is_file():
        raise FileNotFoundError(
            f"Validated projection-target metadata is missing: {target_metadata_path}"
        )
    target_metadata = json.loads(target_metadata_path.read_text())
    # Derive the expected contract from the selected V4 config. A hardcoded
    # per_sample_peak/gt_scale.npy expectation would reject any raw-amplitude
    # cohort whose loader legitimately uses normalize_gt: none.
    v4_data = v4["data"]
    expected_target_contract = {
        "gt_mode": v4_data.get("gt_mode", "binary"),
        "normalize_gt": v4_data.get("normalize_gt", "none"),
        "normalization_scale_filename": v4_data.get("normalization_scale_filename"),
    }
    for key, expected in expected_target_contract.items():
        if target_metadata.get(key) != expected:
            raise RuntimeError(
                f"Projection-target metadata {key}={target_metadata.get(key)!r}, "
                f"expected {expected!r}"
            )

    data = copy.deepcopy(v4["data"])
    data.update(
        {
            "v4_states_root": str(args.v4_states_root),
            "terminal_hidden_root": str(args.terminal_hidden_root),
        }
    )
    common: dict[str, Any] = {
        "experiment": {"name": args.name_prefix},
        "runs_root": "runs",
        "data": data,
        "frozen_backbone": {
            "config": str(args.v4_config),
            "config_sha256": sha256_file(args.v4_config),
            "checkpoint": str(args.v4_checkpoint),
            "checkpoint_sha256": sha256_file(args.v4_checkpoint),
            "expected_epoch": args.expected_epoch,
        },
        "model": {
            "mode": "hard",
            "input_contract": "approximation_space_separated",
            "coordinate_frequencies": 6,
            "hidden_dim": 160,
            "hidden_layers": 3,
            "chunk_size": 65536,
            "boundary_temperature": 0.1,
            "use_views": False,
            "train_view_encoder": False,
            "use_terminal_latent": True,
            "latent_dim": int(v4["model"]["hidden_dim"]),
        },
        "training": {
            "seed": 20260915,
            "epochs": 30,
            "lr": 1.0e-4,
            "weight_decay": 1.0e-5,
            "grad_clip_norm": 1.0,
            "amp_dtype": "bf16",
            "early_stopping_patience": 5,
            "max_peak_gpu_gib": 16.0,
        },
        "loss": {
            "smooth_l1_beta": 0.1,
            "lambda_detail_l1": 1.0,
            "lambda_detail_mse": 0.1,
            # Disabled for unequal multi-source continuous fields: the current
            # decoder implementation defines this mask from the *case-global*
            # half maximum, which can erase a weaker GT source. The explicit
            # global MSE and all-positive-GT source MSE below retain every source
            # without detection censoring. A future morphology term must be
            # source-resolved before it can receive non-zero weight.
            "lambda_support": 0.0,
            "lambda_final_source_mse": 1.0,
            "lambda_final_global_mse": 1.0,
            "source_threshold": 0.0,
            # Retained only to make the disabled implementation fully specified.
            "support_fraction": 0.5,
            "support_temperature": 0.1,
            "tversky_alpha": 0.5,
            "tversky_beta": 0.5,
            # Same non-negativity prior as the M1 corrector, so the whole chain
            # shares one physical contract. Applied identically inside both arms,
            # so the matched pair still differs only in experiment.name and
            # model.mode.
            # A weight of 1 made this term ~1e-6 versus ~1e-2 source loss in
            # the first full LPR hard-Q epoch and permitted positive/negative
            # mass ratios 1.749/0.748 despite a preserved signed mass of 1.001.
            # Scale it into the same optimization decade so improvements cannot
            # be obtained through a large cancelling signed oscillation.
            "lambda_negativity": 1000.0,
        },
        "validation": {
            "threshold": 0.5,
            "dense_every": 1,
            "selection_metric": "ccc",
            "selection_mode": "max",
            "checkpoint_name": "best_dense_val_ccc.pth",
        },
    }
    hard = copy.deepcopy(common)
    hard["experiment"]["name"] = f"{args.name_prefix}_hard_q"
    free = copy.deepcopy(common)
    free["experiment"]["name"] = f"{args.name_prefix}_unconstrained"
    free["model"]["mode"] = "unconstrained"
    assert_matched_pair(hard, free)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hard_path = args.output_dir / f"{args.name_prefix}_hard_q.yaml"
    free_path = args.output_dir / f"{args.name_prefix}_unconstrained.yaml"
    hard_path.write_text(yaml.safe_dump(hard, sort_keys=False))
    free_path.write_text(yaml.safe_dump(free, sort_keys=False))
    receipt = {
        "status": "matched_decoder_pair_materialized",
        "training_or_evaluation_run": False,
        "name_prefix": args.name_prefix,
        "allowed_differences": sorted(ALLOWED_PAIR_DIFFERENCES),
        "hard_config": str(hard_path.resolve()),
        "hard_canonical_sha256": config_hash(hard),
        "unconstrained_config": str(free_path.resolve()),
        "unconstrained_canonical_sha256": config_hash(free),
        "v4_config_sha256": sha256_file(args.v4_config),
        "v4_checkpoint_sha256": sha256_file(args.v4_checkpoint),
        "v4_epoch": args.expected_epoch,
    }
    (args.output_dir / f"{args.name_prefix}_pair_contract.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
