#!/usr/bin/env python3
"""Run the preregistered A/B/C innovation falsification matrix."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.eval_stage2_unified import build_model  # noqa: E402


MODELS = {
    "a": ("model_a_p1_plain", "configs/stage2/falsification/p1_plain_residual.yaml"),
    "b": ("model_b_tc_plain", "configs/stage2/falsification/tc_plain_residual.yaml"),
    "c": (
        "model_c_tc_partition",
        "configs/stage2/falsification/tc_observability_partition.yaml",
    ),
}


def run(command: list[str]) -> None:
    print("[Falsification]", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def parameter_counts(config_path: Path) -> dict[str, int]:
    with open(config_path) as handle:
        cfg = yaml.safe_load(handle)
    data_cfg = cfg.get("data", {})
    if data_cfg.get("shared_dir"):
        os.environ["DU2VOX_SHARED_DIR"] = str(data_cfg["shared_dir"])
    if data_cfg.get("allow_stale_frame_manifest", False):
        os.environ["DU2VOX_ALLOW_STALE_FRAME_MANIFEST"] = "1"
    if data_cfg.get("frame_manifest_sha256"):
        os.environ["DU2VOX_FRAME_MANIFEST_SHA256"] = str(data_cfg["frame_manifest_sha256"])
    model, view_encoder = build_model(cfg, torch.device("cpu"))

    def count(module: torch.nn.Module | None) -> int:
        if module is None:
            return 0
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    lifter = getattr(model, "lifter", None)
    projector = getattr(model, "projector", None)
    model_count = count(model)
    lifter_count = count(lifter)
    projector_count = count(projector)
    view_count = count(view_encoder)
    return {
        "total_trainable": model_count + view_count,
        "inr": model_count - lifter_count - projector_count,
        "lifter": lifter_count,
        "view_encoder": view_count,
        "projector": projector_count,
    }


def aggregate_branch_diagnostics(grouped_path: Path) -> dict[str, object]:
    with open(grouped_path) as handle:
        grouped = json.load(handle)
    keys = (
        "alpha_lambda_l1",
        "alpha_entropy",
        "rho_tc_p1_l1",
        "rho_tc_p1_l2",
        "transport_error",
        "observable_energy_fraction",
        "ambiguous_energy_fraction",
        "raw_observable_norm",
        "projected_observable_norm",
        "observable_projection_ratio",
        "raw_ambiguous_norm",
        "projected_ambiguous_norm",
        "ambiguous_projection_ratio",
        "measurement_error",
        "correction_forward_error",
    )
    rows = grouped.get("per_sample", [])
    summary: dict[str, float] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row]
        if values:
            summary[key] = sum(values) / len(values)
    per_role = {}
    for role in ("bg", "core", "halo", "sentinel", "proposal"):
        role_summary = {}
        for base in ("alpha_lambda_deviation", "rho_tc_minus_p1"):
            key = f"{base}_{role}"
            values = [float(row[key]) for row in rows if key in row]
            if values:
                role_summary[base] = sum(values) / len(values)
        if role_summary:
            per_role[role] = role_summary
    by_depth = {}
    for depth in sorted({str(row.get("depth_tier", "unknown")) for row in rows}):
        group = [row for row in rows if str(row.get("depth_tier", "unknown")) == depth]
        depth_summary = {}
        for key in ("alpha_lambda_l1", "rho_tc_p1_l1", "rho_tc_p1_l2"):
            values = [float(row[key]) for row in group if key in row]
            if values:
                depth_summary[key] = sum(values) / len(values)
        by_depth[depth] = {"n_samples": len(group), **depth_summary}
    return {
        "n_samples": len(rows),
        "mean": summary,
        "per_role_lifting": per_role,
        "per_depth_lifting": by_depth,
        "per_sample": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260722, 20260723, 20260724])
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--physics_weight", type=float, default=0.0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--output_root", default="results/innovation_falsification")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    uv_python = ["uv", "run", "python"]
    for model_key in args.models:
        model_dir_name, template_name = MODELS[model_key]
        model_root = output_root / model_dir_name
        with open(template_name) as handle:
            template = yaml.safe_load(handle)
        if args.physics_weight > 0:
            model_root = output_root / f"{model_dir_name}_phys"
        for seed in args.seeds:
            seed_dir = model_root / f"seed_{seed}"
            best_path = seed_dir / "best.pth"
            if args.skip_existing and best_path.exists():
                print(f"[Falsification] skip existing {seed_dir}")
                continue
            seed_dir.mkdir(parents=True, exist_ok=True)
            cfg = yaml.safe_load(yaml.safe_dump(template))
            cfg["training"]["seed"] = seed
            cfg["data"]["query_base_seed"] = seed
            if args.max_epochs is not None:
                cfg["training"]["max_epochs"] = args.max_epochs
                cfg["training"]["scheduler"]["T_max"] = args.max_epochs
            cfg["loss"]["lambda_correction_physics"] = args.physics_weight
            config_path = seed_dir / "config.yaml"
            with open(config_path, "w") as handle:
                yaml.safe_dump(cfg, handle, sort_keys=False)
            with open(seed_dir / "parameter_counts.json", "w") as handle:
                json.dump(parameter_counts(config_path), handle, indent=2)

            run(
                uv_python
                + [
                    "scripts/train_stage2.py",
                    "--config",
                    str(config_path),
                    "--checkpoint_dir",
                    str(model_root),
                    "--experiment_name",
                    f"seed_{seed}",
                ]
            )
            train_log = Path("logs") / f"seed_{seed}" / "train_log.json"
            if train_log.exists():
                shutil.copy2(train_log, seed_dir / "train_history.json")
            for split in ("val", "test"):
                run(
                    uv_python
                    + [
                        "scripts/eval_stage2_unified.py",
                        "--config",
                        str(config_path),
                        "--checkpoint",
                        str(best_path),
                        "--split",
                        split,
                        "--batch_points",
                        "32768",
                        "--out_json",
                        str(seed_dir / f"{split}_metrics.json"),
                        "--out_csv",
                        str(seed_dir / f"{split}_metrics.csv"),
                    ]
                )
            grouped_path = seed_dir / "grouped_metrics.json"
            run(
                uv_python
                + [
                    "scripts/eval_stage2_v2_grouped.py",
                    "--config",
                    str(config_path),
                    "--checkpoint",
                    str(best_path),
                    "--split",
                    "test",
                    "--batch_points",
                    "32768",
                    "--out_json",
                    str(grouped_path),
                    "--out_csv",
                    str(seed_dir / "grouped_metrics.csv"),
                ]
            )
            with open(seed_dir / "branch_diagnostics.json", "w") as handle:
                json.dump(aggregate_branch_diagnostics(grouped_path), handle, indent=2)

    if set(args.models) >= {"b", "c"}:
        for seed in args.seeds:
            suffix = "_phys" if args.physics_weight > 0 else ""
            b_dir = output_root / f"model_b_tc_plain{suffix}" / f"seed_{seed}"
            c_dir = output_root / f"model_c_tc_partition{suffix}" / f"seed_{seed}"
            if (b_dir / "best.pth").exists() and (c_dir / "best.pth").exists():
                run(
                    uv_python
                    + [
                        "scripts/analyze_partition_equivalence.py",
                        "--plain_config",
                        str(b_dir / "config.yaml"),
                        "--plain_checkpoint",
                        str(b_dir / "best.pth"),
                        "--partition_config",
                        str(c_dir / "config.yaml"),
                        "--partition_checkpoint",
                        str(c_dir / "best.pth"),
                        "--split",
                        "test",
                        "--output",
                        str(c_dir / "partition_equivalence.json"),
                    ]
                )


if __name__ == "__main__":
    main()
