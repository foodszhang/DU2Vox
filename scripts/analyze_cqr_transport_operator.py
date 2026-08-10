#!/usr/bin/env python3
"""Analyze compressed Green identity, spectral energy, and conditioning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.physics.compressed_green_operator import CompressedGreenOperator
from du2vox.physics.compressed_green_operator import load_forward_matrix
from du2vox.physics.measurement_basis import randomized_measurement_basis


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--operator_cache", default=None)
    parser.add_argument("--max_rank", type=int, default=128)
    parser.add_argument("--power-iterations", type=int, default=2)
    parser.add_argument("--out_json", default="diagnosis/cqr_transport_operator.json")
    parser.add_argument("--out_md", default="diagnosis/cqr_transport_operator.md")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    transport = cfg.get("transport", {}) or {}
    cache_path = Path(
        args.operator_cache
        or transport.get("operator_cache", "physics_cache/v2_3k_20k_transport_rank64.npz")
    )
    if not cache_path.is_absolute():
        cache_path = REPO_ROOT / cache_path
    operator = CompressedGreenOperator.load(cache_path)
    forward = load_forward_matrix(Path(cfg["data"]["shared_dir"]) / "system_matrix.A.npz")
    spectrum = randomized_measurement_basis(
        forward,
        rank=args.max_rank,
        oversampling=int(transport.get("oversampling", 16)),
        seed=int(transport.get("seed", 20260722)),
        power_iterations=args.power_iterations,
    )
    total_energy = spectrum.frobenius_energy
    ranks = [rank for rank in (16, 32, 64, 128) if rank <= args.max_rank]
    retained = {
        str(rank): float(np.square(spectrum.singular_values[:rank]).sum() / total_energy)
        for rank in ranks
    }
    gram_condition = float(
        (spectrum.singular_values[0] / max(spectrum.singular_values[-1], 1e-300)) ** 2
    )
    report = {
        "schema_version": 1,
        "cache": str(cache_path),
        "relative_operator_error": operator.relative_operator_error,
        "max_absolute_error": operator.max_absolute_error,
        "per_mode_relative_error": operator.per_mode_relative_error.tolist(),
        "retained_energy_by_rank": retained,
        "rank64_cache_retained_energy": operator.retained_energy_ratio,
        "gram_condition_number_rank_max": gram_condition,
        "spectral_power_iterations": args.power_iterations,
        "optical_modes_finite": bool(
            np.isfinite(operator.measurement_basis).all()
            and np.isfinite(operator.green_node_modes).all()
            and np.isfinite(operator.projected_forward_modes).all()
        ),
        "identity_passed": bool(operator.relative_operator_error <= 1e-4),
    }
    out_json = REPO_ROOT / args.out_json
    out_md = REPO_ROOT / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# CQR Transport Operator Analysis",
        "",
        f"- Identity relative error: `{operator.relative_operator_error:.6e}`",
        f"- Maximum absolute error: `{operator.max_absolute_error:.6e}`",
        f"- Gram condition number (rank {args.max_rank}): `{gram_condition:.6e}`",
        f"- Optical modes finite: **{report['optical_modes_finite']}**",
        "",
        "## Retained energy",
        "",
    ]
    lines.extend(f"- rank {rank}: `{value:.8f}`" for rank, value in retained.items())
    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
