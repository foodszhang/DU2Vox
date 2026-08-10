#!/usr/bin/env python3
"""Precompute and validate the compressed diffusion Green operator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.physics.compressed_green_operator import apply_surface_convention
from du2vox.physics.compressed_green_operator import build_compressed_green_operator
from du2vox.physics.compressed_green_operator import load_forward_matrix
from du2vox.physics.compressed_green_operator import load_surface_index


REPO_ROOT = Path(__file__).resolve().parents[1]


def _metadata(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--oversampling", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--power-iterations", type=int, default=0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-relative-error", type=float, default=1e-4)
    args = parser.parse_args()

    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    shared_dir = Path(cfg["data"]["shared_dir"])
    transport_cfg = cfg.get("transport", {}) or {}
    output = Path(
        args.output
        or transport_cfg.get(
            "operator_cache", f"physics_cache/v2_3k_20k_transport_rank{args.rank}.npz"
        )
    )
    if not output.is_absolute():
        output = REPO_ROOT / output

    matrix_paths = {
        "M": shared_dir / "system_matrix.M.npz",
        "F": shared_dir / "system_matrix.F.npz",
        "A": shared_dir / "system_matrix.A.npz",
        "index": shared_dir / "system_matrix.index.npz",
    }
    print(f"[Transport] loading shared physics from {shared_dir}")
    mass = sp.load_npz(matrix_paths["M"])
    source = sp.load_npz(matrix_paths["F"])
    forward = load_forward_matrix(matrix_paths["A"])
    surface_index = load_surface_index(shared_dir)
    visible_path = shared_dir / "visible_mask.npy"
    visible_mask = np.load(visible_path) if visible_path.exists() else None
    use_visible_mask = bool(cfg["data"].get("use_visible_mask", False))
    forward, surface_index, visible_applied = apply_surface_convention(
        forward, surface_index, visible_mask, use_visible_mask
    )
    print(
        f"[Transport] M={mass.shape}, F={source.shape}, A={forward.shape}, "
        f"rank={args.rank}, oversampling={args.oversampling}, visible={visible_applied}"
    )

    operator = build_compressed_green_operator(
        mass,
        source,
        forward,
        surface_index,
        rank=args.rank,
        oversampling=args.oversampling,
        seed=args.seed,
        power_iterations=args.power_iterations,
        visible_mask_applied=visible_applied,
    )
    frame_path = shared_dir / "frame_manifest.json"
    frame_metadata = json.loads(frame_path.read_text()) if frame_path.exists() else {}
    operator.save(
        output,
        source_matrix_shapes={
            "M": list(mass.shape),
            "F": list(source.shape),
            "A": list(forward.shape),
        },
        file_metadata={name: _metadata(path) for name, path in matrix_paths.items()},
        frame_metadata=frame_metadata,
        oversampling=args.oversampling,
        seed=args.seed,
        power_iterations=args.power_iterations,
    )
    print(f"[Transport] cache={output}")
    print(f"[Transport] relative_operator_error={operator.relative_operator_error:.6e}")
    print(f"[Transport] max_absolute_error={operator.max_absolute_error:.6e}")
    print(f"[Transport] retained_energy_ratio={operator.retained_energy_ratio:.8f}")
    print(
        "[Transport] per_mode_relative_error="
        f"min={operator.per_mode_relative_error.min():.3e}, "
        f"median={np.median(operator.per_mode_relative_error):.3e}, "
        f"max={operator.per_mode_relative_error.max():.3e}"
    )
    if not np.isfinite(operator.relative_operator_error):
        raise RuntimeError("Compressed Green validation produced a non-finite error")
    if operator.relative_operator_error > args.max_relative_error:
        raise RuntimeError(
            f"Operator identity failed: {operator.relative_operator_error:.6e} > "
            f"{args.max_relative_error:.6e}; inspect surface/A/visible/transpose/F conventions"
        )


if __name__ == "__main__":
    main()
