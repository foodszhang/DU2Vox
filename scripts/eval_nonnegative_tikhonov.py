#!/usr/bin/env python3
"""Batched nonnegative graph-Tikhonov baseline on the frozen DE operator."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.sparse
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.bridge.canonical_cross_discretization import (  # noqa: E402
    CanonicalCrossDiscretization,
)
from du2vox.evaluation.continuous_field import continuous_metrics  # noqa: E402
from du2vox.utils.gt_io import (  # noqa: E402
    load_canonical_gt,
    load_normalization_scale,
)
from du2vox.utils.confirmation import (  # noqa: E402
    validate_validation_dataset_receipt,
)


def load_ids(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_freeze(path: Path, config: Path, alpha_relative: float, dataset_receipt: Path) -> None:
    receipt = json.loads(path.read_text())
    if receipt.get("status") != "frozen_on_val300":
        raise RuntimeError("Development-test evaluation requires a valid freeze receipt")
    matches = [
        item
        for item in receipt.get("candidates", [])
        if item.get("name") == "Tikhonov" and item.get("config_sha256") == sha256(config)
    ]
    if len(matches) != 1:
        raise RuntimeError("The Tikhonov config is not uniquely frozen")
    validation = json.loads(Path(matches[0]["validation_artifact"]).read_text())
    if float(validation["alpha_relative"]) != float(alpha_relative):
        raise RuntimeError("Requested Tikhonov alpha differs from validation selection")
    validate_validation_dataset_receipt(receipt, dataset_receipt)


def spectral_norm_squared(
    operator: Callable[[torch.Tensor], torch.Tensor],
    size: int,
    iterations: int,
    device: torch.device,
) -> float:
    vector = torch.randn(size, device=device)
    vector /= torch.linalg.vector_norm(vector)
    eigenvalue = 0.0
    for _ in range(iterations):
        result = operator(vector)
        eigenvalue = float(torch.dot(vector, result))
        vector = result / torch.linalg.vector_norm(result).clamp_min(1e-30)
    return eigenvalue


def sparse_left(matrix: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    return torch.sparse.mm(matrix, rows.T).T


def solve_batch(
    a: torch.Tensor,
    laplacian: torch.Tensor,
    measurement: torch.Tensor,
    *,
    alpha: float,
    step_size: float,
    iterations: int,
) -> torch.Tensor:
    x = torch.zeros(len(measurement), a.shape[1], device=a.device, dtype=torch.float32)
    accelerated = x.clone()
    momentum = 1.0
    for _ in range(iterations):
        residual = accelerated @ a.T - measurement
        gradient = residual @ a
        lap = sparse_left(laplacian, accelerated)
        gradient = gradient + alpha * sparse_left(laplacian.transpose(0, 1), lap)
        updated = torch.clamp(accelerated - step_size * gradient, min=0.0)
        next_momentum = (1.0 + np.sqrt(1.0 + 4.0 * momentum**2)) / 2.0
        accelerated = updated + ((momentum - 1.0) / next_momentum) * (updated - x)
        x = updated
        momentum = next_momentum
    return x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--alpha-relative", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--freeze-receipt", type=Path)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    data = cfg["data"]
    if args.split == "test":
        if args.freeze_receipt is None:
            raise RuntimeError("Development-test evaluation requires --freeze-receipt")
        dataset_receipt = data.get("dataset_receipt")
        if not dataset_receipt:
            raise RuntimeError("Config must declare data.dataset_receipt")
        validate_freeze(
            args.freeze_receipt,
            args.config,
            args.alpha_relative,
            Path(dataset_receipt),
        )
    solver = cfg["solver"]
    ids = load_ids(Path(data[f"{args.split}_split"]))
    if args.max_samples is not None:
        ids = ids[: args.max_samples]
    samples_dir = Path(data["dataset_root"]) / "samples"
    shared_dir = Path(data["shared_dir"])
    canonical = CanonicalCrossDiscretization(
        data["operator_cache"],
        shared_dir=shared_dir,
        factorize=False,
        allow_stale_frame_manifest=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a_np = np.load(shared_dir / "system_matrix.A.npz")["forward_matrix"]
    a = torch.from_numpy(a_np.astype(np.float32, copy=False)).to(device)
    lap_np = scipy.sparse.load_npz(shared_dir / "graph_laplacian_full.Lap.npz").tocoo()
    laplacian = torch.sparse_coo_tensor(
        torch.from_numpy(np.vstack([lap_np.row, lap_np.col])).long(),
        torch.from_numpy(lap_np.data.astype(np.float32)),
        lap_np.shape,
        device=device,
    ).coalesce()
    power_iterations = int(solver.get("power_iterations", 30))
    a_norm_sq = spectral_norm_squared(
        lambda value: a.T @ (a @ value), a.shape[1], power_iterations, device
    )
    l_norm_sq = spectral_norm_squared(
        lambda value: torch.sparse.mm(
            laplacian.transpose(0, 1),
            torch.sparse.mm(laplacian, value[:, None]),
        ).squeeze(1),
        a.shape[1],
        power_iterations,
        device,
    )
    alpha = float(args.alpha_relative) * a_norm_sq / max(l_norm_sq, 1e-30)
    step_size = 1.0 / (a_norm_sq + alpha * l_norm_sq)
    args.predictions_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    batch_size = int(solver.get("batch_size", 20))
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start : start + batch_size]
        measurement = torch.from_numpy(
            np.stack(
                [np.load(samples_dir / sid / "measurement_b.npy") for sid in batch_ids]
            ).astype(np.float32)
        ).to(device)
        if bool(data.get("normalize_b", False)):
            measurement = measurement / measurement.amax(dim=1, keepdim=True).clamp_min(1e-8)
        nodes = (
            solve_batch(
                a,
                laplacian,
                measurement,
                alpha=alpha,
                step_size=step_size,
                iterations=int(solver["iterations"]),
            )
            .cpu()
            .numpy()
        )
        voxels = np.asarray(canonical.p @ nodes.T, dtype=np.float32).T
        for sid, node_values, prediction in zip(batch_ids, nodes, voxels, strict=True):
            scale_filename = data.get("normalization_scale_filename")
            normalization_scale = (
                load_normalization_scale(samples_dir / sid, scale_filename)
                if scale_filename
                else None
            )
            gt, _ = load_canonical_gt(
                samples_dir / sid,
                canonical.operator.valid_flat_indices,
                gt_mode="continuous",
                normalize=data.get("normalize_gt", "none"),
                normalization_scale=normalization_scale,
            )
            row = {
                "sample_id": sid,
                **continuous_metrics(prediction, gt, data_range=float(data.get("data_range", 2.0))),
            }
            rows.append(row)
            np.savez_compressed(
                args.predictions_dir / f"{sid}.npz",
                fem_nodes=node_values.astype(np.float32),
                final_prediction=prediction.astype(np.float32),
                valid_flat_indices=canonical.operator.valid_flat_indices,
                grid_shape=np.asarray(canonical.operator.grid_shape),
            )
        print(f"[{min(start + batch_size, len(ids))}/{len(ids)}]", flush=True)
    keys = [key for key in rows[0] if key != "sample_id"]
    result = {
        "method": "nonnegative_graph_tikhonov",
        "split": args.split,
        "n_samples": len(rows),
        "alpha_relative": float(args.alpha_relative),
        "alpha": alpha,
        "a_norm_squared": a_norm_sq,
        "l_norm_squared": l_norm_sq,
        "iterations": int(solver["iterations"]),
        "normalize_b": bool(data.get("normalize_b", False)),
        "normalize_gt": data.get("normalize_gt", "none"),
        "normalization_scale_filename": data.get("normalization_scale_filename"),
        "data_range": float(data.get("data_range", 2.0)),
        "summary": {key: float(np.nanmean([float(row[key]) for row in rows])) for key in keys},
        "per_sample": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")


if __name__ == "__main__":
    main()
