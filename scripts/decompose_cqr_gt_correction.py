#!/usr/bin/env python3
"""Decompose the oracle GT-minus-P1 correction with the CQR operator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.models.stage2.cqr_observability_projector import CQRObservabilityProjector
from du2vox.physics.compressed_green_operator import CompressedGreenOperator


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_split(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def proposal_at_points(sample_dir: Path, points: np.ndarray) -> np.ndarray:
    heatmap = np.load(sample_dir / "proposal/meas_backproj_heatmap.npy")
    meta = json.loads((sample_dir / "proposal/meas_backproj_meta.json").read_text())
    origin = np.asarray(meta.get("origin_mm", [0, 0, 0]), dtype=np.float64)
    cell = np.asarray(meta["cell_size_mm"], dtype=np.float64)
    indices = np.floor((points - origin) / cell).astype(np.int64)
    valid = ((indices >= 0) & (indices < np.asarray(heatmap.shape))).all(axis=1)
    values = np.zeros(len(points), dtype=np.float32)
    values[valid] = heatmap[tuple(indices[valid].T)]
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--max_samples", type=int, default=4)
    parser.add_argument("--output_dir", default="diagnosis/cqr_gt_decomposition")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    transport = cfg["transport"]
    operator = CompressedGreenOperator.load(transport["operator_cache"])
    with np.load(Path(cfg["data"]["shared_dir"]) / "mesh.npz", allow_pickle=False) as mesh:
        elements = mesh["elements"].astype(np.int64)
    ids = load_split(cfg["data"][f"{args.split}_split"])[: args.max_samples]
    cqr_dir = REPO_ROOT / cfg["data"][f"precomputed_{args.split}_dir"]
    sidecar_dir = REPO_ROOT / transport["sidecar_root"] / args.split
    bridge_dir = REPO_ROOT / cfg["data"][f"{args.split}_bridge_dir"]
    samples_dir = Path(cfg["data"]["samples_dir"])
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    projector = CQRObservabilityProjector(**cfg["observability"])
    rows = []
    for sid in ids:
        cqr_path = cqr_dir / f"{sid}.npz"
        sidecar_path = sidecar_dir / f"{sid}.npz"
        if not cqr_path.exists() or not sidecar_path.exists():
            continue
        with np.load(cqr_path, allow_pickle=False) as cqr, np.load(
            sidecar_path, allow_pickle=False
        ) as sidecar:
            points = cqr["grid_coords"].astype(np.float32)
            prior = cqr["prior_8d"].astype(np.float32)
            tet_ids = cqr["tet_ids"].astype(np.int64)
            role = cqr["role"].astype(np.int64)
            residual_indicator = cqr["residual_indicator"].astype(np.float32)
            gt = cqr["gt_values"].astype(np.float32)
            valid = sidecar["candidate_valid_physics_mask"].astype(bool)
            weights = sidecar["candidate_cell_weight"].astype(np.float32)
        fem = np.sum(prior[:, :4] * prior[:, 4:8], axis=1)
        query_green = np.zeros((len(points), operator.rank), dtype=np.float32)
        vertices = elements[tet_ids[valid]]
        query_green[valid] = np.einsum(
            "nk,rnk->nr", prior[valid, 4:8], operator.green_node_modes[:, vertices]
        )
        a_query = torch.from_numpy(query_green.T * weights[None, :])
        correction = torch.from_numpy(gt - fem)
        observable = projector.project_observable(correction, a_query).detach().numpy()
        ambiguous = projector.project_ambiguous(correction, a_query).detach().numpy()
        coarse = np.load(bridge_dir / sid / "coarse_d.npy").reshape(-1)
        measurement = np.load(samples_dir / sid / "measurement_b.npy").reshape(-1)
        measured_modes = operator.measurement_basis.T @ measurement
        stage1_modes = operator.projected_forward_modes @ coarse
        scale = max(
            0.0,
            float(
                np.dot(measured_modes, stage1_modes)
                / (np.dot(stage1_modes, stage1_modes) + 1e-8)
            ),
        )
        residual_modes = measured_modes - scale * stage1_modes
        adjoint = (a_query.T @ torch.from_numpy(residual_modes).float()).numpy()
        proposal = proposal_at_points(samples_dir / sid, points)
        final_oracle = np.clip(fem + observable + ambiguous, 0, 1)
        np.savez_compressed(
            output_dir / f"{sid}.npz",
            coords_world=points,
            gt=gt,
            stage1_fem=fem,
            p1_baseline=fem,
            role=role,
            residual_indicator=residual_indicator,
            measurement_proposal=proposal,
            adjoint_evidence=adjoint,
            observable_correction=observable,
            ambiguous_correction=ambiguous,
            final_result=final_oracle,
            measurement_scale=np.float64(scale),
        )
        decomposition_error = float(np.max(np.abs(observable + ambiguous - (gt - fem))))
        rows.append(
            {
                "sample_id": sid,
                "measurement_scale": scale,
                "decomposition_max_error": decomposition_error,
                "observable_l2": float(np.linalg.norm(observable)),
                "ambiguous_l2": float(np.linalg.norm(ambiguous)),
                "oracle_final_mse": float(np.mean((final_oracle - gt) ** 2)),
            }
        )
        print(f"[GT decomposition] {sid}: error={decomposition_error:.3e}, scale={scale:.4f}")
    summary = {"schema_version": 1, "split": args.split, "samples": rows}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not rows:
        raise RuntimeError("No CQR GT decompositions were generated")


if __name__ == "__main__":
    main()
