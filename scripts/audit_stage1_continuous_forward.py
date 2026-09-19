#!/usr/bin/env python3
"""Audit raw and normalized Stage1 forward contracts on continuous fields."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def relative_l2(error: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(error, dim=1) / torch.linalg.vector_norm(
        reference, dim=1
    ).clamp_min(1e-30)


def summarize(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = [key for key in rows[0] if key != "sample_id"]
    return {
        key: {
            "min": float(np.min([row[key] for row in rows])),
            "median": float(np.median([row[key] for row in rows])),
            "mean": float(np.mean([row[key] for row in rows])),
            "max": float(np.max([row[key] for row in rows])),
        }
        for key in keys
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    ids = load_ids(args.dataset_root / "splits" / f"{args.split}.txt")
    if args.max_samples is not None:
        ids = ids[: args.max_samples]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matrix = np.load(args.shared_dir / "system_matrix.A.npz")["forward_matrix"]
    a = torch.from_numpy(matrix.astype(np.float32, copy=False)).to(device)
    rows: list[dict[str, float]] = []
    for start in range(0, len(ids), args.batch_size):
        batch_ids = ids[start : start + args.batch_size]
        nodes_raw = torch.from_numpy(
            np.stack(
                [np.load(args.dataset_root / "samples" / sid / "gt_nodes.npy") for sid in batch_ids]
            ).astype(np.float32)
        ).to(device)
        saved_b = torch.from_numpy(
            np.stack(
                [
                    np.load(args.dataset_root / "samples" / sid / "measurement_b.npy")
                    for sid in batch_ids
                ]
            ).astype(np.float32)
        ).to(device)
        gt_scale = torch.tensor(
            [
                float(
                    np.asarray(
                        np.load(args.dataset_root / "samples" / sid / "gt_scale.npy")
                    ).reshape(())
                )
                for sid in batch_ids
            ],
            device=device,
        )
        forward_raw = nodes_raw @ a.t()
        clipped_raw = forward_raw.clamp_min(0.0)
        b_scale = saved_b.amax(dim=1).clamp_min(1e-30)
        y_norm = saved_b / b_scale[:, None]
        x_norm = nodes_raw / gt_scale[:, None]
        forward_fixed = x_norm @ a.t()
        operator_scale = gt_scale / b_scale
        forward_matched = forward_fixed * operator_scale[:, None]
        clipped_fixed = forward_fixed.clamp_min(0.0)
        clipped_matched = forward_matched.clamp_min(0.0)
        alpha = (forward_fixed * y_norm).sum(dim=1).clamp_min(0.0) / forward_fixed.square().sum(
            dim=1
        ).clamp_min(1e-30)
        forward_profiled = forward_fixed * alpha[:, None]
        cold_adjoint = -y_norm @ a
        fixed_target_adjoint = (forward_fixed - y_norm) @ a
        profiled_target_adjoint = alpha[:, None] * ((forward_profiled - y_norm) @ a)
        for index, sid in enumerate(batch_ids):
            negative = forward_raw[index].clamp_max(0.0)
            rows.append(
                {
                    "sample_id": sid,
                    "raw_clipped_saved_rel_l2": float(
                        relative_l2(
                            (clipped_raw - saved_b)[index : index + 1],
                            saved_b[index : index + 1],
                        )[0]
                    ),
                    "raw_negative_energy_fraction": float(
                        negative.square().sum() / forward_raw[index].square().sum().clamp_min(1e-30)
                    ),
                    "normalized_fixed_a_rel_l2": float(
                        relative_l2(
                            (forward_fixed - y_norm)[index : index + 1],
                            y_norm[index : index + 1],
                        )[0]
                    ),
                    "normalized_fixed_a_clipped_rel_l2": float(
                        relative_l2(
                            (clipped_fixed - y_norm)[index : index + 1],
                            y_norm[index : index + 1],
                        )[0]
                    ),
                    "normalized_matched_a_rel_l2": float(
                        relative_l2(
                            (forward_matched - y_norm)[index : index + 1],
                            y_norm[index : index + 1],
                        )[0]
                    ),
                    "normalized_matched_a_clipped_rel_l2": float(
                        relative_l2(
                            (clipped_matched - y_norm)[index : index + 1],
                            y_norm[index : index + 1],
                        )[0]
                    ),
                    "normalized_profiled_a_rel_l2": float(
                        relative_l2(
                            (forward_profiled - y_norm)[index : index + 1],
                            y_norm[index : index + 1],
                        )[0]
                    ),
                    "operator_scale_gt_over_bmax": float(operator_scale[index]),
                    "profiled_scale": float(alpha[index]),
                    "cold_adjoint_rms": float(cold_adjoint[index].square().mean().sqrt()),
                    "cold_adjoint_l2": float(torch.linalg.vector_norm(cold_adjoint[index])),
                    "fixed_target_adjoint_rms": float(
                        fixed_target_adjoint[index].square().mean().sqrt()
                    ),
                    "profiled_target_adjoint_rms": float(
                        profiled_target_adjoint[index].square().mean().sqrt()
                    ),
                }
            )
        print(f"[{min(start + len(batch_ids), len(ids))}/{len(ids)}]", flush=True)

    result = {
        "split": args.split,
        "n_samples": len(rows),
        "device": str(device),
        "summary": summarize(rows),
        "per_sample": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
