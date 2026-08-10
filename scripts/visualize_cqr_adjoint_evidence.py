#!/usr/bin/env python3
"""Visualize CQR GT, P1, proposal, adjoint, and correction decomposition."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PANELS = (
    ("gt", "GT"),
    ("stage1_fem", "Stage 1 FEM"),
    ("p1_baseline", "P1 baseline"),
    ("role", "CQR role"),
    ("residual_indicator", "RGL residual indicator"),
    ("measurement_proposal", "Measurement proposal"),
    ("adjoint_evidence", "Adjoint evidence"),
    ("observable_correction", "Observable correction"),
    ("ambiguous_correction", "Ambiguous correction"),
    ("final_result", "Final result"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default="diagnosis/cqr_gt_decomposition")
    parser.add_argument("--output_dir", default="diagnosis/cqr_adjoint_evidence")
    parser.add_argument("--max_samples", type=int, default=5)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(input_dir.glob("sample_*.npz"))[: args.max_samples]
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            coords = data["coords_world"]
            fig, axes = plt.subplots(2, 5, figsize=(22, 8), constrained_layout=True)
            for axis, (key, title) in zip(axes.flat, PANELS, strict=True):
                values = data[key]
                scatter = axis.scatter(
                    coords[:, 0], coords[:, 2], c=values, s=1.5, cmap="coolwarm", rasterized=True
                )
                axis.set_title(title)
                axis.set_xlabel("x (mm)")
                axis.set_ylabel("z (mm)")
                fig.colorbar(scatter, ax=axis, fraction=0.046)
            fig.suptitle(path.stem)
            fig.savefig(output_dir / f"{path.stem}.png", dpi=160)
            plt.close(fig)
            print(f"[Adjoint visualization] {output_dir / f'{path.stem}.png'}")
    if not paths:
        raise RuntimeError(f"No decomposition NPZ files found in {input_dir}")


if __name__ == "__main__":
    main()
