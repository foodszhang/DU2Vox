#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from du2vox.models.stage2.cqr_residual_inr import CQRResidualINR


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test a CQR precomputed npz")
    parser.add_argument("--npz", required=True)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    path = Path(args.npz)
    data = np.load(path, allow_pickle=False)
    print(f"[Smoke] file={path}")
    for k in data.files:
        print(f"  {k:22s} shape={data[k].shape} dtype={data[k].dtype}")

    required = ["grid_coords_norm", "prior_ext", "gt_values", "valid_mask", "role"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise SystemExit(f"Missing required fields: {missing}")

    valid_idx = np.where(data["valid_mask"])[0]
    if len(valid_idx) == 0:
        raise SystemExit("No valid CQR points found")

    chosen = valid_idx[: min(args.n, len(valid_idx))]
    coords = torch.from_numpy(data["grid_coords_norm"][chosen].astype(np.float32))[None].to(args.device)
    prior = torch.from_numpy(data["prior_ext"][chosen].astype(np.float32))[None].to(args.device)

    model = CQRResidualINR(
        n_freqs=4,
        hidden_dim=32,
        n_hidden_layers=2,
        prior_dim=prior.shape[-1],
        view_feat_dim=0,
    ).to(args.device)
    with torch.no_grad():
        d_hat, fem, residual = model(coords, prior)

    print("[Smoke] model output:")
    print("  d_hat   ", tuple(d_hat.shape), float(d_hat.min()), float(d_hat.max()))
    print("  fem     ", tuple(fem.shape), float(fem.min()), float(fem.max()))
    print("  residual", tuple(residual.shape), float(residual.abs().mean()))
    print("[Smoke] ok")


if __name__ == "__main__":
    main()
