#!/usr/bin/env python3
"""One-sample memory/timing gate for full-resolution exact voxel detail."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.bridge.canonical_cross_discretization import CanonicalCrossDiscretization
from du2vox.models.stage2.complement_voxel_detail import (
    ComplementConstrainedVoxelDetail,
    ExactVoxelComplement,
)
from du2vox.models.stage2.residual_inr import PositionalEncoding
from du2vox.models.stage2.stage2_dataset import load_projection_stack
from scripts.train_error_structured_bridge import build_view_encoder
from scripts.train_iterative_fem_corrector import build_model, load_ids


def gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return values[indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=65536)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    data = cfg["data"]
    os.environ["DU2VOX_SHARED_DIR"] = str(data["shared_dir"])
    os.environ["DU2VOX_ALLOW_STALE_FRAME_MANIFEST"] = "1"
    os.environ["DU2VOX_FRAME_MANIFEST_SHA256"] = str(data["frame_manifest_sha256"])
    device = torch.device("cuda")
    canonical = CanonicalCrossDiscretization(data["operator_cache"], factorize=False)
    exact_q = ExactVoxelComplement(canonical.p)
    p = canonical.p.tocsr()
    node_indices = torch.from_numpy(p.indices.reshape(-1, 4).astype(np.int64)).to(device)
    barycentric = torch.from_numpy(p.data.reshape(-1, 4).astype(np.float32)).to(device)
    coords = torch.from_numpy(canonical.operator.coords_world).to(device)
    from du2vox.utils.frame import FrameManifest

    frame = FrameManifest.load(data["shared_dir"])
    bbox_min = torch.as_tensor(frame.mcx_bbox_min, device=device, dtype=torch.float32)
    bbox_max = torch.as_tensor(frame.mcx_bbox_max, device=device, dtype=torch.float32)
    coords_norm = 2.0 * (coords - bbox_min) / (bbox_max - bbox_min) - 1.0

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    v4 = build_model(cfg).to(device)
    v4.load_state_dict(checkpoint["model"])
    encoder = build_view_encoder(cfg["model"]).to(device)
    encoder.load_state_dict(v4.view_encoder.state_dict())
    encoder.eval().requires_grad_(False)
    del v4

    sid = load_ids(data["val_split"])[0]
    with np.load(
        Path("runs/unified_dual_evidence_fem_v4_2400/val_predictions") / f"{sid}.npz"
    ) as saved:
        x_v4 = torch.from_numpy(saved["step3_fem_nodes"]).to(device)
    gt_volume = np.load(Path(data["samples_dir"]) / sid / "gt_voxels.npy", mmap_mode="r")
    gt_np = (np.asarray(gt_volume).ravel()[canonical.operator.valid_flat_indices] > 0.05).astype(
        np.float32
    )
    pi_gt = np.load(Path(data["projection_targets_dir"]) / f"{sid}.npy").astype(np.float32)
    detail_target_np = gt_np - np.asarray(p @ pi_gt).ravel().astype(np.float32)
    gt = torch.from_numpy(gt_np).to(device)
    detail_target = torch.from_numpy(detail_target_np).to(device)
    projections, _ = load_projection_stack(
        Path(data["samples_dir"]) / sid,
        projection_file=data["projection_file"],
        projection_norm=data["projection_norm"],
        projection_transform=data["projection_transform"],
    )
    images = torch.from_numpy(projections[:, None]).unsqueeze(0).to(device)
    with torch.no_grad():
        encoded = encoder.encode_images(images)

    pe = PositionalEncoding(n_freqs=6, include_input=True).to(device)
    input_dim = pe.out_dim + 1 + 4 + 4 + 3 + 1 + int(cfg["model"]["view_feat_dim"])
    model = ComplementConstrainedVoxelDetail(
        input_dim=input_dim,
        complement=exact_q,
        hidden_dim=160,
        n_hidden_layers=3,
        constrained=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()
    proposals = []
    coarse_chunks = []
    for start in range(0, exact_q.n_voxels, args.chunk_size):
        end = min(start + args.chunk_size, exact_q.n_voxels)
        ids = node_indices[start:end]
        bary = barycentric[start:end]
        local = gather(x_v4, ids)
        coarse = torch.sum(local * bary, dim=-1)
        local_stats = torch.stack(
            [local.mean(-1), local.std(-1, unbiased=False), local.amax(-1) - local.amin(-1)], dim=-1
        )
        boundary = torch.exp(-torch.abs(coarse - 0.5) / 0.1).unsqueeze(-1)
        with torch.no_grad():
            view, _ = encoder.sample_encoded(encoded, coords[start:end][None], coords_vox_norm=None)
        features = torch.cat(
            [
                pe(coords_norm[start:end]),
                coarse[:, None],
                local,
                bary,
                local_stats,
                boundary,
                view.squeeze(0),
            ],
            dim=-1,
        )
        proposals.append(model.proposal(features))
        coarse_chunks.append(coarse)
    proposal = torch.cat(proposals).unsqueeze(0)
    coarse = torch.cat(coarse_chunks).unsqueeze(0)
    detail = model.constrain(proposal)
    final = coarse + detail
    loss = F.smooth_l1_loss(detail, detail_target[None], beta=0.1)
    loss = loss + 0.1 * F.mse_loss(detail, detail_target[None])
    probabilities = torch.sigmoid((final - 0.5) / 0.1)
    intersection = torch.sum(probabilities * gt[None])
    support = 1.0 - (intersection + 1e-6) / (
        intersection
        + 0.5 * torch.sum(probabilities * (1.0 - gt[None]))
        + 0.5 * torch.sum((1.0 - probabilities) * gt[None])
        + 1e-6
    )
    loss = loss + 0.1 * support
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    payload = {
        "sample_id": sid,
        "domain": "full canonical 0.2 mm",
        "n_voxels": exact_q.n_voxels,
        "chunk_size": args.chunk_size,
        "elapsed_seconds": elapsed,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "loss": float(loss.detach()),
        "gradient_l1": float(
            sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
