#!/usr/bin/env python3
"""Recover the byte-exact certified frame manifest from its historical schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


TARGET_SHA256 = "25b77f5d11430ebf4d9735f19534782afb56f7fe5cab525a569028b0f3c02e2b"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _historical_manifest(mesh_path: Path) -> dict[str, object]:
    """Reproduce FMT-SimGen commit 9b65a27's canonical manifest serialization."""
    with np.load(mesh_path, allow_pickle=True) as mesh:
        nodes = mesh["nodes"].astype(np.float64)
    return {
        "subject_id": "subject",
        "world_frame": "mcx_trunk_local_mm",
        "output_dir": "output/shared",
        "mcx_volume": {
            "shape_xyz": [190, 200, 104],
            "shape_zyx": [104, 200, 190],
            "voxel_size_mm": 0.2,
            "origin_world_mm": [0.0, 0.0, 0.0],
            "extent_mm": [38.0, 40.0, 20.8],
            "bbox_world_mm": {
                "min": [0.0, 0.0, 0.0],
                "max": [38.0, 40.0, 20.8],
            },
        },
        "atlas_to_world_offset_mm": [0.0, 0.0, 0.0],
        "crop_bbox_mm": None,
        "segmentation_path": None,
        "segmentation_format": None,
        "label_key": None,
        "label_mapping": {},
        "label_roles": {
            "background_labels": [0],
            "allowed_tumor_labels": [1],
            "forbidden_tumor_labels": [0, 2],
        },
        "volume_center_world_mm": [19.0, 20.0, 10.4],
        "version": 2,
        "frame_contract": {
            "voxel_size_mm": 0.2,
            "grid_shape_xyz": [190, 200, 104],
            "volume_center_world_mm": [19.0, 20.0, 10.4],
            "volume_extents_mm": [38.0, 40.0, 20.8],
            "generated_by": "fmt_simgen.dataset.builder.build_shared_assets",
        },
        "fem_mesh": {
            "file": "mesh.npz",
            "frame": "mcx_trunk_local_mm",
            "n_nodes": int(nodes.shape[0]),
            "bbox_world_mm": {
                "min": nodes.min(0).tolist(),
                "max": nodes.max(0).tolist(),
            },
        },
        "voxel_grid_gt": {
            "shape": [190, 200, 104],
            "spacing_mm": 0.2,
            "offset_world_mm": [0.0, 0.0, 0.0],
            "frame": "mcx_trunk_local_mm",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()

    manifest = _historical_manifest(args.shared_dir / "mesh.npz")
    payload = json.dumps(manifest, indent=2).encode()
    digest = _sha256(payload)
    print(json.dumps({"candidate_sha256": digest, "target_sha256": TARGET_SHA256}))
    if args.restore:
        if digest != TARGET_SHA256:
            raise RuntimeError("Historical candidate does not match certified hash")
        (args.shared_dir / "frame_manifest.json").write_bytes(payload)


if __name__ == "__main__":
    main()
