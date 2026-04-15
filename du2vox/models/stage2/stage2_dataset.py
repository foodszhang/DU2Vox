"""
Stage 2 dataset: bridge output → ROI bbox → query points → 8D prior + GT.

Each sample = one bridge output (coarse_d + roi_tet_indices + roi_info).
Uses LRU cache for FEMBridge to avoid OOM when pre-building all samples.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from du2vox.bridge.fem_bridging import FEMBridge


class Stage2Dataset(Dataset):
    """
    Parameters
    ----------
    bridge_dir      : root of bridge output (sample_XXXX subdirs)
    shared_dir      : path to FMT-SimGen output/shared (mesh.npz)
    samples_dir     : path to FMT-SimGen samples (gt_nodes.npy per sample)
    sample_ids      : list of sample IDs to use
    n_query_points  : random query points per sample per epoch
    roi_padding_mm  : pad ROI bbox by this amount (mm)
    bridge_cache_size: max cached FEMBridge objects (default 16)
    """

    def __init__(
        self,
        bridge_dir: str,
        shared_dir: str,
        samples_dir: str,
        sample_ids: List[str],
        n_query_points: int = 4096,
        roi_padding_mm: float = 1.0,
        bridge_cache_size: int = 16,
    ):
        self.bridge_dir = Path(bridge_dir)
        self.samples_dir = Path(samples_dir)
        self.n_query_points = n_query_points
        self.roi_padding_mm = roi_padding_mm

        # Load shared mesh once
        mesh = np.load(f"{shared_dir}/mesh.npz")
        self._nodes = mesh["nodes"].astype(np.float64)
        self._elements = mesh["elements"]

        self.sample_ids = sample_ids

        # Cache only metadata (coarse_d, roi_info), NOT the FEMBridge
        self._cache = {}
        for sid in sample_ids:
            bd = self.bridge_dir / sid
            self._cache[sid] = {
                "coarse_d": np.load(bd / "coarse_d.npy").astype(np.float32),
                "roi_tet_indices": np.load(bd / "roi_tet_indices.npy"),
                "roi_info": json.loads((bd / "roi_info.json").read_text()),
            }

        # LRU cache for FEMBridge objects
        self._bridge_cache: dict[str, FEMBridge] = {}
        self._cache_order: list[str] = []
        self._bridge_cache_size = bridge_cache_size

    def _get_bridge(self, sid: str) -> FEMBridge:
        """Get FEMBridge from LRU cache (build if not cached)."""
        if sid in self._bridge_cache:
            return self._bridge_cache[sid]

        c = self._cache[sid]
        bridge = FEMBridge(
            self._nodes,
            self._elements,
            roi_tet_indices=c["roi_tet_indices"],
        )

        # Evict if at capacity
        if len(self._bridge_cache) >= self._bridge_cache_size:
            evict_sid = self._cache_order.pop(0)
            del self._bridge_cache[evict_sid]

        self._bridge_cache[sid] = bridge
        self._cache_order.append(sid)
        return bridge

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sid = self.sample_ids[idx]
        c = self._cache[sid]

        # Random query points in (padded) ROI bbox
        bbox = c["roi_info"]["roi_bbox_mm"]
        lo = np.array(bbox["min"]) - self.roi_padding_mm
        hi = np.array(bbox["max"]) + self.roi_padding_mm
        queries = np.random.uniform(lo, hi, (self.n_query_points, 3)).astype(np.float32)

        # 8D prior from Stage 1 coarse (from cached bridge)
        bridge = self._get_bridge(sid)
        prior_8d, valid = bridge.get_prior_features(queries, c["coarse_d"])

        # GT: FEM interpolation of gt_nodes
        gt_nodes = np.load(self.samples_dir / sid / "gt_nodes.npy").astype(np.float32)
        gt_prior, gt_valid = bridge.get_prior_features(queries, gt_nodes)

        # FEM-interpolated GT values
        gt_values = (gt_prior[:, :4] * gt_prior[:, 4:8]).sum(axis=-1)  # [N]
        gt_values[~gt_valid] = 0.0

        return {
            "coords":     torch.from_numpy(queries.astype(np.float32)),
            "prior_8d":   torch.from_numpy(prior_8d.astype(np.float32)),
            "gt":         torch.from_numpy(gt_values.astype(np.float32)),
            "valid":      torch.from_numpy(valid),
            "sample_id":  sid,
        }
