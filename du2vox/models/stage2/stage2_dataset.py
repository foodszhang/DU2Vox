"""
Stage 2 dataset: bridge output → ROI bbox → query points → 8D prior + GT.

Two modes:
- Stage2Dataset:           on-demand computation (slow, for compatibility)
- Stage2DatasetPrecomputed: load precomputed .npz grids (fast, for training)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from du2vox.bridge.fem_bridging import FEMBridge
from du2vox.utils.frame import FrameManifest

# MCX projection angle order (from view_config.json)
MCX_ANGLES = [-90, -60, -30, 0, 30, 60, 90]


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

        # Load shared mesh — rebased to trunk-local via FrameManifest
        self._nodes, self._elements = FrameManifest.load_mesh_nodes(shared_dir)
        self._frame = FrameManifest.load(shared_dir)

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
            # Move to end (most recently used)
            if sid in self._cache_order:
                self._cache_order.remove(sid)
            self._cache_order.append(sid)
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

        # GT: trilinear lookup from gt_voxels.npy (aligns with precompute path)
        from scipy.ndimage import map_coordinates
        gt_voxels = np.load(self.samples_dir / sid / "gt_voxels.npy").astype(np.float32)
        idx_float = self._frame.world_to_gt_index(queries.astype(np.float64))
        gt_values = map_coordinates(
            gt_voxels, idx_float.T, order=1, mode="constant", cval=0.0, prefilter=False,
        ).astype(np.float32)
        shape_arr = np.array(gt_voxels.shape)
        outside = np.any((idx_float < 0) | (idx_float > shape_arr - 1), axis=1)
        gt_values[outside] = 0.0

        return {
            "coords":     torch.from_numpy(queries.astype(np.float32)),
            "prior_8d":   torch.from_numpy(prior_8d.astype(np.float32)),
            "gt":         torch.from_numpy(gt_values.astype(np.float32)),
            "valid":      torch.from_numpy(valid),
            "sample_id":  sid,
        }


class Stage2DatasetPrecomputed(Dataset):
    """
    Precomputed grid dataset — fast, fork-safe, for training.

    Uses an LRU cache for loaded .npz files so repeated accesses within an epoch
    (same sample appearing in multiple batches across epochs) are fast.

    Parameters
    ----------
    precomputed_dir : directory with precomputed/*.npz files
    sample_ids       : list of sample IDs to use
    n_query_points   : random query points per sample per epoch (sampled from valid grid points)
    cache_size       : max number of .npz files to hold in memory (default: 32)
    """

    def __init__(
        self,
        precomputed_dir: str,
        sample_ids: List[str],
        n_query_points: int = 4096,
        cache_size: int = 32,
    ):
        self.precomputed_dir = Path(precomputed_dir)
        self.sample_ids = sample_ids
        self.n_query_points = n_query_points
        self._cache_size = cache_size

        # LRU cache: sid -> loaded npz arrays
        self._npz_cache: dict[str, dict] = {}
        self._cache_order: list[str] = []

    def _load_npz(self, sid: str) -> dict:
        """Load .npz from cache or disk (LRU)."""
        if sid in self._npz_cache:
            # Move to end (most recently used)
            self._cache_order.remove(sid)
            self._cache_order.append(sid)
            return self._npz_cache[sid]

        path = self.precomputed_dir / f"{sid}.npz"
        arrays = dict(np.load(path, allow_pickle=False))

        # Evict if at capacity
        while len(self._npz_cache) >= self._cache_size:
            evict_sid = self._cache_order.pop(0)
            del self._npz_cache[evict_sid]

        self._npz_cache[sid] = arrays
        self._cache_order.append(sid)
        return arrays

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sid = self.sample_ids[idx]
        data = self._load_npz(sid)

        valid_mask = data["valid_mask"]
        valid_indices = np.where(valid_mask)[0]
        n_valid = len(valid_indices)

        if n_valid == 0:
            # Degenerate case: return zeros
            return {
                "coords":   torch.zeros(self.n_query_points, 3, dtype=torch.float32),
                "prior_8d": torch.zeros(self.n_query_points, 8, dtype=torch.float32),
                "gt":       torch.zeros(self.n_query_points, dtype=torch.float32),
                "valid":    torch.zeros(self.n_query_points, dtype=torch.bool),
                "sample_id": sid,
            }

        # Sample n_query_points from valid indices
        if n_valid >= self.n_query_points:
            chosen = np.random.choice(valid_indices, self.n_query_points, replace=False)
        else:
            chosen = np.random.choice(valid_indices, self.n_query_points, replace=True)

        # Use pre-normalized coords if available, otherwise normalize on the fly
        if "grid_coords_norm" in data:
            coords = data["grid_coords_norm"][chosen].copy()
        else:
            # Legacy npz: normalize raw coords to [-1,1] using bbox metadata
            raw = data["grid_coords"][chosen].copy()
            bbox_min = data["bbox_min"]
            bbox_max = data["bbox_max"]
            coords = (2.0 * (raw - bbox_min) / (bbox_max - bbox_min + 1e-8) - 1.0)

        item = {
            "coords":   torch.from_numpy(coords.astype(np.float32)),
            "prior_8d": torch.from_numpy(data["prior_8d"][chosen].copy()),
            "gt":       torch.from_numpy(data["gt_values"][chosen].copy()),
            # Pre-filtered: chosen points are sampled from valid_mask indices only.
            # Every returned point is guaranteed to be inside a ROI/CQR tet.
            "valid":    torch.ones(self.n_query_points, dtype=torch.bool),
            "sample_id": sid,
        }
        if "prior_ext" in data:
            item["prior_ext"] = torch.from_numpy(data["prior_ext"][chosen].copy().astype(np.float32))
        if "role" in data:
            item["role"] = torch.from_numpy(data["role"][chosen].copy().astype(np.int64))
        if "coverage_score" in data:
            item["coverage_score"] = torch.from_numpy(data["coverage_score"][chosen].copy().astype(np.float32))
        if "risk_components" in data:
            item["risk_components"] = torch.from_numpy(data["risk_components"][chosen].copy().astype(np.float32))
        if "query_weight" in data:
            item["query_weight"] = torch.from_numpy(data["query_weight"][chosen].copy().astype(np.float32))
        return item


class Stage2DatasetPrecomputedMultiview(Stage2DatasetPrecomputed):
    """
    Precomputed grid dataset with MCX multi-view projection images.

    Extends Stage2DatasetPrecomputed by loading proj.npz from the sample's
    FMT-SimGen directory for multi-view feature lookup.

    Parameters
    ----------
    precomputed_dir : directory with precomputed/*.npz grid files
    samples_dir      : directory with FMT-SimGen samples (contains proj.npz)
    sample_ids       : list of sample IDs to use
    n_query_points   : random query points per sample per epoch
    cache_size       : max number of .npz files to hold in memory (default: 16)
    """

    def __init__(
        self,
        precomputed_dir: str,
        samples_dir: str,
        sample_ids: List[str],
        n_query_points: int = 4096,
        cache_size: int = 16,
        shared_dir: str | None = None,
    ):
        super().__init__(
            precomputed_dir=precomputed_dir,
            sample_ids=sample_ids,
            n_query_points=n_query_points,
            cache_size=cache_size,
        )
        self.samples_dir = Path(samples_dir)
        # Load frame manifest for MCX coordinate transform
        self.frame = FrameManifest.load(shared_dir) if shared_dir else None

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sid = self.sample_ids[idx]
        data = self._load_npz(sid)

        valid_mask = data["valid_mask"]
        valid_indices = np.where(valid_mask)[0]
        n_valid = len(valid_indices)

        if n_valid == 0:
            return {
                "coords":        torch.zeros(self.n_query_points, 3, dtype=torch.float32),
                "prior_8d":      torch.zeros(self.n_query_points, 8, dtype=torch.float32),
                "gt":            torch.zeros(self.n_query_points, dtype=torch.float32),
                "valid":         torch.zeros(self.n_query_points, dtype=torch.bool),
                "coords_world":  torch.zeros(self.n_query_points, 3, dtype=torch.float32),
                "proj_imgs":     torch.zeros(7, 1, 256, 256, dtype=torch.float32),
                "sample_id":     sid,
            }

        # Sample n_query_points from valid indices
        if n_valid >= self.n_query_points:
            chosen = np.random.choice(valid_indices, self.n_query_points, replace=False)
        else:
            chosen = np.random.choice(valid_indices, self.n_query_points, replace=True)

        # Normalized coords for INR input
        if "grid_coords_norm" in data:
            coords_norm = data["grid_coords_norm"][chosen].copy()
        else:
            raw = data["grid_coords"][chosen].copy()
            bbox_min = data["bbox_min"]
            bbox_max = data["bbox_max"]
            coords_norm = (2.0 * (raw - bbox_min) / (bbox_max - bbox_min + 1e-8) - 1.0)

        # Raw world coords (mm) for projection — always use raw grid_coords
        coords_world = data["grid_coords"][chosen].copy().astype(np.float32)

        # MCX validity: point is inside the physical trunk volume.
        # Projection itself must use coords_world. FMT-SimGen's proj.npz is
        # generated by subtracting the MCX volume center and normalizing by FOV,
        # not by stretching the volume bbox to [-1, 1].
        world_xyz = coords_world  # [N, 3] in trunk-local mm
        if self.frame is not None:
            mcx_bbox_min = self.frame.mcx_bbox_min
            mcx_bbox_max = self.frame.mcx_bbox_max
            mcx_valid = (
                (world_xyz[:, 0] >= mcx_bbox_min[0]) & (world_xyz[:, 0] <= mcx_bbox_max[0]) &
                (world_xyz[:, 1] >= mcx_bbox_min[1]) & (world_xyz[:, 1] <= mcx_bbox_max[1]) &
                (world_xyz[:, 2] >= mcx_bbox_min[2]) & (world_xyz[:, 2] <= mcx_bbox_max[2])
            )
        else:
            mcx_valid = np.ones(len(world_xyz), dtype=bool)

        # Load MCX projection images: [7, 256, 256] in angle order
        proj_path = self.samples_dir / sid / "proj.npz"
        if proj_path.exists():
            proj_data = np.load(proj_path)
            # Stack in the same angle order as MCX_ANGLES
            proj_imgs = np.stack(
                [proj_data[str(angle)].astype(np.float32) for angle in MCX_ANGLES],
                axis=0,
            )  # [7, 256, 256]
        else:
            # Fallback: zeros if proj.npz not available
            proj_imgs = np.zeros((7, 256, 256), dtype=np.float32)

        item = {
            "coords":       torch.from_numpy(coords_norm.astype(np.float32)),
            "prior_8d":    torch.from_numpy(data["prior_8d"][chosen].copy()),
            "gt":          torch.from_numpy(data["gt_values"][chosen].copy()),
            "valid":       torch.ones(self.n_query_points, dtype=torch.bool),
            "coords_world": torch.from_numpy(coords_world),
            "mcx_valid":   torch.from_numpy(mcx_valid),
            "proj_imgs":   torch.from_numpy(proj_imgs).unsqueeze(1),  # [7, 1, 256, 256]
            "sample_id":   sid,
        }
        if "prior_ext" in data:
            item["prior_ext"] = torch.from_numpy(data["prior_ext"][chosen].copy().astype(np.float32))
        if "role" in data:
            item["role"] = torch.from_numpy(data["role"][chosen].copy().astype(np.int64))
        if "coverage_score" in data:
            item["coverage_score"] = torch.from_numpy(data["coverage_score"][chosen].copy().astype(np.float32))
        if "risk_components" in data:
            item["risk_components"] = torch.from_numpy(data["risk_components"][chosen].copy().astype(np.float32))
        if "query_weight" in data:
            item["query_weight"] = torch.from_numpy(data["query_weight"][chosen].copy().astype(np.float32))
        return item
