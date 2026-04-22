"""DU2Vox side frame utilities — read manifest produced by FMT-SimGen."""
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Default path to FMT-SimGen shared directory (may be overridden via env)
DEFAULT_SHARED_DIR = os.environ.get(
    "DU2VOX_SHARED_DIR",
    "/home/foods/pro/FMT-SimGen/output/shared",
)

# Module-level cache for frame constants (avoids re-parsing JSON on every import)
_CACHED_CONSTANTS: Optional[dict] = None
_CACHE_MTIME: Optional[float] = None
_CACHE_PATH: Optional[Path] = None
STALE_THRESHOLD_DAYS: float = 30.0


def get_frame_constants(shared_dir: Optional[str | Path] = None) -> dict:
    """Load and cache frame constants from frame_manifest.json.

    On first call, parses the manifest and caches derived geometry constants.
    If the cached manifest file is older than STALE_THRESHOLD_DAYS, raises
    RuntimeError to prevent silent regressions when FMT-SimGen was updated but
    build_shared_assets was not re-run.

    Parameters
    ----------
    shared_dir : str or Path, optional
        Path to shared directory containing frame_manifest.json.
        Defaults to DU2VOX_SHARED_DIR env var or DEFAULT_SHARED_DIR.

    Returns
    -------
    dict with keys:
        - voxel_size_mm: float (e.g. 0.2)
        - mcx_half_extents: np.ndarray [3] — half of physical volume size
        - mcx_volume_center: np.ndarray [3] — same as half_extents in trunk-local frame
        - mcx_shape_xyz: tuple [3]

    Raises
    ------
    RuntimeError
        If the manifest file does not exist or is older than STALE_THRESHOLD_DAYS.
    """
    global _CACHED_CONSTANTS, _CACHE_MTIME, _CACHE_PATH

    if shared_dir is None:
        shared_dir = DEFAULT_SHARED_DIR
    manifest_path = Path(shared_dir) / "frame_manifest.json"

    # Check staleness: if file changed since last cache, invalidate
    current_mtime = manifest_path.stat().st_mtime
    if _CACHED_CONSTANTS is not None and (
        _CACHE_PATH != manifest_path or _CACHE_MTIME != current_mtime
    ):
        _CACHED_CONSTANTS = None

    if _CACHED_CONSTANTS is not None:
        return _CACHED_CONSTANTS

    if not manifest_path.exists():
        raise RuntimeError(
            f"frame_manifest.json not found at {manifest_path}. "
            f"Run FMT-SimGen build_shared_assets() first."
        )

    # Staleness check: raise if file is older than threshold (guards against
    # genuinely abandoned artifacts; does NOT catch "code changed but no regenerate"
    # since that scenario leaves a fresh mtime).
    import time
    age_days = (time.time() - current_mtime) / 86400.0
    if age_days > STALE_THRESHOLD_DAYS:
        raise RuntimeError(
            f"frame_manifest.json at {manifest_path} is {age_days:.1f} days old "
            f"(threshold={STALE_THRESHOLD_DAYS} days). "
            f"FMT-SimGen frame_contract may have changed — "
            f"re-run build_shared_assets() before using DU2Vox."
        )

    # Prefer frame_contract section (v2 manifest); fall back to legacy layout
    m = json.load(open(manifest_path))
    fc = m.get("frame_contract", {})
    if fc:
        voxel_size_mm = fc["voxel_size_mm"]
        trunk_size = fc["trunk_size_mm"]           # [38, 40, 20.8]
        grid_shape_xyz = tuple(fc["trunk_grid_shape_xyz"])  # (190, 200, 104)
        half_extents = [s / 2.0 for s in trunk_size]       # [19, 20, 10.4]
    else:
        # Legacy v1 layout (derived from config YAML — less trustworthy)
        voxel_size_mm = m["mcx_volume"]["voxel_size_mm"]
        shape_xyz = m["mcx_volume"]["shape_xyz"]
        grid_shape_xyz = tuple(shape_xyz)
        half_extents = [
            shape_xyz[0] * voxel_size_mm / 2,
            shape_xyz[1] * voxel_size_mm / 2,
            shape_xyz[2] * voxel_size_mm / 2,
        ]

    _CACHED_CONSTANTS = {
        "voxel_size_mm": float(voxel_size_mm),
        "mcx_half_extents": np.array(half_extents, dtype=np.float32),
        "mcx_volume_center": np.array(half_extents, dtype=np.float32),
        "mcx_shape_xyz": grid_shape_xyz,
    }
    _CACHE_MTIME = current_mtime
    _CACHE_PATH = manifest_path
    return _CACHED_CONSTANTS


@dataclass
class FrameManifest:
    """Frame metadata loaded from FMT-SimGen's frame_manifest.json."""

    world_frame: str
    atlas_to_world_offset_mm: np.ndarray  # [3] — subtract from atlas to get world
    mcx_bbox_min: np.ndarray  # [3], trunk-local mm
    mcx_bbox_max: np.ndarray  # [3]
    mcx_voxel_size_mm: float
    mcx_shape_xyz: tuple
    gt_offset_world_mm: np.ndarray
    gt_spacing_mm: float
    gt_shape: tuple

    @classmethod
    def load(cls, shared_dir: str | Path) -> "FrameManifest":
        """Load frame_manifest.json from the shared directory."""
        m = json.load(open(Path(shared_dir) / "frame_manifest.json"))
        assert m["world_frame"] == "mcx_trunk_local_mm", (
            f"Unknown frame: {m['world_frame']} (expected mcx_trunk_local_mm)"
        )
        return cls(
            world_frame=m["world_frame"],
            atlas_to_world_offset_mm=np.array(m["atlas_to_world_offset_mm"]),
            mcx_bbox_min=np.array(m["mcx_volume"]["bbox_world_mm"]["min"]),
            mcx_bbox_max=np.array(m["mcx_volume"]["bbox_world_mm"]["max"]),
            mcx_voxel_size_mm=m["mcx_volume"]["voxel_size_mm"],
            mcx_shape_xyz=tuple(m["mcx_volume"]["shape_xyz"]),
            gt_offset_world_mm=np.array(m["voxel_grid_gt"]["offset_world_mm"]),
            gt_spacing_mm=m["voxel_grid_gt"]["spacing_mm"],
            gt_shape=tuple(m["voxel_grid_gt"]["shape"]),
        )

    def atlas_to_world(self, atlas_mm: np.ndarray) -> np.ndarray:
        """atlas_corner_mm → mcx_trunk_local_mm."""
        return np.asarray(atlas_mm, dtype=np.float64) - self.atlas_to_world_offset_mm

    @staticmethod
    def load_mesh_nodes(shared_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
        """Load mesh.npz in trunk-local frame (written at mesh write time by FMT-SimGen).

        This is the ONLY canonical way to read mesh.nodes in DU2Vox.
        FMT-SimGen builder.py now saves mesh.npz in mcx_trunk_local_mm frame,
        so no runtime rebase is needed.
        """
        frame = FrameManifest.load(shared_dir)
        assert frame.world_frame == "mcx_trunk_local_mm", (
            f"mesh.npz frame={frame.world_frame}, expected mcx_trunk_local_mm. "
            f"Regenerate FMT-SimGen shared assets with unified-frame builder."
        )
        mesh = np.load(Path(shared_dir) / "mesh.npz")
        nodes = mesh["nodes"].astype(np.float64)
        assert nodes.max() < 45, (
            f"nodes.max()={nodes.max():.1f} — mesh.npz may be in wrong frame "
            f"(expected trunk-local < 45mm)"
        )
        elements = mesh["elements"]
        return nodes, elements

    # ─── Core transforms ───────────────────────────────────────────────────

    def world_to_mcx_voxel(self, world_mm: np.ndarray) -> np.ndarray:
        """trunk-local mm → MCX voxel (float, for grid_sample)."""
        return np.asarray(world_mm) / self.mcx_voxel_size_mm

    def world_to_mcx_ndc(self, world_mm: np.ndarray) -> np.ndarray:
        """trunk-local mm → [-1, 1] in MCX volume (for F.grid_sample)."""
        # Cache volume size in mm — derived from constants, recomputed only once
        if not hasattr(self, "_vs_mm"):
            self._vs_mm = np.array([
                self.mcx_shape_xyz[i] * self.mcx_voxel_size_mm
                for i in range(3)
            ], dtype=np.float64)
        return 2.0 * np.asarray(world_mm) / self._vs_mm - 1.0

    def world_to_gt_index(self, world_mm: np.ndarray) -> np.ndarray:
        """trunk-local mm → gt_voxels grid fractional index (for trilinear)."""
        return (
            np.asarray(world_mm)
            - self.gt_offset_world_mm
            - self.gt_spacing_mm / 2
        ) / self.gt_spacing_mm
