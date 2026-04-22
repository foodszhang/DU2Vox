"""
Canonical frame contract — MIRROR of fmt_simgen.frame_contract (DU2Vox copy).

This file must be byte-level identical to fmt_simgen/frame_contract.py.
DO NOT edit independently — any change must be mirrored from the source.

version: v2-2026-04-21

Canonical frame: mcx_trunk_local_mm
- Origin = trunk bbox corner (0, 0, 0) in physical mm
- Axes: X=right (+19mm left→right), Y=anterior (+20mm anterior→posterior), Z=dorsal (+10.4mm ventral→dorsal)
- Rotation center for projection = (19.0, 20.0, 10.4) = TRUNK_SIZE_MM / 2.0
- MCX voxel grid: (190, 200, 104) voxels @ 0.2mm = (38, 40, 20.8) mm
"""
from __future__ import annotations

import numpy as np
from typing import Final

# ── Trunk geometry ──────────────────────────────────────────────────────────────

TRUNK_OFFSET_ATLAS_MM: Final[np.ndarray] = np.array(
    [0.0, 34.0, 0.0], dtype=np.float64
)
"""Offset from atlas corner to trunk bbox corner in mm.
Used to convert atlas-corner coordinates to trunk-local coordinates.
Atlas frame: origin at (0,0,0) = top-left-anterior voxel corner.
Trunk-local: origin at trunk bbox corner (0,0,0).
Formula: trunk = atlas_corner - TRUNK_OFFSET_ATLAS_MM"""

TRUNK_SIZE_MM: Final[np.ndarray] = np.array(
    [38.0, 40.0, 20.8], dtype=np.float64
)
"""Physical size of trunk bounding box in mm: X×Y×Z."""

VOXEL_SIZE_MM: Final[float] = 0.2
"""Voxel size for MCX volume and FEM grid in mm."""

TRUNK_GRID_SHAPE: Final[tuple[int, int, int]] = (190, 200, 104)
"""MCX volume grid shape in XYZ order (X=190, Y=200, Z=104)."""

VOLUME_CENTER_WORLD: Final[np.ndarray] = TRUNK_SIZE_MM / 2.0  # (19.0, 20.0, 10.4)
"""Geometric center of the trunk volume in trunk-local mm.
Used as rotation center for orthographic projection (camera looks toward this point)."""

# ── Derived ─────────────────────────────────────────────────────────────────────────

CAMERA_DISTANCE_MM: Final[float] = 200.0
"""Camera-to-rotation-center distance for turntable projection."""

FOV_MM: Final[float] = 80.0
"""Field of view for turntable detector in mm (square detector)."""

DETECTOR_RESOLUTION: Final[tuple[int, int]] = (256, 256)
"""Turntable detector resolution: (width, height) in pixels."""

ANGLES: Final[list[int]] = [-90, -60, -30, 0, 30, 60, 90]
"""Turntable rotation angles in degrees."""


# ── Conversion helpers ────────────────────────────────────────────────────────────

def atlas_corner_to_trunk(atlas_coords: np.ndarray) -> np.ndarray:
    """Convert atlas-corner-mm coordinates to trunk-local-mm coordinates.

    Parameters
    ----------
    atlas_coords : np.ndarray
        [N×3] coordinates in atlas corner frame (mm).

    Returns
    -------
    np.ndarray
        [N×3] coordinates in trunk-local frame (mm).
    """
    return atlas_coords - TRUNK_OFFSET_ATLAS_MM


def trunk_to_atlas_corner(trunk_coords: np.ndarray) -> np.ndarray:
    """Convert trunk-local-mm coordinates to atlas-corner-mm coordinates."""
    return trunk_coords + TRUNK_OFFSET_ATLAS_MM


def world_to_volume_voxel(
    world_coords: np.ndarray,
    voxel_size: float = VOXEL_SIZE_MM,
) -> np.ndarray:
    """Convert trunk-local world mm to integer voxel indices.

    Assumes world origin is at trunk bbox corner (voxel [0,0,0] center = voxel_size/2 mm).
    """
    return np.floor(world_coords / voxel_size).astype(np.int32)


def volume_voxel_to_world(
    voxel_indices: np.ndarray,
    voxel_size: float = VOXEL_SIZE_MM,
) -> np.ndarray:
    """Convert integer voxel indices to trunk-local world mm (voxel centers)."""
    return (voxel_indices + 0.5) * voxel_size


def assert_in_trunk_bbox(
    coords: np.ndarray,
    tol_mm: float = 3.0,
) -> None:
    """Assert that coordinates are within the trunk bounding box (with tolerance).

    Raises
    ------
    AssertionError
        If any coordinate is outside trunk bbox ± tol_mm.
    """
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    assert (lo >= -tol_mm).all(), (
        f"Coordinates below trunk bbox: min={lo}, expected >={-tol_mm}"
    )
    assert (hi <= TRUNK_SIZE_MM + tol_mm).all(), (
        f"Coordinates above trunk bbox: max={hi}, expected <={TRUNK_SIZE_MM + tol_mm}"
    )
