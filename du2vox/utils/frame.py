"""DU2Vox side frame utilities — read manifest produced by FMT-SimGen."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
