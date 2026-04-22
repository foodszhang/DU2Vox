"""
DU2Vox config — mirrors fmt_simgen.config.

Any change to fmt_simgen frame/MCX/DE/view constants MUST be mirrored here.
The CONFIG_HASH cross-check ensures the two packages stay in sync.

version: v2-2026-04-21
"""
import sys
from pathlib import Path

# Add fmt_simgen to path so we can import from it
_FMT_ROOT = Path(__file__).parent.parent.parent.parent / "FMT-SimGen"
if not (_FMT_ROOT / "fmt_simgen").exists():
    _FMT_ROOT = Path(__file__).parent.parent.parent  # try workspace root

sys.path.insert(0, str(_FMT_ROOT))

# Import hash from fmt_simgen.config
from fmt_simgen.config import CONFIG_HASH as UPSTREAM_HASH
from fmt_simgen.config import CONTRACT_VERSION as UPSTREAM_VERSION
from fmt_simgen.config.frame_contract import (
    TRUNK_OFFSET_ATLAS_MM,
    TRUNK_SIZE_MM,
    VOXEL_SIZE_MM,
    TRUNK_GRID_SHAPE,
    VOLUME_CENTER_WORLD,
)

# DU2Vox's own copy of the same constants (must stay in sync)
# These are defined identically here so DU2Vox can use them directly
# without needing to import from fmt_simgen at runtime.
CONFIG_HASH: str = UPSTREAM_HASH  # local alias
CONTRACT_VERSION: str = UPSTREAM_VERSION

# Re-export for convenience
__all__ = [
    "TRUNK_OFFSET_ATLAS_MM",
    "TRUNK_SIZE_MM",
    "VOXEL_SIZE_MM",
    "TRUNK_GRID_SHAPE",
    "VOLUME_CENTER_WORLD",
    "CONFIG_HASH",
    "CONTRACT_VERSION",
    "UPSTREAM_HASH",
    "UPSTREAM_VERSION",
]


class FrameContractMismatch(RuntimeError):
    """Raised when DU2Vox config hash != fmt_simgen upstream hash."""
    pass


def _check_hash():
    """Called on import to verify alignment."""
    from fmt_simgen.config import CONFIG_HASH as current_hash
    if current_hash != UPSTREAM_HASH:
        raise FrameContractMismatch(
            f"DU2Vox config stale: upstream={current_hash}, "
            f"this={UPSTREAM_HASH}. "
            f"Sync du2vox/config/frame_contract.py from fmt_simgen/frame_contract.py "
            f"and rebuild."
        )


_check_hash()
