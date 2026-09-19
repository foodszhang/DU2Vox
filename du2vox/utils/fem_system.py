"""Loading helpers for the canonical FEM forward system."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse
import torch


def load_fem_system_matrix(
    shared_dir: str | Path, *, use_visible_mask: bool = False
) -> torch.Tensor:
    """Load A with the same visible-surface convention used by Stage 1."""

    shared = Path(shared_dir)
    path = shared / "system_matrix.A.npz"
    archive = np.load(path, allow_pickle=True)
    if "forward_matrix" in archive:
        matrix = archive["forward_matrix"].astype(np.float32)
    else:
        matrix = scipy.sparse.load_npz(path).toarray().astype(np.float32)
    if use_visible_mask:
        mask_path = shared / "visible_mask.npy"
        if not mask_path.exists():
            raise FileNotFoundError(f"Requested visible mask does not exist: {mask_path}")
        matrix = matrix[np.load(mask_path).astype(bool)]
    return torch.from_numpy(matrix)
