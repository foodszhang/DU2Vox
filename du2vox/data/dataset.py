"""
FMTSimGen dataset for training.
"""

from pathlib import Path

import numpy as np
import scipy.sparse
import torch
from torch.utils.data import Dataset


def load_npz_as_torch_sparse(path: Path) -> torch.Tensor:
    """Load a sparse .npz file and return as dense torch tensor."""
    mat = scipy.sparse.load_npz(path)
    return torch.tensor(mat.toarray(), dtype=torch.float32)


class FMTSimGenDataset(Dataset):
    """FMT-SimGen dataset for MS-GDUN training.

    Loads shared assets (mesh, system matrix, Laplacians, kNN) once,
    then returns (b, gt) pairs for each sample.

    Normalization and binarization are configurable to support both
    Gaussian source experiments (continuous gt, max-normalized) and
    uniform/binary source experiments (pre-binarized gt).
    """

    def __init__(
        self,
        shared_dir: str | Path,
        samples_dir: str | Path,
        split_file: str | Path,
        normalize_b: bool = True,
        normalize_gt: bool = True,
        binarize_gt: bool = False,
        binarize_threshold: float = 0.05,
    ):
        shared_dir = Path(shared_dir)
        samples_dir = Path(samples_dir)
        split_file = Path(split_file)

        # Load mesh
        mesh = np.load(shared_dir / "mesh.npz")
        self.nodes = torch.tensor(mesh["nodes"], dtype=torch.float32)
        n_nodes = self.nodes.shape[0]
        n_surface = len(mesh["surface_node_indices"])

        # System matrix A [S, N]
        A_sp = scipy.sparse.load_npz(shared_dir / "system_matrix.A.npz")
        self.A = torch.tensor(A_sp.toarray(), dtype=torch.float32)

        # Full-node Laplacians [N, N]
        self.L = load_npz_as_torch_sparse(shared_dir / "graph_laplacian_full.Lap.npz")
        self.L0 = load_npz_as_torch_sparse(shared_dir / "graph_laplacian_full.n_Lap0.npz")
        self.L1 = load_npz_as_torch_sparse(shared_dir / "graph_laplacian_full.n_Lap1.npz")
        self.L2 = load_npz_as_torch_sparse(shared_dir / "graph_laplacian_full.n_Lap2.npz")
        self.L3 = load_npz_as_torch_sparse(shared_dir / "graph_laplacian_full.n_Lap3.npz")

        # kNN indices [N, 32]
        self.knn_idx = torch.tensor(
            np.load(shared_dir / "knn_idx_full.npy"),
            dtype=torch.long,
        )

        # Sensitivity weights
        self.sens_w = torch.norm(self.A, dim=0)
        self.sens_w = self.sens_w / (self.sens_w.max() + 1e-8)

        # Store preprocessing options
        self.normalize_b = normalize_b
        self.normalize_gt = normalize_gt
        self.binarize_gt = binarize_gt
        self.binarize_threshold = binarize_threshold

        # Sample list
        with open(split_file) as f:
            self.sample_ids = [line.strip() for line in f if line.strip()]

        # Preload all samples
        self.b_list: list[torch.Tensor] = []
        self.gt_list: list[torch.Tensor] = []
        for sid in self.sample_ids:
            b_path = samples_dir / sid / "measurement_b.npy"
            gt_path = samples_dir / sid / "gt_nodes.npy"

            # Load and normalize measurement b
            b = torch.tensor(np.load(b_path), dtype=torch.float32).unsqueeze(-1)
            if self.normalize_b:
                b_max = b.max()
                if b_max > 1e-8:
                    b = b / b_max

            # Load and process gt
            gt = np.load(gt_path)
            if self.binarize_gt:
                gt = (gt > self.binarize_threshold).astype(np.float32)
            elif self.normalize_gt:
                gt_max = gt.max()
                if gt_max > 1e-8:
                    gt = gt / gt_max
            gt = torch.tensor(gt, dtype=torch.float32).unsqueeze(-1)

            assert b.shape[0] == n_surface, (
                f"b has {b.shape[0]} surface nodes but mesh has {n_surface}"
            )
            assert gt.shape[0] == n_nodes, (
                f"gt has {gt.shape[0]} nodes but mesh has {n_nodes}"
            )
            self.b_list.append(b)
            self.gt_list.append(gt)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int):
        return {
            "b": self.b_list[idx],
            "gt": self.gt_list[idx],
        }
