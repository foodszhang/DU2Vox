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

    To avoid duplicating shared assets in memory when creating multiple
    dataset instances (e.g. train + val), pass an existing dataset's
    shared assets via the `shared` argument:
        val_set = FMTSimGenDataset(..., shared=train_set)
    """

    def __init__(
        self,
        shared_dir: str | Path | None,
        samples_dir: str | Path,
        split_file: str | Path,
        normalize_b: bool = True,
        normalize_gt: bool = True,
        normalize_gt_mode: str = "per_sample",
        binarize_gt: bool = False,
        binarize_threshold: float = 0.05,
        shared: "FMTSimGenDataset | None" = None,
    ):
        samples_dir = Path(samples_dir)
        split_file = Path(split_file)

        if shared is not None:
            # Reuse shared assets from another dataset instance (avoids duplication)
            self.nodes = shared.nodes
            self.A = shared.A
            self.n_surface = shared.n_surface
            self.visible_mask = shared.visible_mask
            self.L = shared.L
            self.L0 = shared.L0
            self.L1 = shared.L1
            self.L2 = shared.L2
            self.L3 = shared.L3
            self.knn_idx = shared.knn_idx
            self.sens_w = shared.sens_w
        else:
            shared_dir = Path(shared_dir)
            self._load_shared_assets(shared_dir)

        # Store preprocessing options
        self.normalize_b = normalize_b
        self.normalize_gt = normalize_gt
        self.normalize_gt_mode = normalize_gt_mode
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

            b = torch.tensor(np.load(b_path), dtype=torch.float32).unsqueeze(-1)
            if self.normalize_b:
                b_max = b.max()
                if b_max > 1e-8:
                    b = b / b_max

            gt = np.load(gt_path)
            if self.binarize_gt:
                gt = (gt > self.binarize_threshold).astype(np.float32)
            elif self.normalize_gt:
                gt = torch.tensor(gt, dtype=torch.float32).unsqueeze(-1)
                self.gt_list.append(gt)
                self.b_list.append(b)
                continue
            gt = torch.tensor(gt, dtype=torch.float32).unsqueeze(-1)

            assert b.shape[0] == self.n_surface
            assert gt.shape[0] == self.nodes.shape[0]
            self.b_list.append(b)
            self.gt_list.append(gt)

        if self.normalize_gt and self.normalize_gt_mode == "global":
            gt_maxes = [gt.max().item() for gt in self.gt_list]
            self.global_gt_max = float(np.percentile(gt_maxes, 99))
            for i in range(len(self.gt_list)):
                gt = self.gt_list[i]
                if self.global_gt_max > 1e-8:
                    gt = gt / self.global_gt_max
                gt = torch.clamp(gt, min=0.0, max=1.0)
                self.gt_list[i] = gt
        elif self.normalize_gt:
            for i in range(len(self.gt_list)):
                gt = self.gt_list[i]
                gt_max = gt.max()
                if gt_max > 1e-8:
                    gt = gt / gt_max
                gt = torch.clamp(gt, min=0.0, max=1.0)
                self.gt_list[i] = gt

    def _load_shared_assets(self, shared_dir: Path):
        """Load shared assets from disk (called only when shared=None)."""
        mesh = np.load(shared_dir / "mesh.npz")
        self.nodes = torch.tensor(mesh["nodes"], dtype=torch.float32)
        n_surface_full = len(mesh["surface_node_indices"])

        A_path = shared_dir / "system_matrix.A.npz"
        A_data = np.load(A_path, allow_pickle=True)
        if "forward_matrix" in A_data:
            A_arr = A_data["forward_matrix"]
        else:
            A_arr = A_data["arr_0"] if "arr_0" in A_data else scipy.sparse.load_npz(A_path).toarray()
        self.A = torch.tensor(A_arr, dtype=torch.float32)
        self.n_surface = A_arr.shape[0]
        self.visible_mask = None

        self.L = load_npz_as_torch_sparse(shared_dir / "graph_laplacian_full.Lap.npz")
        self.L0 = load_npz_as_torch_sparse(shared_dir / "graph_laplacian_full.n_Lap0.npz")
        self.L1 = load_npz_as_torch_sparse(shared_dir / "graph_laplacian_full.n_Lap1.npz")
        self.L2 = load_npz_as_torch_sparse(shared_dir / "graph_laplacian_full.n_Lap2.npz")
        self.L3 = load_npz_as_torch_sparse(shared_dir / "graph_laplacian_full.n_Lap3.npz")

        self.knn_idx = torch.tensor(
            np.load(shared_dir / "knn_idx_full.npy"),
            dtype=torch.long,
        )

        self.sens_w = torch.norm(self.A, dim=0)
        self.sens_w = self.sens_w / (self.sens_w.max() + 1e-8)

        print(f"  A: {A_arr.shape[0]} x {A_arr.shape[1]} (full surface, no visible_mask crop)")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int):
        return {
            "b": self.b_list[idx],
            "gt": self.gt_list[idx],
        }
