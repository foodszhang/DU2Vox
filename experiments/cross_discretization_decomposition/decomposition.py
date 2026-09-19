"""Fixed-domain P1 projection and support-space oracle utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pyvista as pv
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import binary_erosion, label
from scipy.spatial import cKDTree

from du2vox.bridge.fem_bridging import FEMBridge


@dataclass(frozen=True)
class DomainOperator:
    p: sp.csr_matrix
    valid_flat_indices: np.ndarray
    coords_world: np.ndarray
    grid_shape: tuple[int, int, int]
    spacing_mm: float
    voxel_weight: float
    active_columns: np.ndarray


def gt_voxel_centers(frame) -> np.ndarray:
    """Return GT voxel centers in C-order (x, y, z; z fastest)."""
    axes = [
        frame.gt_offset_world_mm[i]
        + (np.arange(frame.gt_shape[i], dtype=np.float64) + 0.5)
        * frame.gt_spacing_mm
        for i in range(3)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([axis.ravel() for axis in mesh], axis=1)


def _barycentric(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    edge = np.stack(
        [vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0], vertices[:, 3] - vertices[:, 0]],
        axis=2,
    )
    lam123 = np.linalg.solve(edge, (points - vertices[:, 0])[..., None])[..., 0]
    return np.column_stack([1.0 - lam123.sum(axis=1), lam123])


def build_domain_operator(nodes: np.ndarray, elements: np.ndarray, frame) -> DomainOperator:
    """Build sparse P on all GT centers contained by the complete FEM mesh."""
    all_coords = gt_voxel_centers(frame)
    cells = np.column_stack(
        [np.full(len(elements), 4, dtype=np.int64), elements.astype(np.int64)]
    ).ravel()
    grid = pv.UnstructuredGrid(
        cells,
        np.full(len(elements), pv.CellType.TETRA, dtype=np.uint8),
        nodes,
    )
    cell_ids = grid.find_containing_cell(all_coords)
    valid_flat = np.flatnonzero(cell_ids >= 0).astype(np.int64)
    valid_cells = cell_ids[valid_flat].astype(np.int64)
    coords = all_coords[valid_flat]
    node_ids = elements[valid_cells].astype(np.int64)
    bary = _barycentric(coords, nodes[node_ids]).astype(np.float64)
    exact = (bary.min(axis=1) >= -1e-6) & (bary.max(axis=1) <= 1.0 + 1e-6)
    if not exact.all():
        retry = FEMBridge(nodes, elements, roi_tet_indices=None, n_candidates=32)
        retry_cells, retry_bary = retry.locate_points_batch(coords[~exact])
        recovered = retry_cells >= 0
        bad_indices = np.flatnonzero(~exact)
        valid_cells[bad_indices[recovered]] = retry_cells[recovered]
        bary[bad_indices[recovered]] = retry_bary[recovered]
        keep = exact.copy()
        keep[bad_indices[recovered]] = True
        valid_flat = valid_flat[keep]
        valid_cells = valid_cells[keep]
        coords = coords[keep]
        bary = bary[keep]
        node_ids = elements[valid_cells].astype(np.int64)
    if not np.allclose(bary.sum(axis=1), 1.0, atol=2e-10):
        raise RuntimeError("Barycentric rows do not sum to one")
    if float(bary.min()) < -2e-6:
        raise RuntimeError(f"Unexpected negative barycentric coordinate: {bary.min()}")
    bary = np.clip(bary, 0.0, 1.0)
    bary /= bary.sum(axis=1, keepdims=True)
    bary[np.abs(bary) < 1e-14] = 0.0
    rows = np.repeat(np.arange(len(valid_flat), dtype=np.int64), 4)
    p = sp.coo_matrix(
        (bary.ravel(), (rows, node_ids.ravel())),
        shape=(len(valid_flat), len(nodes)),
        dtype=np.float64,
    ).tocsr()
    active = np.flatnonzero(np.asarray(p.getnnz(axis=0)).ravel() > 0).astype(np.int64)
    return DomainOperator(
        p=p,
        valid_flat_indices=valid_flat,
        coords_world=coords.astype(np.float32),
        grid_shape=tuple(int(v) for v in frame.gt_shape),
        spacing_mm=float(frame.gt_spacing_mm),
        voxel_weight=float(frame.gt_spacing_mm**3),
        active_columns=active,
    )


def save_operator(operator: DomainOperator, directory: Path, metadata: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    sp.save_npz(directory / "P_full_fem_gt_centers.npz", operator.p)
    np.savez_compressed(
        directory / "domain_arrays.npz",
        valid_flat_indices=operator.valid_flat_indices,
        coords_world=operator.coords_world,
        grid_shape=np.asarray(operator.grid_shape),
        active_columns=operator.active_columns,
        spacing_mm=np.asarray(operator.spacing_mm),
        voxel_weight=np.asarray(operator.voxel_weight),
    )
    (directory / "operator_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


def load_operator(directory: Path) -> DomainOperator:
    p = sp.load_npz(directory / "P_full_fem_gt_centers.npz").tocsr()
    with np.load(directory / "domain_arrays.npz") as data:
        return DomainOperator(
            p=p,
            valid_flat_indices=data["valid_flat_indices"],
            coords_world=data["coords_world"],
            grid_shape=tuple(int(v) for v in data["grid_shape"]),
            spacing_mm=float(data["spacing_mm"]),
            voxel_weight=float(data["voxel_weight"]),
            active_columns=data["active_columns"],
        )


class MassProjector:
    """Operator-form M-orthogonal projector; M is uniform voxel quadrature."""

    def __init__(self, operator: DomainOperator):
        self.operator = operator
        self.pa = operator.p[:, operator.active_columns].tocsc()
        gram = (self.pa.T @ self.pa).tocsc()
        self.gram = gram
        self.diag_min = float(gram.diagonal().min())
        self.diag_max = float(gram.diagonal().max())
        try:
            lu = spla.splu(gram, permc_spec="COLAMD")
            self.regularization = 0.0
        except RuntimeError:
            ridge = max(self.diag_max, 1.0) * 1e-12
            lu = spla.splu(gram + ridge * sp.eye(gram.shape[0], format="csc"))
            self.regularization = float(ridge)
        self._solve: Callable[[np.ndarray], np.ndarray] = lu.solve

    def coefficients(self, values: np.ndarray) -> np.ndarray:
        rhs = np.asarray(self.pa.T @ np.asarray(values, dtype=np.float64)).ravel()
        active_coeff = self._solve(rhs)
        coeff = np.zeros(self.operator.p.shape[1], dtype=np.float64)
        coeff[self.operator.active_columns] = active_coeff
        return coeff

    def project(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coeff = self.coefficients(values)
        return np.asarray(self.operator.p @ coeff).ravel(), coeff


def decompose(
    operator: DomainOperator,
    projector: MassProjector,
    gt: np.ndarray,
    coarse_d: np.ndarray,
) -> dict[str, np.ndarray | float]:
    gt = np.asarray(gt, dtype=np.float64).ravel()
    fem = np.asarray(operator.p @ np.asarray(coarse_d, dtype=np.float64)).ravel()
    oracle, oracle_coeff = projector.project(gt)
    total = gt - fem
    inv = oracle - fem
    rep = gt - oracle
    rep_corrected = fem + rep
    w = operator.voxel_weight
    def energy(value: np.ndarray) -> float:
        return float(w * np.dot(value, value))
    et, ei, er = energy(total), energy(inv), energy(rep)
    dot = float(w * np.dot(inv, rep))
    closure = total - inv - rep
    return {
        "gt": gt,
        "fem": fem,
        "oracle_inv": oracle,
        "oracle_coeff": oracle_coeff,
        "oracle_rep": rep_corrected,
        "e_total": total,
        "e_inv": inv,
        "e_rep": rep,
        "energy_total": et,
        "energy_inv": ei,
        "energy_rep": er,
        "ratio_inv": ei / et if et > 0 else np.nan,
        "ratio_rep": er / et if et > 0 else np.nan,
        "closure_rel": float(np.sqrt(energy(closure) / max(et, 1e-300))),
        "orthogonality_cos": dot / np.sqrt(max(ei * er, 1e-300)),
        "pythagorean_rel": abs(et - ei - er) / max(et, 1e-300),
    }


def binary_metrics(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred_bin = pred >= threshold
    gt_bin = gt >= threshold
    tp = float(np.count_nonzero(pred_bin & gt_bin))
    fp = float(np.count_nonzero(pred_bin & ~gt_bin))
    fn = float(np.count_nonzero(~pred_bin & gt_bin))
    union = float(np.count_nonzero(pred_bin | gt_bin))
    eps = 1e-12
    return {
        "dice": 2 * tp / (pred_bin.sum() + gt_bin.sum() + eps),
        "iou": tp / (union + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "volume_error": float((pred_bin.sum() - gt_bin.sum()) / max(gt_bin.sum(), 1)),
        "mse": float(np.mean((pred - gt) ** 2)),
        "mae": float(np.mean(np.abs(pred - gt))),
        "mass_error": float((pred.sum() - gt.sum()) / max(abs(gt.sum()), eps)),
        "peak_error": float(pred.max(initial=0.0) - gt.max(initial=0.0)),
    }


def surface_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    valid_flat: np.ndarray,
    shape: tuple[int, int, int],
    spacing: float,
    threshold: float = 0.5,
) -> dict[str, float]:
    full_pred = np.zeros(np.prod(shape), dtype=bool)
    full_gt = np.zeros(np.prod(shape), dtype=bool)
    full_pred[valid_flat] = pred >= threshold
    full_gt[valid_flat] = gt >= threshold
    pred3 = full_pred.reshape(shape)
    gt3 = full_gt.reshape(shape)
    ps = pred3 & ~binary_erosion(pred3)
    gs = gt3 & ~binary_erosion(gt3)
    pxyz = np.argwhere(ps).astype(np.float32) * spacing
    gxyz = np.argwhere(gs).astype(np.float32) * spacing
    diag = float(np.linalg.norm(np.asarray(shape) * spacing))
    if len(pxyz) == 0 or len(gxyz) == 0:
        return {"assd": diag, "hd95": diag}
    p_to_g = cKDTree(gxyz).query(pxyz, k=1)[0]
    g_to_p = cKDTree(pxyz).query(gxyz, k=1)[0]
    return {
        "assd": float((p_to_g.mean() + g_to_p.mean()) / 2),
        "hd95": float(max(np.percentile(p_to_g, 95), np.percentile(g_to_p, 95))),
    }


def component_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    valid_flat: np.ndarray,
    shape: tuple[int, int, int],
    spacing: float,
) -> dict[str, float]:
    arrays = []
    for values in (pred, gt):
        full = np.zeros(np.prod(shape), dtype=bool)
        full[valid_flat] = values >= 0.5
        arrays.append(full.reshape(shape))
    pred3, gt3 = arrays
    pl, pn = label(pred3)
    gl, gn = label(gt3)
    pred_centers = [np.argwhere(pl == i).mean(axis=0) * spacing for i in range(1, pn + 1) if np.count_nonzero(pl == i) >= 3]
    gt_centers = [np.argwhere(gl == i).mean(axis=0) * spacing for i in range(1, gn + 1) if np.count_nonzero(gl == i) >= 3]
    if not gt_centers:
        return {"component_recall": 1.0, "fp_components": float(len(pred_centers)), "localization_error": 0.0, "separation_success": 1.0}
    if not pred_centers:
        return {"component_recall": 0.0, "fp_components": 0.0, "localization_error": float(np.linalg.norm(np.asarray(shape) * spacing)), "separation_success": 0.0}
    dist = np.linalg.norm(np.asarray(gt_centers)[:, None] - np.asarray(pred_centers)[None], axis=-1)
    nearest = dist.argmin(axis=1)
    nearest_dist = dist[np.arange(len(gt_centers)), nearest]
    matched = nearest_dist <= 3.0
    fp = np.count_nonzero(dist.min(axis=0) > 3.0)
    return {
        "component_recall": float(matched.mean()),
        "fp_components": float(fp),
        "localization_error": float(nearest_dist.mean()),
        "separation_success": float(matched.all() and len(set(nearest.tolist())) == len(gt_centers)),
    }
