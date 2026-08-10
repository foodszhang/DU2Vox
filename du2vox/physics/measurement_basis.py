"""Low-rank measurement-space bases without an sklearn dependency."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MeasurementBasisResult:
    basis: np.ndarray
    singular_values: np.ndarray
    retained_energy_ratio: float
    frobenius_energy: float


def _frobenius_energy_chunked(matrix: np.ndarray, chunk_rows: int = 256) -> float:
    energy = 0.0
    for start in range(0, matrix.shape[0], chunk_rows):
        block = np.asarray(matrix[start : start + chunk_rows], dtype=np.float64)
        energy += float(np.einsum("ij,ij->", block, block))
    return energy


def _canonicalize_column_signs(basis: np.ndarray) -> np.ndarray:
    basis = basis.copy()
    pivots = np.argmax(np.abs(basis), axis=0)
    signs = np.sign(basis[pivots, np.arange(basis.shape[1])])
    signs[signs == 0] = 1
    basis *= signs[None, :]
    return basis


def randomized_measurement_basis(
    forward_matrix: np.ndarray,
    rank: int,
    oversampling: int = 16,
    seed: int = 20260722,
    power_iterations: int = 0,
) -> MeasurementBasisResult:
    """Approximate the leading left singular basis of ``forward_matrix``.

    Computation follows the randomized range-finder algorithm.  The small SVD
    is performed in float64; large matrix products retain the input precision.
    """

    matrix = np.asarray(forward_matrix)
    if matrix.ndim != 2:
        raise ValueError(f"forward_matrix must be 2-D, got {matrix.shape}")
    max_rank = min(matrix.shape)
    if not 0 < rank <= max_rank:
        raise ValueError(f"rank must be in [1, {max_rank}], got {rank}")
    sketch_rank = min(max_rank, rank + max(0, int(oversampling)))
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((matrix.shape[1], sketch_rank), dtype=np.float32)
    if matrix.dtype == np.float64:
        omega = omega.astype(np.float64)

    sketch = matrix @ omega
    for _ in range(max(0, int(power_iterations))):
        sketch, _ = np.linalg.qr(sketch, mode="reduced")
        sketch = matrix @ (matrix.T @ sketch)
    q_basis, _ = np.linalg.qr(sketch, mode="reduced")
    reduced = np.asarray(q_basis.T @ matrix, dtype=np.float64)
    small_u, singular_values, _ = np.linalg.svd(reduced, full_matrices=False)
    basis = np.asarray(q_basis @ small_u[:, :rank], dtype=np.float64)
    basis, _ = np.linalg.qr(basis, mode="reduced")
    basis = _canonicalize_column_signs(basis[:, :rank])
    singular_values = singular_values[:rank]
    total_energy = _frobenius_energy_chunked(matrix)
    retained = float(np.square(singular_values).sum() / max(total_energy, np.finfo(np.float64).tiny))
    return MeasurementBasisResult(
        basis=basis,
        singular_values=singular_values,
        retained_energy_ratio=retained,
        frobenius_energy=total_energy,
    )
