"""Compressed diffusion Green operator derived from the generator FEM system."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from du2vox.physics.measurement_basis import MeasurementBasisResult
from du2vox.physics.measurement_basis import randomized_measurement_basis


@dataclass(frozen=True)
class CompressedGreenOperator:
    measurement_basis: np.ndarray
    singular_values: np.ndarray
    surface_index: np.ndarray
    green_node_modes: np.ndarray
    projected_forward_modes: np.ndarray
    rank: int
    visible_mask_applied: bool
    relative_operator_error: float
    max_absolute_error: float
    per_mode_relative_error: np.ndarray
    retained_energy_ratio: float

    def save(
        self,
        path: str | Path,
        *,
        source_matrix_shapes: dict[str, Any],
        file_metadata: dict[str, Any],
        frame_metadata: dict[str, Any],
        oversampling: int,
        seed: int,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            measurement_basis=self.measurement_basis.astype(np.float32),
            singular_values=self.singular_values.astype(np.float64),
            surface_index=self.surface_index.astype(np.int64),
            green_node_modes=self.green_node_modes.astype(np.float32),
            projected_forward_modes=self.projected_forward_modes.astype(np.float32),
            rank=np.int64(self.rank),
            oversampling=np.int64(oversampling),
            seed=np.int64(seed),
            visible_mask_applied=np.bool_(self.visible_mask_applied),
            relative_operator_error=np.float64(self.relative_operator_error),
            max_absolute_error=np.float64(self.max_absolute_error),
            per_mode_relative_error=self.per_mode_relative_error.astype(np.float64),
            retained_energy_ratio=np.float64(self.retained_energy_ratio),
            source_matrix_shapes_json=np.asarray(json.dumps(source_matrix_shapes, sort_keys=True)),
            file_metadata_json=np.asarray(json.dumps(file_metadata, sort_keys=True)),
            frame_metadata_json=np.asarray(json.dumps(frame_metadata, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CompressedGreenOperator":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                measurement_basis=data["measurement_basis"],
                singular_values=data["singular_values"],
                surface_index=data["surface_index"],
                green_node_modes=data["green_node_modes"],
                projected_forward_modes=(
                    data["projected_forward_modes"]
                    if "projected_forward_modes" in data
                    else np.empty((int(data["rank"]), 0), dtype=np.float32)
                ),
                rank=int(data["rank"]),
                visible_mask_applied=bool(data["visible_mask_applied"]),
                relative_operator_error=float(data["relative_operator_error"]),
                max_absolute_error=float(data["max_absolute_error"]),
                per_mode_relative_error=data["per_mode_relative_error"],
                retained_energy_ratio=float(data["retained_energy_ratio"]),
            )


def load_forward_matrix(path: str | Path) -> np.ndarray:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        if "forward_matrix" in data:
            return np.asarray(data["forward_matrix"])
        if {"data", "indices", "indptr", "shape"}.issubset(data.files):
            return sp.csr_matrix(
                (data["data"], data["indices"], data["indptr"]), shape=tuple(data["shape"])
            ).toarray()
        raise ValueError(f"Unsupported forward matrix archive keys at {path}: {data.files}")


def load_surface_index(shared_dir: str | Path) -> np.ndarray:
    path = Path(shared_dir) / "system_matrix.index.npz"
    with np.load(path, allow_pickle=False) as data:
        if "surface_index" not in data:
            raise KeyError(f"surface_index missing from {path}")
        return np.asarray(data["surface_index"], dtype=np.int64)


def _validate_dimensions(
    mass: sp.spmatrix,
    source: sp.spmatrix,
    forward: np.ndarray,
    surface_index: np.ndarray,
) -> None:
    if mass.shape[0] != mass.shape[1]:
        raise ValueError(f"M must be square, got {mass.shape}")
    if source.shape[0] != mass.shape[0]:
        raise ValueError(f"F rows {source.shape[0]} != M size {mass.shape[0]}")
    if forward.shape != (len(surface_index), source.shape[1]):
        raise ValueError(
            f"A shape {forward.shape} != (surface={len(surface_index)}, F cols={source.shape[1]})"
        )
    if np.any(surface_index < 0) or np.any(surface_index >= mass.shape[0]):
        raise ValueError("surface_index contains out-of-range FEM node indices")


def apply_surface_convention(
    forward: np.ndarray,
    surface_index: np.ndarray,
    visible_mask: np.ndarray | None,
    use_visible_mask: bool,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if not use_visible_mask:
        if forward.shape[0] != len(surface_index):
            raise ValueError("Full-surface A rows do not match surface_index")
        return forward, surface_index, False
    if visible_mask is None:
        raise FileNotFoundError("use_visible_mask=true but visible_mask.npy is unavailable")
    mask = np.asarray(visible_mask, dtype=bool).reshape(-1)
    if len(mask) != len(surface_index):
        raise ValueError(f"visible mask length {len(mask)} != surface count {len(surface_index)}")
    if forward.shape[0] == len(surface_index):
        forward = forward[mask]
    elif forward.shape[0] != int(mask.sum()):
        raise ValueError("A rows match neither full nor visible-only surface convention")
    return forward, surface_index[mask], True


def compute_green_node_modes(
    mass: sp.spmatrix,
    surface_index: np.ndarray,
    measurement_basis: np.ndarray,
) -> np.ndarray:
    """Solve ``M.T Z = C_s.T U_R`` and return ``Z.T`` in float64."""

    n_nodes = mass.shape[0]
    rhs = np.zeros((n_nodes, measurement_basis.shape[1]), dtype=np.float64)
    rhs[surface_index] = np.asarray(measurement_basis, dtype=np.float64)
    factor = splu(mass.T.tocsc().astype(np.float64))
    solution = factor.solve(rhs)
    return np.asarray(solution.T, dtype=np.float64)


def validate_compressed_operator(
    green_node_modes: np.ndarray,
    source: sp.spmatrix,
    measurement_basis: np.ndarray,
    forward: np.ndarray,
) -> dict[str, Any]:
    left = np.asarray(green_node_modes, dtype=np.float64) @ source.astype(np.float64)
    right = np.asarray(measurement_basis, dtype=np.float64).T @ np.asarray(forward, dtype=np.float64)
    difference = np.asarray(left) - right
    per_mode_denominator = np.linalg.norm(right, axis=1)
    per_mode = np.linalg.norm(difference, axis=1) / np.maximum(
        per_mode_denominator, np.finfo(np.float64).tiny
    )
    return {
        "relative_operator_error": float(
            np.linalg.norm(difference) / max(np.linalg.norm(right), np.finfo(np.float64).tiny)
        ),
        "max_absolute_error": float(np.max(np.abs(difference))),
        "per_mode_relative_error": per_mode,
    }


def build_compressed_green_operator(
    mass: sp.spmatrix,
    source: sp.spmatrix,
    forward: np.ndarray,
    surface_index: np.ndarray,
    *,
    rank: int = 64,
    oversampling: int = 16,
    seed: int = 20260722,
    power_iterations: int = 0,
    visible_mask_applied: bool = False,
) -> CompressedGreenOperator:
    _validate_dimensions(mass, source, forward, surface_index)
    basis_result: MeasurementBasisResult = randomized_measurement_basis(
        forward,
        rank=rank,
        oversampling=oversampling,
        seed=seed,
        power_iterations=power_iterations,
    )
    green_modes = compute_green_node_modes(mass, surface_index, basis_result.basis)
    projected_forward_modes = np.asarray(green_modes @ source.astype(np.float64))
    validation = validate_compressed_operator(green_modes, source, basis_result.basis, forward)
    return CompressedGreenOperator(
        measurement_basis=basis_result.basis,
        singular_values=basis_result.singular_values,
        surface_index=surface_index,
        green_node_modes=green_modes,
        projected_forward_modes=projected_forward_modes,
        rank=rank,
        visible_mask_applied=visible_mask_applied,
        relative_operator_error=validation["relative_operator_error"],
        max_absolute_error=validation["max_absolute_error"],
        per_mode_relative_error=validation["per_mode_relative_error"],
        retained_energy_ratio=basis_result.retained_energy_ratio,
    )
