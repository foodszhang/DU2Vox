"""Certified cross-discretization operators used by the 3000-case study.

This module deliberately reuses the experiment implementation and its cached sparse
matrix.  It does not rebuild, approximate, or learn either Pi_h or I_h.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path

import numpy as np
import torch

from experiments.cross_discretization_decomposition.decomposition import (
    DomainOperator,
    MassProjector,
    load_operator,
)


EXPECTED_DEFINITION = "0.2 mm GT voxel centers intersect complete FEM tetrahedral domain"
WEIGHTED_INNER_PRODUCT_SCOPE = (
    "fixed uniform 0.2-mm GT-center samples inside the certified FEM domain"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CanonicalCrossDiscretization:
    """Exact cached P1 prolongation and sampled L2 projection from the study."""

    def __init__(
        self,
        cache_dir: str | Path,
        shared_dir: str | Path | None = None,
        *,
        factorize: bool = False,
        allow_stale_frame_manifest: bool | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.operator: DomainOperator = load_operator(self.cache_dir)
        self.metadata = json.loads(
            (self.cache_dir / "operator_metadata.json").read_text()
        )
        if allow_stale_frame_manifest is None:
            # Follow the repo-wide convention from du2vox.utils.frame, so that
            # scripts which set DU2VOX_ALLOW_STALE_FRAME_MANIFEST from a config
            # flag get the same behavior here without threading the argument
            # through every construction site. Explicit True/False still wins.
            allow_stale_frame_manifest = (
                os.environ.get("DU2VOX_ALLOW_STALE_FRAME_MANIFEST", "0") == "1"
            )
        self._validate_metadata(
            Path(shared_dir) if shared_dir else None,
            allow_stale_frame_manifest=allow_stale_frame_manifest,
        )
        self.projector = MassProjector(self.operator) if factorize else None

    def _validate_metadata(
        self,
        shared_dir: Path | None,
        *,
        allow_stale_frame_manifest: bool = False,
    ) -> None:
        meta = self.metadata
        if meta.get("definition") != EXPECTED_DEFINITION:
            raise RuntimeError(f"Unexpected canonical domain: {meta.get('definition')}")
        expected = (
            int(meta["n_valid_voxels"]),
            int(meta["n_fem_nodes"]),
        )
        if self.operator.p.shape != expected:
            raise RuntimeError(
                f"Cached P shape {self.operator.p.shape} does not match {expected}"
            )
        if int(meta["p_nnz"]) != self.operator.p.nnz:
            raise RuntimeError("Cached P nnz does not match certified metadata")
        if len(self.operator.active_columns) != int(meta["n_active_columns"]):
            raise RuntimeError("Cached P active columns do not match metadata")
        if shared_dir is not None:
            assets = {
                "mesh_sha256": shared_dir / "mesh.npz",
            }
            # mesh.npz pins the discretization the cached P was built from, so
            # it is always checked. frame_manifest.json is rewritten on every
            # generation run and its hash moves when unrelated metadata (e.g.
            # the atlas path) changes, so callers comparing across cohorts may
            # opt out of that one check.
            if not allow_stale_frame_manifest:
                assets["frame_manifest_sha256"] = shared_dir / "frame_manifest.json"
            else:
                warnings.warn(
                    "frame_manifest_sha256 check skipped "
                    "(allow_stale_frame_manifest=True); mesh.npz is still verified"
                )
            for key, path in assets.items():
                actual = _sha256(path)
                if actual != meta.get(key):
                    raise RuntimeError(
                        f"{path} hash {actual} differs from certified {meta.get(key)}"
                    )

    @property
    def p(self):
        """Canonical I_h sampled at the fixed valid voxel centers."""

        return self.operator.p

    @property
    def quadrature_weight(self) -> float:
        """Uniform sampled-volume weight defining W = w I on this domain."""

        return float(self.operator.voxel_weight)

    @property
    def inner_product_scope(self) -> str:
        """Scope in which the sampled weighted-L2 statements are certified."""

        return WEIGHTED_INNER_PRODUCT_SCOPE

    @property
    def cache_hashes(self) -> dict[str, str]:
        """SHA256 identities of every file defining the cached operator."""

        names = (
            "P_full_fem_gt_centers.npz",
            "domain_arrays.npz",
            "operator_metadata.json",
        )
        return {name: _sha256(self.cache_dir / name) for name in names}

    def inner_product(self, first: np.ndarray, second: np.ndarray) -> float:
        """Return ``first.T W second`` for the certified uniform sampled domain."""

        lhs = np.asarray(first, dtype=np.float64).ravel()
        rhs = np.asarray(second, dtype=np.float64).ravel()
        if lhs.shape != (self.p.shape[0],) or rhs.shape != lhs.shape:
            raise ValueError(f"Expected two vectors of shape {(self.p.shape[0],)}")
        return float(self.quadrature_weight * np.dot(lhs, rhs))

    def weighted_energy(self, values: np.ndarray) -> float:
        """Return ``values.T W values`` on the certified sampled domain."""

        return self.inner_product(values, values)

    def prolong(self, coefficients: np.ndarray) -> np.ndarray:
        """Apply the certified analytic I_h (the cached sparse P matrix)."""

        return np.asarray(
            self.operator.p @ np.asarray(coefficients, dtype=np.float64)
        ).ravel()

    def project_coefficients(self, values: np.ndarray) -> np.ndarray:
        """Apply ``Pi_h=(P.T W P)^-1 P.T W`` on the fixed domain.

        Here ``W = quadrature_weight * I`` exactly, so the scalar weight cancels
        and the production sparse solve is ``(P.T P)^-1 P.T``.  This equivalence
        is certified only for the fixed, uniformly sampled domain.
        """

        if self.projector is None:
            raise RuntimeError("Projection factorization was not requested")
        return self.projector.coefficients(np.asarray(values, dtype=np.float64))

    def representation_target(
        self, gt_values: np.ndarray, pi_gt_coefficients: np.ndarray
    ) -> np.ndarray:
        """Return rho_gt - I_h Pi_h rho_gt; independent of inverse prediction."""

        gt = np.asarray(gt_values, dtype=np.float64).ravel()
        return gt - self.prolong(pi_gt_coefficients)


def canonical_p1_torch(
    nodal_values: torch.Tensor,
    node_indices: torch.Tensor,
    barycentric: torch.Tensor,
) -> torch.Tensor:
    """Parameter-free P1 interpolation for rows taken directly from cached P."""

    if nodal_values.ndim != 2:
        raise ValueError("nodal_values must have shape [B, N_fem]")
    if node_indices.ndim == 2:
        node_indices = node_indices.unsqueeze(0).expand(nodal_values.shape[0], -1, -1)
    if barycentric.ndim == 2:
        barycentric = barycentric.unsqueeze(0).expand(nodal_values.shape[0], -1, -1)
    gathered = torch.gather(
        nodal_values.unsqueeze(1).expand(-1, node_indices.shape[1], -1),
        2,
        node_indices.long(),
    )
    return torch.sum(gathered * barycentric.to(gathered.dtype), dim=-1)
