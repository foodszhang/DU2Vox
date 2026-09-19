"""Matrix-free voxel-resolved diffusion operators for the P0-C audit.

The operator in this module is deliberately diagnostic.  It does not replace the
canonical reconstruction forward matrix or either canonical cross-discretization
operator.  Its fine-FEM form is

    B_H = C_h M_H^{-1} P_H.T W_v,

where ``C_h`` samples the *unchanged coarse detector nodes*.  All arithmetic in
the operator and iterative solves is float64.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


Array = np.ndarray


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 identity of a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sparse_sha256(matrix: sp.spmatrix) -> str:
    """Hash a sparse matrix canonically (CSR shape, indptr, indices, data)."""

    csr = matrix.tocsr().astype(np.float64)
    csr.sort_indices()
    digest = hashlib.sha256()
    digest.update(np.asarray(csr.shape, dtype=np.int64).tobytes())
    for value in (csr.indptr, csr.indices, csr.data):
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class SolverContract:
    rtol: float = 1e-8
    maxiter: int = 2000
    preconditioner: str = "jacobi"
    dtype: str = "float64"

    def __post_init__(self) -> None:
        if self.rtol <= 0 or self.rtol > 1:
            raise ValueError("rtol must lie in (0, 1]")
        if self.maxiter < 1:
            raise ValueError("maxiter must be positive")
        if self.preconditioner != "jacobi" or self.dtype != "float64":
            raise ValueError("P0-C requires FP64 CG with a Jacobi preconditioner")


@dataclass
class SolveDiagnostic:
    transpose: bool
    iterations: int
    info: int
    relative_residual: float


class IterativeSolveError(RuntimeError):
    """Raised immediately when the preregistered solve contract is not met."""


class VoxelResolvedLinearOperator:
    """Apply a voxel-to-measurement forward model and its exact adjoint.

    Parameters
    ----------
    system_matrix:
        Symmetric diffusion matrix ``M_H``. Non-symmetric matrices are supported
        by using the transpose in ``adjoint``.
    sampling_matrix:
        Sparse fine P1 sampling matrix ``P_H`` with shape ``[V, N_H]``.
    detector_node_indices:
        Original coarse surface nodes. Refinement must retain their numbering.
    complement:
        Callable implementing the unchanged canonical ``Q`` on V-vectors.
    """

    def __init__(
        self,
        system_matrix: sp.spmatrix,
        sampling_matrix: sp.spmatrix,
        detector_node_indices: Array,
        *,
        voxel_weight: float,
        complement: Callable[[Array], Array] | None = None,
        solver: SolverContract | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        self.m = system_matrix.tocsr().astype(np.float64)
        self.p = sampling_matrix.tocsr().astype(np.float64)
        self.detectors = np.asarray(detector_node_indices, dtype=np.int64).ravel()
        self.voxel_weight = float(voxel_weight)
        self.complement = complement
        self.solver_contract = solver or SolverContract()
        self.provenance = dict(provenance or {})
        self.solve_diagnostics: list[SolveDiagnostic] = []
        if self.m.shape[0] != self.m.shape[1]:
            raise ValueError("system_matrix must be square")
        if self.p.shape[1] != self.m.shape[0]:
            raise ValueError("P_H columns must equal the number of FEM nodes")
        if self.detectors.size == 0 or np.any(self.detectors < 0):
            raise ValueError("detector_node_indices must be nonempty and nonnegative")
        if np.any(self.detectors >= self.m.shape[0]):
            raise ValueError("detector index is outside the FEM system")
        if not np.isfinite(self.voxel_weight) or self.voxel_weight <= 0:
            raise ValueError("voxel_weight must be positive and finite")
        diagonal = self.m.diagonal()
        if np.any(~np.isfinite(diagonal)) or np.any(diagonal <= 0):
            raise ValueError("Jacobi preconditioning requires a positive diagonal")
        inverse_diagonal = 1.0 / diagonal
        self._preconditioner = spla.LinearOperator(
            self.m.shape, matvec=lambda value: inverse_diagonal * value
        )

    @property
    def n_voxels(self) -> int:
        return self.p.shape[0]

    @property
    def n_measurements(self) -> int:
        return self.detectors.size

    def _solve(self, rhs: Array, *, transpose: bool) -> Array:
        rhs64 = np.asarray(rhs, dtype=np.float64).ravel()
        matrix = self.m.T.tocsr() if transpose else self.m
        iterations = 0

        def count(_: Array) -> None:
            nonlocal iterations
            iterations += 1

        solution, info = spla.cg(
            matrix,
            rhs64,
            rtol=self.solver_contract.rtol,
            atol=0.0,
            maxiter=self.solver_contract.maxiter,
            M=self._preconditioner,
            callback=count,
        )
        denominator = max(float(np.linalg.norm(rhs64)), np.finfo(np.float64).tiny)
        relative = float(np.linalg.norm(matrix @ solution - rhs64) / denominator)
        diagnostic = SolveDiagnostic(transpose, iterations, int(info), relative)
        self.solve_diagnostics.append(diagnostic)
        if info != 0 or relative > self.solver_contract.rtol:
            raise IterativeSolveError(
                f"CG failed contract: info={info}, iterations={iterations}, "
                f"relative_residual={relative:.3e}"
            )
        return solution

    def forward(self, voxel_values: Array) -> Array:
        values = np.asarray(voxel_values, dtype=np.float64).ravel()
        if values.shape != (self.n_voxels,):
            raise ValueError(f"Expected voxel vector {(self.n_voxels,)}")
        load = np.asarray(self.p.T @ (self.voxel_weight * values)).ravel()
        field = self._solve(load, transpose=False)
        return field[self.detectors]

    def adjoint(self, measurement: Array) -> Array:
        values = np.asarray(measurement, dtype=np.float64).ravel()
        if values.shape != (self.n_measurements,):
            raise ValueError(f"Expected measurement vector {(self.n_measurements,)}")
        load = np.zeros(self.m.shape[0], dtype=np.float64)
        np.add.at(load, self.detectors, values)
        field = self._solve(load, transpose=True)
        return self.voxel_weight * np.asarray(self.p @ field).ravel()

    def complement_forward(self, values: Array) -> Array:
        if self.complement is None:
            raise RuntimeError("No canonical complement action was supplied")
        return self.forward(self.complement(np.asarray(values, dtype=np.float64)))

    def complement_adjoint(self, measurement: Array) -> Array:
        if self.complement is None:
            raise RuntimeError("No canonical complement action was supplied")
        # Canonical Q is self-adjoint only on the certified uniform sampled domain.
        return self.complement(self.adjoint(measurement))

    def matched_score(self, residual: Array, diagonal_norm: Array) -> Array:
        diagonal = np.asarray(diagonal_norm, dtype=np.float64).ravel()
        if diagonal.shape != (self.n_voxels,):
            raise ValueError(f"Expected diagonal vector {(self.n_voxels,)}")
        numerator = self.complement_adjoint(residual)
        epsilon = np.finfo(np.float64).eps * max(float(diagonal.max(initial=0)), 1.0)
        return numerator / np.sqrt(np.maximum(diagonal, 0.0) + epsilon)

    def dot_product_error(self, seed: int = 20260907) -> float:
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(self.n_voxels)
        y = rng.standard_normal(self.n_measurements)
        lhs = float(np.dot(self.forward(x), y))
        rhs = float(np.dot(x, self.adjoint(y)))
        return abs(lhs - rhs) / max(abs(lhs), abs(rhs), np.finfo(float).tiny)

    def audit_metadata(self) -> dict[str, Any]:
        return {
            "definition": "B_H = C_h M_H^-1 P_H^T W_v (matrix-free)",
            "shapes": {
                "M_H": list(self.m.shape),
                "P_H": list(self.p.shape),
                "measurements": self.n_measurements,
            },
            "voxel_weight": self.voxel_weight,
            "solver": asdict(self.solver_contract),
            "hashes": {"M_H": sparse_sha256(self.m), "P_H": sparse_sha256(self.p)},
            "provenance": self.provenance,
            "solve_diagnostics": [asdict(item) for item in self.solve_diagnostics],
        }


class CoarseExtensionOperator:
    """Canonical D0 extension ``B_0=A Pi_h`` and its transpose action."""

    def __init__(self, a: Array, canonical: Any) -> None:
        self.a = np.asarray(a, dtype=np.float64)
        self.canonical = canonical
        if self.a.ndim != 2 or self.a.shape[1] != canonical.p.shape[1]:
            raise ValueError("A and canonical I_h have incompatible shapes")
        if canonical.projector is None:
            raise ValueError("Canonical operator must be constructed with factorize=True")

    def forward(self, values: Array) -> Array:
        return self.a @ self.canonical.project_coefficients(values)

    def complement_forward(self, values: Array) -> Array:
        q = np.asarray(values, dtype=np.float64) - self.canonical.prolong(
            self.canonical.project_coefficients(values)
        )
        return self.forward(q)

    def coarse_action_error(self, coefficients: Array) -> float:
        coefficients = np.asarray(coefficients, dtype=np.float64).ravel()
        expected = self.a @ coefficients
        actual = self.forward(self.canonical.prolong(coefficients))
        return float(
            np.linalg.norm(actual - expected)
            / max(np.linalg.norm(expected), np.finfo(float).tiny)
        )


@dataclass(frozen=True)
class RefinedTetraMesh:
    nodes: Array
    elements: Array
    tissue_labels: Array
    surface_faces: Array
    detector_node_indices: Array
    parent_elements: Array


def _unique_edges(elements: Array) -> tuple[Array, Array]:
    pairs = np.asarray([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
    edges = np.sort(np.asarray(elements, dtype=np.int64)[:, pairs], axis=2)
    unique, inverse = np.unique(edges.reshape(-1, 2), axis=0, return_inverse=True)
    return unique, inverse.reshape(len(elements), 6)


def refine_tetra_mesh(
    nodes: Array,
    elements: Array,
    tissue_labels: Array,
    surface_faces: Array,
    detector_node_indices: Array,
) -> RefinedTetraMesh:
    """Globally refine every tetrahedron into eight conforming children.

    Face subdivision is determined solely by shared global edge midpoints.  The
    remaining central octahedron is split along its shortest diagonal; that choice
    is internal to each parent and cannot break inter-parent conformity.
    """

    nodes = np.asarray(nodes, dtype=np.float64)
    elements = np.asarray(elements, dtype=np.int64)
    labels = np.asarray(tissue_labels)
    faces = np.asarray(surface_faces, dtype=np.int64)
    detectors = np.asarray(detector_node_indices, dtype=np.int64)
    if elements.ndim != 2 or elements.shape[1] != 4:
        raise ValueError("elements must have shape [T, 4]")
    if labels.shape != (len(elements),):
        raise ValueError("one tissue label is required per parent tetrahedron")
    edges, edge_inverse = _unique_edges(elements)
    edge_nodes = np.arange(len(nodes), len(nodes) + len(edges), dtype=np.int64)
    midpoint_ids = edge_nodes[edge_inverse]
    refined_nodes = np.vstack([nodes, 0.5 * (nodes[edges[:, 0]] + nodes[edges[:, 1]])])

    children: list[list[int]] = []
    parents: list[int] = []
    for parent, (tet, mids) in enumerate(zip(elements, midpoint_ids, strict=True)):
        v0, v1, v2, v3 = (int(value) for value in tet)
        m01, m02, m03, m12, m13, m23 = (int(value) for value in mids)
        local = [
            [v0, m01, m02, m03],
            [v1, m01, m12, m13],
            [v2, m02, m12, m23],
            [v3, m03, m13, m23],
        ]
        diagonals = [(m01, m23), (m02, m13), (m03, m12)]
        lengths = [np.linalg.norm(refined_nodes[a] - refined_nodes[b]) for a, b in diagonals]
        selected = int(np.argmin(lengths))
        rings = [
            [m02, m03, m13, m12],
            [m01, m03, m23, m12],
            [m01, m02, m23, m13],
        ]
        a, b = diagonals[selected]
        ring = rings[selected]
        local.extend([[a, b, ring[i], ring[(i + 1) % 4]] for i in range(4)])
        for child in local:
            xyz = refined_nodes[child]
            determinant = np.linalg.det((xyz[1:] - xyz[0]).T)
            if abs(determinant) <= 1e-14:
                raise RuntimeError(f"Degenerate refined tetrahedron in parent {parent}")
            if determinant < 0:
                child[2], child[3] = child[3], child[2]
            children.append(child)
            parents.append(parent)

    edge_lookup = {tuple(edge): int(mid) for edge, mid in zip(edges, edge_nodes, strict=True)}
    refined_faces: list[list[int]] = []
    for a, b, c in faces:
        mab = edge_lookup[tuple(sorted((int(a), int(b))))]
        mac = edge_lookup[tuple(sorted((int(a), int(c))))]
        mbc = edge_lookup[tuple(sorted((int(b), int(c))))]
        refined_faces.extend(
            [[int(a), mab, mac], [int(b), mbc, mab], [int(c), mac, mbc], [mab, mbc, mac]]
        )
    parent_array = np.asarray(parents, dtype=np.int64)
    return RefinedTetraMesh(
        nodes=refined_nodes,
        elements=np.asarray(children, dtype=np.int64),
        tissue_labels=labels[parent_array],
        surface_faces=np.asarray(refined_faces, dtype=np.int64),
        detector_node_indices=detectors.copy(),
        parent_elements=parent_array,
    )


def assemble_diffusion_system(
    mesh: RefinedTetraMesh,
    optical_by_label: Mapping[int, Mapping[str, float]],
    *,
    refractive_index: float = 1.37,
) -> sp.csr_matrix:
    """Assemble ``K+C+B`` using the FMT-SimGen linear-tetrahedron contract."""

    n_nodes = len(mesh.nodes)
    rows: list[Array] = []
    cols: list[Array] = []
    values: list[Array] = []
    local_rows = np.repeat(np.arange(4), 4)
    local_cols = np.tile(np.arange(4), 4)
    label_values: dict[int, tuple[float, float]] = {}
    for label in np.unique(mesh.tissue_labels):
        params = optical_by_label.get(int(label))
        if params is None:
            raise ValueError(f"No optical parameters for tissue label {int(label)}")
        mu_a = float(params["mu_a"])
        mu_sp = float(params.get("mu_s_prime", params.get("mu_sp", 0.0)))
        diffusion = float(params.get("_diffusion", np.nan))
        if not np.isfinite(diffusion):
            diffusion = (
                1e10
                if mu_a < 1e-10 and mu_sp < 1e-10
                else 1.0 / (3.0 * (mu_a + mu_sp))
            )
        label_values[int(label)] = (mu_a, diffusion)
    for start in range(0, len(mesh.elements), 20_000):
        tet = mesh.elements[start : start + 20_000]
        labels = mesh.tissue_labels[start : start + 20_000]
        mu_a = np.asarray([label_values[int(v)][0] for v in labels])
        diffusion = np.asarray([label_values[int(v)][1] for v in labels])
        xyz = mesh.nodes[tet]
        affine = np.concatenate([np.ones((len(tet), 4, 1)), xyz], axis=2)
        determinant = np.abs(np.linalg.det(affine))
        if np.any(determinant <= 1e-15):
            raise RuntimeError("Degenerate tetrahedron encountered during assembly")
        gradients = np.linalg.inv(affine)[:, 1:, :]
        gram = np.einsum("tki,tkj->tij", gradients, gradients)
        stiffness = diffusion[:, None, None] * (determinant[:, None, None] / 6.0) * gram
        mass_template = np.ones((4, 4)) + np.eye(4)
        mass = mu_a[:, None, None] * (determinant[:, None, None] / 120.0) * mass_template
        rows.append(tet[:, local_rows].ravel())
        cols.append(tet[:, local_cols].ravel())
        values.append((stiffness + mass).reshape(-1))
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    data = np.concatenate(values)
    system = sp.coo_matrix((data, (r, c)), shape=(n_nodes, n_nodes)).tocsr()

    reflection = (
        -1.4399 / refractive_index**2
        + 0.7099 / refractive_index
        + 0.6681
        + 0.0636 * refractive_index
    )
    an = (1.0 + reflection) / (1.0 - reflection)
    lr3 = np.repeat(np.arange(3), 3)
    lc3 = np.tile(np.arange(3), 3)
    tri = mesh.surface_faces
    xyz = mesh.nodes[tri]
    double_area = np.linalg.norm(
        np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1
    )
    boundary = (
        (double_area[:, None, None] / (48.0 * an))
        * (np.ones((3, 3)) + np.eye(3))
    )
    boundary_matrix = sp.coo_matrix(
        (boundary.reshape(-1), (tri[:, lr3].ravel(), tri[:, lc3].ravel())),
        shape=(n_nodes, n_nodes),
    ).tocsr()
    return (system + boundary_matrix).tocsr()


def infer_piecewise_optical_coefficients(
    nodes: Array,
    elements: Array,
    tissue_labels: Array,
    stiffness_matrix: sp.spmatrix,
    absorption_matrix: sp.spmatrix,
) -> tuple[dict[int, dict[str, float]], dict[str, float]]:
    """Infer per-label ``D`` and ``mu_a`` from the saved coarse FEM matrices.

    The saved matrices, rather than a possibly stale YAML copy, are the numerical
    authority for a *matched* refinement.  Both matrices depend linearly on the
    piecewise-constant coefficients, so a small Frobenius least-squares system
    recovers them without changing the forward model.
    """

    nodes = np.asarray(nodes, dtype=np.float64)
    elements = np.asarray(elements, dtype=np.int64)
    labels = np.asarray(tissue_labels)
    unique = [int(value) for value in np.unique(labels)]
    n_nodes = len(nodes)
    lr = np.repeat(np.arange(4), 4)
    lc = np.tile(np.arange(4), 4)
    mass_template = np.ones((4, 4)) + np.eye(4)
    stiffness_bases: list[sp.csr_matrix] = []
    absorption_bases: list[sp.csr_matrix] = []
    for label in unique:
        selected = elements[labels == label]
        row_blocks: list[Array] = []
        col_blocks: list[Array] = []
        k_blocks: list[Array] = []
        c_blocks: list[Array] = []
        for start in range(0, len(selected), 20_000):
            tet = selected[start : start + 20_000]
            xyz = nodes[tet]
            affine = np.concatenate([np.ones((len(tet), 4, 1)), xyz], axis=2)
            determinant = np.abs(np.linalg.det(affine))
            gradients = np.linalg.inv(affine)[:, 1:, :]
            gram = np.einsum("tki,tkj->tij", gradients, gradients)
            row_blocks.append(tet[:, lr].ravel())
            col_blocks.append(tet[:, lc].ravel())
            k_blocks.append((determinant[:, None, None] / 6.0 * gram).reshape(-1))
            c_blocks.append(
                (determinant[:, None, None] / 120.0 * mass_template).reshape(-1)
            )
        rows = np.concatenate(row_blocks)
        cols = np.concatenate(col_blocks)
        stiffness_bases.append(
            sp.coo_matrix((np.concatenate(k_blocks), (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
        )
        absorption_bases.append(
            sp.coo_matrix((np.concatenate(c_blocks), (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
        )

    def fit(bases: list[sp.csr_matrix], target: sp.spmatrix) -> tuple[Array, float]:
        gram = np.empty((len(bases), len(bases)), dtype=np.float64)
        rhs = np.empty(len(bases), dtype=np.float64)
        target = target.tocsr().astype(np.float64)
        for i, first in enumerate(bases):
            rhs[i] = float(first.multiply(target).sum())
            for j, second in enumerate(bases):
                gram[i, j] = float(first.multiply(second).sum())
        coefficients = np.linalg.solve(gram, rhs)
        reconstructed = sum(
            (coefficient * basis for coefficient, basis in zip(coefficients, bases, strict=True)),
            start=sp.csr_matrix(target.shape),
        )
        relative = float(
            spla.norm(reconstructed - target) / max(spla.norm(target), np.finfo(float).tiny)
        )
        return coefficients, relative

    diffusion, stiffness_error = fit(stiffness_bases, stiffness_matrix)
    mu_a, absorption_error = fit(absorption_bases, absorption_matrix)
    if np.any(diffusion <= 0) or np.any(mu_a < -1e-12):
        raise RuntimeError("Saved FEM matrices imply nonphysical optical coefficients")
    inferred = {
        label: {"mu_a": float(max(mu_a[i], 0.0)), "_diffusion": float(diffusion[i])}
        for i, label in enumerate(unique)
    }
    return inferred, {
        "stiffness_fit_relative_error": stiffness_error,
        "absorption_fit_relative_error": absorption_error,
    }


def build_refined_sampling_matrix(
    coarse_p: sp.spmatrix,
    coarse_elements: Array,
    voxel_coords: Array,
    refined_mesh: RefinedTetraMesh,
    *,
    tolerance: float = 2e-5,
    chunk_size: int = 100_000,
) -> sp.csr_matrix:
    """Build fine P1 sampling without re-locating points in the full fine mesh.

    The certified coarse ``P`` identifies the parent tetrahedron for every voxel
    row. Each row must retain its four structural entries. Points are then tested
    only against that parent's eight children.
    """

    coarse = coarse_p.tocsr().astype(np.float64)
    elements = np.asarray(coarse_elements, dtype=np.int64)
    coords = np.asarray(voxel_coords, dtype=np.float64)
    if coarse.shape[0] != len(coords):
        raise ValueError("coarse P and voxel coordinates have different row counts")
    row_counts = np.diff(coarse.indptr)
    if not np.all(row_counts == 4):
        raise RuntimeError("Certified coarse P must retain four entries per voxel row")
    parent_lookup = {tuple(sorted(map(int, tet))): index for index, tet in enumerate(elements)}
    parent_ids = np.empty(len(coords), dtype=np.int64)
    for row in range(len(coords)):
        key = tuple(sorted(map(int, coarse.indices[coarse.indptr[row] : coarse.indptr[row + 1]])))
        try:
            parent_ids[row] = parent_lookup[key]
        except KeyError as error:
            raise RuntimeError(f"Could not identify coarse parent for voxel row {row}") from error

    out_indices = np.empty((len(coords), 4), dtype=np.int64)
    out_values = np.empty((len(coords), 4), dtype=np.float64)
    for start in range(0, len(coords), chunk_size):
        stop = min(start + chunk_size, len(coords))
        points = coords[start:stop]
        parents = parent_ids[start:stop]
        assigned = np.zeros(len(points), dtype=bool)
        for child_number in range(8):
            pending = np.flatnonzero(~assigned)
            if pending.size == 0:
                break
            child_ids = refined_mesh.elements[parents[pending] * 8 + child_number]
            xyz = refined_mesh.nodes[child_ids]
            edge = np.transpose(xyz[:, 1:] - xyz[:, :1], (0, 2, 1))
            rhs = (points[pending] - xyz[:, 0])[..., None]
            lam123 = np.linalg.solve(edge, rhs)[..., 0]
            bary = np.column_stack([1.0 - lam123.sum(axis=1), lam123])
            inside = np.all(bary >= -tolerance, axis=1) & np.all(
                bary <= 1.0 + tolerance, axis=1
            )
            selected = pending[inside]
            if selected.size:
                selected_bary = np.clip(bary[inside], 0.0, 1.0)
                selected_bary /= selected_bary.sum(axis=1, keepdims=True)
                out_indices[start + selected] = child_ids[inside]
                out_values[start + selected] = selected_bary
                assigned[selected] = True
        if not assigned.all():
            raise RuntimeError(
                f"Fine P1 location failed for {np.count_nonzero(~assigned)} rows in "
                f"chunk [{start}, {stop})"
            )
    rows = np.repeat(np.arange(len(coords), dtype=np.int64), 4)
    result = sp.coo_matrix(
        (out_values.ravel(), (rows, out_indices.ravel())),
        shape=(len(coords), len(refined_mesh.nodes)),
    ).tocsr()
    if not np.allclose(np.asarray(result.sum(axis=1)).ravel(), 1.0, atol=2e-10):
        raise RuntimeError("Fine P1 rows do not sum to one")
    return result


def estimate_complement_diagonal(
    operator: VoxelResolvedLinearOperator,
    *,
    n_probes: int = 64,
    seed: int = 20260907,
    inverse_noise_std: float = 1.0,
) -> tuple[Array, dict[str, float]]:
    """Estimate ``diag(Q B.T C^-1 B Q)`` with measurement Rademacher probes."""

    if n_probes < 1 or inverse_noise_std <= 0:
        raise ValueError("n_probes and inverse_noise_std must be positive")
    rng = np.random.default_rng(seed)
    diagonal = np.zeros(operator.n_voxels, dtype=np.float64)
    complement_energy = 0.0
    full_energy = 0.0
    for _ in range(n_probes):
        probe = rng.choice(np.asarray([-1.0, 1.0]), size=operator.n_measurements)
        full = operator.adjoint(probe) / inverse_noise_std
        comp = operator.complement(full)
        diagonal += comp * comp
        full_energy += float(np.dot(full, full))
        complement_energy += float(np.dot(comp, comp))
    diagonal /= n_probes
    ratio = np.sqrt(complement_energy / max(full_energy, np.finfo(float).tiny))
    return diagonal, {"n_probes": n_probes, "frobenius_ratio_estimate": float(ratio)}


def randomized_complement_spectrum(
    operator: VoxelResolvedLinearOperator,
    *,
    rank: int = 64,
    oversampling: int = 16,
    power_iterations: int = 2,
    seed: int = 20260907,
    inverse_noise_std: float = 1.0,
) -> dict[str, Any]:
    """Randomized spectrum of ``C_y^-1/2 B_H Q`` in measurement space.

    Working with ``(B_H Q)(B_H Q).T`` keeps the dense subspace in the 7,413-D
    measurement space rather than allocating a voxel-by-80 matrix.
    """

    if rank < 1 or oversampling < 0 or power_iterations < 0:
        raise ValueError("rank must be positive; oversampling/iterations nonnegative")
    width = min(rank + oversampling, operator.n_measurements)
    rng = np.random.default_rng(seed)
    basis = rng.standard_normal((operator.n_measurements, width))

    def covariance_action(values: Array) -> Array:
        result = np.empty_like(values, dtype=np.float64)
        for column in range(values.shape[1]):
            adjoint = operator.complement_adjoint(values[:, column]) / inverse_noise_std
            result[:, column] = operator.complement_forward(adjoint) / inverse_noise_std
        return result

    basis, _ = np.linalg.qr(covariance_action(basis), mode="reduced")
    for _ in range(power_iterations):
        basis, _ = np.linalg.qr(covariance_action(basis), mode="reduced")
    projected = basis.T @ covariance_action(basis)
    eigenvalues = np.linalg.eigvalsh(0.5 * (projected + projected.T))[::-1]
    singular_values = np.sqrt(np.maximum(eigenvalues[:rank], 0.0))
    energy = singular_values**2
    total = max(float(energy.sum()), np.finfo(float).tiny)
    cumulative = {
        str(level): float(energy[: min(level, len(energy))].sum() / total)
        for level in (8, 16, 32, 64)
    }
    return {
        "rank": rank,
        "oversampling": oversampling,
        "power_iterations": power_iterations,
        "singular_values": singular_values.tolist(),
        "captured_energy_cumulative": cumulative,
        "note": "fractions are relative to energy captured by the returned rank",
    }


def save_refined_mesh(mesh: RefinedTetraMesh, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **{key: value for key, value in asdict(mesh).items()})


def load_refined_mesh(path: str | Path) -> RefinedTetraMesh:
    with np.load(path) as data:
        return RefinedTetraMesh(**{key: data[key] for key in RefinedTetraMesh.__annotations__})


def write_metadata(metadata: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(metadata), indent=2, allow_nan=True) + "\n")
