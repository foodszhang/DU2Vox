from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from du2vox.physics.voxel_resolved_forward import (
    CoarseExtensionOperator,
    SolverContract,
    VoxelResolvedLinearOperator,
    build_refined_sampling_matrix,
    estimate_complement_diagonal,
    refine_tetra_mesh,
)


def _two_tet_mesh():
    nodes = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
        dtype=np.float64,
    )
    elements = np.asarray([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    labels = np.asarray([4, 9])
    # Six exterior faces; shared [1,2,3] is intentionally absent.
    faces = np.asarray(
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
    )
    return nodes, elements, labels, faces


def test_edge_midpoint_refinement_is_conforming_and_preserves_contracts():
    nodes, elements, labels, faces = _two_tet_mesh()
    refined = refine_tetra_mesh(nodes, elements, labels, faces, np.asarray([0, 2, 4]))
    assert len(refined.nodes) == len(nodes) + 9  # 6 + 6 edges, 3 shared
    assert refined.elements.shape == (16, 4)
    assert refined.surface_faces.shape == (24, 3)
    np.testing.assert_array_equal(refined.detector_node_indices, [0, 2, 4])
    np.testing.assert_array_equal(refined.tissue_labels[:8], np.full(8, 4))
    np.testing.assert_array_equal(refined.tissue_labels[8:], np.full(8, 9))
    determinants = np.linalg.det(
        np.transpose(
            refined.nodes[refined.elements[:, 1:]]
            - refined.nodes[refined.elements[:, :1]],
            (0, 2, 1),
        )
    )
    assert np.all(determinants > 0)
    # Every child face on the shared coarse face occurs twice.
    shared_plane_faces = []
    for child in refined.elements:
        for face in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
            ids = child[list(face)]
            if np.allclose(refined.nodes[ids].sum(axis=1), 1.0):
                shared_plane_faces.append(tuple(sorted(ids)))
    counts = {face: shared_plane_faces.count(face) for face in set(shared_plane_faces)}
    assert len(counts) == 4
    assert set(counts.values()) == {2}


def test_refined_sampling_rows_and_linear_identity():
    nodes = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    elements = np.asarray([[0, 1, 2, 3]])
    faces = np.asarray([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    refined = refine_tetra_mesh(nodes, elements, np.asarray([1]), faces, np.asarray([0, 1]))
    coords = np.asarray([[0.1, 0.1, 0.1], [0.4, 0.2, 0.2], [0.25, 0.25, 0.25]])
    bary = np.column_stack([1 - coords.sum(axis=1), coords])
    coarse_p = sp.csr_matrix(bary)
    fine_p = build_refined_sampling_matrix(coarse_p, elements, coords, refined, chunk_size=2)
    np.testing.assert_allclose(np.asarray(fine_p.sum(axis=1)).ravel(), 1.0, atol=1e-12)
    affine = 2.0 + refined.nodes @ np.asarray([3.0, -2.0, 0.5])
    expected = 2.0 + coords @ np.asarray([3.0, -2.0, 0.5])
    np.testing.assert_allclose(fine_p @ affine, expected, atol=1e-12)


def test_matrix_free_forward_adjoint_and_complement_match_explicit_matrix():
    m = sp.csr_matrix(np.asarray([[4.0, -1.0, 0.0], [-1.0, 3.0, -0.5], [0.0, -0.5, 2.0]]))
    p = sp.csr_matrix(np.asarray([[1.0, 0, 0], [0.5, 0.5, 0], [0, 0.4, 0.6], [0, 0, 1.0]]))
    detectors = np.asarray([0, 2])
    projection = p @ np.linalg.inv((p.T @ p).toarray()) @ p.T.toarray()
    q_matrix = np.eye(4) - projection
    operator = VoxelResolvedLinearOperator(
        m, p, detectors, voxel_weight=0.125,
        complement=lambda value: q_matrix @ value,
        solver=SolverContract(rtol=1e-12, maxiter=100),
    )
    selector = np.eye(3)[detectors]
    explicit = selector @ np.linalg.inv(m.toarray()) @ p.T.toarray() * 0.125
    rng = np.random.default_rng(17)
    x = rng.standard_normal(4)
    y = rng.standard_normal(2)
    np.testing.assert_allclose(operator.forward(x), explicit @ x, atol=1e-11)
    np.testing.assert_allclose(operator.adjoint(y), explicit.T @ y, atol=1e-11)
    np.testing.assert_allclose(operator.complement_forward(x), explicit @ q_matrix @ x, atol=1e-11)
    assert operator.dot_product_error() <= 1e-11
    diagonal, summary = estimate_complement_diagonal(operator, n_probes=256)
    expected_diagonal = np.diag(q_matrix @ explicit.T @ explicit @ q_matrix)
    np.testing.assert_allclose(diagonal, expected_diagonal, rtol=0.2, atol=1e-12)
    assert 0 <= summary["frobenius_ratio_estimate"] <= 1.0 + 1e-12


class _FakeCanonical:
    def __init__(self, p: np.ndarray):
        self.p = sp.csr_matrix(p)
        self.projector = object()

    def prolong(self, coefficients):
        return np.asarray(self.p @ coefficients).ravel()

    def project_coefficients(self, values):
        return np.linalg.solve((self.p.T @ self.p).toarray(), self.p.T @ values)


def test_d0_extension_annihilates_canonical_complement():
    p = np.asarray([[1, 0], [0.5, 0.5], [0, 1.0]], dtype=float)
    canonical = _FakeCanonical(p)
    a = np.asarray([[2.0, 1.0], [-1.0, 3.0]])
    b0 = CoarseExtensionOperator(a, canonical)
    rng = np.random.default_rng(5)
    value = rng.standard_normal(3)
    np.testing.assert_allclose(b0.complement_forward(value), 0.0, atol=1e-13)
    assert b0.coarse_action_error(rng.standard_normal(2)) < 1e-13
