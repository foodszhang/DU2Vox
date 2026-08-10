"""Second-order four-point quadrature for linear tetrahedra."""

from __future__ import annotations

import numpy as np


TETRA_QUADRATURE_A = 0.5854101966249685
TETRA_QUADRATURE_B = 0.1381966011250105
TETRA_QUADRATURE_BARYCENTRIC = np.asarray(
    [
        [TETRA_QUADRATURE_A, TETRA_QUADRATURE_B, TETRA_QUADRATURE_B, TETRA_QUADRATURE_B],
        [TETRA_QUADRATURE_B, TETRA_QUADRATURE_A, TETRA_QUADRATURE_B, TETRA_QUADRATURE_B],
        [TETRA_QUADRATURE_B, TETRA_QUADRATURE_B, TETRA_QUADRATURE_A, TETRA_QUADRATURE_B],
        [TETRA_QUADRATURE_B, TETRA_QUADRATURE_B, TETRA_QUADRATURE_B, TETRA_QUADRATURE_A],
    ],
    dtype=np.float64,
)


def tetrahedron_volumes(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    vertices = np.asarray(nodes, dtype=np.float64)[np.asarray(elements, dtype=np.int64)]
    edges = np.stack(
        [vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0], vertices[:, 3] - vertices[:, 0]],
        axis=1,
    )
    return np.abs(np.linalg.det(edges)) / 6.0


def four_point_tetra_quadrature(
    nodes: np.ndarray,
    elements: np.ndarray,
    tet_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return world points, barycentric coordinates, and physical weights."""

    tet_ids = np.asarray(tet_ids, dtype=np.int64)
    vertices = np.asarray(nodes, dtype=np.float64)[np.asarray(elements, dtype=np.int64)[tet_ids]]
    barycentric = np.broadcast_to(
        TETRA_QUADRATURE_BARYCENTRIC[None, :, :], (len(tet_ids), 4, 4)
    ).copy()
    points = np.einsum("tqk,tkd->tqd", barycentric, vertices)
    volumes = tetrahedron_volumes(nodes, np.asarray(elements, dtype=np.int64)[tet_ids])
    weights = np.broadcast_to((volumes / 4.0)[:, None], (len(tet_ids), 4)).copy()
    return points, barycentric, weights


def local_p1_source_mass(volume: np.ndarray) -> np.ndarray:
    """Return the exact P1 local source mass matrices for tetra volumes."""

    volume = np.asarray(volume, dtype=np.float64).reshape(-1)
    template = np.ones((4, 4), dtype=np.float64) + np.eye(4, dtype=np.float64)
    return volume[:, None, None] * template[None, :, :] / 20.0
