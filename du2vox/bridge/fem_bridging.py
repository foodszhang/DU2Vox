"""
FEM bridge: BVH spatial index, barycentric coordinates, 8D prior features.

Uses a KDTree on tet centroids for fast candidate lookup, then validates
with exact barycentric coordinate computation.
"""

import numpy as np
from scipy.spatial import KDTree

import numba


@numba.njit(cache=True)
def _barycentric_batch(
    queries: np.ndarray,
    tet_vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Batch barycentric coordinate via np.linalg.solve.

    T[:, 0] = v1 - v0  (edge from v0 to v1)
    T[:, 1] = v2 - v0  (edge from v0 to v2)
    T[:, 2] = v3 - v0  (edge from v0 to v3)
    rhs     = p  - v0

    Returns
    -------
    bary  : [M, 4]  (λ0, λ1, λ2, λ3)
    inside: [M] bool
    """
    M = queries.shape[0]
    bary = np.zeros((M, 4), dtype=np.float64)
    inside = np.zeros(M, dtype=np.bool_)

    for i in numba.prange(M):
        p = queries[i, :]
        v0 = tet_vertices[i, 0, :]
        T = np.empty((3, 3), dtype=np.float64)
        T[:, 0] = tet_vertices[i, 1, :] - v0
        T[:, 1] = tet_vertices[i, 2, :] - v0
        T[:, 2] = tet_vertices[i, 3, :] - v0
        rhs = p - v0

        det = (
            T[0, 0] * (T[1, 1] * T[2, 2] - T[1, 2] * T[2, 1])
            - T[0, 1] * (T[1, 0] * T[2, 2] - T[1, 2] * T[2, 0])
            + T[0, 2] * (T[1, 0] * T[2, 1] - T[1, 1] * T[2, 0])
        )
        if abs(det) < 1e-12:
            continue

        lam123 = np.linalg.solve(T, rhs)
        l1, l2, l3 = lam123[0], lam123[1], lam123[2]
        l0 = 1.0 - l1 - l2 - l3

        tol = -1e-8
        if l0 >= tol and l1 >= tol and l2 >= tol and l3 >= tol:
            bary[i, 0] = l0
            bary[i, 1] = l1
            bary[i, 2] = l2
            bary[i, 3] = l3
            inside[i] = True

    return bary, inside


def barycentric_coords(
    p: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
    tol: float = -1e-6,
) -> tuple[np.ndarray, bool]:
    """
    Compute barycentric coordinates of point p in tetrahedron (v0,v1,v2,v3).

    Solves [v1-v0, v2-v0, v3-v0] @ [λ1, λ2, λ3] = p - v0,
    then λ0 = 1 - λ1 - λ2 - λ3.

    Returns:
        bary: [4] — (λ0, λ1, λ2, λ3)
        inside: bool — True if all λi >= tol (point inside tet)
    """
    T = np.stack([v1 - v0, v2 - v0, v3 - v0], axis=1)  # [3, 3]
    try:
        lam123 = np.linalg.solve(T, p - v0)
    except np.linalg.LinAlgError:
        return np.zeros(4), False
    lam0 = 1.0 - lam123.sum()
    bary = np.array([lam0, lam123[0], lam123[1], lam123[2]])
    inside = bool(np.all(bary >= tol))
    return bary, inside


def barycentric_coords_batch(
    ps: np.ndarray,
    v0s: np.ndarray,
    v1s: np.ndarray,
    v2s: np.ndarray,
    v3s: np.ndarray,
    tol: float = -1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Batched barycentric coordinate computation.

    Args:
        ps: [M, 3] query points
        v0s, v1s, v2s, v3s: [M, 3] tet vertices

    Returns:
        bary: [M, 4]
        inside: [M] bool
    """
    M = ps.shape[0]
    T = np.stack([v1s - v0s, v2s - v0s, v3s - v0s], axis=2)  # [M, 3, 3]
    rhs = ps - v0s  # [M, 3]

    lam123 = np.zeros((M, 3))
    for i in range(M):
        try:
            lam123[i] = np.linalg.solve(T[i], rhs[i])
        except np.linalg.LinAlgError:
            lam123[i] = np.array([-1.0, -1.0, -1.0])  # degenerate tet → outside

    lam0 = 1.0 - lam123.sum(axis=1, keepdims=True)  # [M, 1]
    bary = np.concatenate([lam0, lam123], axis=1)  # [M, 4]
    inside = np.all(bary >= tol, axis=1)
    return bary, inside


class FEMBridge:
    """
    BVH-like spatial index over FEM tetrahedra for point location and prior extraction.

    Uses a KDTree on tet centroids to find candidates, then validates
    containment with exact barycentric coordinate computation.

    Usage:
        bridge = FEMBridge(nodes, elements, roi_tet_indices)
        tet_indices, bary_coords = bridge.locate_points_batch(queries)
        prior_8d, valid_mask = bridge.get_prior_features(queries, coarse_d)
    """

    def __init__(
        self,
        nodes: np.ndarray,
        elements: np.ndarray,
        roi_tet_indices: np.ndarray | None = None,
        n_candidates: int = 16,
    ):
        """
        Args:
            nodes: [N_nodes, 3] node coordinates in mm.
            elements: [N_tets, 4] tetrahedral connectivity.
            roi_tet_indices: If given, build index over only these tets.
            n_candidates: Number of nearest centroids to check per query point.
        """
        self.nodes = nodes.astype(np.float64)
        self.elements = elements
        self.n_candidates = n_candidates

        if roi_tet_indices is not None:
            self.active_tet_indices = roi_tet_indices.astype(np.int64)
        else:
            self.active_tet_indices = np.arange(len(elements), dtype=np.int64)

        # Compute centroids of active tets
        active_verts = nodes[elements[self.active_tet_indices]]  # [M, 4, 3]
        self.centroids = active_verts.mean(axis=1)  # [M, 3]

        # Build KDTree on centroids
        self.kdtree = KDTree(self.centroids)

        # Precompute tet bounding boxes for fast rejection
        self._tet_verts = active_verts  # [M, 4, 3]

        # ROI tet set for O(1) membership testing
        if roi_tet_indices is not None:
            self._roi_set = set(roi_tet_indices.tolist())
        else:
            self._roi_set = None

    def locate_point(self, query: np.ndarray) -> tuple[int, np.ndarray | None]:
        """
        Locate a single query point in the indexed tetrahedra.

        Returns:
            (global_tet_idx, bary [4]) if found, or (-1, None) if not in any tet.
        """
        _, cand_local = self.kdtree.query(query, k=min(self.n_candidates, len(self.centroids)))
        if np.isscalar(cand_local):
            cand_local = [cand_local]

        for local_idx in cand_local:
            verts = self._tet_verts[local_idx]
            bary, inside = barycentric_coords(query, verts[0], verts[1], verts[2], verts[3])
            if inside:
                return int(self.active_tet_indices[local_idx]), bary

        return -1, None

    def locate_points_batch(
        self, queries: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Locate M query points in the indexed tetrahedra.

        Args:
            queries: [M, 3]

        Returns:
            tet_indices: [M] global tet indices, -1 if not found
            bary_coords: [M, 4] barycentric coordinates, zeros if not found
        """
        M = queries.shape[0]
        tet_indices = np.full(M, -1, dtype=np.int64)
        bary_coords = np.zeros((M, 4), dtype=np.float64)

        k = min(self.n_candidates, len(self.centroids))
        _, cand_locals = self.kdtree.query(queries, k=k)  # [M, k]

        if k == 1:
            cand_locals = cand_locals[:, np.newaxis]

        for i in range(M):
            for local_idx in cand_locals[i]:
                verts = self._tet_verts[local_idx]
                bary, inside = barycentric_coords(
                    queries[i], verts[0], verts[1], verts[2], verts[3]
                )
                if inside:
                    tet_indices[i] = int(self.active_tet_indices[local_idx])
                    bary_coords[i] = bary
                    break

        return tet_indices, bary_coords

    def get_prior_features(
        self,
        queries: np.ndarray,
        coarse_d: np.ndarray,
        K: int = 16,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract 8D prior features for M query points (numpy + numba accelerated).

        prior_8d[i] = [d_v0, d_v1, d_v2, d_v3, λ0, λ1, λ2, λ3]
        where v0..v3 are vertices of the containing tet and λi are
        barycentric coordinates.

        Args:
            queries: [M, 3] query points in mm.
            coarse_d: [N_nodes] Stage 1 coarse distribution values.
            K: Number of nearest centroid candidates per query point.

        Returns:
            prior_8d: [M, 8] — zeros for points outside ROI
            valid_mask: [M] bool — True if inside a ROI tet
        """
        M = queries.shape[0]
        queries = queries.astype(np.float64)
        prior_8d = np.zeros((M, 8), dtype=np.float64)
        valid = np.zeros(M, dtype=bool)
        remaining = np.ones(M, dtype=bool)

        k = min(K, len(self.centroids))
        _, knn = self.kdtree.query(queries, k=k)
        if k == 1:
            knn = knn[:, np.newaxis]

        max_candidates = 32
        cand_local = np.full((M, max_candidates), -1, dtype=np.int64)
        cand_count = np.zeros(M, dtype=np.int64)

        for i in range(M):
            seen: set[int] = set()
            cnt = 0
            for ki in range(k):
                local_idx = int(knn[i, ki])
                if local_idx in seen:
                    continue
                seen.add(local_idx)
                tet_global = int(self.active_tet_indices[local_idx])
                if self._roi_set is not None and tet_global not in self._roi_set:
                    continue
                cand_local[i, cnt] = local_idx
                cnt += 1
                if cnt >= max_candidates:
                    break
            cand_count[i] = cnt

        for slot in range(max_candidates):
            has_cand = (cand_count > slot) & remaining
            idx = np.where(has_cand)[0]
            if len(idx) == 0:
                break

            tet_local_slot = cand_local[idx, slot]
            verts = self._tet_verts[tet_local_slot]  # [len(idx), 4, 3]

            bary, inside = _barycentric_batch(queries[idx], verts)

            hit_local = np.where(inside)[0]
            if len(hit_local) == 0:
                continue

            global_idx = idx[hit_local]
            global_tet = self.active_tet_indices[tet_local_slot[hit_local]]
            node_ids = self.elements[global_tet]  # [H, 4]

            prior_8d[global_idx, :4] = coarse_d[node_ids]
            prior_8d[global_idx, 4:] = bary[hit_local]
            valid[global_idx] = True
            remaining[global_idx] = False

        return prior_8d.astype(np.float32), valid.astype(bool)


def compute_prior_cache(
    nodes: np.ndarray,
    elements: np.ndarray,
    coarse_d: np.ndarray,
    roi_tet_indices: np.ndarray,
    query_points: np.ndarray,
    n_candidates: int = 16,
) -> dict:
    """
    Compute and return prior features for a set of query points.

    Args:
        nodes: [N_nodes, 3]
        elements: [N_tets, 4]
        coarse_d: [N_nodes]
        roi_tet_indices: ROI tet subset
        query_points: [M, 3]
        n_candidates: KDTree search breadth

    Returns:
        dict with prior_8d, valid_mask, tet_indices, bary_coords
    """
    bridge = FEMBridge(nodes, elements, roi_tet_indices, n_candidates=n_candidates)
    tet_indices, bary_coords = bridge.locate_points_batch(query_points)
    prior_8d, valid_mask = bridge.get_prior_features(query_points, coarse_d)
    return {
        "prior_8d": prior_8d,
        "valid_mask": valid_mask,
        "tet_indices": tet_indices,
        "bary_coords": bary_coords,
    }
