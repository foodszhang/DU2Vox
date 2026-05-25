from __future__ import annotations

from collections import defaultdict

import numpy as np


def robust_minmax(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values, dtype=np.float32)
    lo = float(np.percentile(values[finite], 1.0))
    hi = float(np.percentile(values[finite], 99.0))
    if hi - lo < eps:
        return np.zeros_like(values, dtype=np.float32)
    out = (values - lo) / (hi - lo + eps)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def compute_tet_gradients(
    nodes: np.ndarray,
    tets: np.ndarray,
    node_values: np.ndarray,
) -> dict[str, np.ndarray]:
    nodes = np.asarray(nodes, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    node_values = np.asarray(node_values, dtype=np.float64)

    tet_values = node_values[tets]
    tet_grad = np.zeros((len(tets), 3), dtype=np.float32)

    for tet_idx, tet in enumerate(tets):
        pts = nodes[tet]
        vals = tet_values[tet_idx]
        dmat = np.stack(
            [pts[1] - pts[0], pts[2] - pts[0], pts[3] - pts[0]],
            axis=1,
        )
        du = np.asarray([vals[1] - vals[0], vals[2] - vals[0], vals[3] - vals[0]], dtype=np.float64)
        try:
            tet_grad[tet_idx] = np.linalg.solve(dmat.T, du).astype(np.float32)
        except np.linalg.LinAlgError:
            tet_grad[tet_idx] = 0.0

    tet_max = tet_values.max(axis=1).astype(np.float32)
    tet_min = tet_values.min(axis=1).astype(np.float32)
    return {
        "tet_grad": tet_grad,
        "tet_grad_norm": np.linalg.norm(tet_grad, axis=1).astype(np.float32),
        "tet_mean": tet_values.mean(axis=1).astype(np.float32),
        "tet_max": tet_max,
        "tet_min": tet_min,
        "tet_range": (tet_max - tet_min).astype(np.float32),
        "tet_var": tet_values.var(axis=1).astype(np.float32),
    }


def build_tet_neighbors(tets: np.ndarray) -> list[list[int]]:
    tets = np.asarray(tets, dtype=np.int64)
    face_to_tets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    face_indices = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]
    for tet_idx, tet in enumerate(tets):
        for face in face_indices:
            key = tuple(sorted(int(tet[i]) for i in face))
            face_to_tets[key].append(tet_idx)

    neighbors: list[set[int]] = [set() for _ in range(len(tets))]
    for tet_list in face_to_tets.values():
        if len(tet_list) < 2:
            continue
        for tet_idx in tet_list:
            neighbors[tet_idx].update(other for other in tet_list if other != tet_idx)
    return [sorted(items) for items in neighbors]


def compute_grad_jump_score(tet_grad: np.ndarray, neighbors: list[list[int]]) -> np.ndarray:
    tet_grad = np.asarray(tet_grad, dtype=np.float32)
    jump = np.zeros(len(tet_grad), dtype=np.float32)
    for tet_idx, neigh in enumerate(neighbors):
        if not neigh:
            continue
        diff = tet_grad[tet_idx][None, :] - tet_grad[np.asarray(neigh, dtype=np.int64)]
        jump[tet_idx] = float(np.linalg.norm(diff, axis=1).mean())
    return robust_minmax(jump)


def compute_recovery_error_score(tet_grad: np.ndarray, neighbors: list[list[int]]) -> np.ndarray:
    tet_grad = np.asarray(tet_grad, dtype=np.float32)
    err = np.zeros(len(tet_grad), dtype=np.float32)
    for tet_idx, neigh in enumerate(neighbors):
        if neigh:
            idx = np.asarray([tet_idx, *neigh], dtype=np.int64)
            recovered = tet_grad[idx].mean(axis=0)
        else:
            recovered = tet_grad[tet_idx]
        err[tet_idx] = float(np.linalg.norm(tet_grad[tet_idx] - recovered))
    return robust_minmax(err)


def compute_transition_score(tet_value: np.ndarray, tau_core: float, tau_halo: float) -> np.ndarray:
    tet_value = np.asarray(tet_value, dtype=np.float32)
    if tau_core <= tau_halo:
        return ((tet_value >= tau_halo) & (tet_value < tau_core)).astype(np.float32)

    score = np.zeros_like(tet_value, dtype=np.float32)
    halo = (tet_value >= tau_halo) & (tet_value < tau_core)
    score[halo] = (tet_value[halo] - tau_halo) / (tau_core - tau_halo)
    core_side = tet_value >= tau_core
    score[core_side] = np.maximum(0.0, 1.0 - (tet_value[core_side] - tau_core) / (1.0 - tau_core + 1e-8))
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def compute_lifting_indicators(
    nodes: np.ndarray,
    tets: np.ndarray,
    node_values: np.ndarray,
    tau_core: float,
    tau_halo: float,
    weights: dict | None = None,
) -> dict[str, np.ndarray]:
    weights = weights or {}
    w_grad = float(weights.get("grad", 0.25))
    w_jump = float(weights.get("jump", 0.30))
    w_recovery = float(weights.get("recovery", 0.25))
    w_transition = float(weights.get("transition", 0.20))

    base = compute_tet_gradients(nodes, tets, node_values)
    neighbors = build_tet_neighbors(tets)
    tet_grad_norm = robust_minmax(base["tet_grad_norm"])
    grad_jump_score = compute_grad_jump_score(base["tet_grad"], neighbors)
    recovery_error_score = compute_recovery_error_score(base["tet_grad"], neighbors)
    transition_score = compute_transition_score(base["tet_max"], tau_core=tau_core, tau_halo=tau_halo)
    residual_indicator = robust_minmax(
        w_grad * tet_grad_norm
        + w_jump * grad_jump_score
        + w_recovery * recovery_error_score
        + w_transition * transition_score
    )

    return {
        "tet_grad_norm": tet_grad_norm.astype(np.float32),
        "grad_jump_score": grad_jump_score.astype(np.float32),
        "recovery_error_score": recovery_error_score.astype(np.float32),
        "transition_score": transition_score.astype(np.float32),
        "residual_indicator": residual_indicator.astype(np.float32),
        "tet_mean": base["tet_mean"].astype(np.float32),
        "tet_max": base["tet_max"].astype(np.float32),
        "tet_range": base["tet_range"].astype(np.float32),
        "tet_var": base["tet_var"].astype(np.float32),
    }
