from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class QueryRole(IntEnum):
    BG = 0
    CORE = 1
    HALO = 2
    SENTINEL = 3


@dataclass
class CoverageFieldConfig:
    tau_core: float = 0.50
    tau_weak: float = 0.08
    halo_layers: int = 1
    sentinel_score_quantile: float = 0.85
    eps: float = 1e-8


def _norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo + eps)).astype(np.float32)


def _node_to_tets(elements: np.ndarray, n_nodes: int) -> list[list[int]]:
    node_to_tets: list[list[int]] = [[] for _ in range(n_nodes)]
    for ti, tet in enumerate(elements):
        for node in tet:
            node_to_tets[int(node)].append(ti)
    return node_to_tets


def _dilate_tet_mask(seed: np.ndarray, elements: np.ndarray, layers: int) -> np.ndarray:
    out = seed.astype(bool).copy()
    if layers <= 0 or not out.any():
        return out

    n_nodes = int(elements.max()) + 1
    node_to_tets = _node_to_tets(elements, n_nodes)
    frontier = set(np.where(out)[0].tolist())

    for _ in range(layers):
        new_frontier: set[int] = set()
        for ti in frontier:
            for node in elements[ti]:
                new_frontier.update(node_to_tets[int(node)])
        if not new_frontier:
            break
        out[np.asarray(sorted(new_frontier), dtype=np.int64)] = True
        frontier = new_frontier

    return out


def _coerce_cfg(cfg: CoverageFieldConfig | dict | None) -> CoverageFieldConfig:
    if cfg is None:
        return CoverageFieldConfig()
    if isinstance(cfg, CoverageFieldConfig):
        return cfg
    allowed = CoverageFieldConfig.__annotations__
    return CoverageFieldConfig(**{k: v for k, v in cfg.items() if k in allowed})


def compute_coverage_field(
    coarse_d: np.ndarray,
    elements: np.ndarray,
    roi_tet_indices: np.ndarray | None = None,
    cfg: CoverageFieldConfig | dict | None = None,
) -> dict[str, np.ndarray]:
    """
    Convert Stage-1 FEM node values into tet-level coverage metadata.

    Role definition:
      CORE: high-confidence FEM support.
      HALO: immediate neighborhood around CORE.
      SENTINEL: weak or locally uncertain structures outside CORE/HALO.
      BG: background-control region.
    """
    cfg = _coerce_cfg(cfg)
    coarse_d = coarse_d.astype(np.float32)
    elements = elements.astype(np.int64)

    vals = coarse_d[elements]  # [T, 4]
    tet_mean = vals.mean(axis=1).astype(np.float32)
    tet_max = vals.max(axis=1).astype(np.float32)
    tet_min = vals.min(axis=1).astype(np.float32)
    tet_range = (tet_max - tet_min).astype(np.float32)
    tet_var = vals.var(axis=1).astype(np.float32)

    mean_n = _norm(tet_mean, cfg.eps)
    max_n = _norm(tet_max, cfg.eps)
    range_n = _norm(tet_range, cfg.eps)
    var_n = _norm(tet_var, cfg.eps)
    weak_n = ((tet_max >= cfg.tau_weak) & (tet_max < cfg.tau_core)).astype(np.float32)

    coverage_score = (
        0.45 * max_n
        + 0.25 * range_n
        + 0.20 * var_n
        + 0.10 * weak_n
    ).astype(np.float32)

    role = np.full(len(elements), int(QueryRole.BG), dtype=np.int64)

    core_mask = tet_max >= cfg.tau_core
    if roi_tet_indices is not None and len(roi_tet_indices) > 0:
        roi_mask = np.zeros(len(elements), dtype=bool)
        roi_mask[np.asarray(roi_tet_indices, dtype=np.int64)] = True
        core_mask = core_mask | (roi_mask & (tet_max >= cfg.tau_weak))

    core_halo_mask = _dilate_tet_mask(core_mask, elements, int(cfg.halo_layers))
    halo_mask = core_halo_mask & (~core_mask)
    non_core_halo = ~(core_mask | halo_mask)

    if np.any(non_core_halo):
        cut = float(np.quantile(coverage_score[non_core_halo], cfg.sentinel_score_quantile))
    else:
        cut = float(np.quantile(coverage_score, cfg.sentinel_score_quantile))

    sentinel_mask = non_core_halo & ((tet_max >= cfg.tau_weak) | (coverage_score >= cut))

    role[core_mask] = int(QueryRole.CORE)
    role[halo_mask] = int(QueryRole.HALO)
    role[sentinel_mask] = int(QueryRole.SENTINEL)

    risk_components = np.stack([mean_n, max_n, range_n, var_n], axis=1).astype(np.float32)

    return {
        "tet_mean": tet_mean,
        "tet_max": tet_max,
        "tet_range": tet_range,
        "tet_var": tet_var,
        "coverage_score": coverage_score,
        "risk_components": risk_components,
        "role": role,
    }


def role_query_weights(role: np.ndarray) -> np.ndarray:
    w = np.ones_like(role, dtype=np.float32)
    w[role == int(QueryRole.HALO)] = 1.15
    w[role == int(QueryRole.SENTINEL)] = 1.20
    w[role == int(QueryRole.BG)] = 0.90
    return w.astype(np.float32)
