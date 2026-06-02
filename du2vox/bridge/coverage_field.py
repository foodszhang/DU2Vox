from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class QueryRole(IntEnum):
    BG = 0
    CORE = 1
    HALO = 2
    SENTINEL = 3
    PROPOSAL = 4


def correction_band_distance(role: np.ndarray) -> np.ndarray:
    out = np.zeros_like(role, dtype=np.float32)
    out[role == int(QueryRole.BG)] = 0.0
    out[role == int(QueryRole.CORE)] = 0.5
    out[role == int(QueryRole.HALO)] = 1.0
    out[role == int(QueryRole.SENTINEL)] = 0.0
    out[role == int(QueryRole.PROPOSAL)] = 0.75
    return out.astype(np.float32)


@dataclass
class CoverageFieldConfig:
    tau_core: float = 0.50
    tau_weak: float = 0.08
    tau_boundary_low: float = 0.20
    tau_boundary_high: float = 0.50
    use_boundary_halo: bool = False
    halo_layers: int = 1
    max_halo_tets_ratio: float = 0.40
    sentinel_score_quantile: float = 0.85
    core_from_roi_weak: bool = True
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


def coverage_cfg_from_cqr(cqr_cfg: dict | None) -> CoverageFieldConfig:
    if not cqr_cfg:
        return CoverageFieldConfig()
    if "prolongation" in cqr_cfg:
        prolongation = cqr_cfg.get("prolongation") or {}
        return CoverageFieldConfig(
            tau_core=float(prolongation.get("tau_core_band", CoverageFieldConfig.tau_core)),
            tau_weak=float(prolongation.get("tau_halo_band", CoverageFieldConfig.tau_weak)),
            halo_layers=int(prolongation.get("halo_layers", CoverageFieldConfig.halo_layers)),
            core_from_roi_weak=bool(
                prolongation.get("use_weak_roi_as_core", CoverageFieldConfig.core_from_roi_weak)
            ),
        )
    return _coerce_cfg(cqr_cfg.get("coverage", {}))


def _limit_mask_by_score(mask: np.ndarray, score: np.ndarray, max_ratio: float) -> np.ndarray:
    if max_ratio <= 0 or max_ratio >= 1 or not np.any(mask):
        return mask
    max_count = max(1, int(round(len(mask) * max_ratio)))
    idx = np.where(mask)[0]
    if len(idx) <= max_count:
        return mask
    keep = idx[np.argsort(score[idx])[-max_count:]]
    out = np.zeros_like(mask, dtype=bool)
    out[keep] = True
    return out


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
    boundary_score = (range_n * max_n).astype(np.float32)

    role = np.full(len(elements), int(QueryRole.BG), dtype=np.int64)

    core_mask = tet_max >= cfg.tau_core
    support_mask = np.zeros(len(elements), dtype=bool)
    if roi_tet_indices is not None and len(roi_tet_indices) > 0:
        roi_mask = np.zeros(len(elements), dtype=bool)
        roi_mask[np.asarray(roi_tet_indices, dtype=np.int64)] = True
        support_mask = roi_mask & (tet_max >= cfg.tau_weak)
        if cfg.core_from_roi_weak:
            core_mask = core_mask | support_mask

    core_halo_mask = _dilate_tet_mask(core_mask, elements, int(cfg.halo_layers))
    halo_mask = core_halo_mask & (~core_mask)
    if not cfg.core_from_roi_weak:
        halo_mask = halo_mask | (support_mask & (~core_mask))
    if cfg.use_boundary_halo:
        range_cut = float(np.quantile(tet_range, 0.60))
        boundary_mask = (
            (tet_max >= cfg.tau_boundary_low)
            & (tet_max < cfg.tau_boundary_high)
            & (tet_range >= range_cut)
        )
        halo_mask = (halo_mask | boundary_mask) & (~core_mask)
        halo_mask = _limit_mask_by_score(halo_mask, boundary_score + coverage_score, cfg.max_halo_tets_ratio)
    non_core_halo = ~(core_mask | halo_mask)

    if np.any(non_core_halo):
        cut = float(np.quantile(coverage_score[non_core_halo], cfg.sentinel_score_quantile))
    else:
        cut = float(np.quantile(coverage_score, cfg.sentinel_score_quantile))

    sentinel_mask = non_core_halo & ((tet_max >= cfg.tau_weak) | (coverage_score >= cut))

    role[core_mask] = int(QueryRole.CORE)
    role[halo_mask] = int(QueryRole.HALO)
    role[sentinel_mask] = int(QueryRole.SENTINEL)

    weak_support_score = ((tet_max >= cfg.tau_weak) & (tet_max < cfg.tau_core)).astype(np.float32)
    band_boundary_score = np.maximum(boundary_score, halo_mask.astype(np.float32))
    band_boundary_score[core_mask] = np.maximum(band_boundary_score[core_mask], 0.5)
    local_variation_score = (0.5 * range_n + 0.5 * var_n).astype(np.float32)
    correction_demand_score = _norm(
        weak_support_score + band_boundary_score + local_variation_score,
        cfg.eps,
    )

    band_distance_score = correction_band_distance(role)

    risk_components = np.stack([mean_n, max_n, range_n, var_n], axis=1).astype(np.float32)

    return {
        "tet_mean": tet_mean,
        "tet_max": tet_max,
        "tet_range": tet_range,
        "tet_var": tet_var,
        "coverage_score": coverage_score,
        "boundary_score": boundary_score,
        "correction_demand_score": correction_demand_score.astype(np.float32),
        "band_distance_score": band_distance_score.astype(np.float32),
        "risk_components": risk_components,
        "role": role,
    }


def role_query_weights(role: np.ndarray) -> np.ndarray:
    w = np.ones_like(role, dtype=np.float32)
    w[role == int(QueryRole.BG)] = 0.75
    w[role == int(QueryRole.CORE)] = 1.0
    w[role == int(QueryRole.HALO)] = 1.25
    w[role == int(QueryRole.SENTINEL)] = 1.0
    w[role == int(QueryRole.PROPOSAL)] = 1.0
    return w.astype(np.float32)
