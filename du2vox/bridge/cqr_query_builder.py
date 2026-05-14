from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from du2vox.bridge.coverage_field import (
    CoverageFieldConfig,
    QueryRole,
    compute_coverage_field,
    role_query_weights,
)
from du2vox.bridge.fem_bridging import FEMBridge


@dataclass
class CQRQueryConfig:
    n_query_points: int = 32768
    n_candidates: int = 16
    seed: int = 0
    ratios: dict[str, float] = field(
        default_factory=lambda: {
            "core": 0.25,
            "halo": 0.45,
            "sentinel": 0.20,
            "bg": 0.10,
        }
    )
    coverage: CoverageFieldConfig = field(default_factory=CoverageFieldConfig)
    bg_score_quantile_max: float = 0.40


def _coerce_config(cfg: CQRQueryConfig | dict | None) -> CQRQueryConfig:
    if cfg is None:
        return CQRQueryConfig()
    if isinstance(cfg, CQRQueryConfig):
        return cfg

    cfg = dict(cfg)
    coverage_cfg = cfg.pop("coverage", None)
    allowed = CQRQueryConfig.__annotations__
    out = CQRQueryConfig(**{k: v for k, v in cfg.items() if k in allowed})
    if coverage_cfg is not None:
        out.coverage = CoverageFieldConfig(
            **{k: v for k, v in coverage_cfg.items() if k in CoverageFieldConfig.__annotations__}
        )
    return out


def _normalize_ratios(ratios: dict[str, float]) -> dict[str, float]:
    keys = ["core", "halo", "sentinel", "bg"]
    vals = {k: max(0.0, float(ratios.get(k, 0.0))) for k in keys}
    total = sum(vals.values())
    if total <= 0:
        return {"core": 0.25, "halo": 0.45, "sentinel": 0.20, "bg": 0.10}
    return {k: vals[k] / total for k in keys}


def _counts(n: int, ratios: dict[str, float]) -> dict[str, int]:
    ratios = _normalize_ratios(ratios)
    out = {k: int(round(n * v)) for k, v in ratios.items()}
    out["halo"] += n - sum(out.values())
    return out


def _dirichlet4(rng: np.random.Generator, n: int) -> np.ndarray:
    x = rng.exponential(scale=1.0, size=(n, 4)).astype(np.float64)
    x /= x.sum(axis=1, keepdims=True) + 1e-12
    return x.astype(np.float32)


def _sample_points_in_tets(
    rng: np.random.Generator,
    nodes: np.ndarray,
    elements: np.ndarray,
    tet_indices: np.ndarray,
    n_points: int,
    probability: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    tet_indices = np.asarray(tet_indices, dtype=np.int64)
    if len(tet_indices) == 0 or n_points <= 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    p = None
    if probability is not None:
        p = np.asarray(probability, dtype=np.float64)
        p = np.clip(p, 1e-8, None)
        p = p / p.sum()

    local = rng.choice(len(tet_indices), size=n_points, replace=True, p=p)
    chosen_tets = tet_indices[local]
    bary = _dirichlet4(rng, n_points).astype(np.float64)
    verts = nodes[elements[chosen_tets]].astype(np.float64)
    pts = (verts * bary[:, :, None]).sum(axis=1).astype(np.float32)
    return pts, chosen_tets.astype(np.int64)


class CQRQueryBuilder:
    """
    Build a coverage-aware query cloud and extended Stage-2 prior.

    prior_ext layout:
      0:8   original FEM prior [d_v0..d_v3, lambda_0..lambda_3]
      8     coverage_score
      9:13  risk_components [tet_mean, tet_max, tet_range, tet_var], normalized
    """

    def __init__(
        self,
        nodes: np.ndarray,
        elements: np.ndarray,
        config: CQRQueryConfig | dict | None = None,
    ):
        self.nodes = nodes.astype(np.float64)
        self.elements = elements.astype(np.int64)
        self.config = _coerce_config(config)

    def _make_pools(self, field: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        role = field["role"]
        score = field["coverage_score"]
        all_tets = np.arange(len(role), dtype=np.int64)

        core = np.where(role == int(QueryRole.CORE))[0].astype(np.int64)
        halo = np.where(role == int(QueryRole.HALO))[0].astype(np.int64)
        sentinel = np.where(role == int(QueryRole.SENTINEL))[0].astype(np.int64)

        bg_mask = role == int(QueryRole.BG)
        if np.any(bg_mask):
            bg_cut = np.quantile(score[bg_mask], self.config.bg_score_quantile_max)
            bg = np.where(bg_mask & (score <= bg_cut))[0].astype(np.int64)
        else:
            bg = np.zeros((0,), dtype=np.int64)

        if len(core) == 0:
            core = np.argsort(score)[-max(1, len(score) // 100):].astype(np.int64)
        if len(halo) == 0:
            halo = core
        if len(sentinel) == 0:
            remain = np.setdiff1d(all_tets, core, assume_unique=False)
            sentinel = remain[np.argsort(score[remain])[-max(1, len(score) // 100):]] if len(remain) else core
        if len(bg) == 0:
            bg = np.argsort(score)[:max(1, len(score) // 100)].astype(np.int64)

        return {"core": core, "halo": halo, "sentinel": sentinel, "bg": bg}

    def build(
        self,
        coarse_d: np.ndarray,
        roi_tet_indices: np.ndarray | None,
        n_query_points: int | None = None,
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        cfg = self.config
        n_query = int(n_query_points or cfg.n_query_points)
        rng = np.random.default_rng(cfg.seed if seed is None else seed)

        field = compute_coverage_field(
            coarse_d=coarse_d,
            elements=self.elements,
            roi_tet_indices=roi_tet_indices,
            cfg=cfg.coverage,
        )
        pools = self._make_pools(field)
        counts = _counts(n_query, cfg.ratios)

        pts_chunks = []
        tet_chunks = []
        role_chunks = []
        role_value = {
            "core": int(QueryRole.CORE),
            "halo": int(QueryRole.HALO),
            "sentinel": int(QueryRole.SENTINEL),
            "bg": int(QueryRole.BG),
        }

        score = field["coverage_score"]
        for name in ["core", "halo", "sentinel", "bg"]:
            pool = pools[name]
            n = counts[name]
            prob = 1.0 - score[pool] if name == "bg" else score[pool]
            pts, tids = _sample_points_in_tets(
                rng=rng,
                nodes=self.nodes,
                elements=self.elements,
                tet_indices=pool,
                n_points=n,
                probability=prob,
            )
            pts_chunks.append(pts)
            tet_chunks.append(tids)
            role_chunks.append(np.full(len(tids), role_value[name], dtype=np.int64))

        points = np.concatenate(pts_chunks, axis=0).astype(np.float32)
        tet_ids = np.concatenate(tet_chunks, axis=0).astype(np.int64)
        role = np.concatenate(role_chunks, axis=0).astype(np.int64)

        perm = rng.permutation(len(points))
        points = points[perm]
        tet_ids = tet_ids[perm]
        role = role[perm]

        active_tets = np.unique(tet_ids).astype(np.int64)
        bridge = FEMBridge(
            self.nodes,
            self.elements,
            roi_tet_indices=active_tets,
            n_candidates=cfg.n_candidates,
        )
        prior_8d, valid_mask = bridge.get_prior_features(points, coarse_d, K=cfg.n_candidates)

        coverage_score = field["coverage_score"][tet_ids].astype(np.float32)
        risk_components = field["risk_components"][tet_ids].astype(np.float32)
        query_weight = role_query_weights(role)
        prior_ext = np.concatenate(
            [prior_8d.astype(np.float32), coverage_score[:, None], risk_components],
            axis=1,
        ).astype(np.float32)

        return {
            "query_points": points,
            "prior_8d": prior_8d.astype(np.float32),
            "prior_ext": prior_ext,
            "valid_mask": valid_mask.astype(bool),
            "tet_ids": tet_ids,
            "role": role,
            "coverage_score": coverage_score,
            "risk_components": risk_components,
            "query_weight": query_weight.astype(np.float32),
            "role_counts": np.bincount(role, minlength=4).astype(np.int64),
        }
