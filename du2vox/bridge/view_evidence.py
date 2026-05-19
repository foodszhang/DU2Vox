from __future__ import annotations

import numpy as np

from du2vox.models.stage2.view_encoder import ANGLES, FOV_MM, MCX_VOLUME_CENTER_WORLD


def project_world_to_uv(points_world: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_world, dtype=np.float32)
    centered = points - np.asarray(MCX_VOLUME_CENTER_WORLD, dtype=np.float32)
    angle = np.deg2rad(angle_deg)
    x_rot = centered[:, 0] * np.cos(angle) + centered[:, 2] * np.sin(angle)
    y_rot = centered[:, 1]
    half_fov = FOV_MM / 2.0
    u_ndc = x_rot / half_fov
    v_ndc = y_rot / half_fov
    visible = (np.abs(u_ndc) <= 1.0) & (np.abs(v_ndc) <= 1.0)
    return u_ndc.astype(np.float32), v_ndc.astype(np.float32), visible


def _normalize_view(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    lo = float(np.percentile(img, 1.0))
    hi = float(np.percentile(img, 99.5))
    if hi <= lo + 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)


def _bilinear_sample(img: np.ndarray, u_ndc: np.ndarray, v_ndc: np.ndarray, visible: np.ndarray) -> np.ndarray:
    height, width = img.shape
    x = (u_ndc + 1.0) * 0.5 * width - 0.5
    y = (v_ndc + 1.0) * 0.5 * height - 0.5

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    valid = visible & (x0 >= 0) & (x1 < width) & (y0 >= 0) & (y1 < height)
    out = np.zeros_like(u_ndc, dtype=np.float32)
    if not np.any(valid):
        return out

    xv = x[valid]
    yv = y[valid]
    x0v = x0[valid]
    x1v = x1[valid]
    y0v = y0[valid]
    y1v = y1[valid]
    wx = xv - x0v
    wy = yv - y0v
    out[valid] = (
        img[y0v, x0v] * (1.0 - wx) * (1.0 - wy)
        + img[y0v, x1v] * wx * (1.0 - wy)
        + img[y1v, x0v] * (1.0 - wx) * wy
        + img[y1v, x1v] * wx * wy
    )
    return out.astype(np.float32)


def compute_view_evidence(points_world: np.ndarray, proj_imgs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Score world points by lightweight projection-image evidence.

    Returns:
        view_score: [N] top-2 mean normalized projection intensity.
        view_stats: [N, 5] mean, max, top2_mean, visible_count, consistency.
    """
    proj_imgs = np.asarray(proj_imgs, dtype=np.float32)
    if proj_imgs.ndim != 3:
        raise ValueError(f"proj_imgs must have shape [V,H,W], got {proj_imgs.shape}")

    n_points = len(points_world)
    samples = np.zeros((n_points, len(ANGLES)), dtype=np.float32)
    visible_all = np.zeros((n_points, len(ANGLES)), dtype=bool)
    for view_idx, angle in enumerate(ANGLES):
        img = _normalize_view(proj_imgs[view_idx])
        u_ndc, v_ndc, visible = project_world_to_uv(points_world, angle)
        samples[:, view_idx] = _bilinear_sample(img, u_ndc, v_ndc, visible)
        visible_all[:, view_idx] = visible

    mean_score = samples.mean(axis=1)
    max_score = samples.max(axis=1)
    top2 = np.sort(samples, axis=1)[:, -2:].mean(axis=1)
    visible_count = visible_all.sum(axis=1).astype(np.float32)
    consistency = (samples >= np.percentile(samples, 75.0)).sum(axis=1).astype(np.float32) / max(len(ANGLES), 1)
    view_stats = np.stack([mean_score, max_score, top2, visible_count, consistency], axis=1).astype(np.float32)
    return top2.astype(np.float32), view_stats


def load_proj_npz(path: str) -> np.ndarray:
    data = np.load(path)
    return np.stack([data[str(angle)].astype(np.float32) for angle in ANGLES], axis=0)
