"""Volume-weighted diagnostics for continuous fields represented on FEM nodes."""

from __future__ import annotations

import numpy as np


def lumped_nodal_volumes(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """Return the P1 lumped mass diagonal (one quarter of each incident tet volume)."""

    xyz = np.asarray(nodes, dtype=np.float64)
    tets = np.asarray(elements, dtype=np.int64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("nodes must have shape [N, 3]")
    if tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError("elements must have shape [E, 4]")
    edge1 = xyz[tets[:, 1]] - xyz[tets[:, 0]]
    edge2 = xyz[tets[:, 2]] - xyz[tets[:, 0]]
    edge3 = xyz[tets[:, 3]] - xyz[tets[:, 0]]
    volumes = np.abs(np.einsum("ij,ij->i", np.cross(edge1, edge2), edge3)) / 6.0
    if not np.all(np.isfinite(volumes)) or np.any(volumes <= 0.0):
        raise ValueError("mesh contains non-positive or non-finite tetrahedron volumes")
    weights = np.zeros(len(xyz), dtype=np.float64)
    np.add.at(weights, tets.ravel(), np.repeat(volumes / 4.0, 4))
    if np.any(weights <= 0.0):
        raise ValueError("mesh contains nodes with zero incident volume")
    return weights


def _fields(
    prediction: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    gt = np.asarray(target, dtype=np.float64).reshape(-1)
    mass = np.asarray(weights, dtype=np.float64).reshape(-1)
    if pred.shape != gt.shape or pred.shape != mass.shape:
        raise ValueError("prediction, target, and weights must have the same length")
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(gt)):
        raise ValueError("fields must be finite")
    if not np.all(np.isfinite(mass)) or np.any(mass <= 0.0):
        raise ValueError("weights must be finite and strictly positive")
    return pred, gt, mass


def weighted_ccc(prediction: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    """Lin's concordance correlation under the normalized lumped-volume measure."""

    pred, gt, mass = _fields(prediction, target, weights)
    probability = mass / mass.sum()
    pred_mean = float(np.dot(probability, pred))
    gt_mean = float(np.dot(probability, gt))
    pred_centered = pred - pred_mean
    gt_centered = gt - gt_mean
    pred_var = float(np.dot(probability, pred_centered**2))
    gt_var = float(np.dot(probability, gt_centered**2))
    covariance = float(np.dot(probability, pred_centered * gt_centered))
    denominator = pred_var + gt_var + (pred_mean - gt_mean) ** 2
    return float(2.0 * covariance / denominator) if denominator > 0.0 else float("nan")


def fem_volume_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    nodes: np.ndarray,
) -> dict[str, float]:
    """Score two nodal P1 fields with the frozen lumped-volume inner product.

    ``shape_error`` removes one non-negative global scale from the prediction. It
    therefore measures the remaining spatial-field mismatch rather than overall
    mass calibration. It still includes displacement and source-composition errors,
    which are separately exposed by COM and source metrics.
    """

    pred, gt, mass = _fields(prediction, target, weights)
    xyz = np.asarray(nodes, dtype=np.float64)
    if xyz.shape != (len(pred), 3):
        raise ValueError("nodes must have shape [N, 3]")
    error = pred - gt
    gt_energy = float(np.dot(mass, gt**2))
    error_energy = float(np.dot(mass, error**2))
    pred_mass = float(np.dot(mass, pred))
    gt_mass = float(np.dot(mass, gt))
    pred_energy = float(np.dot(mass, pred**2))
    cross = float(np.dot(mass, pred * gt))
    shape_scale = max(cross / pred_energy, 0.0) if pred_energy > 0.0 else 0.0
    shape_residual = shape_scale * pred - gt

    def center_of_mass(field: np.ndarray, total: float) -> np.ndarray:
        if total <= 1e-30:
            return np.full(3, np.nan)
        return np.sum(xyz * (mass * field)[:, None], axis=0) / total

    pred_com = center_of_mass(pred, pred_mass)
    gt_com = center_of_mass(gt, gt_mass)
    com_error = float(np.linalg.norm(pred_com - gt_com))
    return {
        "wrel_l2": float(np.sqrt(error_energy / max(gt_energy, 1e-30))),
        "mass_ratio": pred_mass / max(gt_mass, 1e-30),
        "mass_relative_error": abs(pred_mass - gt_mass) / max(abs(gt_mass), 1e-30),
        "weighted_ccc": weighted_ccc(pred, gt, mass),
        "com_error_mm": com_error,
        "shape_error": float(
            np.sqrt(np.dot(mass, shape_residual**2) / max(gt_energy, 1e-30))
        ),
        "shape_optimal_scale": shape_scale,
        "prediction_mass": pred_mass,
        "target_mass": gt_mass,
    }


def _irregular_radial_limit(normalized: np.ndarray, focus: dict) -> np.ndarray:
    rho = np.linalg.norm(normalized, axis=1, keepdims=True)
    unit = normalized / np.maximum(rho, 1e-6)
    params = focus.get("params", focus)
    phases = np.asarray(params.get("irregular_phases", [0.0, 1.7, 3.1]))
    amplitude = float(params.get("irregularity", 0.2))
    modulation = (
        np.sin(3.0 * unit[:, 0] + phases[0])
        + np.sin(4.0 * unit[:, 1] + phases[1])
        + np.sin(5.0 * unit[:, 2] + phases[2])
    ) / 3.0
    return np.clip(1.0 + amplitude * modulation, 0.65, 1.45)


def historical_mcx_focus_field(
    nodes: np.ndarray, focus: dict, *, truncate: bool = True
) -> np.ndarray:
    """Evaluate one historical D1-Q MCX Gaussian component at FEM nodes.

    With ``truncate=True`` this reproduces the component shape used before maximum
    composition. Untruncated templates are used only for source attribution so a
    declared sub-resolution weak component cannot disappear from the audit.
    """

    xyz = np.asarray(nodes, dtype=np.float64)
    center = np.asarray(focus["center"], dtype=np.float64)
    params = focus.get("params", focus)
    intensity = float(params.get("intensity", 1.0))
    delta = xyz - center
    shape = focus.get("shape", "sphere")
    radius = float(params.get("radius", focus.get("radius", 1.0)))
    if shape == "sphere":
        distance = np.linalg.norm(delta, axis=1) / radius
    else:
        axes = np.asarray(
            [
                float(params.get(name, focus.get(name, radius)))
                for name in ("rx", "ry", "rz")
            ]
        )
        normalized = delta / axes
        if shape == "irregular":
            distance = np.linalg.norm(normalized, axis=1) / np.maximum(
                _irregular_radial_limit(normalized, focus), 1e-6
            )
        elif shape == "ellipsoid":
            distance = np.linalg.norm(normalized, axis=1)
        else:
            raise ValueError(f"unsupported historical focus shape: {shape}")
    values = np.exp(-0.5 * distance**2)
    if truncate:
        values = np.where(distance <= 3.0, values, 0.0)
    return intensity * values


def source_partition(
    nodes: np.ndarray,
    foci: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Return intensity-independent territories and component templates.

    Every FEM node belongs to its closest normalized source shape. Untruncated
    templates keep a declared component measurable even when a small, weak source
    has no mesh node inside the historical 3-sigma MCX support.
    """

    if not foci:
        raise ValueError("at least one focus is required")
    components = np.stack(
        [historical_mcx_focus_field(nodes, focus, truncate=False) for focus in foci]
    )
    intensities = np.asarray(
        [float(focus.get("params", focus).get("intensity", 1.0)) for focus in foci]
    )
    # Territory ownership is deliberately intensity-independent. Using the
    # amplitude-weighted winner can erase a nearby weak source completely even
    # though that source is part of the declared GT.
    shapes = components / intensities[:, None]
    labels = np.argmax(shapes, axis=0)
    return labels, components


def source_composition_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    nodes: np.ndarray,
    foci: list[dict],
) -> dict[str, float | list[float]]:
    """Mass-composition and weak-source errors in fixed known-source territories."""

    pred, gt, mass = _fields(prediction, target, weights)
    labels, component_templates = source_partition(nodes, foci)
    pred_masses = np.asarray(
        [np.dot(mass[labels == i], pred[labels == i]) for i in range(len(foci))]
    )
    template_masses = np.asarray(
        [np.dot(mass, component_templates[i]) for i in range(len(foci))]
    )
    target_total = float(np.dot(mass, gt))
    gt_masses = target_total * template_masses / max(float(template_masses.sum()), 1e-30)
    pred_fractions = pred_masses / max(float(pred_masses.sum()), 1e-30)
    gt_fractions = gt_masses / max(float(gt_masses.sum()), 1e-30)
    composition_error = (
        0.5 * float(np.sum(np.abs(pred_fractions - gt_fractions)))
        if len(foci) > 1
        else float("nan")
    )
    intensities = np.asarray(
        [float(focus.get("params", focus).get("intensity", 1.0)) for focus in foci]
    )
    weak_index = int(np.argmin(intensities))
    if len(foci) == 1:
        weak_ratio = float("nan")
        weak_relative_error = float("nan")
        weak_fraction_error = float("nan")
    else:
        weak_ratio = float(pred_masses[weak_index] / max(gt_masses[weak_index], 1e-30))
        weak_relative_error = abs(weak_ratio - 1.0)
        weak_fraction_error = abs(
            float(pred_fractions[weak_index] - gt_fractions[weak_index])
        )
    return {
        "source_composition_error": composition_error,
        "weak_source_mass_ratio": weak_ratio,
        "weak_source_error": weak_relative_error,
        "weak_source_fraction_error": weak_fraction_error,
        "source_prediction_masses": pred_masses.tolist(),
        "source_target_masses": gt_masses.tolist(),
        "source_prediction_fractions": pred_fractions.tolist(),
        "source_target_fractions": gt_fractions.tolist(),
    }
