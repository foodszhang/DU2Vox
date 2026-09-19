import numpy as np

from du2vox.evaluation.fem_volume import (
    fem_volume_metrics,
    lumped_nodal_volumes,
    source_composition_metrics,
)


def test_lumped_nodal_volumes_single_tetrahedron():
    nodes = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    weights = lumped_nodal_volumes(nodes, np.asarray([[0, 1, 2, 3]]))
    np.testing.assert_allclose(weights, np.full(4, 1.0 / 24.0))
    np.testing.assert_allclose(weights.sum(), 1.0 / 6.0)


def test_fem_volume_metrics_identical_and_scale_separation():
    nodes = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    weights = np.asarray([1.0, 2.0, 3.0, 4.0])
    target = np.asarray([0.0, 0.2, 0.7, 1.0])
    exact = fem_volume_metrics(target, target, weights, nodes)
    assert exact["wrel_l2"] == 0.0
    assert exact["mass_ratio"] == 1.0
    assert exact["weighted_ccc"] == 1.0
    assert exact["com_error_mm"] == 0.0
    assert exact["shape_error"] == 0.0

    scaled = fem_volume_metrics(0.5 * target, target, weights, nodes)
    np.testing.assert_allclose(scaled["wrel_l2"], 0.5)
    np.testing.assert_allclose(scaled["mass_ratio"], 0.5)
    np.testing.assert_allclose(scaled["shape_error"], 0.0, atol=1e-15)
    np.testing.assert_allclose(scaled["shape_optimal_scale"], 2.0)


def test_source_composition_detects_weak_source_loss():
    nodes = np.asarray(
        [[-2.0, 0, 0], [-1.9, 0, 0], [2.0, 0, 0], [2.1, 0, 0]], dtype=float
    )
    foci = [
        {"center": [-2, 0, 0], "shape": "sphere", "params": {"radius": 0.5, "intensity": 2}},
        {"center": [2, 0, 0], "shape": "sphere", "params": {"radius": 0.5, "intensity": 1}},
    ]
    target = np.asarray([1.0, 1.0, 0.5, 0.5])
    prediction = np.asarray([1.0, 1.0, 0.0, 0.0])
    metrics = source_composition_metrics(
        prediction, target, np.ones(4), nodes, foci
    )
    np.testing.assert_allclose(metrics["source_composition_error"], 1.0 / 3.0)
    np.testing.assert_allclose(metrics["weak_source_mass_ratio"], 0.0)
    np.testing.assert_allclose(metrics["weak_source_error"], 1.0)


def test_single_source_composition_is_not_counted_as_perfect_multisource_result():
    nodes = np.asarray([[0.0, 0, 0], [0.1, 0, 0]], dtype=float)
    focus = {
        "center": [0, 0, 0],
        "shape": "sphere",
        "params": {"radius": 0.5, "intensity": 1},
    }
    metrics = source_composition_metrics(
        np.ones(2), np.ones(2), np.ones(2), nodes, [focus]
    )
    assert np.isnan(metrics["source_composition_error"])
    assert np.isnan(metrics["weak_source_error"])
