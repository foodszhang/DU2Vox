import numpy as np

from du2vox.evaluation.continuous_field import (
    SSIM3DProtocol,
    all_gt_source_metrics,
    continuous_metrics,
    masked_ssim3d,
)


def test_amplitude_sensitive_metrics_do_not_normalize_inputs() -> None:
    target = np.asarray([0.0, 0.25, 0.5, 1.0])
    prediction = 2.0 * target
    metrics = continuous_metrics(prediction, target, data_range=2.0)
    assert metrics["relative_l2"] == 1.0
    assert metrics["integrated_ratio_signed"] == 2.0
    assert metrics["ccc"] < 1.0


def test_masked_true_3d_ssim_identity_and_scale_error() -> None:
    shape = (11, 12, 13)
    valid = np.arange(np.prod(shape), dtype=np.int64)
    rng = np.random.default_rng(9)
    target = rng.random(len(valid))
    protocol = SSIM3DProtocol(data_range=2.0, window_size=11, gaussian_sigma_vox=1.5)
    identity = masked_ssim3d(target, target, valid, shape, protocol=protocol)
    scaled = masked_ssim3d(0.5 * target, target, valid, shape, protocol=protocol)
    assert np.isclose(identity, 1.0, atol=1e-12)
    assert scaled < identity


def test_mask_excludes_invalid_shared_zero_background() -> None:
    shape = (13, 13, 13)
    valid = np.ravel_multi_index(([6, 6], [6, 7], [6, 6]), shape)
    target = np.asarray([1.0, 0.5])
    prediction = np.asarray([0.0, 0.0])
    score = masked_ssim3d(prediction, target, valid, shape)
    assert score < 0.5


def test_all_gt_sources_are_scored_without_detection_censoring() -> None:
    coords = np.asarray(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [4.0, 0.0, 0.0], [4.5, 0.0, 0.0]]
    )
    foci = [
        {
            "center": [0.0, 0.0, 0.0],
            "params": {
                "rx": 1.0,
                "ry": 1.0,
                "rz": 1.0,
                "intensity": 1.0,
                "rotation_matrix": np.eye(3).tolist(),
            },
        },
        {
            "center": [4.0, 0.0, 0.0],
            "params": {
                "rx": 1.0,
                "ry": 1.0,
                "rz": 1.0,
                "intensity": 0.5,
                "rotation_matrix": np.eye(3).tolist(),
            },
        },
    ]
    target = np.asarray([1.0, 0.8, 0.5, 0.4])
    prediction = np.asarray([1.0, 0.8, 0.0, 0.0])
    metrics = all_gt_source_metrics(prediction, target, coords, foci)
    assert metrics["all_source_detection_recall"] == 0.5
    assert metrics["all_source_peak_relative_error"] == 0.5
    assert np.isfinite(metrics["all_source_contrast_log_error"])


def test_legacy_axis_aligned_focus_metadata_is_supported() -> None:
    coords = np.asarray([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [3.0, 0.0, 0.0]])
    foci = [
        {
            "center": [0.0, 0.0, 0.0],
            "shape": "sphere",
            "radius": 0.5,
            "rx": None,
            "ry": None,
            "rz": None,
            "params": {"radius": 0.5, "intensity": 1.4},
        }
    ]
    target = np.asarray([1.0, 0.7, 0.0])
    metrics = all_gt_source_metrics(target, target, coords, foci)
    assert metrics["all_source_peak_relative_error"] == 0.0
    assert metrics["all_source_detection_recall"] == 1.0
