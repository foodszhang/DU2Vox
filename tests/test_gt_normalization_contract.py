from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from du2vox.data.error_structured_dataset import ErrorStructuredFixedDomainDataset
from du2vox.utils.gt_io import (
    canonical_gt_values,
    load_normalization_scale,
)


def test_explicit_full_volume_peak_overrides_valid_domain_peak() -> None:
    volume = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    valid = np.asarray([0, 1], dtype=np.int64)

    legacy, legacy_scale = canonical_gt_values(
        volume, valid, gt_mode="continuous", normalize="per_sample_peak"
    )
    contracted, contracted_scale = canonical_gt_values(
        volume,
        valid,
        gt_mode="continuous",
        normalize="per_sample_peak",
        normalization_scale=1.0,
    )

    assert legacy_scale == 0.5
    assert np.array_equal(legacy, np.asarray([0.0, 1.0], dtype=np.float32))
    assert contracted_scale == 1.0
    assert np.array_equal(contracted, np.asarray([0.0, 0.5], dtype=np.float32))


def test_normalization_scale_file_must_be_positive_scalar(tmp_path: Path) -> None:
    sample = tmp_path / "sample_0000"
    sample.mkdir()
    np.save(sample / "gt_scale.npy", np.asarray(1.25, dtype=np.float32))
    assert load_normalization_scale(sample) == 1.25

    np.save(sample / "gt_scale.npy", np.asarray([1.0, 2.0], dtype=np.float32))
    with pytest.raises(ValueError, match="must be scalar"):
        load_normalization_scale(sample)


def test_projection_metadata_rejects_missing_scale_contract(tmp_path: Path) -> None:
    metadata = {
        "gt_mode": "continuous",
        "normalize_gt": "per_sample_peak",
        "binary_threshold": 0.05,
        "sample_mapping": {"train": ["sample_0000"]},
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))

    dataset = ErrorStructuredFixedDomainDataset.__new__(ErrorStructuredFixedDomainDataset)
    dataset.projection_targets_dir = tmp_path
    dataset.sample_ids = ["sample_0000"]
    dataset.gt_mode = "continuous"
    dataset.normalize_gt = "per_sample_peak"
    dataset.normalization_scale_filename = "gt_scale.npy"
    dataset.binary_threshold = 0.05

    with pytest.raises(RuntimeError, match="normalization_scale_filename"):
        dataset._validate_projection_target_contract()


def test_measurement_and_gt_normalization_preserve_forward_scale(tmp_path: Path) -> None:
    sample = tmp_path / "sample_0000"
    sample.mkdir()
    np.save(sample / "measurement_b.npy", np.asarray([2.0, 4.0], dtype=np.float32))
    np.save(sample / "gt_scale.npy", np.asarray(1.5, dtype=np.float32))

    dataset = ErrorStructuredFixedDomainDataset.__new__(ErrorStructuredFixedDomainDataset)
    dataset.samples_dir = tmp_path
    dataset.visible_mask = None
    dataset.normalize_measurement = True
    dataset.measurement_operator_scaling = "matched_gt_measurement"
    dataset.normalize_gt = "per_sample_peak"
    dataset.normalization_scale_filename = "gt_scale.npy"

    measurement, operator_scale = dataset.load_measurement_with_operator_scale("sample_0000")
    assert np.array_equal(measurement, np.asarray([0.5, 1.0], dtype=np.float32))
    assert operator_scale == 1.5 / 4.0

    dataset.measurement_operator_scaling = "none"
    _, legacy_scale = dataset.load_measurement_with_operator_scale("sample_0000")
    assert legacy_scale == 1.0
