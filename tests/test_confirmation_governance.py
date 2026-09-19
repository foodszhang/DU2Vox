from __future__ import annotations

import json
from pathlib import Path

import pytest

from du2vox.utils.confirmation import (
    require_confirmation_permission,
    validate_sealed_manifest,
    validate_validation_dataset_receipt,
    write_sealed_manifest,
)


def _manifest(samples_dir: Path) -> dict[str, object]:
    return {
        "dataset_name": "sealed_confirmation",
        "generation_date": "2026-09-01T00:00:00+00:00",
        "generator_commit_hash": "abc",
        "generator_config_sha256": "def",
        "global_random_seed": 7,
        "sample_ids": ["sample_0000"],
        "number_of_samples": 1,
        "mesh_version": {},
        "measurement_configuration": {},
        "gt_semantics": {},
        "samples_dir": str(samples_dir.resolve()),
    }


def test_confirmation_requires_explicit_permission(tmp_path: Path) -> None:
    samples = tmp_path / "confirmation" / "samples"
    manifest_path = tmp_path / "confirmation_manifest.json"
    write_sealed_manifest(_manifest(samples), manifest_path)
    with pytest.raises(PermissionError, match="allow-confirmation-eval"):
        require_confirmation_permission(
            samples_dir=samples,
            allow_confirmation_eval=False,
            manifest_path=manifest_path,
        )
    result = require_confirmation_permission(
        samples_dir=samples,
        allow_confirmation_eval=True,
        manifest_path=manifest_path,
    )
    assert result is not None


def test_manifest_hash_tampering_is_detected(tmp_path: Path) -> None:
    samples = tmp_path / "confirmation" / "samples"
    manifest_path = tmp_path / "confirmation_manifest.json"
    write_sealed_manifest(_manifest(samples), manifest_path)
    value = json.loads(manifest_path.read_text())
    value["global_random_seed"] = 8
    manifest_path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="hash mismatch"):
        validate_sealed_manifest(manifest_path)


def test_validation_freeze_is_bound_to_exact_dataset_receipt(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset_freeze.json"
    dataset_path.write_text(json.dumps({"status": "frozen", "sealed_confirmation_accessed": False}))
    from du2vox.utils.confirmation import sha256_file

    validation = {
        "dataset_receipt": str(dataset_path.resolve()),
        "dataset_receipt_sha256": sha256_file(dataset_path),
    }
    assert validate_validation_dataset_receipt(validation, dataset_path)["status"] == "frozen"

    dataset_path.write_text(json.dumps({"status": "frozen", "sealed_confirmation_accessed": True}))
    with pytest.raises(RuntimeError, match="different dataset receipt"):
        validate_validation_dataset_receipt(validation, dataset_path)
