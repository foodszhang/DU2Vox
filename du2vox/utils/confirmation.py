"""Sealed simulation-confirmation data governance helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically for an auditable SHA256 digest."""

    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("ascii")


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA256 digest without interpreting file contents."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sealed_manifest(manifest: dict[str, Any], manifest_path: str | Path) -> str:
    """Write canonical manifest JSON and its adjacent ``.sha256`` file."""

    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(manifest))
    digest = sha256_file(path)
    hash_path = path.with_suffix(".sha256")
    hash_path.write_text(f"{digest}  {path.name}\n")
    return digest


def validate_sealed_manifest(
    manifest_path: str | Path, hash_path: str | Path | None = None
) -> dict[str, Any]:
    """Validate the adjacent hash and return the parsed sealed manifest."""

    path = Path(manifest_path)
    expected_path = Path(hash_path) if hash_path else path.with_suffix(".sha256")
    if not path.is_file() or not expected_path.is_file():
        raise FileNotFoundError("Confirmation manifest or SHA256 seal is missing")
    expected = expected_path.read_text().split()[0]
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"Confirmation manifest hash mismatch: expected {expected}, got {actual}"
        )
    manifest = json.loads(path.read_text())
    required = {
        "dataset_name",
        "generation_date",
        "generator_commit_hash",
        "generator_config_sha256",
        "global_random_seed",
        "sample_ids",
        "number_of_samples",
        "mesh_version",
        "measurement_configuration",
        "gt_semantics",
        "samples_dir",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise RuntimeError(f"Confirmation manifest is incomplete: {missing}")
    if len(manifest["sample_ids"]) != int(manifest["number_of_samples"]):
        raise RuntimeError("Confirmation sample count does not match sample_ids")
    return manifest


def require_confirmation_permission(
    *,
    samples_dir: str | Path,
    allow_confirmation_eval: bool,
    manifest_path: str | Path,
) -> dict[str, Any] | None:
    """Reject access to the sealed cohort unless the explicit flag is present."""

    requested = Path(samples_dir).resolve()
    manifest_file = Path(manifest_path)
    if not manifest_file.is_file():
        if "confirmation" in requested.as_posix().lower():
            raise FileNotFoundError("Refusing access to an unsealed confirmation-like dataset")
        return None
    manifest = validate_sealed_manifest(manifest_file)
    sealed = Path(manifest["samples_dir"]).resolve()
    if requested == sealed and not allow_confirmation_eval:
        raise PermissionError(
            "Refusing sealed confirmation evaluation without --allow-confirmation-eval"
        )
    return manifest if requested == sealed else None


def validate_validation_dataset_receipt(
    validation_receipt: dict[str, Any], dataset_receipt_path: str | Path
) -> dict[str, Any]:
    """Bind a validation freeze to the exact frozen development dataset."""

    dataset_path = Path(dataset_receipt_path).resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset receipt is missing: {dataset_path}")
    actual_hash = sha256_file(dataset_path)
    if validation_receipt.get("dataset_receipt_sha256") != actual_hash:
        raise RuntimeError("Validation freeze is bound to a different dataset receipt")
    recorded_path = validation_receipt.get("dataset_receipt")
    if recorded_path is None or Path(recorded_path).resolve() != dataset_path:
        raise RuntimeError("Validation freeze dataset-receipt path mismatch")
    dataset_receipt = json.loads(dataset_path.read_text())
    if dataset_receipt.get("status") != "frozen":
        raise RuntimeError("Dataset receipt is not frozen")
    if dataset_receipt.get("sealed_confirmation_accessed") is not False:
        raise RuntimeError("Dataset receipt does not certify unopened confirmation data")
    return dataset_receipt
