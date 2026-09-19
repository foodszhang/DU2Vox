#!/usr/bin/env python3
"""Freeze validation-selected LPR checkpoints before development-test access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_candidate(value: str) -> tuple[str, Path, Path, Path]:
    name, remainder = value.split("=", 1)
    config, checkpoint, validation = remainder.split(":", 2)
    return name, Path(config), Path(checkpoint), Path(validation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate", action="append", type=parse_candidate, required=True
    )
    parser.add_argument("--dataset-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    entries = []
    for name, config, checkpoint, validation in args.candidate:
        payload = json.loads(validation.read_text())
        if payload.get("n_samples") != 300 or payload.get("split") != "val":
            raise RuntimeError(f"{name}: validation artifact is not complete val300")
        entries.append(
            {
                "name": name,
                "config": str(config.resolve()),
                "config_sha256": sha256(config),
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256(checkpoint),
                "validation_artifact": str(validation.resolve()),
                "validation_artifact_sha256": sha256(validation),
                "validation_summary": payload.get("summary"),
            }
        )
    result = {
        "status": "frozen_on_val300",
        "dataset_receipt": str(args.dataset_receipt.resolve()),
        "dataset_receipt_sha256": sha256(args.dataset_receipt),
        "sealed_confirmation_accessed": False,
        "development_test_accessed_before_freeze": False,
        "candidates": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "candidates": [e["name"] for e in entries]}, indent=2))


if __name__ == "__main__":
    main()
