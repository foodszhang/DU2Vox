#!/usr/bin/env python3
"""Precompute canonical Pi_h rho_gt coefficients for ESCB."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.bridge.canonical_cross_discretization import (
    CanonicalCrossDiscretization,
)
from du2vox.utils.gt_io import (
    GT_MODES,
    GT_NORMALIZATIONS,
    load_canonical_gt,
    load_normalization_scale,
)


def load_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--operator-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-stale-frame-manifest",
        action="store_true",
        help=(
            "Skip the frame_manifest.json hash check when opening the operator "
            "cache. Use only when mesh.npz still matches the cache (that check "
            "is always enforced); frame_manifest.json is rewritten on every "
            "generation run and its hash moves with unrelated metadata."
        ),
    )
    parser.add_argument(
        "--gt-mode",
        choices=GT_MODES,
        default="binary",
        help=(
            "binary: threshold the GT volume (historical D0 contract, teaches "
            "source support only). continuous: keep the raw intensity so a "
            "spatially varying source keeps its profile and per-focus amplitude."
        ),
    )
    parser.add_argument(
        "--normalize-gt",
        choices=GT_NORMALIZATIONS,
        default="none",
        help=(
            "per_sample_peak divides each target by its own peak (peak -> 1), "
            "removing the cohort's 0.63-1.75 peak spread. Source-to-source "
            "contrast ratios are preserved because the divisor is shared."
        ),
    )
    parser.add_argument(
        "--normalization-scale-filename",
        help=(
            "Sample-local scalar divisor (for example gt_scale.npy). Required "
            "for the D1-Q MCX-canonical contract so every discretization uses "
            "the authoritative full-volume peak."
        ),
    )
    args = parser.parse_args()

    if args.normalization_scale_filename and args.normalize_gt != "per_sample_peak":
        parser.error("--normalization-scale-filename requires --normalize-gt per_sample_peak")

    canonical = CanonicalCrossDiscretization(
        args.operator_cache,
        shared_dir=args.shared_dir,
        factorize=True,
        allow_stale_frame_manifest=args.allow_stale_frame_manifest,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, list[str]] = {}
    sample_scales: dict[str, float] = {}
    done = 0
    for split in args.splits:
        ids = load_ids(args.dataset_root / "splits" / f"{split}.txt")
        mapping[split] = ids
        for sid in ids:
            if args.max_samples is not None and done >= args.max_samples:
                break
            sample_dir = args.dataset_root / "samples" / sid
            explicit_scale = (
                load_normalization_scale(sample_dir, args.normalization_scale_filename)
                if args.normalization_scale_filename
                else None
            )
            output = args.output_dir / f"{sid}.npy"
            if output.exists() and not args.overwrite:
                if explicit_scale is not None:
                    sample_scales[sid] = explicit_scale
                elif args.normalize_gt == "per_sample_peak":
                    _, scale = load_canonical_gt(
                        sample_dir,
                        canonical.operator.valid_flat_indices,
                        gt_mode=args.gt_mode,
                        normalize=args.normalize_gt,
                    )
                    sample_scales[sid] = scale
                else:
                    sample_scales[sid] = 1.0
                done += 1
                continue
            gt, scale = load_canonical_gt(
                sample_dir,
                canonical.operator.valid_flat_indices,
                gt_mode=args.gt_mode,
                normalize=args.normalize_gt,
                normalization_scale=explicit_scale,
            )
            coefficients = canonical.project_coefficients(gt.astype(np.float64)).astype(np.float32)
            np.save(output, coefficients)
            sample_scales[sid] = scale
            done += 1
            if done % 25 == 0:
                print(f"[{done}] {split}/{sid}", flush=True)
        if args.max_samples is not None and done >= args.max_samples:
            break

    metadata = {
        "target": (
            "(gt_voxels > 0.05).astype(float64)"
            if args.gt_mode == "binary"
            else "gt_voxels (continuous intensity).astype(float64)"
        ),
        "gt_mode": args.gt_mode,
        "normalize_gt": args.normalize_gt,
        "normalization_scale_filename": args.normalization_scale_filename,
        "binary_threshold": 0.05,
        # Per-sample divisor applied under per_sample_peak normalization; the
        # original intensity is recovered as value * scale.
        "sample_scale": sample_scales,
        "projection": "MassProjector.coefficients from decomposition experiment",
        "operator_cache": str(args.operator_cache.resolve()),
        "operator_metadata": canonical.metadata,
        "sample_mapping": mapping,
        "n_written_or_reused": done,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"n_targets": done, "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
