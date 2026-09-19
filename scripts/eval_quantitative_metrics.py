#!/usr/bin/env python3
"""Quantitative-source metrics for DU2Vox reconstructions.

Dice-style overlap metrics cannot express whether a reconstruction recovered
the *fluorescence intensity* of a spatially varying source. This script scores
saved predictions against the continuous GT on the quantities the
quantitative-source study actually asks about:

  - PSNR / Pearson / CCC            (continuous field fidelity)
  - relative L2 / MSE               (scale-sensitive error)
  - R_peak                          (per-source peak amplitude recovery)
  - R_int                           (per-source integrated fluorescence recovery)
  - contrast ratio recovery         (multi-source relative intensity)

Layers are scored identically so a coarse-only and a final dual-space
reconstruction of the same sample are directly comparable.

Predictions are read from ``*_fem`` arrays written by
``du2vox.evaluation.error_structured.evaluate_dense``; those are indexed in the
canonical valid-voxel domain, matching the GT projection used here.

CPU only: no model, no GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.utils.frame import FrameManifest
from du2vox.utils.gt_io import (
    GT_MODES,
    GT_NORMALIZATIONS,
    load_canonical_gt,
    load_normalization_scale,
)

# Layers saved by evaluate_dense, in increasing refinement order.
LAYERS = {
    "coarse": "stage1_fem",
    "step1": "step1_fem",
    "step2": "step2_fem",
    "final": "step3_fem",
}

SIGNAL_FRACTION = 0.01
"""A voxel counts as signal when GT exceeds this fraction of the sample peak."""


def psnr(predicted: np.ndarray, target: np.ndarray, peak: float) -> float:
    """PSNR in dB against an explicit dynamic-range peak."""
    mse = float(np.mean((predicted - target) ** 2))
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * np.log10((peak**2) / mse))


def pearson(predicted: np.ndarray, target: np.ndarray) -> float:
    """Pearson correlation; NaN when either side is constant."""
    p = predicted - predicted.mean()
    t = target - target.mean()
    denom = float(np.linalg.norm(p) * np.linalg.norm(t))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(p, t) / denom)


def ccc(predicted: np.ndarray, target: np.ndarray) -> float:
    """Lin's concordance correlation coefficient (penalizes scale/offset drift)."""
    mp, mt = float(predicted.mean()), float(target.mean())
    vp, vt = float(predicted.var()), float(target.var())
    cov = float(np.mean((predicted - mp) * (target - mt)))
    denom = vp + vt + (mp - mt) ** 2
    if denom <= 0.0:
        return float("nan")
    return float(2.0 * cov / denom)


def voxel_centers(frame: FrameManifest, flat_indices: np.ndarray) -> np.ndarray:
    """World-mm coordinates of the canonical valid-voxel centers."""
    shape = np.asarray(frame.gt_shape, dtype=np.int64)
    spacing = float(frame.gt_spacing_mm)
    offset = np.asarray(frame.gt_offset_world_mm, dtype=np.float64)
    idx = np.stack(np.unravel_index(flat_indices, shape), axis=1).astype(np.float64)
    return offset + idx * spacing


def focus_masks(centers_mm: np.ndarray, tumor: dict) -> list[tuple[str, np.ndarray]]:
    """Assign each voxel to the nearest focus whose 3-sigma ball contains it.

    D1-Q samples the Gaussian sigma through the focus ``radius`` (or the
    ellipsoid half-axes), and the generator truncates at 3 sigma, so the same
    scale is used to define each source's support for amplitude/integral
    attribution. Voxels outside every ball stay unassigned.
    """
    foci = tumor.get("foci", [])
    if not foci:
        return []
    centers = np.asarray([f["center"] for f in foci], dtype=np.float64)
    scales = []
    for f in foci:
        if f.get("shape") == "sphere":
            scales.append(float(f["radius"]))
        else:
            axes = [f.get(k) for k in ("rx", "ry", "rz")]
            axes = [float(a) for a in axes if a is not None]
            scales.append(max(axes) if axes else float(f.get("radius", 1.0)))
    scales_arr = np.asarray(scales, dtype=np.float64)

    deltas = centers_mm[:, None, :] - centers[None, :, :]
    dist = np.linalg.norm(deltas, axis=2)
    nearest = np.argmin(dist, axis=1)
    radius = 3.0 * scales_arr[nearest]

    masks: list[tuple[str, np.ndarray]] = []
    for i, focus in enumerate(foci):
        inside = (nearest == i) & (dist[:, i] <= np.maximum(radius, 1e-6))
        if inside.any():
            masks.append((f"focus{i}", inside))
    return masks


def score_layer(
    predicted: np.ndarray,
    gt: np.ndarray,
    signal: np.ndarray,
    masks: list[tuple[str, np.ndarray]],
    peak: float,
    centers_mm: np.ndarray | None = None,
) -> dict[str, float]:
    """Score one reconstruction layer over the signal region."""
    out: dict[str, float] = {}
    p_sig, g_sig = predicted[signal], gt[signal]

    # Dice on the 0.5 level set of each field normalized by its own peak.
    # This is scale-invariant, so it is comparable across a binary-trained
    # model and a continuous one, and across D0 and D1-Q. For a binary GT
    # (D0) it reduces exactly to the historical support Dice, because
    # normalizing 0/1 leaves 0/1 and the 0.5 level set is the support.
    # For a continuous GT it asks whether the reconstruction concentrates
    # intensity where the source does, rather than merely overlapping it.
    if p_sig.size and p_sig.max() > 0 and g_sig.max() > 0:
        p_bin = p_sig >= 0.5 * p_sig.max()
        g_bin = g_sig >= 0.5 * g_sig.max()
        tp = float(np.count_nonzero(p_bin & g_bin))
        out["dice_halfmax"] = float(2 * tp / (p_bin.sum() + g_bin.sum() + 1e-12))
        # Localization: distance between the intensity centroids.
        pw, gw = np.clip(p_sig, 0, None), np.clip(g_sig, 0, None)
        if pw.sum() > 0 and gw.sum() > 0 and centers_mm is not None:
            c_pred = (centers_mm[signal] * pw[:, None]).sum(0) / pw.sum()
            c_true = (centers_mm[signal] * gw[:, None]).sum(0) / gw.sum()
            out["centroid_error_mm"] = float(np.linalg.norm(c_pred - c_true))

    if p_sig.size:
        out["psnr_db"] = psnr(p_sig, g_sig, peak)
        out["pearson"] = pearson(p_sig, g_sig)
        out["ccc"] = ccc(p_sig, g_sig)
        err = p_sig - g_sig
        out["relative_l2"] = float(np.linalg.norm(err) / max(np.linalg.norm(g_sig), 1e-30))
        out["mse"] = float(np.mean(err**2))
        # Ratio of total recovered to total true fluorescence over the region.
        out["integrated_ratio"] = float(p_sig.sum() / max(g_sig.sum(), 1e-30))

    # Per-source amplitude and integral recovery.
    peaks, integrals = [], []
    for name, mask in masks:
        p_i, g_i = predicted[mask], gt[mask]
        if g_i.size == 0:
            continue
        g_peak = float(g_i.max())
        if g_peak > 0:
            r_peak = float(p_i.max()) / g_peak
            peaks.append(r_peak)
            out[f"rpeak_{name}"] = r_peak
        g_sum = float(g_i.sum())
        if g_sum > 0:
            r_int = float(p_i.sum()) / g_sum
            integrals.append(r_int)
            out[f"rint_{name}"] = r_int
    if peaks:
        out["rpeak_mean"] = float(np.mean(peaks))
        out["rpeak_median"] = float(np.median(peaks))
        out["rpeak_abs_err_mean"] = float(np.mean(np.abs(np.asarray(peaks) - 1.0)))
    if integrals:
        out["rint_mean"] = float(np.mean(integrals))
        out["rint_median"] = float(np.median(integrals))
        out["rint_abs_err_mean"] = float(np.mean(np.abs(np.asarray(integrals) - 1.0)))

    # Contrast recovery across source pairs (multi-focus samples only).
    # Contrast is a ratio of source *amplitudes*, so it is compared peak to
    # peak. An earlier min/max formulation was dominated by a single near-zero
    # voxel and produced unbounded values on coarse reconstructions that do not
    # separate the sources; peak ratios stay bounded and are the quantity the
    # study actually reports.
    if len(masks) >= 2:
        gt_peaks = np.asarray(
            [gt[m].max() if gt[m].size else 0.0 for _, m in masks], dtype=np.float64
        )
        pred_peaks = np.asarray(
            [predicted[m].max() if predicted[m].size else 0.0 for _, m in masks],
            dtype=np.float64,
        )
        order = np.argsort(-gt_peaks)
        pred_scale = float(pred_peaks.max()) if pred_peaks.size else 0.0
        ratios_true, ratios_pred, errors, log_errors = [], [], [], []
        undetected = 0
        for a in range(len(order)):
            for b in range(a + 1, len(order)):
                i, j = int(order[a]), int(order[b])
                g_hi, g_lo = gt_peaks[i], gt_peaks[j]
                if g_hi <= 0 or g_lo <= 0:
                    continue
                if pred_peaks[j] <= 1e-2 * max(pred_scale, 1e-12):
                    # The lower-intensity source is essentially absent from the
                    # prediction. Its ratio is undefined rather than merely
                    # large, so it is counted separately instead of polluting
                    # the error statistics with an enormous number.
                    undetected += 1
                    continue
                r_true = float(g_hi / g_lo)
                r_pred = float(pred_peaks[i] / pred_peaks[j])
                ratios_true.append(r_true)
                ratios_pred.append(r_pred)
                errors.append(abs(r_pred - r_true))
                # A zero prediction on the dominant source makes r_pred == 0,
                # whose log is undefined; keep the finite-domain absolute error
                # but leave this pair out of the log statistic.
                if r_pred > 0.0:
                    log_errors.append(abs(float(np.log(r_pred) - np.log(r_true))))
        out["contrast_ratio_undetected"] = int(undetected)
        if errors:
            # Guard against non-finite entries (e.g. a degenerate prediction)
            # so one pathological pair cannot turn a summary statistic into NaN.
            err_arr = np.asarray([e for e in errors if np.isfinite(e)])
            log_arr = np.asarray([e for e in log_errors if np.isfinite(e)])
            true_arr = np.asarray([r for r in ratios_true if np.isfinite(r)])
            pred_arr = np.asarray([r for r in ratios_pred if np.isfinite(r)])
            if err_arr.size:
                out["contrast_ratio_true_mean"] = float(true_arr.mean())
                out["contrast_ratio_pred_mean"] = float(pred_arr.mean())
                out["contrast_ratio_abs_err_mean"] = float(err_arr.mean())
                out["contrast_ratio_abs_err_median"] = float(np.median(err_arr))
                # Log-space error stays comparable across ratio magnitudes.
                # Computed separately: a ratio that overflowed to inf is
                # non-finite and drops out, so this array can be shorter (or
                # empty) even when err_arr is populated.
                if log_arr.size:
                    out["contrast_ratio_log_err_mean"] = float(log_arr.mean())
                out["contrast_ratio_n_pairs"] = int(err_arr.size)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--layers",
        nargs="+",
        default=list(LAYERS),
        help=f"Subset of {list(LAYERS)} to score",
    )
    parser.add_argument(
        "--gt-mode",
        choices=GT_MODES,
        default="continuous",
        help=(
            "continuous scores the actual fluorescence intensity (the quantity "
            "this study reports); binary reproduces the historical support-only "
            "target for comparison."
        ),
    )
    parser.add_argument(
        "--normalize-gt",
        choices=GT_NORMALIZATIONS,
        default="none",
        help=(
            "Must match how the evaluated model was supervised, because the "
            "prediction is on that same scale. Exp A (a D0-trained model) "
            "outputs raw-scale values, so use 'none'. Exp B trained on "
            "per-sample-peak-normalized targets, so use 'per_sample_peak'. "
            "Ratio metrics (R_peak, R_int, relative L2, contrast) are "
            "invariant to this choice; MSE and PSNR are not."
        ),
    )
    parser.add_argument(
        "--normalization-scale-filename",
        help="Sample-local scalar divisor, e.g. gt_scale.npy.",
    )
    args = parser.parse_args()

    frame = FrameManifest.load(args.shared_dir)
    ids = [line.strip() for line in args.split_file.read_text().splitlines() if line.strip()]
    if args.max_samples is not None:
        ids = ids[: args.max_samples]

    per_sample: list[dict] = []
    skipped: list[str] = []
    for sid in ids:
        pred_path = args.predictions_dir / f"{sid}.npz"
        if not pred_path.exists():
            skipped.append(sid)
            continue

        with np.load(pred_path) as z:
            valid = z["valid_flat_indices"]
            sample_dir = args.dataset_root / "samples" / sid
            normalization_scale = (
                load_normalization_scale(sample_dir, args.normalization_scale_filename)
                if args.normalization_scale_filename
                else None
            )
            gt, gt_scale = load_canonical_gt(
                sample_dir,
                valid,
                gt_mode=args.gt_mode,
                normalize=args.normalize_gt,
                normalization_scale=normalization_scale,
            )
            gt = gt.astype(np.float64)
            centers = voxel_centers(frame, valid)
            with open(args.dataset_root / "samples" / sid / "tumor_params.json") as f:
                tumor = json.load(f)
            masks = focus_masks(centers, tumor)
            peak = float(gt.max()) if gt.size else 0.0
            signal = gt > SIGNAL_FRACTION * peak if peak > 0 else gt > 0

            row: dict = {
                "sample_id": sid,
                "num_foci": int(tumor.get("num_foci", len(tumor.get("foci", [])))),
                "depth_tier": tumor.get("depth_tier", "unknown"),
                "gt_peak": peak,
                # Divisor applied under per_sample_peak; 1.0 when unnormalized.
                # Multiply normalized values by this to recover raw intensity.
                "gt_scale": float(gt_scale),
                "n_signal_voxels": int(signal.sum()),
            }
            for layer in args.layers:
                key = LAYERS[layer]
                if key not in z.files:
                    continue
                pred = np.asarray(z[key]).astype(np.float64)
                metrics = score_layer(pred, gt, signal, masks, peak, centers)
                for name, value in metrics.items():
                    row[f"{layer}_{name}"] = value
            per_sample.append(row)

    if not per_sample:
        raise SystemExit("No predictions scored; check --predictions-dir")

    # Aggregate: mean/median across samples, per layer.
    summary: dict = {}
    for layer in args.layers:
        prefix = f"{layer}_"
        keys = sorted(
            {
                k[len(prefix) :]
                for row in per_sample
                for k in row
                if k.startswith(prefix)
                and not k.startswith(prefix + "rpeak_focus")
                and not k.startswith(prefix + "rint_focus")
            }
        )
        layer_summary = {}
        for key in keys:
            values = np.asarray(
                [row[f"{prefix}{key}"] for row in per_sample if f"{prefix}{key}" in row],
                dtype=np.float64,
            )
            values = values[np.isfinite(values)]
            if values.size:
                layer_summary[key] = {
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "n": int(values.size),
                }
        summary[layer] = layer_summary

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            {
                "n_samples": len(per_sample),
                "skipped": skipped,
                "layers": args.layers,
                "gt_mode": args.gt_mode,
                "normalize_gt": args.normalize_gt,
                "signal_threshold_fraction": SIGNAL_FRACTION,
                "summary": summary,
                "per_sample": per_sample,
            },
            f,
            indent=2,
        )

    print(f"scored {len(per_sample)} samples, skipped {len(skipped)}")
    print(f"-> {args.output}\n")
    print(f"{'metric':<30s} " + " ".join(f"{ly:>12s}" for ly in args.layers))
    print("-" * (30 + 13 * len(args.layers)))
    all_keys = sorted({k for ly in summary for k in summary[ly]})
    for key in all_keys:
        cells = []
        for ly in args.layers:
            entry = summary.get(ly, {}).get(key)
            cells.append(f"{entry['mean']:>12.4f}" if entry else f"{'-':>12s}")
        print(f"{key:<30s} " + " ".join(cells))


if __name__ == "__main__":
    main()
