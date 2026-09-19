#!/usr/bin/env python3
"""Gaussian source profile analysis for the quantitative-source study.

Dice answers "where is the source"; it says nothing about whether the
reconstructed intensity *profile* is right. This extracts 1-D cuts through each
source centre and compares the GT profile against the coarse-only and final
reconstructions, so peak recovery, decay width and over-smoothing are visible
directly rather than inferred from an aggregate score.

Reports per-source numbers over the whole split (amplitude and width recovery)
plus PNG panels for a few representative cases.

CPU only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.utils.frame import FrameManifest  # noqa: E402
from du2vox.utils.gt_io import (  # noqa: E402
    load_canonical_gt,
    load_normalization_scale,
)

LAYERS = {"coarse": "stage1_fem", "final": "step3_fem"}
AXES = {"x": 0, "y": 1, "z": 2}


def profile_along(
    centers: np.ndarray,
    values: np.ndarray,
    centroid: np.ndarray,
    axis: int,
    half_width_mm: float,
    transverse_radius_mm: float = 1.0,
    n_bins: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean value vs offset along one axis, sampled from a thin cylinder.

    A full slab (constraining only the axis coordinate) is dominated by
    background: a ~0.5 mm source occupies a tiny fraction of a 6 mm slab, so the
    binned mean is diluted toward zero and can even go negative where the
    residual model dips below zero. Restricting the transverse distance to a
    small radius keeps the cut inside the source.
    """
    offset_axis = centers[:, axis] - centroid[axis]
    transverse_sq = np.sum((centers - centroid) ** 2, axis=1) - offset_axis**2
    sel = (np.abs(offset_axis) <= half_width_mm) & (transverse_sq <= transverse_radius_mm**2)
    if not np.any(sel):
        return np.array([]), np.array([])
    edges = np.linspace(-half_width_mm, half_width_mm, n_bins + 1)
    idx = np.clip(np.digitize(offset_axis[sel], edges) - 1, 0, n_bins - 1)
    out = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    np.add.at(out, idx, values[sel])
    np.add.at(counts, idx, 1.0)
    mid = 0.5 * (edges[:-1] + edges[1:])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(counts > 0, out / np.maximum(counts, 1), np.nan)
    return mid, mean


def fwhm(x: np.ndarray, y: np.ndarray) -> float:
    """Full width at half maximum in mm; NaN if the profile never reaches half."""
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    x, y = x[ok], y[ok]
    peak = y.max()
    if peak <= 0:
        return float("nan")
    above = y >= 0.5 * peak
    if above.sum() < 2:
        return float("nan")
    return float(x[above].max() - x[above].min())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--n-figures", type=int, default=6)
    parser.add_argument("--half-width-mm", type=float, default=3.0)
    parser.add_argument(
        "--normalization-scale-filename",
        help="Sample-local scalar divisor, e.g. gt_scale.npy.",
    )
    args = parser.parse_args()

    frame = FrameManifest.load(args.shared_dir)
    ids = [line.strip() for line in args.split_file.read_text().splitlines() if line.strip()]
    if args.max_samples is not None:
        ids = ids[: args.max_samples]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_sample: list[dict] = []
    fig_data: list[tuple] = []

    for sid in ids:
        pred_path = args.predictions_dir / f"{sid}.npz"
        if not pred_path.exists():
            continue
        sample_dir = args.dataset_root / "samples" / sid
        with open(sample_dir / "tumor_params.json") as f:
            tumor = json.load(f)
        foci = tumor.get("foci", [])
        if not foci:
            continue

        with np.load(pred_path) as z:
            valid = z["valid_flat_indices"]
            preds = {
                name: np.asarray(z[key]).astype(np.float64)
                for name, key in LAYERS.items()
                if key in z.files
            }
            normalization_scale = (
                load_normalization_scale(sample_dir, args.normalization_scale_filename)
                if args.normalization_scale_filename
                else None
            )
            gt, _ = load_canonical_gt(
                sample_dir,
                valid,
                gt_mode="continuous",
                normalize="per_sample_peak",
                normalization_scale=normalization_scale,
            )
            gt = gt.astype(np.float64)

            shape = frame.gt_shape
            spacing = float(frame.gt_spacing_mm)
            offset = np.asarray(frame.gt_offset_world_mm, dtype=np.float64)
            idx = np.stack(np.unravel_index(valid, shape), axis=1).astype(np.float64)
            centers = offset + idx * spacing

            row: dict = {"sample_id": sid, "num_foci": len(foci)}
            for fi, focus in enumerate(foci):
                c = np.asarray(focus["center"], dtype=np.float64)
                for axis_name, axis in AXES.items():
                    x, g_prof = profile_along(centers, gt, c, axis, args.half_width_mm)
                    if x.size == 0:
                        continue
                    row[f"f{fi}_{axis_name}_gt_peak"] = float(np.nanmax(g_prof))
                    row[f"f{fi}_{axis_name}_gt_fwhm"] = fwhm(x, g_prof)
                    for name, pred in preds.items():
                        _, p_prof = profile_along(centers, pred, c, axis, args.half_width_mm)
                        pk_g = np.nanmax(g_prof)
                        if pk_g > 0:
                            row[f"f{fi}_{axis_name}_{name}_peak_ratio"] = float(
                                np.nanmax(p_prof) / pk_g
                            )
                        wg = fwhm(x, g_prof)
                        wp = fwhm(x, p_prof)
                        if np.isfinite(wg) and wg > 0 and np.isfinite(wp):
                            row[f"f{fi}_{axis_name}_{name}_fwhm_ratio"] = float(wp / wg)
                if len(fig_data) < args.n_figures and fi == 0:
                    fig_data.append((sid, fi, c, centers, gt, preds))
            per_sample.append(row)

    if not per_sample:
        raise SystemExit("No samples scored")

    # Aggregate the per-axis ratios.
    summary: dict = {}
    for name in LAYERS:
        for kind in ("peak_ratio", "fwhm_ratio"):
            vals = [
                v
                for row in per_sample
                for k, v in row.items()
                if k.endswith(f"_{name}_{kind}") and np.isfinite(v)
            ]
            if vals:
                a = np.asarray(vals)
                summary[f"{name}_{kind}"] = {
                    "mean": float(a.mean()),
                    "median": float(np.median(a)),
                    "n": int(a.size),
                }
    gt_fwhm = [
        v for row in per_sample for k, v in row.items() if k.endswith("_gt_fwhm") and np.isfinite(v)
    ]
    if gt_fwhm:
        summary["gt_fwhm_mm"] = {
            "mean": float(np.mean(gt_fwhm)),
            "median": float(np.median(gt_fwhm)),
            "n": len(gt_fwhm),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            {"n_samples": len(per_sample), "summary": summary, "per_sample": per_sample},
            f,
            indent=2,
        )

    print(f"profiled {len(per_sample)} samples over {len(LAYERS)} layers")
    for key, s in sorted(summary.items()):
        print(f"  {key:<28s} mean={s['mean']:.4f} median={s['median']:.4f} n={s['n']}")

    if args.figures_dir and fig_data:
        args.figures_dir.mkdir(parents=True, exist_ok=True)
        for sid, fi, c, centers, gt, preds in fig_data:
            fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
            for ax, (axis_name, axis) in zip(axes, AXES.items()):
                x, g_prof = profile_along(centers, gt, c, axis, args.half_width_mm)
                ax.plot(x, g_prof, "k-", lw=2.2, label="GT")
                for name, pred in preds.items():
                    _, p_prof = profile_along(centers, pred, c, axis, args.half_width_mm)
                    ax.plot(x, p_prof, "--", lw=1.6, label=name)
                ax.axhline(0.5, color="gray", ls=":", lw=1)
                ax.set_title(f"{sid} focus{fi}  {axis_name}-cut")
                ax.set_xlabel("offset from centre (mm)")
                ax.set_ylabel("normalised intensity")
                ax.grid(alpha=0.25)
            axes[0].legend(fontsize=8)
            fig.tight_layout()
            out = args.figures_dir / f"{sid}_profile.png"
            fig.savefig(out, dpi=140)
            plt.close(fig)
            print(f"  figure -> {out}")

    print(f"\n-> {args.output}")


if __name__ == "__main__":
    main()
