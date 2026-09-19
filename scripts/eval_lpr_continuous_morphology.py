#!/usr/bin/env python3
"""Unified source-resolved morphology evaluation for the continuous LPR cohort.

Every method is lifted to the *same* canonical voxel grid through the frozen
prolongation operator ``p``, so Tikhonov, the Stage-1/M0 nodal state, and the
P1-coefficient oracle are directly comparable.

For each GT source the 50% and 20% iso-intensity supports are defined from that
source's own local peak, and the prediction is scored against its *matched*
local peak inside the same GT-defined Mahalanobis ROI. Weak sources therefore
cannot be erased by a global threshold.

Raw ``continuous_metrics`` are retained as the continuous-field auxiliary
(notably ``relative_l2``); nothing is clamped or independently normalized.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.bridge.canonical_cross_discretization import (  # noqa: E402
    CanonicalCrossDiscretization,
)
from du2vox.evaluation.continuous_field import continuous_metrics  # noqa: E402
from du2vox.evaluation.source_morphology import (  # noqa: E402
    SourceMorphologyProtocol,
    aggregate_source_rows,
    source_morphology_metrics,
)
from du2vox.utils.frame import FrameManifest  # noqa: E402
from du2vox.utils.gt_io import load_canonical_gt, load_normalization_scale  # noqa: E402

FIGURE_METHOD = "stage1_m0"
VIEWS = ("XY", "XZ", "YZ")

NORMALIZATIONS = ("raw", "per_sample_peak", "gt_scale_file")
"""How a prediction/target pair is put on one scale before scoring.

Each option rescales with ONE explicit divisor so the pair stays internally
consistent. Scoring a per-case-normalized prediction against a raw target (or
the reverse) is a scale *mismatch* and silently corrupts ``ccc``,
``relative_l2``, ``peak_relative_error_global`` and ``positive_mass_ratio``.

``raw``
    Nothing is rescaled. Correct for a cohort whose measurement and target
    share the raw physical amplitude (the LPR cohort).
``per_sample_peak``
    Both arrays are divided by the target's own peak, so every case becomes a
    unit-peak normalized field. Mechanically identical across cohorts and needs
    no saved scale file.
``gt_scale_file``
    Only the target is divided, by the cohort's saved per-case scalar
    (``gt_scale.npy``). Required by a per-case-normalized training contract: the
    model output already lives in that normalized space while the raw GT on
    disk does not.
"""

DATA_RANGE = {"raw": 2.0, "per_sample_peak": 1.0, "gt_scale_file": 1.0}
"""Fixed PSNR data range implied by each scaling, per each cohort's protocol."""


def load_ids(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if line]


def parse_prediction(value: str) -> tuple[str, Path, str]:
    """Parse an aligned dense prediction as ``NAME=DIRECTORY:NPZ_KEY``."""

    if "=" not in value or ":" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=DIRECTORY:NPZ_KEY")
    name, location = value.split("=", 1)
    directory, key = location.rsplit(":", 1)
    if not name or not directory or not key:
        raise argparse.ArgumentTypeError("Expected NAME=DIRECTORY:NPZ_KEY")
    return name, Path(directory), key


def load_dense_prediction(directory: Path, key: str, sample_id: str) -> np.ndarray:
    path = directory / f"{sample_id}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as archive:
        if key not in archive:
            raise KeyError(f"{path} does not contain {key!r}")
        return np.asarray(archive[key], dtype=np.float64).reshape(-1)


def lift(canonical: CanonicalCrossDiscretization, nodal: np.ndarray) -> np.ndarray:
    nodal = np.asarray(nodal, dtype=np.float32).reshape(-1)
    if nodal.shape[0] != canonical.p.shape[1]:
        raise RuntimeError(
            f"nodal state has {nodal.shape[0]} values, prolongation expects {canonical.p.shape[1]}"
        )
    return np.asarray(canonical.p @ nodal, dtype=np.float64).reshape(-1)


def strip_masks(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]


def normalize_pair(
    prediction: np.ndarray,
    target: np.ndarray,
    sample_dir: Path,
    mode: str,
    gt_scale_filename: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Put one (prediction, target) pair on a single, explicitly recorded scale.

    Returns the rescaled pair plus the applied divisors, so the convention is
    auditable from the output JSON alone.
    """

    if mode not in NORMALIZATIONS:
        raise ValueError(f"normalization must be one of {NORMALIZATIONS}, got {mode!r}")

    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)

    if mode == "raw":
        target_divisor = prediction_divisor = 1.0
    elif mode == "per_sample_peak":
        peak = float(target.max()) if target.size else 0.0
        if not np.isfinite(peak) or peak <= 0.0:
            raise RuntimeError(f"per_sample_peak needs a positive target peak in {sample_dir}")
        target_divisor = prediction_divisor = peak
    else:  # gt_scale_file
        target_divisor = load_normalization_scale(sample_dir, gt_scale_filename)
        prediction_divisor = 1.0

    scaled_target = target / target_divisor
    scaled_prediction = prediction / prediction_divisor
    record = {
        "mode": mode,
        "target_divisor": float(target_divisor),
        "prediction_divisor": float(prediction_divisor),
        "data_range": float(DATA_RANGE[mode]),
    }
    return scaled_prediction, scaled_target, record


def _to_mask_volume(
    flat_mask: np.ndarray, valid_flat_indices: np.ndarray, grid_shape: tuple[int, int, int]
) -> np.ndarray:
    volume = np.zeros(int(np.prod(grid_shape)), dtype=bool)
    volume[np.asarray(valid_flat_indices, dtype=np.int64)] = flat_mask
    return volume.reshape(grid_shape)


def _to_field_volume(
    flat_values: np.ndarray, valid_flat_indices: np.ndarray, grid_shape: tuple[int, int, int]
) -> np.ndarray:
    volume = np.full(int(np.prod(grid_shape)), np.nan, dtype=np.float64)
    volume[np.asarray(valid_flat_indices, dtype=np.int64)] = flat_values
    return volume.reshape(grid_shape)


def _slice(volume: np.ndarray, view: str, center: tuple[int, int, int]) -> np.ndarray:
    kx, ky, kz = center
    if view == "XY":
        return volume[:, :, kz].T
    if view == "XZ":
        return volume[:, ky, :].T
    return volume[kx, :, :].T


def _axis_labels(view: str) -> tuple[str, str]:
    if view == "XY":
        return "x (mm)", "y (mm)"
    if view == "XZ":
        return "x (mm)", "z (mm)"
    return "y (mm)", "z (mm)"


def _contour(axis, mask2d: np.ndarray, *, color: str, style: str, label: str) -> None:
    """Draw the 0.5 level of a boolean support using the panel's mm extent.

    Without an explicit ``extent`` matplotlib autoscales the axes to the
    contour's *pixel* indices, which collapses the mm-space imshow underneath
    into a tiny corner. The extent is therefore read back from that image.
    """
    if not np.any(mask2d):
        return
    extent = axis.images[0].get_extent() if axis.images else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        axis.contour(
            mask2d.astype(np.float64),
            levels=[0.5],
            colors=[color],
            linestyles=[style],
            linewidths=1.2,
            extent=extent,
        )


def render_case(
    *,
    sid: str,
    num_foci: int,
    gt_volume: np.ndarray,
    gt_masks: dict[str, np.ndarray],
    method_fields: dict[str, np.ndarray],
    method_masks: dict[str, dict[str, np.ndarray]],
    center_voxel: tuple[int, int, int],
    extent_mm: dict[str, tuple[float, float]],
    output: Path,
) -> None:
    columns = ["GT"] + list(method_fields)
    figure, axes = plt.subplots(
        len(VIEWS), len(columns), figsize=(4.2 * len(columns), 4.0 * len(VIEWS)), squeeze=False
    )
    for row, view in enumerate(VIEWS):
        for column, name in enumerate(columns):
            axis = axes[row][column]
            field = gt_volume if name == "GT" else method_fields[name]
            panel = _slice(field, view, center_voxel)
            axis.imshow(
                panel,
                origin="lower",
                cmap="magma",
                extent=extent_mm[view],
                vmin=0.0,
                vmax=float(np.nanmax(gt_volume)) or 1.0,
                interpolation="nearest",
                aspect="equal",
            )
            _contour(
                axis,
                _slice(gt_masks["gt_50"], view, center_voxel),
                color="white",
                style="solid",
                label="GT 50%",
            )
            _contour(
                axis,
                _slice(gt_masks["gt_20"], view, center_voxel),
                color="white",
                style="dashed",
                label="GT 20%",
            )
            if name != "GT":
                _contour(
                    axis,
                    _slice(method_masks[name]["pred_50"], view, center_voxel),
                    color="red",
                    style="solid",
                    label="pred 50%",
                )
                _contour(
                    axis,
                    _slice(method_masks[name]["pred_20"], view, center_voxel),
                    color="red",
                    style="dashed",
                    label="pred 20%",
                )
            axis.set_title(f"{name} | {view}", fontsize=10)
            x_label, y_label = _axis_labels(view)
            axis.set_xlabel(x_label, fontsize=8)
            axis.set_ylabel(y_label, fontsize=8)
            axis.tick_params(labelsize=7)
    figure.suptitle(
        f"{sid} | GT sources={num_foci} | white=GT 20%/50%, red=prediction 20%/50%",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=130)
    plt.close(figure)


def select_figure_cases(rows: list[dict], count: int) -> list[str]:
    """Mechanical selection by the figure method's mean per-case Dice@50."""

    scored = []
    for row in rows:
        metrics = row["methods"].get(FIGURE_METHOD)
        if metrics is None:
            continue
        values = [source["dice_50"] for source in metrics["morphology"]["per_source"]]
        if not values:
            continue
        scored.append((float(np.mean(values)), row["sample_id"], row["num_foci"]))
    if not scored:
        return []
    scored.sort()
    selected: list[str] = []
    for fraction in (0.25, 0.5, 0.75):
        index = int(round(fraction * (len(scored) - 1)))
        candidate = scored[index][1]
        if candidate not in selected:
            selected.append(candidate)
    multi = [entry for entry in scored if entry[2] >= 2]
    if multi:
        candidate = multi[len(multi) // 2][1]
        if candidate not in selected:
            selected.append(candidate)
    return selected[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--operator-cache", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--bridge-dir", type=Path, help="Stage-1/M0 per-sample coarse_d.npy")
    parser.add_argument("--tikhonov-dir", type=Path, help="per-sample {sid}.npz with fem_nodes")
    parser.add_argument(
        "--prediction",
        action="append",
        type=parse_prediction,
        default=[],
        help="Aligned dense prediction as NAME=DIRECTORY:NPZ_KEY; repeatable.",
    )
    parser.add_argument(
        "--include-p1-oracle", action="store_true", help="lift gt_nodes.npy as the P1 oracle"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path)
    parser.add_argument("--figure-count", type=int, default=4)
    parser.add_argument(
        "--normalization",
        choices=NORMALIZATIONS,
        default="raw",
        help=(
            "Scale convention applied to every (prediction, target) pair. "
            "'raw' for a raw-amplitude cohort; 'gt_scale_file' for a "
            "per-case-normalized cohort whose sample dir carries the saved scale."
        ),
    )
    parser.add_argument(
        "--gt-scale-filename",
        default="gt_scale.npy",
        help="Per-case scalar used by --normalization gt_scale_file",
    )
    parser.add_argument("--freeze-receipt", type=Path)
    parser.add_argument("--dataset-receipt", type=Path)
    args = parser.parse_args()

    if args.split == "test":
        if args.freeze_receipt is None or args.dataset_receipt is None:
            raise RuntimeError(
                "Development-test evaluation requires --freeze-receipt and --dataset-receipt"
            )
        receipt = json.loads(args.freeze_receipt.read_text())
        if receipt.get("status") != "frozen_on_val300":
            raise RuntimeError("Invalid validation-freeze status")
        if receipt.get("sealed_confirmation_accessed") is not False:
            raise RuntimeError("Receipt does not certify unopened confirmation data")

    dense_predictions = {name: (directory, key) for name, directory, key in args.prediction}
    if len(dense_predictions) != len(args.prediction):
        raise RuntimeError("Prediction method names must be unique")
    if (
        args.bridge_dir is None
        and args.tikhonov_dir is None
        and not args.include_p1_oracle
        and not dense_predictions
    ):
        raise RuntimeError("At least one prediction source is required")

    ids = load_ids(args.dataset_root / "splits" / f"{args.split}.txt")
    samples_dir = args.dataset_root / "samples"
    protocol = SourceMorphologyProtocol()

    canonical = CanonicalCrossDiscretization(
        args.operator_cache,
        shared_dir=args.shared_dir,
        factorize=False,
    )
    operator = canonical.operator
    frame = FrameManifest.load(args.shared_dir)
    valid_flat_indices = operator.valid_flat_indices
    grid_shape = operator.grid_shape
    spacing = float(operator.spacing_mm)
    offset = np.asarray(frame.gt_offset_world_mm, dtype=np.float64)
    coords = np.asarray(operator.coords_world, dtype=np.float64)

    rows: list[dict] = []
    render_cache: dict[str, dict] = {}
    for index, sid in enumerate(ids, start=1):
        sample_dir = samples_dir / sid
        gt, _ = load_canonical_gt(
            sample_dir,
            valid_flat_indices,
            gt_mode="continuous",
            normalize="none",
        )
        gt = np.asarray(gt, dtype=np.float64).reshape(-1)
        tumor = json.loads((sample_dir / "tumor_params.json").read_text())
        foci = tumor["foci"]

        predictions: dict[str, np.ndarray] = {}
        if args.bridge_dir is not None:
            predictions["stage1_m0"] = lift(
                canonical, np.load(args.bridge_dir / sid / "coarse_d.npy")
            )
        if args.tikhonov_dir is not None:
            with np.load(args.tikhonov_dir / f"{sid}.npz") as archive:
                predictions["tikhonov"] = lift(canonical, archive["fem_nodes"])
        if args.include_p1_oracle:
            predictions["p1_oracle"] = lift(canonical, np.load(sample_dir / "gt_nodes.npy"))
        for name, (directory, key) in dense_predictions.items():
            predictions[name] = load_dense_prediction(directory, key, sid)

        case = {"sample_id": sid, "num_foci": int(tumor.get("num_foci", len(foci))), "methods": {}}
        # Masks are deliberately NOT kept here: 300 cases x 3 methods x sources x
        # full-domain boolean masks would be tens of GB. Figures re-derive them for
        # the few mechanically selected cases only.
        for name, prediction in predictions.items():
            scaled_prediction, scaled_gt, normalization = normalize_pair(
                prediction,
                gt,
                sample_dir,
                args.normalization,
                args.gt_scale_filename,
            )
            occurrence = source_morphology_metrics(
                scaled_prediction,
                scaled_gt,
                coords,
                foci,
                valid_flat_indices,
                grid_shape,
                spacing,
                offset,
                protocol=protocol,
            )
            continuous = continuous_metrics(
                scaled_prediction, scaled_gt, data_range=normalization["data_range"]
            )
            case["methods"][name] = {
                "normalization": normalization,
                "morphology": occurrence,
                "continuous": {
                    "relative_l2": continuous["relative_l2"],
                    "ccc": continuous["ccc"],
                    "pearson": continuous["pearson"],
                    "positive_mass_ratio": continuous["positive_mass_ratio"],
                    "peak_relative_error_global": continuous["peak_relative_error_global"],
                },
            }
        rows.append(case)
        print(f"[{index}/{len(ids)}] {sid}", flush=True)

    method_names = sorted({name for row in rows for name in row["methods"]})
    summary: dict[str, dict] = {}
    for name in method_names:
        all_sources = [
            source for row in rows for source in row["methods"][name]["morphology"]["per_source"]
        ]
        weak_sources = [source for source in all_sources if source["is_weak_source"]]
        summary[name] = {
            "overall": aggregate_source_rows(strip_masks(all_sources)),
            "weak_sources": aggregate_source_rows(strip_masks(weak_sources)),
            "continuous": {
                key: float(np.nanmean([row["methods"][name]["continuous"][key] for row in rows]))
                for key in rows[0]["methods"][name]["continuous"]
            },
        }

    if args.figures_dir is not None:
        selected = select_figure_cases(rows, args.figure_count)
        by_id = {row["sample_id"]: row for row in rows}
        for sid in selected:
            render_cache[sid] = by_id[sid]

    payload = {
        "protocol": {
            "domain": "canonical valid voxel centers via the frozen prolongation operator",
            "lift": "p @ nodal_state (identical operator for every method)",
            "roi_mahalanobis": protocol.roi_mahalanobis,
            "iso_fractions": [protocol.tight_fraction, protocol.loose_fraction],
            "detection_fraction": protocol.detection_fraction,
            "weak_source_rule": (
                f"intensity * {protocol.weak_source_ratio} < max intensity within the case"
            ),
            "peak_definition": "each source uses its own local peak (GT and prediction)",
            "masked_ssim3d": False,
        },
        "normalization": {
            "mode": args.normalization,
            "gt_scale_filename": (
                args.gt_scale_filename if args.normalization == "gt_scale_file" else None
            ),
            "data_range": float(DATA_RANGE[args.normalization]),
            "note": (
                "One explicit divisor per pair; see per_sample[].methods[].normalization "
                "for the applied values. Scale-invariant metrics (pearson, morphology) "
                "are unaffected by the choice."
            ),
        },
        "split": args.split,
        "n_samples": len(rows),
        "methods": method_names,
        "sources": {
            "stage1_m0": str(args.bridge_dir) if args.bridge_dir else None,
            "tikhonov": str(args.tikhonov_dir) if args.tikhonov_dir else None,
            "p1_oracle": "gt_nodes.npy" if args.include_p1_oracle else None,
            "dense_predictions": {
                name: {"directory": str(directory), "npz_key": key}
                for name, (directory, key) in dense_predictions.items()
            },
        },
        "figure_cases": sorted(render_cache),
        "summary": summary,
        "per_sample": [
            {
                "sample_id": row["sample_id"],
                "num_foci": row["num_foci"],
                "methods": {
                    name: {
                        "normalization": entry["normalization"],
                        "morphology": {
                            "n_sources": entry["morphology"]["n_sources"],
                            "per_source": strip_masks(entry["morphology"]["per_source"]),
                        },
                        "continuous": entry["continuous"],
                    }
                    for name, entry in row["methods"].items()
                },
            }
            for row in rows
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")

    if args.figures_dir is not None:
        for sid in sorted(render_cache):
            row = render_cache[sid]
            tumor = json.loads((samples_dir / sid / "tumor_params.json").read_text())
            gt, _ = load_canonical_gt(
                samples_dir / sid, valid_flat_indices, gt_mode="continuous", normalize="none"
            )
            predictions = {}
            if args.bridge_dir is not None:
                predictions["stage1_m0"] = lift(
                    canonical, np.load(args.bridge_dir / sid / "coarse_d.npy")
                )
            if args.tikhonov_dir is not None:
                with np.load(args.tikhonov_dir / f"{sid}.npz") as archive:
                    predictions["tikhonov"] = lift(canonical, archive["fem_nodes"])
            if args.include_p1_oracle:
                predictions["p1_oracle"] = lift(
                    canonical, np.load(samples_dir / sid / "gt_nodes.npy")
                )
            for name, (directory, key) in dense_predictions.items():
                predictions[name] = load_dense_prediction(directory, key, sid)

            # Figures must use the same scale convention as the metrics, otherwise
            # the shared colour scale silently mixes two amplitude spaces.
            if predictions:
                scaled_predictions: dict[str, np.ndarray] = {}
                scaled_gt = gt
                for name, prediction in predictions.items():
                    scaled_prediction, scaled_gt, _ = normalize_pair(
                        prediction,
                        gt,
                        samples_dir / sid,
                        args.normalization,
                        args.gt_scale_filename,
                    )
                    scaled_predictions[name] = scaled_prediction
                predictions = scaled_predictions
                gt = scaled_gt

            method_masks: dict[str, dict[str, np.ndarray]] = {}
            gt_masks = {"gt_50": None, "gt_20": None}
            for name, prediction in predictions.items():
                occurrence = source_morphology_metrics(
                    prediction,
                    gt,
                    coords,
                    tumor["foci"],
                    valid_flat_indices,
                    grid_shape,
                    spacing,
                    offset,
                    protocol=protocol,
                    include_masks=True,
                )
                union = {
                    key: np.zeros_like(valid_flat_indices, dtype=bool)
                    for key in ("pred_50", "pred_20")
                }
                for source in occurrence["per_source"]:
                    union["pred_50"] |= source["_pred_50_mask"]
                    union["pred_20"] |= source["_pred_20_mask"]
                    if gt_masks["gt_50"] is None:
                        gt_masks["gt_50"] = np.zeros_like(valid_flat_indices, dtype=bool)
                        gt_masks["gt_20"] = np.zeros_like(valid_flat_indices, dtype=bool)
                    gt_masks["gt_50"] |= source["_gt_50_mask"]
                    gt_masks["gt_20"] |= source["_gt_20_mask"]
                method_masks[name] = union

            centers = np.asarray([focus["center"] for focus in tumor["foci"]], dtype=np.float64)
            center_world = centers.mean(axis=0)
            center_voxel = tuple(
                int(
                    np.clip(
                        round((center_world[axis] - offset[axis]) / spacing - 0.5),
                        0,
                        grid_shape[axis] - 1,
                    )
                )
                for axis in range(3)
            )
            extent_mm = {
                "XY": (
                    offset[0],
                    offset[0] + grid_shape[0] * spacing,
                    offset[1],
                    offset[1] + grid_shape[1] * spacing,
                ),
                "XZ": (
                    offset[0],
                    offset[0] + grid_shape[0] * spacing,
                    offset[2],
                    offset[2] + grid_shape[2] * spacing,
                ),
                "YZ": (
                    offset[1],
                    offset[1] + grid_shape[1] * spacing,
                    offset[2],
                    offset[2] + grid_shape[2] * spacing,
                ),
            }
            render_case(
                sid=sid,
                num_foci=row["num_foci"],
                gt_volume=_to_field_volume(gt, valid_flat_indices, grid_shape),
                gt_masks={
                    key: _to_mask_volume(value, valid_flat_indices, grid_shape)
                    for key, value in gt_masks.items()
                },
                method_fields={
                    name: _to_field_volume(value, valid_flat_indices, grid_shape)
                    for name, value in predictions.items()
                },
                method_masks={
                    name: {
                        key: _to_mask_volume(value, valid_flat_indices, grid_shape)
                        for key, value in masks.items()
                    }
                    for name, masks in method_masks.items()
                },
                center_voxel=center_voxel,
                extent_mm=extent_mm,
                output=args.figures_dir / f"{sid}.png",
            )
            print(f"[figure] {args.figures_dir / f'{sid}.png'}", flush=True)


if __name__ == "__main__":
    main()
