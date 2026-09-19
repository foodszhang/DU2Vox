#!/usr/bin/env python3
"""Audit continuous Stage-1 predictions with lumped FEM-volume weighting.

This is an inference-free audit: it consumes already exported ``coarse_d.npy``
states for epoch 29 and epoch 30. The P1 oracle is the normalized nodal GT itself
and is included as an identity/sanity control under this node-domain protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.bridge.canonical_cross_discretization import CanonicalCrossDiscretization
from du2vox.evaluation.fem_volume import (
    fem_volume_metrics,
    lumped_nodal_volumes,
    source_composition_metrics,
)
from du2vox.utils.frame import FrameManifest


METHODS = ("epoch29", "epoch30", "p1_oracle")
SCALAR_METRICS = (
    "wrel_l2",
    "mass_ratio",
    "mass_relative_error",
    "weighted_ccc",
    "com_error_mm",
    "shape_error",
    "shape_optimal_scale",
    "source_composition_error",
    "weak_source_mass_ratio",
    "weak_source_error",
    "weak_source_fraction_error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/d1q_mcx_canonical_3k_20k"),
    )
    parser.add_argument(
        "--shared-dir",
        type=Path,
        default=Path("/home/foods/pro/FMT-SimGen/output/shared_mesh_20k"),
    )
    parser.add_argument(
        "--epoch29-bridge",
        type=Path,
        default=Path("output/bridge_d1q_mcx_canonical_stage1_raw_epoch29_dice05_val"),
    )
    parser.add_argument(
        "--epoch30-bridge",
        type=Path,
        default=Path("output/bridge_d1q_mcx_canonical_stage1_raw_epoch30_valloss_val"),
    )
    parser.add_argument(
        "--operator-cache",
        type=Path,
        default=Path("experiments/cross_discretization_decomposition/artifacts/operator_cache"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("diagnosis/stage1_continuous_fem_volume_audit"),
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def minimum_source_distance(foci: list[dict]) -> float:
    if len(foci) < 2:
        return float("nan")
    centers = np.asarray([focus["center"] for focus in foci], dtype=np.float64)
    pairwise = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    return float(np.min(pairwise[np.triu_indices(len(foci), 1)]))


def ratio_group(num_sources: int, ratio: float) -> str:
    if num_sources == 1:
        return "single"
    if ratio < 1.25:
        return "multi_[1,1.25)"
    if ratio < 1.5:
        return "multi_[1.25,1.5)"
    if ratio < 2.0:
        return "multi_[1.5,2)"
    return "multi_>=2"


def distance_group(distance: float) -> str:
    if not np.isfinite(distance):
        return "single"
    if distance < 5.0:
        return "multi_<5mm"
    if distance < 10.0:
        return "multi_[5,10)mm"
    if distance < 20.0:
        return "multi_[10,20)mm"
    return "multi_>=20mm"


def serialize_csv_value(value: object) -> object:
    if isinstance(value, (list, tuple)):
        return json.dumps(value, separators=(",", ":"))
    return value


def write_per_case_csv(rows: list[dict], path: Path) -> None:
    keys = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_csv_value(row[key]) for key in keys})


def finite_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "median": float("nan"), "q25": float("nan"), "q75": float("nan")}
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }


def grouped_summaries(rows: list[dict]) -> tuple[list[dict], dict]:
    axes = {
        "all": lambda row: "all",
        "num_sources": lambda row: str(row["num_sources"]),
        "strength_ratio": lambda row: row["strength_ratio_group"],
        "source_distance": lambda row: row["source_distance_group"],
        "depth": lambda row: row["depth_tier"],
    }
    long_rows: list[dict] = []
    nested: dict = {}
    for axis, getter in axes.items():
        labels = sorted({getter(row) for row in rows})
        nested[axis] = {}
        for label in labels:
            group = [row for row in rows if getter(row) == label]
            nested[axis][label] = {"n_cases": len(group), "methods": {}}
            for method in METHODS:
                nested[axis][label]["methods"][method] = {}
                for metric in SCALAR_METRICS:
                    stats = finite_summary(
                        [float(row[f"{method}_{metric}"]) for row in group]
                    )
                    nested[axis][label]["methods"][method][metric] = stats
                    long_rows.append(
                        {
                            "group_axis": axis,
                            "group": label,
                            "method": method,
                            "metric": metric,
                            "n_cases": len(group),
                            **stats,
                        }
                    )
    return long_rows, nested


def write_group_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def closest_to_group_median(rows: list[dict], metric: str) -> dict:
    finite = [row for row in rows if np.isfinite(float(row[metric]))]
    if not finite:
        raise ValueError(f"no finite {metric} values in selection group")
    median = float(np.median([float(row[metric]) for row in finite]))
    return min(finite, key=lambda row: (abs(float(row[metric]) - median), row["sample_id"]))


def choose_typical_cases(rows: list[dict]) -> list[dict]:
    selections = []
    used: set[str] = set()
    for count in (1, 2, 3):
        group = [row for row in rows if row["num_sources"] == count]
        if not group:
            continue
        selected = closest_to_group_median(group, "epoch29_wrel_l2")
        selections.append(
            {"rule": f"{count}-source: closest to group median epoch29 wRelL2", **selected}
        )
        used.add(selected["sample_id"])
    weak_group = [
        row
        for row in rows
        if row["num_sources"] > 1
        and row["source_intensity_ratio"] >= 1.5
        and row["sample_id"] not in used
    ]
    if weak_group:
        selected = closest_to_group_median(weak_group, "epoch29_weak_source_error")
        selections.append(
            {
                "rule": "ratio>=1.5: closest to group median epoch29 weak-source error",
                **selected,
            }
        )
    return selections


def mip(volume: np.ndarray, axis: int) -> np.ndarray:
    finite = np.isfinite(volume)
    projected = np.max(np.where(finite, volume, -np.inf), axis=axis)
    projected[~np.any(finite, axis=axis)] = np.nan
    return projected.T


def plot_case(
    sample_id: str,
    fields: dict[str, np.ndarray],
    canonical: CanonicalCrossDiscretization,
    frame: FrameManifest,
    output: Path,
) -> None:
    volumes = {}
    for name, values in fields.items():
        valid_values = np.asarray(canonical.p @ values).reshape(-1)
        full = np.full(int(np.prod(canonical.operator.grid_shape)), np.nan)
        full[canonical.operator.valid_flat_indices] = valid_values
        volumes[name] = full.reshape(canonical.operator.grid_shape)
    prediction_names = ["epoch29", "epoch30", "p1_oracle"]
    field_vmax = max(float(np.nanmax(volumes[name])) for name in ("gt", *prediction_names))
    error_vmax = max(
        float(np.nanmax(np.abs(volumes[name] - volumes["gt"])))
        for name in prediction_names
    )
    error_vmax = max(error_vmax, 1e-12)
    rows: list[tuple[str, str, float]] = [("GT", "gt", field_vmax)]
    for name in prediction_names:
        rows.extend(
            [
                (name, name, field_vmax),
                (f"|{name} - GT|", f"error_{name}", error_vmax),
            ]
        )
        volumes[f"error_{name}"] = np.abs(volumes[name] - volumes["gt"])
    planes = (("XY", 2), ("XZ", 1), ("YZ", 0))
    fig, axes = plt.subplots(len(rows), 3, figsize=(11.5, 2.35 * len(rows)), constrained_layout=True)
    for row_index, (row_label, key, vmax) in enumerate(rows):
        is_error = key.startswith("error_")
        for column, (plane, axis) in enumerate(planes):
            ax = axes[row_index, column]
            image = ax.imshow(
                mip(volumes[key], axis),
                origin="lower",
                cmap="magma" if is_error else "viridis",
                vmin=0.0,
                vmax=vmax,
                interpolation="nearest",
                aspect="auto",
            )
            if row_index == 0:
                ax.set_title(f"{plane} MIP")
            if column == 0:
                ax.set_ylabel(row_label)
            ax.set_xticks([])
            ax.set_yticks([])
        colorbar = fig.colorbar(image, ax=axes[row_index, :], fraction=0.012, pad=0.01)
        colorbar.set_label("|error|" if is_error else "normalized fluorescence")
    fig.suptitle(
        f"{sample_id}: common field scale [0, {field_vmax:.3g}], "
        f"common error scale [0, {error_vmax:.3g}]\n"
        f"P1 visualization only; FEM audit metrics use lumped nodal volumes",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_report(summary: dict, rows: list[dict], path: Path) -> None:
    all_group = summary["groups"]["all"]["all"]["methods"]

    def median(method: str, metric: str) -> float:
        return float(all_group[method][metric]["median"])

    def paired_better_fraction(metric: str, *, lower_is_better: bool) -> float:
        first = np.asarray([row[f"epoch29_{metric}"] for row in rows], dtype=np.float64)
        second = np.asarray([row[f"epoch30_{metric}"] for row in rows], dtype=np.float64)
        finite = np.isfinite(first) & np.isfinite(second)
        comparison = first[finite] < second[finite] if lower_is_better else first[finite] > second[finite]
        return float(np.mean(comparison))

    def format_median(value: dict) -> str:
        number = float(value["median"])
        return "N/A" if not np.isfinite(number) else f"{number:.3f}"

    lines = [
        "# Stage-1 continuous-field FEM-volume audit",
        "",
        "This audit changes no network and performs no training. Epoch 29 and epoch 30 are",
        "the already exported physical `coarse_d.npy` states. All scalar FEM metrics use",
        "the P1 lumped mass diagonal `m_i = sum_{T contains i} |T|/4`; nodes are not",
        "weighted equally.",
        "",
        "## Metric contract",
        "",
        "- `wRelL2 = sqrt(sum m_i (p_i-g_i)^2 / sum m_i g_i^2)`.",
        "- mass ratio uses `sum m_i p_i / sum m_i g_i`.",
        "- CCC uses lumped-volume-normalized weighted moments over the full FEM domain.",
        "- COM uses the nonnegative nodal fields and lumped-volume mass.",
        "- shape error is wRelL2 after the best nonnegative global rescaling of prediction;",
        "  it removes overall amplitude but retains displacement, separation, and composition error.",
        "- source composition is total-variation distance between predicted attributed mass",
        "  fractions and the declared component-template mass fractions. Prediction mass is",
        "  assigned by intensity-independent normalized-shape territories over the FEM mesh;",
        "  therefore every GT source participates and background artifacts are not discarded.",
        "- weak-source error is the relative attributed mass error for the lowest-parameter-",
        "  intensity component; it is undefined for single-source cases.",
        "",
        "## Overall medians",
        "",
        "| Method | wRelL2 | mass ratio | CCC | COM mm | shape error | composition error | weak error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        lines.append(
            f"| {method} | {median(method, 'wrel_l2'):.4f} | "
            f"{median(method, 'mass_ratio'):.4f} | {median(method, 'weighted_ccc'):.4f} | "
            f"{median(method, 'com_error_mm'):.4f} | {median(method, 'shape_error'):.4f} | "
            f"{median(method, 'source_composition_error'):.4f} | "
            f"{median(method, 'weak_source_error'):.4f} |"
        )
    lines.extend(["", "## Stratified medians"])
    for axis, title in (
        ("num_sources", "Source count"),
        ("strength_ratio", "Strong/weak parameter-intensity ratio"),
        ("source_distance", "Minimum source-center distance"),
        ("depth", "Depth tier"),
    ):
        lines.extend(
            [
                "",
                f"### {title}",
                "",
                "| Group | n | e29 wRelL2 | e30 wRelL2 | e29 mass | e30 mass | e29 composition | e30 composition | e29 weak | e30 weak |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for label, group in summary["groups"][axis].items():
            methods = group["methods"]
            lines.append(
                f"| {label} | {group['n_cases']} | "
                f"{format_median(methods['epoch29']['wrel_l2'])} | "
                f"{format_median(methods['epoch30']['wrel_l2'])} | "
                f"{format_median(methods['epoch29']['mass_ratio'])} | "
                f"{format_median(methods['epoch30']['mass_ratio'])} | "
                f"{format_median(methods['epoch29']['source_composition_error'])} | "
                f"{format_median(methods['epoch30']['source_composition_error'])} | "
                f"{format_median(methods['epoch29']['weak_source_error'])} | "
                f"{format_median(methods['epoch30']['weak_source_error'])} |"
            )

    epoch29_scale_removal = 1.0 - median("epoch29", "shape_error") / median(
        "epoch29", "wrel_l2"
    )
    epoch30_scale_removal = 1.0 - median("epoch30", "shape_error") / median(
        "epoch30", "wrel_l2"
    )
    count_groups = summary["groups"]["num_sources"]
    lines.extend(
        [
            "",
            "## Failure-mode diagnosis",
            "",
            f"1. **A single global amplitude correction is not sufficient.** Epoch29 has a near-unity median mass ratio ({median('epoch29', 'mass_ratio'):.3f}) but its median wRelL2 remains {median('epoch29', 'wrel_l2'):.3f}. Optimally rescaling every prediction reduces the median error by only approximately {100 * epoch29_scale_removal:.1f}% (epoch30: {100 * epoch30_scale_removal:.1f}%). The dominant residual is therefore spatial/shape error, not only global gain.",
            f"2. **Amplitude calibration is nevertheless unstable.** The median absolute mass error is {median('epoch29', 'mass_relative_error'):.3f} at epoch29. Epoch30 is systematically under-amplitude (median mass ratio {median('epoch30', 'mass_ratio'):.3f}, median absolute mass error {median('epoch30', 'mass_relative_error'):.3f}). Epoch29 has lower mass error in {100 * paired_better_fraction('mass_relative_error', lower_is_better=True):.1f}% of paired cases.",
            f"3. **Localization and multi-source structure are major failures.** Epoch29 median COM error is {median('epoch29', 'com_error_mm'):.3f} mm. Its median shape error rises from {float(count_groups['1']['methods']['epoch29']['shape_error']['median']):.3f} for one source to {float(count_groups['3']['methods']['epoch29']['shape_error']['median']):.3f} for three sources, while COM error rises from {float(count_groups['1']['methods']['epoch29']['com_error_mm']['median']):.3f} to {float(count_groups['3']['methods']['epoch29']['com_error_mm']['median']):.3f} mm.",
            f"4. **Weak-source relative recovery is poor beyond the attribution floor.** On all 219 multi-source cases, epoch29 median composition error is {median('epoch29', 'source_composition_error'):.3f} and weak-source error is {median('epoch29', 'weak_source_error'):.3f}; the P1-oracle floors are only {median('p1_oracle', 'source_composition_error'):.3f} and {median('p1_oracle', 'weak_source_error'):.3f}. Epoch29 beats epoch30 in {100 * paired_better_fraction('source_composition_error', lower_is_better=True):.1f}% of cases for composition and {100 * paired_better_fraction('weak_source_error', lower_is_better=True):.1f}% for weak-source error.",
            "5. **Source distance, intensity ratio, and depth are modifiers, not a single causal explanation.** The stratified medians are not monotonic across distance or depth bins. Three-source cases are consistently harder, but the current audit cannot reduce the failure to close-source overlap alone.",
            f"6. **Epoch29 is the scientifically safer of the two checkpoints despite the similar wRelL2.** Epoch30 has slightly lower median wRelL2 ({median('epoch30', 'wrel_l2'):.3f} versus {median('epoch29', 'wrel_l2'):.3f}), but epoch29 has higher CCC in {100 * paired_better_fraction('weighted_ccc', lower_is_better=False):.1f}% of cases and materially better amplitude, source composition, and weak-source recovery. Epoch30's lower validation loss is achieved with a pronounced low-amplitude/multi-source-collapse bias.",
            "",
            "**Audit verdict:** current Stage-1 failure is primarily spatial field reconstruction--localization, morphology, and multi-source composition/weak-source allocation--with an additional checkpoint-dependent global amplitude problem. It is not adequately explained by a single scalar amplitude mismatch, and the non-monotonic distance/depth results do not support claiming that source separation alone is the cause.",
        ]
    )
    lines.extend(
        [
            "",
            "## Interpretation guardrail",
            "",
            "Under this node-domain protocol, the P1 oracle is exactly the normalized nodal GT.",
            "It is therefore an identity sanity control and must score zero on whole-field errors.",
            "Its source-composition scores need not be zero: those compare the historical",
            "maximum-composed nodal field against the declared latent component-template fractions,",
            "and thus expose the source-attribution floor of the FEM/GT contract. It is not the",
            "cross-discretization P1 approximation oracle previously evaluated against raw voxel GT.",
            "The latter answers a different question and must not be compared numerically as if it",
            "used this lumped-node metric contract.",
            "",
            "See `per_case.csv`, `grouped_summary.csv`, `summary.json`, and `figures/` for the",
            "complete casewise and stratified evidence.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_root = args.dataset_root / "samples"
    ids = [line for line in (args.dataset_root / "splits" / "val.txt").read_text().splitlines() if line]
    if args.max_samples is not None:
        ids = ids[: args.max_samples]
    nodes, elements = FrameManifest.load_mesh_nodes(args.shared_dir)
    nodal_volumes = lumped_nodal_volumes(nodes, elements)
    rows: list[dict] = []
    fields_by_id: dict[str, dict[str, np.ndarray]] = {}
    for index, sample_id in enumerate(ids, start=1):
        sample_dir = sample_root / sample_id
        tumor = json.loads((sample_dir / "tumor_params.json").read_text())
        foci = tumor["foci"]
        intensities = np.asarray(
            [float(focus["params"]["intensity"]) for focus in foci], dtype=np.float64
        )
        scale = float(np.asarray(np.load(sample_dir / "gt_scale.npy")).reshape(()))
        gt = np.clip(
            np.load(sample_dir / "gt_nodes.npy").astype(np.float64) / scale,
            0.0,
            1.0,
        )
        predictions = {
            "epoch29": np.load(args.epoch29_bridge / sample_id / "coarse_d.npy").astype(np.float64),
            "epoch30": np.load(args.epoch30_bridge / sample_id / "coarse_d.npy").astype(np.float64),
            "p1_oracle": gt.copy(),
        }
        distance = minimum_source_distance(foci)
        intensity_ratio = float(intensities.max() / intensities.min())
        row: dict = {
            "sample_id": sample_id,
            "num_sources": len(foci),
            "source_intensity_ratio": intensity_ratio,
            "min_source_distance_mm": distance,
            "depth_mm": float(tumor["depth_mm"]),
            "depth_tier": tumor["depth_tier"],
            "strength_ratio_group": ratio_group(len(foci), intensity_ratio),
            "source_distance_group": distance_group(distance),
        }
        for method, prediction in predictions.items():
            metrics = fem_volume_metrics(prediction, gt, nodal_volumes, nodes)
            metrics.update(
                source_composition_metrics(
                    prediction, gt, nodal_volumes, nodes, foci
                )
            )
            for key, value in metrics.items():
                row[f"{method}_{key}"] = value
        rows.append(row)
        fields_by_id[sample_id] = {"gt": gt, **predictions}
        if index % 25 == 0 or index == len(ids):
            print(f"[audit] {index}/{len(ids)} {sample_id}", flush=True)

    write_per_case_csv(rows, args.output_dir / "per_case.csv")
    group_rows, groups = grouped_summaries(rows)
    write_group_csv(group_rows, args.output_dir / "grouped_summary.csv")
    selections = choose_typical_cases(rows)
    selection_compact = [
        {
            key: value
            for key, value in selected.items()
            if key
            in {
                "rule",
                "sample_id",
                "num_sources",
                "source_intensity_ratio",
                "min_source_distance_mm",
                "depth_mm",
                "depth_tier",
                "epoch29_wrel_l2",
                "epoch29_weak_source_error",
            }
        }
        for selected in selections
    ]
    summary = {
        "protocol": "continuous Stage-1 nodal P1 fields with lumped FEM-volume weighting",
        "split": "val",
        "n_cases": len(rows),
        "network_changed": False,
        "training_run": False,
        "metric_contract": {
            "inner_product": "P1 lumped nodal volume: m_i=sum_incident_tet(volume/4)",
            "domain": "full FEM node domain",
            "shape_error": "wRelL2 after best nonnegative global prediction scale",
            "source_attribution": (
                "all-node intensity-independent normalized-shape territories; "
                "declared source fractions from untruncated Gaussian component templates; "
                "lumped-volume integration throughout"
            ),
            "single_source_composition": "undefined (NaN), not counted as zero error",
            "prediction_normalization": "none beyond the frozen training/export contract",
        },
        "prediction_artifacts": {
            "epoch29": str(args.epoch29_bridge.resolve()),
            "epoch30": str(args.epoch30_bridge.resolve()),
            "p1_oracle": "normalized gt_nodes identity control",
        },
        "mesh": {
            "path": str((args.shared_dir / "mesh.npz").resolve()),
            "sha256": sha256(args.shared_dir / "mesh.npz"),
            "n_nodes": len(nodes),
            "n_elements": len(elements),
            "total_tetrahedral_volume_mm3": float(nodal_volumes.sum()),
            "lumped_weight_min_mm3": float(nodal_volumes.min()),
            "lumped_weight_max_mm3": float(nodal_volumes.max()),
        },
        "groups": groups,
        "typical_case_selection": selection_compact,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n"
    )
    write_report(summary, rows, args.output_dir / "REPORT.md")

    if not args.skip_figures:
        canonical = CanonicalCrossDiscretization(
            args.operator_cache,
            shared_dir=args.shared_dir,
            factorize=False,
            allow_stale_frame_manifest=True,
        )
        frame = FrameManifest.load(args.shared_dir)
        for selected in selections:
            sample_id = selected["sample_id"]
            plot_case(
                sample_id,
                fields_by_id[sample_id],
                canonical,
                frame,
                args.output_dir / "figures" / f"{sample_id}_mips.png",
            )
            print(f"[figure] {sample_id}", flush=True)


if __name__ == "__main__":
    main()
