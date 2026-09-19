#!/usr/bin/env python3
"""Run the fixed-domain DU2Vox support-space oracle decomposition."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.ndimage import binary_erosion, distance_transform_edt, map_coordinates

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from du2vox.utils.frame import FrameManifest  # noqa: E402
from experiments.cross_discretization_decomposition.decomposition import (  # noqa: E402
    MassProjector,
    binary_metrics,
    build_domain_operator,
    component_metrics,
    decompose,
    load_operator,
    save_operator,
    surface_metrics,
)

EXP_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS = EXP_DIR / "artifacts"
SHARED = Path("/home/foods/pro/FMT-SimGen/output/shared_mesh_20k")
DATASET = Path("/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k")
BRIDGES = {
    "train": REPO / "output/bridge_v2_3k_train_balanced_v2",
    "val": REPO / "output/bridge_v2_3k_val_balanced_v2",
    "test": REPO / "output/bridge_v2_3k_test_balanced_v2",
}
LEGACY = {
    split: REPO / f"precomputed/cqr_v2_3k_rgl_main/{split}"
    for split in ("train", "val", "test")
}
RECON_NAMES = ("stage1_p1", "oracle_inverse", "oracle_representation", "gt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ids() -> list[tuple[str, str]]:
    out = []
    for split in ("train", "val", "test"):
        ids = [line.strip() for line in (DATASET / f"splits/{split}.txt").read_text().splitlines() if line.strip()]
        out.extend((split, sid) for sid in ids)
    return out


def focus_metadata(sample_dir: Path) -> dict[str, float | int]:
    params = json.loads((sample_dir / "tumor_params.json").read_text())
    foci = params.get("foci", [])
    centers = np.asarray([f["center"] for f in foci], dtype=np.float64)
    volumes, intensities, radii = [], [], []
    for focus in foci:
        fp = focus.get("params", {}) or {}
        radius = float(focus.get("radius") or fp.get("radius") or 1.0)
        rx = float(focus.get("rx") or fp.get("rx") or radius)
        ry = float(focus.get("ry") or fp.get("ry") or radius)
        rz = float(focus.get("rz") or fp.get("rz") or radius)
        volumes.append(4.0 * math.pi * rx * ry * rz / 3.0)
        radii.append(max(rx, ry, rz))
        intensities.append(float(fp.get("intensity") or 1.0))
    if len(centers) > 1:
        dist = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
        dist[dist == 0] = np.inf
        min_distance = float(dist.min())
    else:
        min_distance = float("nan")
    return {
        "num_foci": len(foci),
        "depth_mm": float(params.get("depth_mm", np.nan)),
        "total_source_volume_mm3": float(sum(volumes)),
        "min_source_volume_mm3": float(min(volumes, default=np.nan)),
        "max_source_radius_mm": float(max(radii, default=np.nan)),
        "min_inter_source_distance_mm": min_distance,
        "min_intensity": float(min(intensities, default=np.nan)),
        "mean_intensity": float(np.mean(intensities)) if intensities else float("nan"),
        "depth_tier": params.get("depth_tier", "unknown"),
        "source_type": params.get("source_type", "unknown"),
        "tumor_params": params,
    }


def weak_recall(pred: np.ndarray, gt: np.ndarray, coords: np.ndarray, params: dict) -> float:
    foci = params.get("foci", [])
    if not foci:
        return float("nan")
    focus = min(foci, key=lambda f: float((f.get("params", {}) or {}).get("intensity") or 1.0))
    fp = focus.get("params", {}) or {}
    r = float(focus.get("radius") or fp.get("radius") or 1.0)
    scales = np.asarray([
        float(focus.get("rx") or fp.get("rx") or r),
        float(focus.get("ry") or fp.get("ry") or r),
        float(focus.get("rz") or fp.get("rz") or r),
    ])
    inside = np.sum(((coords - np.asarray(focus["center"])) / scales) ** 2, axis=1) <= 1.0
    target = inside & (gt >= 0.5)
    return float(np.mean(pred[target] >= 0.5)) if target.any() else float("nan")


def error_structure(values: np.ndarray, gt_full: np.ndarray, valid_flat: np.ndarray, shape: tuple[int, ...], spacing: float) -> dict[str, float]:
    out = {}
    abs_values = np.abs(values)
    for threshold in (1e-4, 1e-3, 1e-2, 5e-2):
        out[f"near_zero_le_{threshold:g}"] = float(np.mean(abs_values <= threshold))
        out[f"support_ge_{threshold:g}"] = float(np.mean(abs_values >= threshold))
    if np.std(values) > 1e-15:
        out["skewness"] = float(stats.skew(values, bias=False))
        out["kurtosis"] = float(stats.kurtosis(values, fisher=True, bias=False))
    else:
        out["skewness"] = out["kurtosis"] = 0.0

    full = np.full(np.prod(shape), np.nan, dtype=np.float32)
    full[valid_flat] = values.astype(np.float32)
    volume = full.reshape(shape)
    gradients = []
    lag_products = defaultdict(list)
    mean = float(np.mean(values))
    variance = float(np.var(values))
    for axis in range(3):
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        a, b = volume[tuple(left)], volume[tuple(right)]
        mask = np.isfinite(a) & np.isfinite(b)
        gradients.append(((b[mask] - a[mask]) / spacing).astype(np.float32))
        for lag in (1, 2, 3, 5, 10):
            la = [slice(None)] * 3
            lb = [slice(None)] * 3
            la[axis] = slice(None, -lag)
            lb[axis] = slice(lag, None)
            aa, bb = volume[tuple(la)], volume[tuple(lb)]
            lm = np.isfinite(aa) & np.isfinite(bb)
            if lm.any() and variance > 1e-15:
                lag_products[lag].append(float(np.mean((aa[lm] - mean) * (bb[lm] - mean)) / variance))
    grad = np.concatenate(gradients)
    out["gradient_abs_mean"] = float(np.mean(np.abs(grad)))
    out["gradient_near_zero_1e-2"] = float(np.mean(np.abs(grad) <= 1e-2))
    for lag, vals in lag_products.items():
        out[f"autocorr_{lag * spacing:.1f}mm"] = float(np.mean(vals))

    gt_binary = gt_full >= 0.5
    boundary = gt_binary & ~binary_erosion(gt_binary)
    distance = distance_transform_edt(~boundary, sampling=spacing)
    valid_distance = distance.ravel()[valid_flat]
    energy = values**2
    total = float(energy.sum())
    for radius in (0.2, 0.4, 0.6, 1.0, 2.0):
        out[f"boundary_energy_le_{radius:.1f}mm"] = float(energy[valid_distance <= radius].sum() / max(total, 1e-300))
    return out


def fit_distributions(values: np.ndarray) -> list[dict[str, float | str]]:
    values = values[np.isfinite(values)]
    if len(values) > 1_000_000:
        values = np.random.default_rng(20260812).choice(values, 1_000_000, replace=False)
    fits = []
    for name, distribution in (("normal", stats.norm), ("laplace", stats.laplace), ("generalized_gaussian", stats.gennorm)):
        parameters = distribution.fit(values)
        log_likelihood = float(np.sum(distribution.logpdf(values, *parameters)))
        k = len(parameters)
        fits.append({
            "distribution": name,
            "n": len(values),
            "parameters": json.dumps([float(v) for v in parameters]),
            "log_likelihood": log_likelihood,
            "aic": float(2 * k - 2 * log_likelihood),
            "bic": float(k * np.log(len(values)) - 2 * log_likelihood),
        })
    return fits


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summaries(rows: list[dict], keys: list[str]) -> list[dict]:
    result = []
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))])
        if not len(values):
            continue
        q = np.percentile(values, [5, 25, 50, 75, 95])
        result.append({
            "metric": key, "n": len(values), "mean": values.mean(), "std": values.std(),
            "p05": q[0], "p25": q[1], "median": q[2], "p75": q[3], "p95": q[4], "iqr": q[3] - q[1],
        })
    return result


def oracle_rows(sample_id: str, split: str, result: dict, operator, meta: dict) -> list[dict]:
    rows = []
    recon = {
        "stage1_p1": result["fem"],
        "oracle_inverse": result["oracle_inv"],
        "oracle_representation": result["oracle_rep"],
        "gt": result["gt"],
    }
    for name, pred in recon.items():
        metrics = binary_metrics(pred, result["gt"])
        metrics.update(surface_metrics(pred, result["gt"], operator.valid_flat_indices, operator.grid_shape, operator.spacing_mm))
        metrics.update(component_metrics(pred, result["gt"], operator.valid_flat_indices, operator.grid_shape, operator.spacing_mm))
        metrics["weak_recall"] = weak_recall(pred, result["gt"], operator.coords_world, meta["tumor_params"])
        rows.append({"sample_id": sample_id, "split": split, "reconstruction": name, **metrics})
    return rows


def legacy_row(split: str, sid: str, result: dict, elements: np.ndarray, frame: FrameManifest) -> list[dict]:
    path = LEGACY[split] / f"{sid}.npz"
    if not path.exists():
        return []
    with np.load(path, allow_pickle=False) as data:
        required = {"grid_coords", "valid_mask", "tet_ids", "prior_8d", "gt_values"}
        if not required.issubset(data.files):
            return []
        valid = data["valid_mask"].astype(bool) & (data["tet_ids"] >= 0)
        coords = data["grid_coords"][valid]
        bary = data["prior_8d"][valid, 4:8]
        node_ids = elements[data["tet_ids"][valid].astype(np.int64)]
        baseline = np.sum(data["prior_8d"][valid, :4] * bary, axis=1)
        inv = np.sum(result["oracle_coeff"][node_ids] * bary, axis=1)
        idx = frame.world_to_gt_index(coords)
        rep_full = np.zeros(np.prod(result["gt_full_shape"]), dtype=np.float32)
        rep_full[result["valid_flat_indices"]] = result["e_rep"].astype(np.float32)
        rep_at_coords = map_coordinates(rep_full.reshape(result["gt_full_shape"]), idx.T, order=1, mode="constant", cval=0.0, prefilter=False)
        gt = (data["gt_values"][valid] > 0.05).astype(np.float64)
        rows = []
        for name, pred in (("stage1_p1", baseline), ("oracle_inverse", inv), ("oracle_representation", baseline + rep_at_coords), ("gt", gt)):
            rows.append({"sample_id": sid, "split": split, "reconstruction": name, **binary_metrics(pred, gt)})
        return rows


def plots(artifact_dir: Path, energy_rows: list[dict], oracle: list[dict], samples: dict[str, dict], operator) -> None:
    fig_dir = artifact_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    rinv = np.asarray([r["ratio_inv"] for r in energy_rows])
    rrep = np.asarray([r["ratio_rep"] for r in energy_rows])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(rinv, bins=50, alpha=0.7, label="inverse")
    axes[0].hist(rrep, bins=50, alpha=0.7, label="representation")
    axes[0].set(xlabel="energy ratio", ylabel="cases", title="Error-energy fractions")
    axes[0].legend()
    axes[1].scatter(rinv, rrep, s=5, alpha=0.3)
    axes[1].set(xlabel="r_inv", ylabel="r_rep", title="Per-case decomposition")
    fig.tight_layout()
    fig.savefig(fig_dir / "01_energy_decomposition.png", dpi=180)
    plt.close(fig)

    grouped = defaultdict(list)
    for row in oracle:
        for metric in ("dice", "precision", "recall", "weak_recall", "localization_error", "hd95", "assd"):
            if np.isfinite(float(row.get(metric, np.nan))):
                grouped[(row["reconstruction"], metric)].append(float(row[metric]))
    metrics = ("dice", "precision", "recall", "weak_recall", "localization_error", "hd95")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, metric in zip(axes.ravel(), metrics):
        vals = [np.mean(grouped[(name, metric)]) for name in RECON_NAMES]
        ax.bar(range(4), vals, color=["#777777", "#377eb8", "#e41a1c", "#4daf4a"])
        ax.set_xticks(range(4), ["P1", "inv", "rep", "GT"], rotation=20)
        ax.set_title(metric)
    fig.tight_layout()
    fig.savefig(fig_dir / "03_oracle_metrics.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter([r["depth_mm"] for r in energy_rows], rinv, s=5, alpha=0.3, label="inverse")
    axes[0].scatter([r["depth_mm"] for r in energy_rows], rrep, s=5, alpha=0.3, label="representation")
    axes[0].set(xlabel="source depth (mm)", ylabel="energy ratio")
    axes[0].legend()
    axes[1].scatter([r["total_source_volume_mm3"] for r in energy_rows], rinv, s=5, alpha=0.3, label="inverse")
    axes[1].scatter([r["total_source_volume_mm3"] for r in energy_rows], rrep, s=5, alpha=0.3, label="representation")
    axes[1].set(xlabel="total source volume (mm3)", ylabel="energy ratio")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "05_source_dependence.png", dpi=180)
    plt.close(fig)

    if samples:
        n = len(samples)
        fig, axes = plt.subplots(n * 3, 7, figsize=(17, 3.2 * n * 3), squeeze=False)
        fields = ("gt", "fem", "e_total", "e_inv", "e_rep", "oracle_inv", "oracle_rep")
        for case_i, (sid, result) in enumerate(samples.items()):
            volumes = {}
            for field in fields:
                full = np.full(np.prod(operator.grid_shape), np.nan, dtype=np.float32)
                full[operator.valid_flat_indices] = result[field]
                volumes[field] = full.reshape(operator.grid_shape)
            gt_idx = np.argwhere(np.nan_to_num(volumes["gt"]) >= 0.5)
            center = np.rint(gt_idx.mean(axis=0)).astype(int) if len(gt_idx) else np.asarray(operator.grid_shape) // 2
            for view, axis in enumerate((2, 1, 0)):
                for col, field in enumerate(fields):
                    image = np.take(volumes[field], center[axis], axis=axis).T
                    vmax = 1.0 if not field.startswith("e_") else max(0.1, float(np.nanmax(np.abs(image))))
                    cmap = "viridis" if not field.startswith("e_") else "coolwarm"
                    axes[case_i * 3 + view, col].imshow(image, origin="lower", cmap=cmap, vmin=-vmax if field.startswith("e_") else 0, vmax=vmax)
                    axes[case_i * 3 + view, col].set_title(f"{sid} {('axial','coronal','sagittal')[view]}\n{field}")
                    axes[case_i * 3 + view, col].axis("off")
        fig.tight_layout()
        fig.savefig(fig_dir / "02_spatial_decomposition.png", dpi=140)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    artifact = args.artifact_dir
    tables = artifact / "tables"
    cache = artifact / "operator_cache"
    artifact.mkdir(parents=True, exist_ok=True)
    tables.mkdir(exist_ok=True)

    frame = FrameManifest.load(SHARED)
    mesh = np.load(SHARED / "mesh.npz")
    nodes = mesh["nodes"].astype(np.float64)
    elements = mesh["elements"].astype(np.int64)
    if (cache / "P_full_fem_gt_centers.npz").exists() and not args.overwrite:
        operator = load_operator(cache)
    else:
        print("[operator] locating all GT voxel centers in the complete FEM mesh", flush=True)
        operator = build_domain_operator(nodes, elements, frame)
        metadata = {
            "definition": "0.2 mm GT voxel centers intersect complete FEM tetrahedral domain",
            "gt_centers": True,
            "n_full_voxels": int(np.prod(frame.gt_shape)),
            "n_valid_voxels": int(operator.p.shape[0]),
            "n_fem_nodes": int(operator.p.shape[1]),
            "n_active_columns": int(len(operator.active_columns)),
            "p_nnz": int(operator.p.nnz),
            "voxel_weight_mm3": operator.voxel_weight,
            "mesh_sha256": sha256(SHARED / "mesh.npz"),
            "frame_manifest_sha256": sha256(SHARED / "frame_manifest.json"),
        }
        save_operator(operator, cache, metadata)
    projector = MassProjector(operator)
    operator_report = json.loads((cache / "operator_metadata.json").read_text())
    operator_report.update({
        "gram_shape": list(projector.gram.shape), "gram_nnz": int(projector.gram.nnz),
        "gram_diag_min": projector.diag_min, "gram_diag_max": projector.diag_max,
        "ridge_regularization": projector.regularization,
    })
    (cache / "operator_metadata.json").write_text(json.dumps(operator_report, indent=2) + "\n")

    all_ids = load_ids()
    if args.max_samples:
        all_ids = all_ids[: args.max_samples]
    energy_rows, raw_rows, oracle, legacy, structure_rows = [], [], [], [], []
    inv_pool, rep_pool = [], []
    case_cache = {}
    started = time.time()
    for index, (split, sid) in enumerate(all_ids, 1):
        sample_dir = DATASET / "samples" / sid
        gt_full_raw = np.load(sample_dir / "gt_voxels.npy").astype(np.float64)
        gt_support = (gt_full_raw.ravel()[operator.valid_flat_indices] > 0.05).astype(np.float64)
        gt_raw = gt_full_raw.ravel()[operator.valid_flat_indices]
        coarse = np.load(BRIDGES[split] / sid / "coarse_d.npy").astype(np.float64)
        meta = focus_metadata(sample_dir)
        result = decompose(operator, projector, gt_support, coarse)
        raw = decompose(operator, projector, gt_raw, coarse)
        common = {key: value for key, value in meta.items() if key != "tumor_params"}
        energy_rows.append({
            "sample_id": sid, "split": split, **common,
            **{key: result[key] for key in ("energy_total", "energy_inv", "energy_rep", "ratio_inv", "ratio_rep", "closure_rel", "orthogonality_cos", "pythagorean_rel")},
        })
        raw_rows.append({
            "sample_id": sid, "split": split, **common,
            **{key: raw[key] for key in ("energy_total", "energy_inv", "energy_rep", "ratio_inv", "ratio_rep", "closure_rel", "orthogonality_cos", "pythagorean_rel")},
        })
        oracle.extend(oracle_rows(sid, split, result, operator, meta))
        rng = np.random.default_rng(20260812 + index)
        chosen = rng.choice(len(gt_support), min(512, len(gt_support)), replace=False)
        inv_pool.append(result["e_inv"][chosen].astype(np.float32))
        rep_pool.append(result["e_rep"][chosen].astype(np.float32))
        struct_inv = error_structure(result["e_inv"], gt_full_raw > 0.05, operator.valid_flat_indices, operator.grid_shape, operator.spacing_mm)
        struct_rep = error_structure(result["e_rep"], gt_full_raw > 0.05, operator.valid_flat_indices, operator.grid_shape, operator.spacing_mm)
        structure_rows.extend([
            {"sample_id": sid, "split": split, "component": "inverse", **struct_inv},
            {"sample_id": sid, "split": split, "component": "representation", **struct_rep},
        ])
        result["gt_full_shape"] = operator.grid_shape
        result["valid_flat_indices"] = operator.valid_flat_indices
        legacy.extend(legacy_row(split, sid, result, elements, frame))
        if index <= 2:
            case_cache[sid] = result
        if index % 10 == 0 or index == len(all_ids):
            print(f"[{index}/{len(all_ids)}] {sid} elapsed={time.time()-started:.1f}s", flush=True)

    sanity_keys = ["closure_rel", "orthogonality_cos", "pythagorean_rel"]
    worst = {key: float(np.nanmax(np.abs([r[key] for r in energy_rows]))) for key in sanity_keys}
    if worst["closure_rel"] > 1e-10 or worst["orthogonality_cos"] > 1e-7 or worst["pythagorean_rel"] > 1e-7:
        write_csv(tables / "numerical_validation.csv", energy_rows)
        raise RuntimeError(f"Projection sanity gate failed: {worst}")

    write_csv(tables / "per_case_error_energy_support.csv", energy_rows)
    write_csv(tables / "per_case_error_energy_raw_intensity_sensitivity.csv", raw_rows)
    write_csv(tables / "per_case_oracle_metrics_fixed_domain.csv", oracle)
    write_csv(tables / "per_case_error_structure.csv", structure_rows)
    write_csv(tables / "legacy_cqr_domain_metrics_sensitivity.csv", legacy)
    energy_summary = summaries(energy_rows, ["energy_total", "energy_inv", "energy_rep", "ratio_inv", "ratio_rep", *sanity_keys])
    raw_summary = summaries(raw_rows, ["energy_total", "energy_inv", "energy_rep", "ratio_inv", "ratio_rep", *sanity_keys])
    write_csv(tables / "error_energy_summary_support.csv", energy_summary)
    write_csv(tables / "error_energy_summary_raw_intensity.csv", raw_summary)
    oracle_summary = []
    for recon in RECON_NAMES:
        subset = [row for row in oracle if row["reconstruction"] == recon]
        for summary in summaries(subset, ["dice", "iou", "precision", "recall", "weak_recall", "volume_error", "mse", "mae", "mass_error", "peak_error", "assd", "hd95", "component_recall", "fp_components", "localization_error", "separation_success"]):
            oracle_summary.append({"reconstruction": recon, **summary})
    write_csv(tables / "oracle_metric_summary.csv", oracle_summary)
    structure_summary = []
    structure_keys = sorted(set(structure_rows[0]) - {"sample_id", "split", "component"})
    for component in ("inverse", "representation"):
        for row in summaries([r for r in structure_rows if r["component"] == component], structure_keys):
            structure_summary.append({"component": component, **row})
    write_csv(tables / "error_structure_summary.csv", structure_summary)

    distribution_rows = []
    for component, values in (("inverse", np.concatenate(inv_pool)), ("representation", np.concatenate(rep_pool))):
        for fit in fit_distributions(values.astype(np.float64)):
            distribution_rows.append({"component": component, **fit})
    write_csv(tables / "distribution_fits.csv", distribution_rows)

    correlations = []
    for prop in ("depth_mm", "total_source_volume_mm3", "min_source_volume_mm3", "num_foci", "min_inter_source_distance_mm", "min_intensity"):
        for ratio in ("ratio_inv", "ratio_rep"):
            pairs = np.asarray([(r[prop], r[ratio]) for r in energy_rows if np.isfinite(float(r[prop])) and np.isfinite(float(r[ratio]))], dtype=float)
            if len(pairs) > 2:
                pearson = stats.pearsonr(pairs[:, 0], pairs[:, 1])
                spearman = stats.spearmanr(pairs[:, 0], pairs[:, 1])
                correlations.append({"property": prop, "ratio": ratio, "n": len(pairs), "pearson_r": pearson.statistic, "pearson_p": pearson.pvalue, "spearman_r": spearman.statistic, "spearman_p": spearman.pvalue})
    write_csv(tables / "source_property_correlations.csv", correlations)

    plots(artifact, energy_rows, oracle, case_cache, operator)
    run_meta = {
        "primary_claim_scope": "binary source-support space only",
        "target": "gt_voxels > 0.05",
        "domain": operator_report["definition"],
        "n_samples": len(all_ids),
        "splits_unchanged": True,
        "stage2_training_performed": False,
        "experiment_f": "skipped: no verified matched coarse/default/fine meshes and reconstructions for this dataset",
        "runtime_seconds": time.time() - started,
        "numerical_gate_worst_absolute": worst,
        "raw_intensity_is_sensitivity_only": True,
        "legacy_cqr_is_historical_metric_restriction_only": True,
    }
    (artifact / "run_metadata.json").write_text(json.dumps(run_meta, indent=2) + "\n")
    print(json.dumps(run_meta, indent=2))


if __name__ == "__main__":
    main()
