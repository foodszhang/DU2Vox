#!/usr/bin/env python3
"""Audit the on-disk contracts required by transport-observability CQR.

This script is deliberately read-only with respect to datasets, bridges, caches,
and checkpoints.  Its only outputs are the JSON and Markdown audit reports.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
KEY_CQR_FILES = (
    "du2vox/bridge/coverage_field.py",
    "du2vox/bridge/cqr_query_builder.py",
    "du2vox/bridge/fem_lift_indicators.py",
    "du2vox/bridge/measurement_proposal.py",
    "du2vox/models/stage2/cqr_residual_inr.py",
    "scripts/precompute_stage2_cqr.py",
    "scripts/eval_stage2_unified.py",
)
CQR_KEYS = (
    "grid_coords",
    "grid_coords_norm",
    "prior_8d",
    "prior_ext",
    "prior_prolong",
    "prior_lift",
    "gt_values",
    "valid_mask",
    "tet_ids",
    "role",
    "query_src_tag",
    "correction_band",
    "prolongation_value",
    "residual_indicator",
    "query_weight",
    "coverage_score",
    "risk_components",
)
ROLE_TO_SOURCE = {0: 2, 1: 0, 2: 1, 4: 3}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _array_stats(array: np.ndarray) -> dict[str, Any]:
    array = np.asarray(array)
    out: dict[str, Any] = {"shape": list(array.shape), "dtype": str(array.dtype)}
    if array.size == 0:
        out.update({"finite_fraction": None, "min": None, "max": None, "mean": None})
        return out
    if np.issubdtype(array.dtype, np.number) or array.dtype == np.bool_:
        finite = np.isfinite(array)
        out["finite_fraction"] = float(finite.mean())
        if np.any(finite):
            values = array[finite].astype(np.float64, copy=False)
            out.update(
                min=float(values.min()), max=float(values.max()), mean=float(values.mean())
            )
    return out


def _npz_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    with np.load(path, allow_pickle=False) as data:
        return {
            "path": str(path),
            "exists": True,
            "arrays": {key: _array_stats(data[key]) for key in data.files},
        }


def _load_split(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _pick_ids(ids: list[str], count: int = 3) -> list[str]:
    if len(ids) <= count:
        return ids
    indices = np.linspace(0, len(ids) - 1, count, dtype=int)
    return [ids[index] for index in indices]


def _sample_assets(samples_dir: Path, ids: list[str], data_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    projection_name = str(data_cfg.get("projection_file", "proj.npz"))
    for sid in ids:
        root = samples_dir / sid
        record: dict[str, Any] = {"sample_id": sid, "root": str(root), "exists": root.exists()}
        for name in ("measurement_b.npy", "gt_nodes.npy", "gt_voxels.npy"):
            path = root / name
            record[name] = (
                {"path": str(path), "exists": True, **_array_stats(np.load(path, mmap_mode="r"))}
                if path.exists()
                else {"path": str(path), "exists": False}
            )
        projection = _npz_stats(root / projection_name)
        if projection.get("exists"):
            with np.load(root / projection_name, allow_pickle=False) as data:
                numeric = [np.asarray(data[key]) for key in data.files if np.asarray(data[key]).dtype.kind in "fiu"]
                if numeric:
                    stack = np.concatenate([item.reshape(-1) for item in numeric])
                    projection["raw_combined"] = _array_stats(stack)
                    eps = float(data_cfg.get("projection_eps", 1e-8))
                    if stack.ndim == 1:
                        projection["configured_normalization"] = {
                            "mode": data_cfg.get("projection_norm", "none"),
                            "transform": data_cfg.get("projection_transform", "none"),
                            "note": "per-view normalization is applied by the dataset loader",
                            "global_max_denominator": float(max(np.max(np.abs(stack)), eps)),
                        }
        record[projection_name] = projection
        proposal = root / "proposal" / "meas_backproj_heatmap.npy"
        record["proposal/meas_backproj_heatmap.npy"] = (
            {"path": str(proposal), "exists": True, **_array_stats(np.load(proposal, mmap_mode="r"))}
            if proposal.exists()
            else {"path": str(proposal), "exists": False}
        )
        proposal_meta = root / "proposal" / "meas_backproj_meta.json"
        record["proposal/meas_backproj_meta.json"] = {
            "path": str(proposal_meta),
            "exists": proposal_meta.exists(),
        }
        records.append(record)
    return records


def _matrix_stats(path: Path) -> tuple[dict[str, Any], Any | None]:
    if not path.exists():
        return {"path": str(path), "exists": False}, None
    try:
        matrix = sp.load_npz(path)
    except ValueError:
        with np.load(path, allow_pickle=False) as data:
            keys = data.files
            if len(keys) != 1:
                return {
                    "path": str(path),
                    "exists": True,
                    "load_error": f"Dense matrix NPZ has ambiguous keys: {keys}",
                }, None
            matrix = np.asarray(data[keys[0]])
        values = matrix
        return (
            {
                "path": str(path),
                "exists": True,
                "array_key": keys[0],
                "shape": list(matrix.shape),
                "dtype": str(matrix.dtype),
                "format": "dense",
                "nnz": int(np.count_nonzero(matrix)),
                "finite": bool(np.isfinite(values).all()),
                "min": float(values.min()) if values.size else None,
                "max": float(values.max()) if values.size else None,
            },
            matrix,
        )
    values = matrix.data
    return (
        {
            "path": str(path),
            "exists": True,
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
            "format": matrix.format,
            "nnz": int(matrix.nnz),
            "finite": bool(np.isfinite(values).all()),
            "min": float(values.min()) if values.size else None,
            "max": float(values.max()) if values.size else None,
        },
        matrix,
    )


def _checkpoint_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    out: dict[str, Any] = {"path": str(path), "exists": True, "size_bytes": path.stat().st_size}
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            out["top_level_keys"] = sorted(map(str, checkpoint.keys()))
            for key in ("epoch", "best_metric", "best_val_dice", "config"):
                if key in checkpoint and isinstance(checkpoint[key], (str, int, float, bool, type(None))):
                    out[key] = checkpoint[key]
            state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
            if isinstance(state, dict):
                out["state_tensor_count"] = sum(torch.is_tensor(value) for value in state.values())
    except Exception as exc:  # audit should report unreadable metadata, not abort
        out["load_error"] = f"{type(exc).__name__}: {exc}"
    return out


def _audit_shared(shared_dir: Path, use_visible_mask: bool, measurement_sizes: list[int]) -> dict[str, Any]:
    mesh_path = shared_dir / "mesh.npz"
    mesh: dict[str, np.ndarray] = {}
    mesh_report = {"path": str(mesh_path), "exists": mesh_path.exists()}
    if mesh_path.exists():
        with np.load(mesh_path, allow_pickle=False) as loaded:
            mesh = {key: loaded[key] for key in loaded.files}
        mesh_report["arrays"] = {key: _array_stats(value) for key, value in mesh.items()}

    matrices: dict[str, Any] = {}
    loaded_matrices: dict[str, sp.spmatrix] = {}
    for letter in ("M", "F", "A"):
        stats, matrix = _matrix_stats(shared_dir / f"system_matrix.{letter}.npz")
        matrices[letter] = stats
        if matrix is not None:
            loaded_matrices[letter] = matrix

    index_path = shared_dir / "system_matrix.index.npz"
    index_report = _npz_stats(index_path)
    visible_path = shared_dir / "visible_mask.npy"
    visible_report: dict[str, Any] = {"path": str(visible_path), "exists": visible_path.exists()}
    visible = None
    if visible_path.exists():
        visible = np.load(visible_path)
        visible_report.update(_array_stats(visible), true_count=int(np.asarray(visible, dtype=bool).sum()))

    surface_mesh = np.asarray(mesh.get("surface_node_indices", []), dtype=np.int64)
    surface_index = np.array([], dtype=np.int64)
    if index_path.exists():
        with np.load(index_path, allow_pickle=False) as index_data:
            if "surface_index" in index_data:
                surface_index = np.asarray(index_data["surface_index"], dtype=np.int64)
    a_rows = loaded_matrices["A"].shape[0] if "A" in loaded_matrices else None
    full_surface_count = int(len(surface_index) or len(surface_mesh))
    visible_count = int(np.asarray(visible, dtype=bool).sum()) if visible is not None else None
    convention_checks = {
        "stage1_use_visible_mask": bool(use_visible_mask),
        "measurement_sizes": measurement_sizes,
        "a_rows": a_rows,
        "full_surface_count": full_surface_count,
        "visible_count": visible_count,
        "mesh_and_index_surface_equal": bool(
            len(surface_mesh) == len(surface_index) and np.array_equal(surface_mesh, surface_index)
        ) if len(surface_mesh) and len(surface_index) else None,
        "a_is_full_surface": bool(a_rows == full_surface_count) if a_rows is not None else None,
        "measurements_are_full_surface": bool(
            measurement_sizes and all(size == full_surface_count for size in measurement_sizes)
        ),
    }
    convention_checks["full_surface_convention_confirmed"] = bool(
        not use_visible_mask
        and convention_checks["a_is_full_surface"]
        and convention_checks["measurements_are_full_surface"]
    )
    return {
        "directory": str(shared_dir),
        "mesh": mesh_report,
        "matrices": matrices,
        "index": index_report,
        "visible_mask": visible_report,
        "convention": convention_checks,
        "frame_manifest": _load_json(shared_dir / "frame_manifest.json"),
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    try:
        return {"path": str(path), "exists": True, "content": json.loads(path.read_text())}
    except Exception as exc:
        return {"path": str(path), "exists": True, "error": f"{type(exc).__name__}: {exc}"}


def _audit_cqr_split(directory: Path, ids: list[str], min_files: int = 5) -> dict[str, Any]:
    existing = [directory / f"{sid}.npz" for sid in ids if (directory / f"{sid}.npz").exists()]
    if len(existing) < min_files and directory.exists():
        existing = sorted(directory.glob("*.npz"))
    selected = _pick_ids([path.stem for path in existing], min_files)
    records = []
    for stem in selected:
        path = directory / f"{stem}.npz"
        with np.load(path, allow_pickle=False) as data:
            missing_keys = [key for key in CQR_KEYS if key not in data]
            arrays = {key: _array_stats(data[key]) for key in CQR_KEYS if key in data}
            valid = np.asarray(data["valid_mask"], dtype=bool) if "valid_mask" in data else np.ones(0, dtype=bool)
            tet_ids = np.asarray(data["tet_ids"], dtype=np.int64) if "tet_ids" in data else np.full(len(valid), -1)
            role = np.asarray(data["role"], dtype=np.int64) if "role" in data else np.full(len(valid), -1)
            band = (
                np.asarray(data["correction_band"], dtype=np.int64)
                if "correction_band" in data
                else np.full(len(valid), -1)
            )
            source = (
                np.asarray(data["query_src_tag"], dtype=np.int64)
                if "query_src_tag" in data
                else np.full(len(valid), -1)
            )
            negative = tet_ids < 0
            expected_source = np.array([ROLE_TO_SOURCE.get(int(value), -1) for value in band])
            valid_tets = tet_ids[(tet_ids >= 0) & valid]
            unique, counts = np.unique(valid_tets, return_counts=True)
            prior_nonzero_negative = None
            if "prior_8d" in data and np.any(negative):
                prior_nonzero_negative = int(np.any(np.asarray(data["prior_8d"])[negative] != 0, axis=1).sum())
            records.append(
                {
                    "path": str(path),
                    "candidate_pool_size": int(len(valid)),
                    "valid_count": int(valid.sum()),
                    "missing_keys": missing_keys,
                    "arrays": arrays,
                    "unique_valid_tets": int(len(unique)),
                    "queries_per_tet": _array_stats(counts),
                    "negative_tet_count": int(negative.sum()),
                    "negative_tet_marked_valid": int((negative & valid).sum()) if len(valid) else 0,
                    "negative_tet_with_nonzero_prior": prior_nonzero_negative,
                    "role_band_mismatch": int((role != band).sum()),
                    "band_source_mismatch": int((expected_source != source).sum()),
                    "role_counts": np.bincount(np.clip(role, 0, None), minlength=5).tolist()
                    if len(role)
                    else [],
                }
            )
    return {
        "directory": str(directory),
        "exists": directory.exists(),
        "expected_split_samples": len(ids),
        "npz_files_found": len(existing),
        "required_minimum_audited": min_files,
        "samples_audited": len(records),
        "records": records,
        "blocked": len(records) < min_files,
        "blocker": (
            f"Need at least {min_files} CQR NPZ files, found {len(records)}"
            if len(records) < min_files
            else None
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    git = report["git"]
    lines = [
        "# CQR Transport Asset Audit",
        "",
        "## Git and baseline",
        "",
        f"- Branch: `{git['branch']}`",
        f"- HEAD: `{git['head']}`",
        f"- Dirty files: {len(git['dirty_files'])}",
        f"- Resolved Stage 1 checkpoint: `{report['checkpoints']['resolved_stage1_checkpoint']}`",
        f"- Resolved Stage 2 baseline: `{report['checkpoints']['resolved_baseline_checkpoint']}`",
        "",
        "## Dataset",
        "",
    ]
    for split, count in report["dataset"]["split_counts"].items():
        lines.append(f"- {split}: {count}")
    lines.extend(
        [
            f"- Total: {report['dataset']['total_split_count']}",
            f"- Dataset root contract valid: {report['dataset']['required_v2_3k_paths']}",
            "",
            "## Shared physics",
            "",
        ]
    )
    shared = report["shared_physics"]
    for name, matrix in shared["matrices"].items():
        lines.append(
            f"- {name}: shape={matrix.get('shape')}, nnz={matrix.get('nnz')}, dtype={matrix.get('dtype')}"
        )
    convention = shared["convention"]
    lines.extend(
        [
            f"- Surface nodes: {convention['full_surface_count']}",
            f"- Visible nodes: {convention['visible_count']}",
            f"- Full-surface convention confirmed: **{convention['full_surface_convention_confirmed']}**",
            "",
            "## Existing CQR candidate pools",
            "",
        ]
    )
    for split, value in report["cqr_npz"].items():
        lines.append(
            f"- {split}: dir_exists={value['exists']}, files={value['npz_files_found']}, "
            f"audited={value['samples_audited']}, blocked={value['blocked']}"
        )
        if value["blocker"]:
            lines.append(f"  - Blocker: {value['blocker']}")
    lines.extend(["", "## Blockers", ""])
    blockers = report["blockers"]
    if blockers:
        lines.extend(f"- {item}" for item in blockers)
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "The JSON companion contains per-array shapes, dtypes, ranges, file metadata, "
            "role/band/source checks, and checkpoint keys.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-md", default="diagnosis/cqr_transport_assets.md")
    parser.add_argument("--out-json", default="diagnosis/cqr_transport_assets.json")
    parser.add_argument("--samples-per-data-split", type=int, default=3)
    parser.add_argument("--cqr-files-per-split", type=int, default=5)
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open() as handle:
        cfg = yaml.safe_load(handle)
    data_cfg = cfg["data"]

    split_ids: dict[str, list[str]] = {}
    split_paths: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = Path(data_cfg[f"{split}_split"])
        split_paths[split] = str(path)
        split_ids[split] = _load_split(path)

    sampled_ids: list[str] = []
    for split in ("train", "val", "test"):
        sampled_ids.extend(_pick_ids(split_ids[split], args.samples_per_data_split))
    sampled_ids = list(dict.fromkeys(sampled_ids))
    sample_records = _sample_assets(Path(data_cfg["samples_dir"]), sampled_ids, data_cfg)
    measurement_sizes = [
        int(np.prod(record["measurement_b.npy"]["shape"]))
        for record in sample_records
        if record["measurement_b.npy"].get("exists")
    ]

    baseline_candidates = [
        REPO_ROOT / "checkpoints/stage2/cqr_v2_3k_rgl_main_sparse/best.pth",
        REPO_ROOT / "runs/cqr_v2_3k_rgl_main_multiview_sparse/checkpoints/best.pth",
    ]
    stage1_candidates = [
        REPO_ROOT / "runs/stage1_fmt_simgen_v2_3k_20k_balanced_v2_eval/checkpoints/best.pth"
    ]
    resolved_baseline = next((path for path in baseline_candidates if path.exists()), None)
    resolved_stage1 = next((path for path in stage1_candidates if path.exists()), None)

    cqr_reports = {}
    for split in ("train", "val", "test"):
        directory = REPO_ROOT / data_cfg[f"precomputed_{split}_dir"]
        cqr_reports[split] = _audit_cqr_split(
            directory, split_ids[split], min_files=args.cqr_files_per_split
        )

    dataset_root = Path(data_cfg["dataset_root"])
    required_paths = {
        "dataset_root": dataset_root,
        "samples": Path(data_cfg["samples_dir"]),
        "manifest": dataset_root / "dataset_manifest.json",
        "shared": Path(data_cfg["shared_dir"]),
    }
    required_paths.update({f"split_{name}": Path(path) for name, path in split_paths.items()})
    blockers = [
        f"CQR {split} precomputed assets unavailable: {value['blocker']}"
        for split, value in cqr_reports.items()
        if value["blocked"]
    ]
    if resolved_baseline is None:
        blockers.append("No Stage 2 baseline checkpoint exists at either required candidate path")
    if resolved_stage1 is None:
        blockers.append("Stage 1 balanced_v2 checkpoint is missing")

    report = {
        "schema_version": 1,
        "config": str(config_path),
        "git": {
            "branch": _git("branch", "--show-current"),
            "head": _git("rev-parse", "HEAD"),
            "dirty_files": _git("status", "--short").splitlines(),
            "key_cqr_files": {name: (REPO_ROOT / name).exists() for name in KEY_CQR_FILES},
        },
        "dataset": {
            "paths": {name: {"path": str(path), "exists": path.exists()} for name, path in required_paths.items()},
            "required_v2_3k_paths": all(path.exists() for path in required_paths.values()),
            "split_counts": {name: len(ids) for name, ids in split_ids.items()},
            "total_split_count": sum(map(len, split_ids.values())),
            "sample_assets": sample_records,
        },
        "shared_physics": _audit_shared(
            Path(data_cfg["shared_dir"]),
            use_visible_mask=False,
            measurement_sizes=measurement_sizes,
        ),
        "checkpoints": {
            "stage1_candidates": [_checkpoint_summary(path) for path in stage1_candidates],
            "baseline_candidates": [_checkpoint_summary(path) for path in baseline_candidates],
            "resolved_stage1_checkpoint": str(resolved_stage1) if resolved_stage1 else None,
            "resolved_baseline_checkpoint": str(resolved_baseline) if resolved_baseline else None,
        },
        "cqr_npz": cqr_reports,
        "blockers": blockers,
    }

    out_json = REPO_ROOT / args.out_json
    out_md = REPO_ROOT / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=_json_value) + "\n")
    out_md.write_text(_markdown(report))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    for blocker in blockers:
        print(f"[BLOCKED] {blocker}")


if __name__ == "__main__":
    main()
