#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).parent.parent))

from du2vox.bridge.coverage_field import compute_coverage_field
from du2vox.bridge.fem_bridging import FEMBridge
from du2vox.utils.frame import FrameManifest


def load_split(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_op_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return {
        "name": cfg.get("name", Path(path).stem),
        "tau_active": float(cfg.get("tau_active", 0.5)),
        "tau_core": float(cfg.get("tau_core", 0.5)),
        "roi_dilation_layers": int(cfg.get("roi_dilation_layers", 1)),
        "min_component_size": int(cfg.get("min_component_size", 0)),
    }


def sample_seed(sample_id: str) -> int:
    return int(zlib.crc32(sample_id.encode("utf-8")) & 0xFFFFFFFF)


def gt_index_to_world(frame: FrameManifest, idx: np.ndarray) -> np.ndarray:
    return (
        np.asarray(idx, dtype=np.float64) * frame.gt_spacing_mm
        + frame.gt_offset_world_mm
        + frame.gt_spacing_mm / 2
    ).astype(np.float32)


def make_bbox_grid(bbox: dict[str, list[float]], spacing: float, padding: float) -> np.ndarray:
    lo = np.asarray(bbox["min"], dtype=np.float32) - float(padding)
    hi = np.asarray(bbox["max"], dtype=np.float32) + float(padding)
    axes = [np.arange(lo[i], hi[i] + spacing * 0.5, spacing, dtype=np.float32) for i in range(3)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=1).astype(np.float32)


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred_bin = pred >= 0.5
    gt_bin = gt >= 0.5
    tp = float((pred_bin & gt_bin).sum())
    fp = float((pred_bin & ~gt_bin).sum())
    fn = float((~pred_bin & gt_bin).sum())
    return {
        "dice": 2.0 * tp / (float(pred_bin.sum() + gt_bin.sum()) + 1e-8),
        "precision": tp / (tp + fp + 1e-8),
        "recall": tp / (tp + fn + 1e-8),
    }


def sample_gt_values(frame: FrameManifest, gt_voxels: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = frame.world_to_gt_index(points)
    shape = np.asarray(gt_voxels.shape)
    inside = ~np.any((idx < 0) | (idx > shape - 1), axis=1)
    values = map_coordinates(gt_voxels, idx.T, order=1, mode="constant", cval=0.0, prefilter=False)
    values = values.astype(np.float32)
    values[~inside] = 0.0
    return values, inside


def locate_gt_positive_roles(
    frame: FrameManifest,
    full_bridge: FEMBridge,
    field: dict[str, np.ndarray],
    roi_mask: np.ndarray,
    gt_voxels: np.ndarray,
    sample_id: str,
    max_gt_points: int,
) -> dict[str, float]:
    pos_idx = np.argwhere(gt_voxels >= 0.5)
    if len(pos_idx) == 0:
        return {
            "gt_positive_covered_by_roi": 0.0,
            "gt_positive_outside_roi": 0.0,
            "mean_gt_pos_ratio_in_core": 0.0,
            "mean_gt_pos_ratio_in_halo": 0.0,
            "mean_gt_pos_ratio_in_sentinel": 0.0,
        }
    if len(pos_idx) > max_gt_points:
        rng = np.random.default_rng(sample_seed(sample_id))
        pos_idx = pos_idx[rng.choice(len(pos_idx), size=max_gt_points, replace=False)]
    points = gt_index_to_world(frame, pos_idx)
    tet_ids, _ = full_bridge.locate_points_batch(points)
    inside_mesh = tet_ids >= 0
    in_roi = np.zeros(len(points), dtype=bool)
    in_roi[inside_mesh] = roi_mask[tet_ids[inside_mesh]]
    role = np.full(len(points), -1, dtype=np.int64)
    role[inside_mesh] = field["role"][tet_ids[inside_mesh]]
    denom = max(len(points), 1)
    return {
        "gt_positive_covered_by_roi": float(in_roi.mean()),
        "gt_positive_outside_roi": float(1.0 - in_roi.mean()),
        "mean_gt_pos_ratio_in_core": float((role == 1).sum() / denom),
        "mean_gt_pos_ratio_in_halo": float((role == 2).sum() / denom),
        "mean_gt_pos_ratio_in_sentinel": float((role == 3).sum() / denom),
    }


def diagnose_one(
    name: str,
    bridge_dir: Path,
    op: dict[str, Any],
    sample_ids: list[str],
    samples_dir: Path,
    nodes: np.ndarray,
    elements: np.ndarray,
    frame: FrameManifest,
    grid_spacing: float,
    padding: float,
    max_gt_points: int,
) -> dict[str, float | int | str]:
    full_bridge = FEMBridge(nodes, elements)
    rows = []
    n_tets = len(elements)
    for sid in sample_ids:
        bd = bridge_dir / sid
        if not (bd / "roi_tet_indices.npy").exists():
            continue
        coarse_d = np.load(bd / "coarse_d.npy").astype(np.float32)
        roi_tets = np.load(bd / "roi_tet_indices.npy").astype(np.int64)
        roi_info = json.loads((bd / "roi_info.json").read_text())
        gt_voxels = np.load(samples_dir / sid / "gt_voxels.npy").astype(np.float32)
        roi_mask = np.zeros(n_tets, dtype=bool)
        roi_mask[roi_tets] = True
        field = compute_coverage_field(
            coarse_d,
            elements,
            roi_tet_indices=roi_tets,
            cfg={"tau_core": op["tau_core"], "tau_weak": op["tau_active"]},
        )
        role = field["role"]
        gt_role = locate_gt_positive_roles(frame, full_bridge, field, roi_mask, gt_voxels, sid, max_gt_points)

        points = make_bbox_grid(roi_info["roi_bbox_mm"], grid_spacing, padding)
        bridge = FEMBridge(nodes, elements, roi_tets)
        prior_8d, valid = bridge.get_prior_features(points, coarse_d)
        gt, gt_inside = sample_gt_values(frame, gt_voxels, points)
        valid = valid & gt_inside
        fem = (prior_8d[:, :4] * prior_8d[:, 4:8]).sum(axis=1)
        metrics = binary_metrics(fem[valid], gt[valid]) if valid.any() else {"dice": 0.0, "precision": 0.0, "recall": 0.0}
        rows.append(
            {
                "roi_tet_ratio": len(roi_tets) / n_tets,
                "core_ratio": float((role == 1).mean()),
                "halo_ratio": float((role == 2).mean()),
                "sentinel_ratio": float((role == 3).mean()),
                "bg_ratio": float((role == 0).mean()),
                "outside_ratio": float((~roi_mask).mean()),
                "mean_gt_pos_ratio_in_roi": gt_role["gt_positive_covered_by_roi"],
                "fem_dice_05": metrics["dice"],
                "fem_precision_05": metrics["precision"],
                "fem_recall_05": metrics["recall"],
                **gt_role,
            }
        )

    out: dict[str, float | int | str] = {"name": name, "n_samples": len(rows)}
    for key in [
        "roi_tet_ratio",
        "gt_positive_covered_by_roi",
        "gt_positive_outside_roi",
        "core_ratio",
        "halo_ratio",
        "sentinel_ratio",
        "bg_ratio",
        "outside_ratio",
        "mean_gt_pos_ratio_in_roi",
        "mean_gt_pos_ratio_in_core",
        "mean_gt_pos_ratio_in_halo",
        "mean_gt_pos_ratio_in_sentinel",
        "fem_dice_05",
        "fem_precision_05",
        "fem_recall_05",
    ]:
        vals = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        out[f"{key}_mean"] = float(vals.mean()) if len(vals) else 0.0
        if key == "roi_tet_ratio":
            out[f"{key}_std"] = float(vals.std()) if len(vals) else 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Stage1 operating point bridge coverage")
    parser.add_argument("--bridge_dirs", nargs="+", required=True)
    parser.add_argument("--op_configs", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--split_file", required=True)
    parser.add_argument("--shared_dir", required=True)
    parser.add_argument("--samples_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--grid_spacing_mm", type=float, default=1.0)
    parser.add_argument("--padding_mm", type=float, default=1.0)
    parser.add_argument("--max_gt_points", type=int, default=20000)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    if not (len(args.bridge_dirs) == len(args.op_configs) == len(args.names)):
        raise SystemExit("--bridge_dirs, --op_configs, and --names must have equal length")
    sample_ids = load_split(args.split_file)
    if args.max_samples is not None:
        sample_ids = sample_ids[: args.max_samples]
    nodes, elements = FrameManifest.load_mesh_nodes(args.shared_dir)
    frame = FrameManifest.load(args.shared_dir)

    rows = []
    for bridge_dir, op_config, name in zip(args.bridge_dirs, args.op_configs, args.names):
        rows.append(
            diagnose_one(
                name=name,
                bridge_dir=Path(bridge_dir),
                op=load_op_config(op_config),
                sample_ids=sample_ids,
                samples_dir=Path(args.samples_dir),
                nodes=nodes,
                elements=elements,
                frame=frame,
                grid_spacing=args.grid_spacing_mm,
                padding=args.padding_mm,
                max_gt_points=args.max_gt_points,
            )
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Stage1OP] wrote {out_csv}")


if __name__ == "__main__":
    main()
