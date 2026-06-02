#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).parent.parent))

from du2vox.bridge.coverage_field import correction_band_distance
from du2vox.bridge.coverage_field import QueryRole
from du2vox.bridge.cqr_query_builder import CQRQueryBuilder
from du2vox.bridge.fem_lift_indicators import compute_lifting_indicators
from du2vox.bridge.fem_bridging import FEMBridge
from du2vox.bridge.measurement_proposal import (
    load_measurement_proposal,
    sample_measurement_proposal_points,
)
from du2vox.utils.frame import FrameManifest


def load_split(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def sample_seed(sid: str) -> int:
    return int(zlib.crc32(sid.encode("utf-8")) & 0xFFFFFFFF)


def normalize_coords(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = points.min(axis=0).astype(np.float32)
    hi = points.max(axis=0).astype(np.float32)
    coords_norm = (2.0 * (points - lo) / (hi - lo + 1e-8) - 1.0).astype(np.float32)
    return coords_norm, lo, hi


def get_lifting_config(cqr_cfg: dict) -> tuple[float, float, dict]:
    lifting = cqr_cfg.get("lifting", {}) or {}
    coverage = cqr_cfg.get("coverage", {}) or {}
    prolongation = cqr_cfg.get("prolongation", {}) or {}
    tau_core = float(
        lifting.get(
            "tau_core",
            prolongation.get("tau_core_band", coverage.get("tau_core", 0.65)),
        )
    )
    tau_halo = float(
        lifting.get(
            "tau_halo",
            prolongation.get("tau_halo_band", coverage.get("tau_weak", 0.18)),
        )
    )
    return tau_core, tau_halo, lifting.get("weights", {}) or {}


def gt_index_to_world(frame: FrameManifest, idx: np.ndarray) -> np.ndarray:
    return (
        np.asarray(idx, dtype=np.float64) * frame.gt_spacing_mm
        + frame.gt_offset_world_mm
        + frame.gt_spacing_mm / 2
    ).astype(np.float32)


def apply_oracle_sentinel(
    data: dict[str, np.ndarray],
    gt_voxels: np.ndarray,
    frame: FrameManifest,
    nodes: np.ndarray,
    elements: np.ndarray,
    coarse_d: np.ndarray,
    n_candidates: int,
    oracle_cfg: dict,
    seed: int,
) -> None:
    if not oracle_cfg.get("enabled", False):
        return
    ratio = float(oracle_cfg.get("oracle_sentinel_ratio", 0.05))
    n_oracle = int(round(len(data["grid_coords"]) * ratio))
    if n_oracle <= 0:
        return

    pos_idx = np.argwhere(gt_voxels >= float(oracle_cfg.get("gt_threshold", 0.5)))
    if len(pos_idx) == 0:
        return
    rng = np.random.default_rng(seed + 17)
    chosen = pos_idx[rng.choice(len(pos_idx), size=n_oracle, replace=len(pos_idx) < n_oracle)]
    points = gt_index_to_world(frame, chosen)

    replace = np.where(data["role"] == 3)[0]
    if len(replace) == 0:
        replace = np.arange(len(data["grid_coords"]))[-n_oracle:]
    if len(replace) > n_oracle:
        replace = replace[:n_oracle]
    points = points[: len(replace)]

    data["grid_coords"][replace] = points
    data["grid_coords_norm"], data["bbox_min"], data["bbox_max"] = normalize_coords(data["grid_coords"])
    bridge = FEMBridge(nodes, elements, n_candidates=n_candidates)
    prior_8d, valid = bridge.get_prior_features(points, coarse_d, K=n_candidates)
    data["prior_8d"][replace] = prior_8d
    if "prior_ext" in data:
        data["prior_ext"][replace, :8] = prior_8d
    if "prior_prolong" in data:
        data["prior_prolong"][replace, :8] = prior_8d
        data["prior_prolong"][replace, 8] = (prior_8d[:, :4] * prior_8d[:, 4:8]).sum(axis=1)
    if "prior_lift" in data:
        data["prior_lift"][replace, :8] = prior_8d
        data["prior_lift"][replace, 8] = (prior_8d[:, :4] * prior_8d[:, 4:8]).sum(axis=1)
    if "prolongation_value" in data:
        data["prolongation_value"][replace] = (prior_8d[:, :4] * prior_8d[:, 4:8]).sum(axis=1)
    data["gt_values"][replace] = 1.0
    data["valid_mask"][replace] = valid
    data["role"][replace] = 3
    if "correction_band" in data:
        data["correction_band"][replace] = 3


def precompute_one(
    sid: str,
    bridge_dir: Path,
    samples_dir: Path,
    nodes: np.ndarray,
    elements: np.ndarray,
    frame: FrameManifest,
    n_query_points: int,
    n_candidates: int,
    cqr_cfg: dict,
) -> dict[str, np.ndarray]:
    bd = bridge_dir / sid
    coarse_d = np.load(bd / "coarse_d.npy").astype(np.float32)
    roi_tet_indices = np.load(bd / "roi_tet_indices.npy").astype(np.int64)
    cqr_cfg = dict(cqr_cfg)
    proposal_cfg = cqr_cfg.get("measurement_proposal", {}) or {}
    if proposal_cfg.get("enabled", False):
        heatmap, meta, _ = load_measurement_proposal(
            samples_dir / sid,
            proposal_subdir=proposal_cfg.get("proposal_subdir", "proposal"),
            proposal_filename=proposal_cfg.get("proposal_filename", "meas_backproj_heatmap.npy"),
            proposal_meta_filename=proposal_cfg.get("proposal_meta_filename", "meas_backproj_meta.json"),
        )
        ratios = cqr_cfg.get("ratios", {}) or {}
        proposal_ratio = float(ratios.get("proposal", proposal_cfg.get("proposal_ratio", 0.0)))
        n_proposal = int(round(n_query_points * max(0.0, proposal_ratio)))
        rng = np.random.default_rng(sample_seed(sid) + 2027)
        cqr_cfg["proposal_points"] = sample_measurement_proposal_points(
            heatmap,
            meta,
            n_points=max(n_proposal, 1),
            rng=rng,
        )
    if cqr_cfg.get("sentinel", {}).get("mode") == "view_guided":
        from du2vox.bridge.view_evidence import compute_view_evidence, load_proj_npz

        proj_path = samples_dir / sid / "proj.npz"
        if not proj_path.exists():
            raise FileNotFoundError(f"view_guided sentinel requires {proj_path}")
        centroids = nodes[elements].mean(axis=1).astype(np.float32)
        view_score, _ = compute_view_evidence(centroids, load_proj_npz(str(proj_path)))
        cqr_cfg["view_evidence"] = view_score

    tau_core, tau_halo, lifting_weights = get_lifting_config(cqr_cfg)
    lift_ind = compute_lifting_indicators(
        nodes=nodes,
        tets=elements,
        node_values=coarse_d,
        tau_core=tau_core,
        tau_halo=tau_halo,
        weights=lifting_weights,
    )

    builder = CQRQueryBuilder(
        nodes=nodes,
        elements=elements,
        config={
            **cqr_cfg,
            "n_query_points": int(n_query_points),
            "n_candidates": int(n_candidates),
        },
    )
    cqr = builder.build(
        coarse_d=coarse_d,
        roi_tet_indices=roi_tet_indices,
        n_query_points=n_query_points,
        seed=sample_seed(sid),
        lift_indicators=lift_ind,
    )

    points = cqr["query_points"].astype(np.float32)
    coords_norm, bbox_min, bbox_max = normalize_coords(points)

    gt_voxels = np.load(samples_dir / sid / "gt_voxels.npy").astype(np.float32)
    idx_float = frame.world_to_gt_index(points)
    gt_values = map_coordinates(
        gt_voxels,
        idx_float.T,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype(np.float32)

    shape_arr = np.array(gt_voxels.shape)
    outside_gt = np.any((idx_float < 0) | (idx_float > shape_arr - 1), axis=1)
    valid = cqr["valid_mask"].astype(bool) & (~outside_gt)
    gt_values[outside_gt] = 0.0
    tet_ids = cqr["tet_ids"].astype(np.int64)
    valid_tet = tet_ids >= 0
    q_tet_grad_norm = np.zeros(len(points), dtype=np.float32)
    q_grad_jump_score = np.zeros(len(points), dtype=np.float32)
    q_recovery_error_score = np.zeros(len(points), dtype=np.float32)
    q_transition_score = np.zeros(len(points), dtype=np.float32)
    q_residual_indicator = np.zeros(len(points), dtype=np.float32)
    q_tet_grad_norm[valid_tet] = lift_ind["tet_grad_norm"][tet_ids[valid_tet]]
    q_grad_jump_score[valid_tet] = lift_ind["grad_jump_score"][tet_ids[valid_tet]]
    q_recovery_error_score[valid_tet] = lift_ind["recovery_error_score"][tet_ids[valid_tet]]
    q_transition_score[valid_tet] = lift_ind["transition_score"][tet_ids[valid_tet]]
    q_residual_indicator[valid_tet] = lift_ind["residual_indicator"][tet_ids[valid_tet]]
    correction_band = cqr["correction_band"].astype(np.int64)
    band_distance_score = correction_band_distance(correction_band)
    prolongation_value = cqr["prolongation_value"].astype(np.float32)
    query_src_tag = np.full_like(correction_band, -1, dtype=np.int64)
    query_src_tag[correction_band == int(QueryRole.CORE)] = 0
    query_src_tag[correction_band == int(QueryRole.HALO)] = 1
    query_src_tag[correction_band == int(QueryRole.BG)] = 2
    query_src_tag[correction_band == int(QueryRole.PROPOSAL)] = 3
    # prior_lift layout is fixed mainline input:
    # 0:8 prior_8d, 8 prolongation_value, 9 tet_grad_norm,
    # 10 grad_jump_score, 11 recovery_error_score, 12 transition_score,
    # 13 residual_indicator, 14 band_distance_score.
    prior_lift = np.concatenate(
        [
            cqr["prior_8d"].astype(np.float32),
            prolongation_value[:, None],
            q_tet_grad_norm[:, None],
            q_grad_jump_score[:, None],
            q_recovery_error_score[:, None],
            q_transition_score[:, None],
            q_residual_indicator[:, None],
            band_distance_score[:, None],
        ],
        axis=1,
    ).astype(np.float32)

    out = {
        "grid_coords": points,
        "grid_coords_norm": coords_norm,
        "prior_8d": cqr["prior_8d"].astype(np.float32),
        "prior_ext": cqr["prior_ext"].astype(np.float32),
        "prior_prolong": cqr["prior_prolong"].astype(np.float32),
        "prior_lift": prior_lift,
        "gt_values": gt_values.astype(np.float32),
        "valid_mask": valid.astype(bool),
        "grid_shape": np.array([len(points), 1, 1], dtype=np.int32),
        "bbox_min": bbox_min.astype(np.float32),
        "bbox_max": bbox_max.astype(np.float32),
        "tet_ids": tet_ids,
        "tet_id": tet_ids,
        "role": cqr["role"].astype(np.int64),
        "query_src_tag": query_src_tag,
        "correction_band": correction_band,
        "prolongation_value": prolongation_value,
        "correction_demand_score": cqr["correction_demand_score"].astype(np.float32),
        "band_distance_score": band_distance_score,
        "tet_grad_norm": q_tet_grad_norm,
        "grad_jump_score": q_grad_jump_score,
        "recovery_error_score": q_recovery_error_score,
        "transition_score": q_transition_score,
        "residual_indicator": q_residual_indicator,
        "coverage_score": cqr["coverage_score"].astype(np.float32),
        "view_evidence_score": cqr["view_evidence_score"].astype(np.float32),
        "sentinel_score": cqr["sentinel_score"].astype(np.float32),
        "risk_components": cqr["risk_components"].astype(np.float32),
        "query_weight": cqr["query_weight"].astype(np.float32),
        "coverage_role_counts": cqr["role_counts"].astype(np.int64),
    }
    apply_oracle_sentinel(
        out,
        gt_voxels,
        frame,
        nodes,
        elements,
        coarse_d,
        n_candidates,
        cqr_cfg.get("oracle", {}),
        sample_seed(sid),
    )
    out["coverage_role_counts"] = np.bincount(out["role"], minlength=5).astype(np.int64)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute CQR Stage-2 query clouds")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_query_points", type=int, default=None)
    parser.add_argument("--n_candidates", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    split_file = cfg["data"][f"{args.split}_split"]
    sample_ids = load_split(split_file)
    if args.max_samples:
        sample_ids = sample_ids[: args.max_samples]

    bridge_dir = Path(cfg["data"].get(f"{args.split}_bridge_dir", cfg["data"].get("bridge_dir", "")))
    shared_dir = Path(cfg["data"]["shared_dir"])
    samples_dir = Path(cfg["data"]["samples_dir"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_query = args.n_query_points
    if n_query is None:
        n_query = int(
            cfg["data"].get(
                "precompute_query_points",
                max(32768, int(cfg["data"].get("n_query_points", 4096)) * 8),
            )
        )

    cqr_cfg = cfg.get("cqr", {})

    print(f"[CQR] config={cqr_cfg}")
    print(f"[CQR] Loading mesh: {shared_dir}")
    nodes, elements = FrameManifest.load_mesh_nodes(shared_dir)
    frame = FrameManifest.load(shared_dir)
    print(f"[CQR] nodes={len(nodes)}, tets={len(elements)}")
    print(f"[CQR] split={args.split}, samples={len(sample_ids)}, n_query={n_query}")
    print(f"[CQR] bridge_dir={bridge_dir}")
    print(f"[CQR] output_dir={output_dir}")
    proposal_cfg = cqr_cfg.get("measurement_proposal", {}) or {}
    if proposal_cfg.get("enabled", False):
        print(
            "[CQR] measurement_proposal="
            f"enabled ratio={cqr_cfg.get('ratios', {}).get('proposal', proposal_cfg.get('proposal_ratio', 0.0))} "
            f"file={proposal_cfg.get('proposal_subdir', 'proposal')}/"
            f"{proposal_cfg.get('proposal_filename', 'meas_backproj_heatmap.npy')}"
        )
    print("-" * 80)

    total = 0.0
    done = 0
    for i, sid in enumerate(sample_ids, 1):
        out_path = output_dir / f"{sid}.npz"
        if out_path.exists() and not args.overwrite:
            print(f"[{i}/{len(sample_ids)}] {sid}: exists, skip")
            continue

        t0 = time.perf_counter()
        data = precompute_one(
            sid=sid,
            bridge_dir=bridge_dir,
            samples_dir=samples_dir,
            nodes=nodes,
            elements=elements,
            frame=frame,
            n_query_points=n_query,
            n_candidates=args.n_candidates,
            cqr_cfg=cqr_cfg,
        )
        np.savez_compressed(out_path, **data)
        elapsed = time.perf_counter() - t0
        total += elapsed
        done += 1

        valid = data["valid_mask"]
        counts = data["coverage_role_counts"].tolist()
        band_counts = np.bincount(data["correction_band"], minlength=5).tolist()
        demand = data["correction_demand_score"]
        prolong = data["prolongation_value"]
        residual = data["residual_indicator"]
        band = data["correction_band"]

        def band_mean(values: np.ndarray, band_id: int) -> float:
            mask = valid & (band == band_id)
            return float(values[mask].mean()) if np.any(mask) else 0.0

        print(
            f"[{i}/{len(sample_ids)}] {sid}: "
            f"points={len(valid)}, valid={int(valid.sum())}/{len(valid)} "
            f"({100*valid.mean():.1f}%), roles(bg/core/halo/sentinel/proposal)={counts}, "
            f"prior_lift_dim={data['prior_lift'].shape[-1]}, "
            f"band(bg/core/halo/sentinel/proposal)={band_counts}, "
            f"demand(bg/core/halo)="
            f"({band_mean(demand, 0):.3f},{band_mean(demand, 1):.3f},{band_mean(demand, 2):.3f}), "
            f"residual(bg/core/halo)="
            f"({band_mean(residual, 0):.3f},{band_mean(residual, 1):.3f},{band_mean(residual, 2):.3f}), "
            f"prolong(bg/core/halo)="
            f"({band_mean(prolong, 0):.3f},{band_mean(prolong, 1):.3f},{band_mean(prolong, 2):.3f}), "
            f"t={elapsed:.1f}s"
        )

    print("-" * 80)
    print(f"[CQR] Done: processed={done}, total={total:.1f}s, avg={total/max(done,1):.2f}s/sample")


if __name__ == "__main__":
    main()
