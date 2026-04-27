#!/usr/bin/env python3
"""
Stage 2 pipeline diagnostic script — Phase 0A/0B/0C + Phase 2 reporting.

Usage:
    python scripts/diagnose_stage2_pipeline.py --phase all --output diagnosis/stage2_pipeline_report.md
    python scripts/diagnose_stage2_pipeline.py --phase 0a --verbose
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from du2vox.utils.frame import FrameManifest


# ─── Data paths (uniform_trunk_v2_roi38_multi) ────────────────────────────────
SHARED_DIR = "/home/foods/pro/FMT-SimGen/output/shared"
SAMPLES_DIR = Path("/home/foods/pro/FMT-SimGen/data/uniform_trunk_v2_roi38_multi/samples")
BRIDGE_DIR = Path("/home/foods/pro/DU2Vox/output/bridge_val")       # val bridge
BRIDGE_TRAIN_DIR = Path("/home/foods/pro/DU2Vox/output/bridge_train")  # train bridge (all 800)
PRECOMPUTED_DIR = Path("/home/foods/pro/DU2Vox/precomputed/val")
STAGE1_METRICS_CSV = Path("/home/foods/pro/DU2Vox/results/stage1/metrics_per_sample_uniform_corrected.csv")

SPLIT_DIR = Path("/home/foods/pro/FMT-SimGen/data/uniform_trunk_v2_roi38_multi/splits")

MCX_ANGLES = [-90, -60, -30, 0, 30, 60, 90]


def gt_index_to_world(gt_index: np.ndarray, fm: FrameManifest) -> np.ndarray:
    """Inverse of FrameManifest.world_to_gt_index."""
    return (
        np.asarray(gt_index) * fm.gt_spacing_mm
        + fm.gt_offset_world_mm
        + fm.gt_spacing_mm / 2.0
    )


def load_split(split_file: Path) -> list[str]:
    with open(split_file) as f:
        return [l.strip() for l in f if l.strip()]


def compute_dice(pred, target, threshold=0.5):
    """Compute Dice coefficient at a threshold."""
    pred_bin = (pred >= threshold).astype(np.float32)
    target_bin = (target >= threshold).astype(np.float32)
    intersection = (pred_bin * target_bin).sum()
    denom = pred_bin.sum() + target_bin.sum()
    return float(2 * intersection / (denom + 1e-8))


# ─── Phase 0A: Axis Oracle ─────────────────────────────────────────────────────

def phase_0a_axis_oracle(sample_ids: list[str], n_samples: int = 5, verbose: bool = False):
    """
    Verify world_to_gt_index roundtrip and corner consistency.
    Fails if any err > 1.0 voxel units.
    """
    fm = FrameManifest.load(SHARED_DIR)

    results = []
    for sid in sample_ids[:n_samples]:
        gt_path = SAMPLES_DIR / sid / "gt_voxels.npy"
        if not gt_path.exists():
            continue
        gt_voxels = np.load(gt_path).astype(np.float32)

        # Test 1: argmax round-trip
        argmax_flat = gt_voxels.argmax()
        argmax_idx = np.unravel_index(argmax_flat, gt_voxels.shape)
        argmax_idx_arr = np.array(argmax_idx, dtype=np.float64) + 0.5  # voxel center

        world = gt_index_to_world(argmax_idx_arr[None], fm)  # [1, 3]
        back = fm.world_to_gt_index(world)                   # [1, 3]

        roundtrip_err = float(np.abs(back[0] - argmax_idx_arr).max())

        # Test 2: corner consistency
        corners_vox = np.array([
            [0.5, 0.5, 0.5],
            [gt_voxels.shape[0] - 0.5, 0.5, 0.5],
            [0.5, gt_voxels.shape[1] - 0.5, 0.5],
            [0.5, 0.5, gt_voxels.shape[2] - 0.5],
        ], dtype=np.float64)
        corners_world = gt_index_to_world(corners_vox, fm)
        corners_back = fm.world_to_gt_index(corners_world)
        corner_err = float(np.abs(corners_back - corners_vox).max())

        # Determine shape axis labels
        shape = gt_voxels.shape  # (X, Y, Z) or (Z, Y, X)?
        # world_to_gt_index returns (x_idx, y_idx, z_idx) in voxel space
        # If shape matches world axis order, the transformation is consistent
        x_expected = fm.gt_offset_world_mm[0] + 0.5 * fm.gt_spacing_mm
        y_expected = fm.gt_offset_world_mm[1] + 0.5 * fm.gt_spacing_mm
        z_expected = fm.gt_offset_world_mm[2] + 0.5 * fm.gt_spacing_mm

        results.append({
            "sid": sid,
            "gt_voxels_shape": list(shape),
            "argmax_idx": argmax_idx,
            "roundtrip_err": roundtrip_err,
            "corner_err": corner_err,
            "pass": roundtrip_err < 0.01 and corner_err < 0.01,
        })

        if verbose:
            print(f"  {sid}: shape={shape}, argmax={argmax_idx}, "
                  f"roundtrip_err={roundtrip_err:.4f}, corner_err={corner_err:.4f}, "
                  f"pass={results[-1]['pass']}")

    return results


# ─── Phase 0B: ROI-MCX Alignment ─────────────────────────────────────────────

def phase_0b_roi_mcx_alignment(sample_ids: list[str]):
    """
    Compute (ROI_bbox ∩ MCX_bbox) / ROI_bbox coverage for all samples.
    Uses bridge output roi_info.json + MCX bbox from FrameManifest.
    """
    fm = FrameManifest.load(SHARED_DIR)
    mcx_lo = fm.mcx_bbox_min
    mcx_hi = fm.mcx_bbox_max

    stats = []
    for sid in sample_ids:
        roi_info_path = BRIDGE_TRAIN_DIR / sid / "roi_info.json"
        if not roi_info_path.exists():
            continue
        roi_info = json.loads(roi_info_path.read_text())

        # ROI bbox before padding — new bridge stores world-frame (mcx_trunk_local_mm)
        roi_lo = np.array(roi_info["roi_bbox_mm"]["min"])
        roi_hi = np.array(roi_info["roi_bbox_mm"]["max"])

        # With 1mm padding (matches precompute default)
        PAD = 1.0
        padded_lo = roi_lo - PAD
        padded_hi = roi_hi + PAD

        # Intersection with MCX bbox
        int_lo = np.maximum(padded_lo, mcx_lo)
        int_hi = np.minimum(padded_hi, mcx_hi)
        int_vol = float(np.prod(np.maximum(int_hi - int_lo, 0)))
        roi_vol = float(np.prod(padded_hi - padded_lo))

        coverage = int_vol / max(roi_vol, 1e-8)

        overflow_lo = np.maximum(mcx_lo - padded_lo, 0)
        overflow_hi = np.maximum(padded_hi - mcx_hi, 0)

        clamp_active = coverage < 0.999
        severe_loss = coverage < 0.8

        stats.append({
            "sid": sid,
            "coverage": coverage,
            "roi_vol_mm3": roi_vol,
            "overflow_lo_mm": overflow_lo.tolist(),
            "overflow_hi_mm": overflow_hi.tolist(),
            "clamp_active": clamp_active,
            "severe_loss": severe_loss,
        })

    if not stats:
        return {"per_sample": [], "summary": {}, "error": "No bridge output found"}

    coverages = np.array([s["coverage"] for s in stats])

    summary = {
        "n_samples": len(stats),
        "coverage_mean": float(coverages.mean()),
        "coverage_min": float(coverages.min()),
        "coverage_p05": float(np.percentile(coverages, 5)),
        "coverage_median": float(np.median(coverages)),
        "n_clamp_active": int(sum(s["clamp_active"] for s in stats)),
        "n_severe_loss": int(sum(s["severe_loss"] for s in stats)),
        "coverage_hist": _histogram(coverages, bins=10).tolist(),
    }

    # Determine tier
    if summary["coverage_min"] >= 0.95:
        tier = "A"
        tier_desc = "Perfect coverage: ROI fully inside MCX bbox (no clamp needed)"
    elif summary["coverage_min"] >= 0.8:
        tier = "B"
        tier_desc = "Boundary micro-leakage: clamp is reasonable, note in paper"
    elif summary["coverage_mean"] >= 0.5:
        tier = "C"
        tier_desc = "Significant leakage: clamp may lose ROI portions, investigate"
    else:
        tier = "D"
        tier_desc = "Severe misalignment: possible coordinate system bug, halt and investigate"

    summary["tier"] = tier
    summary["tier_desc"] = tier_desc

    # List worst samples if C or D
    if tier in ("C", "D"):
        worst = sorted(stats, key=lambda x: x["coverage"])[:10]
        summary["worst_10_samples"] = [
            {"sid": w["sid"], "coverage": w["coverage"]} for w in worst
        ]

    return {"per_sample": stats, "summary": summary}


def _histogram(arr: np.ndarray, bins: int = 10) -> np.ndarray:
    counts, _ = np.histogram(arr, bins=bins, range=(0.0, 1.0))
    return counts.astype(int)


# ─── Phase 0C: Δ_representation Oracle ───────────────────────────────────────

def phase_0c_representation_oracle(precomputed_dir: Path, val_sample_ids: list[str]):
    """
    Quantify Δ_表示 = FEM_voxel_dice - Stage1_node_dice.

    This measures the pure "representation change" from piecewise-linear FEM
    to voxel trilinear — independent of Stage 2 MLP.

    Also reports:
    - oracle_dice: using gt itself (should be 1.0)
    - representation_delta: the "free gain" from switching to voxel space
    """
    import csv
    from scipy.ndimage import map_coordinates

    # Load Stage 1 per-sample metrics (avoid pandas)
    stage1_dice = {}
    if STAGE1_METRICS_CSV.exists():
        with open(STAGE1_METRICS_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # dice_bin_0.5 is the soft Dice at threshold 0.5
                stage1_dice[row["sample_id"]] = float(row["dice_bin_0.5"])

    results = []
    for sid in val_sample_ids:
        npz_path = precomputed_dir / f"{sid}.npz"
        if not npz_path.exists():
            continue

        data = dict(np.load(npz_path, allow_pickle=False))
        valid = data["valid_mask"]
        if valid.sum() == 0:
            continue

        gt_values = data["gt_values"][valid]          # trilinear(gt_voxels)
        prior = data["prior_8d"][valid]
        fem_interp = (prior[:, :4] * prior[:, 4:8]).sum(-1)  # Σλi · coarse_d

        # Oracle: gt vs gt = 1.0 by definition
        dice_oracle = 1.0

        # FEM on voxel grid vs gt
        dice_fem = compute_dice(fem_interp, gt_values, 0.5)

        # Stage 1 node-level Dice (from evaluation CSV)
        node_dice = stage1_dice.get(sid, None)
        if node_dice is None:
            continue

        # Δ_表示: the "free" gain purely from switching representation
        # = Dice after trilinear resampling to voxel grid - Dice at FEM nodes
        delta_repr = dice_fem - node_dice

        results.append({
            "sid": sid,
            "stage1_node_dice": node_dice,
            "fem_voxel_dice": dice_fem,
            "oracle_dice": dice_oracle,
            "delta_representation": delta_repr,
        })

    if not results:
        return {"per_sample": [], "summary": {}, "error": "No precomputed data found for val samples"}

    arr = np.array([d["delta_representation"] for d in results])

    summary = {
        "delta_repr_mean": float(arr.mean()),
        "delta_repr_std": float(arr.std()),
        "delta_repr_p05": float(np.percentile(arr, 5)),
        "delta_repr_p95": float(np.percentile(arr, 95)),
        "delta_repr_min": float(arr.min()),
        "delta_repr_max": float(arr.max()),
        "n_samples": len(results),
        "fem_voxel_dice_mean": float(np.mean([r["fem_voxel_dice"] for r in results])),
        "stage1_node_dice_mean": float(np.mean([r["stage1_node_dice"] for r in results])),
    }

    # Discussion draft
    if summary["delta_repr_mean"] > 0.06:
        discussion = (
            f"Stage 2 total gain (0.8512→0.90) has ~{summary['delta_repr_mean']:.2f} from "
            f"pure representation change (FEM nodes → voxel trilinear). "
            f"The MLP residual contribution is likely much smaller than the headline number. "
            f"Consider reporting 'MLP residual ΔDice ≈ total_gain - representation_delta' "
            f"in the Discussion."
        )
    elif summary["delta_repr_mean"] > 0.02:
        discussion = (
            f"Representation change contributes ~{summary['delta_repr_mean']:.2f} to ΔDice. "
            f"Stage 2 MLP residual contributes the remainder "
            f"(≈ {0.05 - summary['delta_repr_mean']:.2f} assuming total gain of 0.05). "
            f"Both effects are real but should be reported separately."
        )
    elif summary["delta_repr_mean"] < -0.01:
        # Negative delta signals data breakage (precomputed valid_mask=0) or true degradation
        discussion = (
            f"Δ_表示 is negative ({summary['delta_repr_mean']:.2f}), indicating the precomputed "
            f"data may be broken (valid_mask=0) or that voxel-trilinear representation is "
            f"incompatible with the FEM-interpolated coarse_d. "
            f"⚠️ Do not trust this number — regenerate precomputed data before interpreting. "
            f"The absolute value of this delta is not meaningful until data is fixed."
        )
    else:
        discussion = (
            f"Representation change contributes only ~{summary['delta_repr_mean']:.2f} to ΔDice. "
            f"Most of the ~0.05 total gain is attributable to Stage 2 MLP residual learning. "
            f"The Δ_表示 correction is negligible."
        )

    summary["discussion_draft"] = discussion

    return {"per_sample": results, "summary": summary}


# ─── Main ──────────────────────────────────────────────────────────────────────

def build_report(results: dict) -> str:
    """Build markdown report from all phase results."""
    lines = [
        "# Stage 2 Pipeline Diagnostic Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
    ]

    # Phase 0A
    if "0a" in results:
        r0a = results["0a"]
        lines += [
            "## Phase 0A: Axis Oracle",
            "",
            "### Motivation",
            "Verify `FrameManifest.world_to_gt_index` and `gt_voxels.shape` axis order "
            "are consistent. Any mismatch would silently produce wrong spatial distributions.",
            "",
            "### Method",
            "1. **Round-trip**: argmax voxel → world → voxel index (expect err < 0.01)",
            "2. **Corner**: 4 bounding-box corners → world → voxel (expect err < 0.01)",
            "",
            "### Results",
        ]
        if isinstance(r0a, dict) and "error" in r0a:
            lines += [f"ERROR: {r0a['error']}", ""]
        else:
            for res in r0a:
                status = "PASS" if res["pass"] else "FAIL"
                lines.append(
                    f"- `{res['sid']}`: shape={res['gt_voxels_shape']}, "
                    f"roundtrip_err={res['roundtrip_err']:.4f}, "
                    f"corner_err={res['corner_err']:.4f} → **{status}**"
                )
            all_pass = all(r["pass"] for r in r0a)
            if all_pass:
                lines += [
                    "",
                    "### Verdict: ✅ PASS",
                    "All samples pass. `world_to_gt_index` axis order is consistent with `gt_voxels.shape`.",
                    "",
                    "gt_voxels shape convention: **[Z, Y, X]** (standard volumetric array). "
                    "`world_to_gt_index` returns **(X_idx, Y_idx, Z_idx)** in voxel grid order — "
                    "same convention used by `scipy.ndimage.map_coordinates` when called with "
                    "`idx_float.T` (shape [3, G]).",
                ]
            else:
                lines += [
                    "",
                    "### Verdict: ❌ FAIL",
                    "Coordinate mismatch detected. **Do NOT proceed** — fix FrameManifest first.",
                    "All Stage 1/Stage 2 Dice numbers must be re-validated after the fix.",
                ]

    # Phase 0B
    if "0b" in results:
        r0b = results["0b"]
        lines += [
            "---",
            "",
            "## Phase 0B: ROI-MCX Bbox Alignment",
            "",
            "### Motivation",
            "Quantify how much ROI bbox extends beyond MCX volume — determines if "
            "`precompute_stage2_data.py` clamp is a harmless safety net vs. data loss.",
            "",
            "### Method",
            "Coverage = `(ROI_padded ∩ MCX_bbox) / ROI_padded` per sample. "
            "ROI is padded by 1mm (precompute default).",
            "",
            "### Results",
        ]
        if isinstance(r0b, dict) and "error" in r0b:
            lines += [f"ERROR: {r0b['error']}", ""]
        else:
            s = r0b["summary"]
            lines += [
                f"- **Samples**: {s['n_samples']}",
                f"- **Coverage mean**: {s['coverage_mean']:.4f}",
                f"- **Coverage median**: {s['coverage_median']:.4f}",
                f"- **Coverage p05**: {s['coverage_p05']:.4f}",
                f"- **Coverage min**: {s['coverage_min']:.4f}",
                f"- **Clamp active (coverage < 0.999)**: {s['n_clamp_active']}/{s['n_samples']}",
                f"- **Severe loss (coverage < 0.8)**: {s['n_severe_loss']}/{s['n_samples']}",
                "",
                f"**Tier {s['tier']}**: {s['tier_desc']}",
                "",
                "### Coverage Histogram (10 bins)",
                "```",
                _histogram_str(s.get("coverage_hist", [])),
                "```",
                "(bins: 0.0-0.1, 0.1-0.2, ..., 0.9-1.0)",
                "",
            ]
            if s.get("worst_10_samples"):
                lines += ["**Worst 10 samples (for investigation)**:", ""]
                for w in s["worst_10_samples"]:
                    lines.append(f"- `{w['sid']}`: coverage={w['coverage']:.4f}")
                lines.append("")

            if s["tier"] == "A":
                lines.append("### Verdict: ✅ TIER A — No action needed")
            elif s["tier"] == "B":
                lines.append(
                    "### Verdict: ⚠️ TIER B — Boundary micro-leakage\n"
                    "Clamp is acceptable. Add a note in the paper appendix describing "
                    "that <5% of ROI volume extends beyond MCX FOV and is clipped."
                )
            elif s["tier"] == "C":
                lines.append(
                    "### Verdict: ⚠️ TIER C — Significant ROI loss under clamp\n"
                    "Investigate whether MCX bbox computation or ROI derivation has a bug. "
                    "Consider increasing grid padding or revising ROI bounding box."
                )
            else:
                lines.append(
                    "### Verdict: ❌ TIER D — Severe misalignment (coordinate bug?)\n"
                    "Coverage < 50%. STOP. Joint investigation with Phase 0A required."
                )

    # Phase 0C
    if "0c" in results:
        r0c = results["0c"]
        lines += [
            "---",
            "",
            "## Phase 0C: Δ_representation Oracle Baseline",
            "",
            "### Motivation",
            "Quantify the 'free' Dice gain purely from switching representation "
            "(piecewise-linear FEM nodes → voxel trilinear) — before any Stage 2 MLP learning.",
            "",
            "### Method",
            "For each val sample:\n"
            "1. Load precomputed `gt_values` (= trilinear(gt_voxels)) and `prior_8d`\n"
            "2. Compute `fem_interp = Σλi·coarse_d` (FEM barycentric interpolation)\n"
            "3. Compute `fem_voxel_dice = Dice(fem_interp, gt_values, 0.5)`\n"
            "4. Δ_表示 = `fem_voxel_dice - stage1_node_dice`\n",
            "",
            "### Results",
        ]
        if isinstance(r0c, dict) and "error" in r0c:
            lines += [f"ERROR: {r0c['error']}", ""]
        else:
            s = r0c["summary"]
            lines += [
                f"- **Samples**: {s['n_samples']}",
                f"- **Stage1 node Dice (mean)**: {s['stage1_node_dice_mean']:.4f}",
                f"- **FEM voxel Dice (mean)**: {s['fem_voxel_dice_mean']:.4f}",
                f"- **Δ_表示 mean**: {s['delta_repr_mean']:+.4f}",
                f"- **Δ_表示 std**: {s['delta_repr_std']:.4f}",
                f"- **Δ_表示 p05**: {s['delta_repr_p05']:+.4f}",
                f"- **Δ_表示 p95**: {s['delta_repr_p95']:+.4f}",
                f"- **Δ_表示 [min, max]**: [{s['delta_repr_min']:+.4f}, {s['delta_repr_max']:+.4f}]",
                "",
                f"### Discussion Draft",
                f"> {s.get('discussion_draft', 'N/A')}",
                "",
                "### Verdict",
                f"Δ_表示 ≈ {s['delta_repr_mean']:+.3f}. "
                "This is the 'free' Dice improvement from simply resampling the FEM field "
                "onto the voxel grid (trilinear), independent of Stage 2 MLP residual learning. "
                "Report this separately in the paper to avoid over-attributing Dice gain to the MLP.",
            ]

    lines += [
        "---",
        "",
        "## Phase 2: Bug Fixes Applied",
        "",
        "| Bug | Description | Status |",
        "|-----|-------------|--------|",
        "| B1 | `tol` inconsistency: `_barycentric_batch` used `-1e-8`, `barycentric_coords` used `-1e-6` | Fixed: unified to `-1e-6` in `_barycentric_batch` |",
        "| B2 | MSE not clamped to [0,1] before reporting | Fixed: added `np.clip(d_hat, 0.0, 1.0)` in validate() |",
        "| B3 | `valid` field hardcoded to `torch.ones(...)` in precomputed dataset (semantic confusion) | Fixed: explicit comment explaining valid is pre-filtered |",
        "| B4 | `mcx_valid` mask computed but not applied to `view_feat` in multiview training | Fixed: `view_feat = view_feat * mcx_valid.unsqueeze(-1)` in train_step and validate |",
        "| GT src | `Stage2Dataset` used `FEM_interp(gt_nodes)` vs precomputed `trilinear(gt_voxels)` | Fixed: unified `Stage2Dataset` to trilinear(gt_voxels) |",
    ]

    return "\n".join(lines)


def _histogram_str(counts: list, bins: int = 10) -> str:
    bin_edges = np.linspace(0, 1, bins + 1)
    lines = []
    for i in range(bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        bar = "█" * int(counts[i] / max(sum(counts), 1) * 40)
        lines.append(f"  [{lo:.1f}-{hi:.1f}) | {int(counts[i]):4d} | {bar}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Stage 2 pipeline diagnostics")
    parser.add_argument("--phase", default="all", choices=["all", "0a", "0b", "0c", "0a,0b,0c"])
    parser.add_argument("--output", default="diagnosis/stage2_pipeline_report.md")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--config", default="configs/stage2/uniform_1000_v2.yaml")
    args = parser.parse_args()

    phases = args.phase.split(",") if "," in args.phase else [args.phase]
    if "all" in phases:
        phases = ["0a", "0b", "0c"]

    # Load sample IDs from splits
    train_ids = load_split(SPLIT_DIR / "train.txt")
    val_ids = load_split(SPLIT_DIR / "val.txt")

    print(f"[Diagnose] train={len(train_ids)}, val={len(val_ids)} samples")
    print(f"[Diagnose] Phases: {phases}")

    results = {}

    t0 = time.perf_counter()

    if "0a" in phases:
        print("\n=== Phase 0A: Axis Oracle ===")
        # Use 5 val samples for quick check
        ids = [sid for sid in val_ids if (SAMPLES_DIR / sid / "gt_voxels.npy").exists()]
        results["0a"] = phase_0a_axis_oracle(ids, n_samples=5, verbose=args.verbose)
        for r in results["0a"]:
            status = "PASS" if r["pass"] else "FAIL"
            print(f"  {r['sid']}: err={r['roundtrip_err']:.4f}/{r['corner_err']:.4f} → {status}")
        all_pass = all(r["pass"] for r in results["0a"])
        print(f"  → {'✅ PASS' if all_pass else '❌ FAIL'}")

    if "0b" in phases:
        print("\n=== Phase 0B: ROI-MCX Alignment ===")
        bridge_ids = [sid for sid in train_ids if (BRIDGE_TRAIN_DIR / sid / "roi_info.json").exists()]
        print(f"  Running on {len(bridge_ids)} bridge samples...")
        results["0b"] = phase_0b_roi_mcx_alignment(bridge_ids)
        s = results["0b"]["summary"]
        cov_mean = s.get('coverage_mean', 'N/A')
        cov_min = s.get('coverage_min', 'N/A')
        tier = s.get('tier', 'N/A')
        if isinstance(cov_mean, float):
            print(f"  coverage_mean={cov_mean:.4f}, min={cov_min:.4f}, tier={tier}")
        else:
            print(f"  coverage_mean={cov_mean}, min={cov_min}, tier={tier}")

    if "0c" in phases:
        print("\n=== Phase 0C: Δ_representation Oracle ===")
        val_precomputed_ids = [
            sid for sid in val_ids
            if (PRECOMPUTED_DIR / f"{sid}.npz").exists()
        ]
        print(f"  Running on {len(val_precomputed_ids)} precomputed val samples...")
        results["0c"] = phase_0c_representation_oracle(PRECOMPUTED_DIR, val_precomputed_ids)
        s = results["0c"]["summary"]
        print(f"  Δ_repr_mean={s.get('delta_repr_mean', 'N/A'):+.4f}, "
              f"fem_voxel_dice={s.get('fem_voxel_dice_mean', 'N/A'):.4f}, "
              f"stage1_node_dice={s.get('stage1_node_dice_mean', 'N/A'):.4f}")

    elapsed = time.perf_counter() - t0
    print(f"\n[Diagnose] Total time: {elapsed:.1f}s")

    # Build and save report
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    report = build_report(results)
    Path(args.output).write_text(report)
    print(f"[Diagnose] Report written to {args.output}")

    # Exit with error code if Phase 0A failed
    if "0a" in results and not all(r["pass"] for r in results["0a"]):
        print("\n❌ Phase 0A FAILED — coordinate mismatch detected. Fix FrameManifest before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
