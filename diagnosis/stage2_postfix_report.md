# Stage 2 Pipeline Diagnostic Report

Generated: 2026-04-27 11:05:03

---

## Phase 0A: Axis Oracle

### Motivation
Verify `FrameManifest.world_to_gt_index` and `gt_voxels.shape` axis order are consistent. Any mismatch would silently produce wrong spatial distributions.

### Method
1. **Round-trip**: argmax voxel → world → voxel index (expect err < 0.01)
2. **Corner**: 4 bounding-box corners → world → voxel (expect err < 0.01)

### Results
- `sample_0003`: shape=[190, 190, 190], roundtrip_err=0.0000, corner_err=0.0000 → **PASS**
- `sample_0006`: shape=[190, 190, 190], roundtrip_err=0.0000, corner_err=0.0000 → **PASS**
- `sample_0011`: shape=[190, 190, 190], roundtrip_err=0.0000, corner_err=0.0000 → **PASS**
- `sample_0025`: shape=[190, 190, 190], roundtrip_err=0.0000, corner_err=0.0000 → **PASS**
- `sample_0027`: shape=[190, 190, 190], roundtrip_err=0.0000, corner_err=0.0000 → **PASS**

### Verdict: ✅ PASS
All samples pass. `world_to_gt_index` axis order is consistent with `gt_voxels.shape`.

gt_voxels shape convention: **[Z, Y, X]** (standard volumetric array). `world_to_gt_index` returns **(X_idx, Y_idx, Z_idx)** in voxel grid order — same convention used by `scipy.ndimage.map_coordinates` when called with `idx_float.T` (shape [3, G]).
---

## Phase 0B: ROI-MCX Bbox Alignment

### Motivation
Quantify how much ROI bbox extends beyond MCX volume — determines if `precompute_stage2_data.py` clamp is a harmless safety net vs. data loss.

### Method
Coverage = `(ROI_padded ∩ MCX_bbox) / ROI_padded` per sample. ROI is padded by 1mm (precompute default).

### Results
- **Samples**: 800
- **Coverage mean**: 0.2515
- **Coverage median**: 0.2629
- **Coverage p05**: 0.0000
- **Coverage min**: 0.0000
- **Clamp active (coverage < 0.999)**: 800/800
- **Severe loss (coverage < 0.8)**: 800/800

**Tier D**: Severe misalignment: possible coordinate system bug, halt and investigate

### Coverage Histogram (10 bins)
```
  [0.0-0.1) |   90 | ████
  [0.1-0.2) |  102 | █████
  [0.2-0.3) |  366 | ██████████████████
  [0.3-0.4) |  185 | █████████
  [0.4-0.5) |   55 | ██
  [0.5-0.6) |    2 | 
  [0.6-0.7) |    0 | 
  [0.7-0.8) |    0 | 
  [0.8-0.9) |    0 | 
  [0.9-1.0) |    0 | 
```
(bins: 0.0-0.1, 0.1-0.2, ..., 0.9-1.0)

**Worst 10 samples (for investigation)**:

- `sample_0001`: coverage=0.0000
- `sample_0074`: coverage=0.0000
- `sample_0076`: coverage=0.0000
- `sample_0088`: coverage=0.0000
- `sample_0133`: coverage=0.0000
- `sample_0153`: coverage=0.0000
- `sample_0160`: coverage=0.0000
- `sample_0171`: coverage=0.0000
- `sample_0183`: coverage=0.0000
- `sample_0205`: coverage=0.0000

### Verdict: ❌ TIER D — Severe misalignment (coordinate bug?)
Coverage < 50%. STOP. Joint investigation with Phase 0A required.
---

## Phase 0C: Δ_representation Oracle Baseline

### Motivation
Quantify the 'free' Dice gain purely from switching representation (piecewise-linear FEM nodes → voxel trilinear) — before any Stage 2 MLP learning.

### Method
For each val sample:
1. Load precomputed `gt_values` (= trilinear(gt_voxels)) and `prior_8d`
2. Compute `fem_interp = Σλi·coarse_d` (FEM barycentric interpolation)
3. Compute `fem_voxel_dice = Dice(fem_interp, gt_values, 0.5)`
4. Δ_表示 = `fem_voxel_dice - stage1_node_dice`


### Results
- **Samples**: 200
- **Stage1 node Dice (mean)**: 0.7506
- **FEM voxel Dice (mean)**: 0.0201
- **Δ_表示 mean**: -0.7305
- **Δ_表示 std**: 0.1798
- **Δ_表示 p05**: -0.8944
- **Δ_表示 p95**: -0.3995
- **Δ_表示 [min, max]**: [-0.9383, +0.4437]

### Discussion Draft
> Δ_表示 is negative (-0.73), indicating the precomputed data may be broken (valid_mask=0) or that voxel-trilinear representation is incompatible with the FEM-interpolated coarse_d. ⚠️ Do not trust this number — regenerate precomputed data before interpreting. The absolute value of this delta is not meaningful until data is fixed.

### Verdict
Δ_表示 ≈ -0.731. This is the 'free' Dice improvement from simply resampling the FEM field onto the voxel grid (trilinear), independent of Stage 2 MLP residual learning. Report this separately in the paper to avoid over-attributing Dice gain to the MLP.
---

## Phase 2: Bug Fixes Applied

| Bug | Description | Status |
|-----|-------------|--------|
| B1 | `tol` inconsistency: `_barycentric_batch` used `-1e-8`, `barycentric_coords` used `-1e-6` | Fixed: unified to `-1e-6` in `_barycentric_batch` |
| B2 | MSE not clamped to [0,1] before reporting | Fixed: added `np.clip(d_hat, 0.0, 1.0)` in validate() |
| B3 | `valid` field hardcoded to `torch.ones(...)` in precomputed dataset (semantic confusion) | Fixed: explicit comment explaining valid is pre-filtered |
| B4 | `mcx_valid` mask computed but not applied to `view_feat` in multiview training | Fixed: `view_feat = view_feat * mcx_valid.unsqueeze(-1)` in train_step and validate |
| GT src | `Stage2Dataset` used `FEM_interp(gt_nodes)` vs precomputed `trilinear(gt_voxels)` | Fixed: unified `Stage2Dataset` to trilinear(gt_voxels) |