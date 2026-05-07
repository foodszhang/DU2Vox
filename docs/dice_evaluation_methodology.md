# Stage 1 / Stage 2 Dice Evaluation Methodology

**Date: 2026-05-07**
**Status: Final**

---

## 1. Three Dice Variants in DU2Vox

| Variant | Formula | Threshold? | Domain | Reported As |
|---|---|---|---|---|
| **Soft Dice** | `2·Σ(p·g) / (Σp + Σg)` | No — continuous | Full domain | `dice` |
| **Binary Dice** | `2·TP / (2·TP + FP + FN)` | Yes — both pred & gt binarized | Full domain | `dice_bin_0.5` etc. |
| **Binary Dice ROI** | Same as Binary Dice | Yes | ROI grid points only | `voxel_dice`, `stage2_dice` |

All Stage 2 evaluation uses **Binary Dice @ 0.5 on ROI grid points**.

---

## 2. Why Soft Dice ≠ Binary Dice

**Example — Stage 1 Mesh:**
```
Soft Dice (mesh):  0.8462
Binary Dice @ 0.5: 0.6108  ← 23% gap
```

Root cause: mesh predictions contain negative values (AdaptiveThreshold output range `(-∞, +∞)` after sigmoid gate). After clamping to [0, ∞):

- Clamped pred range: [0.0, 4.24]
- At GT==1 (tumor) nodes: mean pred = 1.84, median = 1.98
- At GT==0 (background) nodes: ~100% are exactly 0.0000; ~0.4% produce small positives

**Threshold sweep (Stage 1 Mesh):**
```
Thresh   Dice     TP      FP      FN
 0.05   0.530   5735    9501    672
 0.10   0.547   5686    8691    721
 0.15   0.556   5639    8222    768
 0.20   0.566   5609    7822    798
 0.30   0.575   5560    7369    847
 0.50   0.590   5481    6697    926   ← optimal in this range
```

- Low threshold captures more TP but also more background FP → Dice lower
- 0.5 is near-optimal; Dice is monotonically increasing up to 0.5
- Soft Dice captures partial credit from low-but-nonzero predictions at tumor nodes

**GT is binary at mesh nodes** (binarized at 0.05 in dataset, so GT ∈ {0, 1}).

---

## 3. Evaluation Domains

| Metric | Domain | Size | Easy Negatives? |
|---|---|---|---|
| **Stage 1 Mesh Dice** | All FEM nodes | ~20k nodes/sample | Yes — 99.8% are background |
| **Stage 1 Voxel Dice** | ROI grid points only | ~80k valid points/sample | No — only tumor-proximal points |
| **Stage 2 Dice** | ROI grid points only | ~80k valid points/sample | No — same domain as Stage 1 Voxel |

ROI = tetrahedra where `coarse_d > τ` (Stage 1 threshold-derived region).
Grid points outside ROI have `valid_mask = False` and are excluded from Stage 2 evaluation.

---

## 4. Full-Grid vs ROI Dice

Full-grid Dice: predict 0 outside ROI (zero-filled), evaluate on ALL ~563k grid points.
ROI Dice: evaluate on ~80k valid (inside-ROI) points only.

```
Metric                          ROI Dice   Full-Grid
FEM baseline (coarse interp)    0.5962     0.5905
Stage 2 DE-only                 0.6082     0.6013
Stage 2 Multiview               0.6557     0.6487
```

**Full-grid is ~0.007 lower** because background GT contains微量 floating-point values from trilinear interpolation (not absolute 0). Zero-fill is more conservative than the model's faint background predictions, slightly reducing intersection relative to ROI-only.

**Δ between methods is stable across both domains:**
- DE-only Δ vs FEM: +0.012 (ROI) / +0.011 (full-grid)
- Multiview Δ vs FEM: +0.060 (ROI) / +0.058 (full-grid)

---

## 5. Stage 1 Mesh vs Voxel Dice Cannot Be Compared Directly

Stage 1 Mesh Dice (full mesh, soft) = 0.8462 vs Stage 1 Voxel Dice (ROI, binary @0.5) = 0.5962.

Three compounding differences:
1. **Soft vs binary** (23% gap from threshold)
2. **Full domain vs ROI** (99.8% vs ~14% background fraction)
3. **Different GT sources** — mesh GT from `gt_nodes.npy` (binary), voxel GT from trilinear interpolation of `gt_voxels.npy` (float)

The correct Stage 1 ↔ Stage 2 comparison is **Stage 1 Voxel Dice (FEM interp)** = **FEM baseline** = 0.5962.

---

## 6. Final Results Summary

### Complete Dice Table @ threshold=0.5

```
Metric                                   ROI Dice   Full-Grid
-----------------------------------------------------------------
Stage 1 Mesh (binary @ 0.5, all nodes)    N/A       0.6108
Stage 1 Voxel = FEM interp baseline       0.5962     0.5905
Stage 2 DE-only                          0.6082     0.6013
Stage 2 Multiview                        0.6557     0.6487

Δ DE-only vs FEM                          +0.0120   +0.0108
Δ Multiview vs FEM                        +0.0596   +0.0582
```

### Per-Foci Breakdown (ROI Dice)

```
Scope      N    FEM Baseline  DE-only   Multiview  ΔDE    ΔMV
Overall   200     0.5962      0.6082    0.6557    +0.012 +0.060
1-Foci     66     0.6596      0.6715    0.7476    +0.012 +0.088
2-Foci     73     0.5648      0.5974    0.6321    +0.033 +0.067
3-Foci     61     0.5650      0.5525    0.5845    -0.013 +0.019
```

### Stage 1 Mesh Dice @ Different Thresholds (per-foci)

```
Foci     N    Soft      Bin@0.5   Bin@0.3
1-Foci   66   0.9555    0.6808    0.6666
2-Foci   73   0.7941    0.5791    0.5624
3-Foci   61   0.7903    0.5730    0.5622
Overall 200   0.8462    0.6108     —
```

---

## 7. Configuration and Checkpoint References

| Experiment | Config | Checkpoint | Best Epoch | S2 Dice | FEM Dice | ΔDice |
|---|---|---|---|---|---|---|
| Stage 1 (mesh+voxel) | `configs/stage1/uniform_1000_20k.yaml` | `runs/stage1_uniform_1000_20k/checkpoints/best.pth` | — | 0.6108 (mesh) | 0.5962 (voxel) | — |
| DE-only | `configs/stage2/de_only_20k.yaml` | `checkpoints/stage2/de_only_20k_v3/best.pth` | 6 | 0.6082 | 0.5962 | +0.012 |
| Multiview | `configs/stage2/full_multiview_20k.yaml` | `checkpoints/stage2/mv_fixed_ext2/best.pth` | 44 | 0.6557 | 0.5962 | +0.060 |

Data source: `precomputed/val_20k/` + `output/bridge_20k_val/`
