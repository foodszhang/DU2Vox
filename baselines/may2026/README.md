# DU2Vox Baseline Results — May 2026

**Archived: 2026-05-07**

These are the authoritative baseline results used for the final evaluation
conclusions. They are frozen here to prevent accidental modification when
code changes.

---

## Quick Eval Commands

```bash
# ── Stage 1 ───────────────────────────────────────────────────────────────
# Mesh Dice (binary @0.5)
uv run python scripts/eval_du2vox.py stage1 \
    --config baselines/may2026/configs/stage1/uniform_1000_20k.yaml \
    --checkpoint baselines/may2026/checkpoints/stage1/uniform_1000_20k/best.pth \
    --voxel \
    --output baselines/may2026/results/stage1_eval.json

# ── Stage 2 DE-only ───────────────────────────────────────────────────────
uv run python scripts/eval_du2vox.py stage2 \
    --config baselines/may2026/configs/stage2/de_only_20k.yaml \
    --checkpoint baselines/may2026/checkpoints/stage2/de_only_20k_v3/best.pth \
    --output baselines/may2026/results/de_only_eval.json

# ── Stage 2 Multiview ────────────────────────────────────────────────────
uv run python scripts/eval_du2vox.py stage2 \
    --config baselines/may2026/configs/stage2/full_multiview_20k.yaml \
    --checkpoint baselines/may2026/checkpoints/stage2/mv_fixed_ext2/best.pth \
    --multiview \
    --output baselines/may2026/results/multiview_eval.json

# ── Side-by-side comparison (with 3D renders) ───────────────────────────
uv run python scripts/compare_stage1_stage2_voxel.py \
    --stage1_json baselines/may2026/results/stage1_voxel.json \
    --de_only_json baselines/may2026/results/de_only_eval.json \
    --de_only_config baselines/may2026/configs/stage2/de_only_20k.yaml \
    --de_only_checkpoint baselines/may2026/checkpoints/stage2/de_only_20k_v3/best.pth \
    --multiview_json baselines/may2026/results/multiview_eval.json \
    --multiview_config baselines/may2026/configs/stage2/full_multiview_20k.yaml \
    --multiview_checkpoint baselines/may2026/checkpoints/stage2/mv_fixed_ext2/best.pth \
    --shared_dir /home/foods/pro/FMT-SimGen/output/shared_mesh_20k \
    --samples_dir /home/foods/pro/FMT-SimGen/data/uniform_1000_20k/samples \
    --bridge_dir output/bridge_20k_val \
    --precomputed_dir precomputed/val_20k \
    --output_dir baselines/may2026/results/comparison
```

---

## Key Results

```
Metric                          ROI Dice   Full-Grid
───────────────────────────────────────────────────────
Stage 1 Mesh (binary @0.5)      N/A       0.6108
Stage 1 Voxel = FEM baseline    0.5962    0.5905
Stage 2 DE-only                 0.6082    0.6013
Stage 2 Multiview               0.6557    0.6487

Δ DE-only vs FEM     +0.012   +0.011
Δ Multiview vs FEM   +0.060   +0.058
```

### Per-Foci Breakdown (ROI Dice @0.5)

| Scope  | N  | FEM Baseline | DE-only | ΔDE   | Multiview | ΔMV   |
|--------|----|-------------|---------|-------|-----------|-------|
| Overall|200 | 0.5962      | 0.6082  |+0.012 | 0.6557    |+0.060 |
| 1-Foci | 66 | 0.6596      | 0.6715  |+0.012 | 0.7476    |+0.088 |
| 2-Foci | 73 | 0.5648      | 0.5974  |+0.033 | 0.6321    |+0.067 |
| 3-Foci | 61 | 0.5650      | 0.5525  |-0.013 | 0.5845    |+0.019 |

### Per-Depth Breakdown (ROI Dice @0.5)

| Metric | Shallow | Medium | Deep |
|--------|---------|--------|------|
| FEM Baseline | 0.6019 | 0.5956 | 0.5908 |
| DE-only       | 0.6015 | 0.6109 | 0.6114 |
| Multiview     | 0.6615 | 0.6428 | 0.6684 |

---

## File Inventory

### Checkpoints
| Experiment | Path | Size |
|---|---|---|
| Stage 1 | `checkpoints/stage1/uniform_1000_20k/best.pth` | ~0.6 MB |
| DE-only | `checkpoints/stage2/de_only_20k_v3/best.pth` | ~1.5 MB |
| Multiview | `checkpoints/stage2/mv_fixed_ext2/best.pth` | ~3.4 MB |

### Training Logs
| Experiment | Path | Best Epoch |
|---|---|---|
| DE-only | `logs/de_only_20k_v3_train_log.json` | Epoch 6 (Δ=+0.013) |
| Multiview | `logs/mv_fixed_ext2_train_log.json` | Epoch 44 (Δ=+0.064) |

### Evaluation Results
| File | Description |
|---|---|
| `FINAL_stage1_mesh_dice_binary05.json` | Stage 1 Mesh Dice @0.5 = **0.6108** |
| `FINAL_stage1_voxel_fem_baseline.json` | Stage 1 Voxel = FEM = **0.5962** |
| `FINAL_stage2_deonly_roi_dice.json` | DE-only = **0.6082**, Δ=**+0.012** |
| `FINAL_stage2_multiview_roi_dice.json` | Multiview = **0.6557**, Δ=**+0.060** |
| `FINAL_comparison_summary.json` | Per-foci/per-depth breakdown (JSON) |
| `FINAL_comparison_summary.md` | Per-foci/per-depth breakdown (Markdown) |

### Configs
| File | Description |
|---|---|
| `configs/stage1/uniform_1000_20k.yaml` | Stage 1 config |
| `configs/stage2/de_only_20k.yaml` | DE-only config |
| `configs/stage2/full_multiview_20k.yaml` | Multiview config |

### Figures
| File | Description |
|---|---|
| `figures/sample_0141_compare.png` | 1-Foci representative |
| `figures/sample_0546_compare.png` | 2-Foci representative |
| `figures/sample_0830_compare.png` | 3-Foci representative |

---

## Dice Methodology

See `docs/dice_evaluation_methodology.md` for the full explanation of
soft vs binary Dice, evaluation domains (mesh/ROI/full-grid), and the
threshold sweep analysis. Key points:

- **Soft Dice 0.85 → binary @0.5 = 0.61** for Stage 1 Mesh (threshold drops 23%)
- **ROI Dice** evaluates on ~80k valid grid points (inside ROI tets)
- **Full-grid Dice** evaluates on all ~563k grid points, zero-filled outside ROI
- Stage 2 always uses **ROI Dice @0.5** as the primary metric

---

## Data Dependencies

All results use:
- Bridge output: `output/bridge_20k_val/` (Stage 1→2)
- Precomputed data: `precomputed/val_20k/` (200 validation samples)
- Source data: `FMT-SimGen/output/shared_mesh_20k/`, `FMT-SimGen/data/uniform_1000_20k/`
