# DU2Vox Baseline Results — May 2026

**Preserved: 2026-05-07**

These are the authoritative baseline results used for the final evaluation
conclusions. They are frozen here to prevent accidental modification when
code changes.

## Files

| File | Description |
|------|-------------|
| `FINAL_stage1_mesh_dice_binary05.json` | Stage 1 Mesh Dice @ binary threshold=0.5 = **0.6108** |
| `FINAL_stage1_voxel_fem_baseline.json` | Stage 1 Voxel Dice = FEM interp = **0.5962** (baseline) |
| `FINAL_stage2_deonly_roi_dice.json` | Stage 2 DE-only ROI Dice = **0.6082**, ΔDice = **+0.012** |
| `FINAL_stage2_multiview_roi_dice.json` | Stage 2 Multiview ROI Dice = **0.6557**, ΔDice = **+0.060** |
| `FINAL_comparison_summary.json` | Per-foci / per-depth breakdown (JSON) |
| `FINAL_comparison_summary.md` | Per-foci / per-depth breakdown (Markdown table) |
| `figures/` | Representative sample renders |

## Key Results

```
Metric                          ROI Dice   Full-Grid
Stage 1 Mesh (binary @0.5)       N/A       0.6108
Stage 1 Voxel = FEM baseline    0.5962    0.5905
Stage 2 DE-only                 0.6082    0.6013
Stage 2 Multiview               0.6557    0.6487

Δ DE-only vs FEM     +0.012   +0.011
Δ Multiview vs FEM   +0.060   +0.058
```

## Checkpoints

| Experiment | Checkpoint |
|---|---|
| Stage 1 | `runs/stage1_uniform_1000_20k/checkpoints/best.pth` |
| DE-only | `checkpoints/stage2/de_only_20k_v3/best.pth` (epoch 6) |
| Multiview | `checkpoints/stage2/mv_fixed_ext2/best.pth` (epoch 44) |

## Dice Methodology

See `docs/dice_evaluation_methodology.md` for the full explanation of
soft vs binary Dice, evaluation domains (mesh/ROI/full-grid), and the
threshold sweep analysis.
