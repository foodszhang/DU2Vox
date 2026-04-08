# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DU2Vox is a two-stage framework for Fluorescence Molecular Tomography (FMT) reconstruction:
- **Stage 1** (active): MS-GDUN / GCAIN — 6-stage unrolled network with multi-scale graph attention for coarse relative distribution initialization
- **Stage 2** (planned): Residual INR — FEM-consistent prior-guided voxel distribution generation

## Commands

All commands use `uv run` from the project root to ensure correct dependencies.

```bash
# Train stage 1 (Gaussian source)
uv run python scripts/train_stage1.py --config configs/stage1/gaussian_1000.yaml
uv run python scripts/train_stage1.py --config configs/stage1/gaussian_1000.yaml --resume runs/gcain_gaussian_1000/checkpoints/latest.pth

# Train stage 1 (Uniform source)
uv run python scripts/train_stage1.py --config configs/stage1/uniform_1000.yaml

# Evaluate single model (prints per-foci and per-depth tables when dataset_manifest.json is present)
uv run python scripts/eval_stage1.py --config configs/stage1/gaussian_1000.yaml --checkpoint runs/gcain_gaussian_1000/checkpoints/best.pth
uv run python scripts/eval_stage1.py --config configs/stage1/uniform_1000.yaml --checkpoint runs/gcain_uniform_1000/checkpoints/best.pth

# Comprehensive evaluation: generates all tables + figures in results/stage1/
uv run python scripts/evaluate_stage1.py \
    --checkpoint_g runs/gcain_gaussian_1000/checkpoints/best.pth \
    --checkpoint_u runs/gcain_uniform_1000/checkpoints/best.pth \
    --config_g configs/stage1/gaussian_1000.yaml \
    --config_u configs/stage1/uniform_1000.yaml \
    --shared_dir /home/foods/pro/FMT-SimGen/output/shared \
    --output_dir results/stage1/

# Visualize GT vs prediction
uv run python scripts/visualize.py --config configs/stage1/gaussian_1000.yaml --checkpoint runs/gcain_gaussian_1000/checkpoints/best.pth --n_samples 4
```

Checkpoints are saved under `runs/{experiment_name}/checkpoints/`:
- `checkpoints/latest.pth` — updated every epoch
- `checkpoints/best.pth` — best Dice_bin@0.3

## Architecture

### Stage 1 Model (GCAIN_full)

6 `BasicBlock` stages run sequentially, each containing:
1. **InputBlock**: concatenates `[x, LᵀL x, AᵀA x - Aᵀ b]` → [B, N, 3]
2. **GCN sequence**: 3→8→16→8→feat_dim (default 6)
3. **MultiScale GCN**: 4 parallel branches on Laplacians L0–L3 (different spectral scales)
4. **MultiScaleKNNGraphAttention**: cross-attention across the 4 scales using kNN + sensitivity weights
5. **GradientProjection**: projects attention features → scalar gradient
6. **UpdateBlock**: adaptive threshold sparse update `x = sign(u) * softplus(|u| - λ(feat))`

Cold start: X₀ = zeros. Forward pass: `X₀ → Block₁ → ... → Block₆ → X₆`

### Data Flow

```
b [B, S, 1]  (surface measurements)
     ↓
GCAIN_full (6 stages)
     ↓
pred [B, N, 1]  (voxel predictions at mesh nodes)
```

Shared assets (loaded once from FMT-SimGen output):
- `mesh.npz`: node coordinates [N, 3], surface indices
- `system_matrix.A.npz`: forward matrix [S, N]
- `graph_laplacian_full.*.npz`: L, L0, L1, L2, L3 [N, N]
- `knn_idx_full.npy`: kNN indices [N, 32]

Per-sample (preloaded):
- `measurement_b.npy`: surface measurement [S, 1]
- `gt_nodes.npy`: ground truth distribution [N, 1], per-sample max-normalized

### Loss

Combined: `0.7 * Tversky(α=0.1, β=0.9) + 0.3 * weighted_MSE`
- Tversky heavily penalizes FN (β=0.9, α=0.1) → aggressive tumor ROI expansion
- Weighted MSE: 70% weight on foreground (gt > 0.05)

### Evaluation Metrics

Metrics computed per-batch: Dice (soft), Dice_bin@0.3, Dice_bin@0.1, Recall@0.3, Precision@0.3, Recall@0.1, Precision@0.1, Location Error (mm), MSE

When `dataset_manifest.json` is present at `output/dataset_manifest.json`, `data/dataset_manifest.json`, or `{samples_dir}/../dataset_manifest.json`, results are grouped by:
- **Foci count**: 1-foci, 2-foci, 3-foci
- **Depth tier**: shallow, medium, deep
- **Cross**: foci × depth

### Key Config Notes

Two experiment configs exist for different source types:
- `gaussian_1000.yaml` — Gaussian source distribution; normalizes GT, does not binarize
- `uniform_1000.yaml` — Uniform/binary source distribution; binarizes GT at threshold 0.05

Config `data.data_root` points to FMT-SimGen output directory. The config's `n_nodes` and `n_surface` are overridden by actual mesh dimensions at runtime (with a warning if they differ).

## Directory Structure

```
du2vox/                  # Python package
├── models/stage1/       # GCAIN_full, BasicBlock, GCNBlock, MultiScaleKNNGraphAttention
├── models/stage2/       # Residual INR (WIP)
├── models/baselines/    # Comparison methods (WIP)
├── data/                # FMTSimGenDataset, sparse matrix loaders
├── losses/              # TverskyLoss, weighted_mse_loss, dice/location_error metrics
├── evaluation/           # metrics.py, per_foci.py (grouped eval)
└── visualization/       # 3D Tecplot-style rendering
scripts/
├── train_stage1.py       # Single-model training
├── eval_stage1.py        # Single-model evaluation
├── evaluate_stage1.py    # Comprehensive eval: tables + figures for both sources
├── render_stage1_figures.py  # Standalone figure rendering
├── export_nifti.py       # Export predictions to NIfTI format
└── visualize.py          # GT vs prediction 3D visualization
configs/stage1/
├── gaussian_1000.yaml    # Gaussian source: 800 train / 200 val
└── uniform_1000.yaml     # Uniform/binary source: 800 train / 200 val
runs/                     # Experiment outputs (checkpoints, logs)
results/stage1/           # Evaluation outputs (tables, figures, NIfTIs)
checkpoints/              # Legacy checkpoint storage (deprecated)
```

## Dependencies

Uses `.venv` (Python 3.12). Key packages: PyTorch, NumPy, SciPy, PyYAML, matplotlib.
