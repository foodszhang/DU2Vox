# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DU2Vox is a two-stage framework for Fluorescence Molecular Tomography (FMT) reconstruction:
- **Stage 1** (active): MS-GDUN / GCAIN — 6-stage unrolled network with multi-scale graph attention for coarse relative distribution initialization
- **Stage 2** (planned): Residual INR — FEM-consistent prior-guided voxel distribution generation

## Commands

All commands use `uv run` from the project root to ensure correct dependencies.

**Bash execution rules (mandatory):**
- Never use `python -c "..."` for multi-line code — write to a `.py` file first, then run with `uv run python <file>.py`
- Single-line `python -c "..."` is allowed only for trivial one-liners without newlines
- Use `uv run python scripts/...` for all scripts

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

# Extract CSV metrics from a training log
grep '^\[CSV\]' runs/gcain_gaussian_1000/train_*.log
```

Checkpoints are saved under `runs/{experiment_name}/checkpoints/`:
- `latest.pth` — updated every epoch (stores epoch, optimizer, scheduler state)
- `best.pth` — best Dice_bin@0.3
- `epoch_{N:03d}.pth` — milestone checkpoints every `logging.milestone_every` epochs

The active config is also copied to `runs/{experiment_name}/config.yaml` for reproducibility.

## Architecture

### Stage 1 Model (GCAIN_full)

6 `BasicBlock` stages run sequentially, each containing:
1. **InputBlock** (`blocks.py`): concatenates `[x, LᵀL x, AᵀA x - Aᵀ b]` → [B, N, 3]
2. **GCN sequence** (`blocks.py:GCNBlock`): 3→8→16→8→feat_dim (default 6); spectral GCN `out = LeakyReLU(L @ X @ W + b)`
3. **MultiScale GCN** (`gcain.py:GCNMultiScal`): 4 parallel two-layer GCN branches, one per Laplacian (L0–L3), producing `x_l0, x_l1, x_l2, x_l3` independently
4. **MultiScaleKNNGraphAttention** (`msgc.py`): 3 cross-attention heads where L0/L1/L2 are **queries** and L3 is the shared **key/value** source; outputs fused via learned scale gate (softmax α over 3 scales)
5. **GradientProjection** (`gcain.py`): linear projection feat_dim → feat_dim//2 → 1
6. **UpdateBlock** (`blocks.py`): computes `u = x - grad`, then applies soft shrinkage `u * sigmoid(k * (|u| - λ(feat)))` where λ is predicted from features (gradient-friendly; not hard sign/softplus)

Cold start: X₀ = zeros. Forward pass: `X₀ → Block₁ → ... → Block₆ → X₆`

### Data Flow

```
b [B, S, 1]  (surface measurements)
     ↓
GCAIN_full (6 stages)
     ↓
pred [B, N, 1]  (voxel predictions at mesh nodes)
```

Shared assets (loaded once from FMT-SimGen output, moved to GPU at training time):
- `mesh.npz`: node coordinates [N, 3], surface indices
- `system_matrix.A.npz`: forward matrix [S, N]
- `graph_laplacian_full.*.npz`: L, L0, L1, L2, L3 [N, N]
- `knn_idx_full.npy`: kNN indices [N, 32]
- `sens_w`: computed as `||A[:, i]||_2` per node, normalized — used in `SensitivityWeighting` to weight K and V in attention

Per-sample (preloaded into CPU memory at dataset init):
- `measurement_b.npy`: surface measurement [S, 1]
- `gt_nodes.npy`: ground truth distribution [N, 1]

### Loss

Two distinct criteria selected by `loss.type` in config:

**Gaussian** (`loss.type: "gaussian"`):
`0.5 * Tversky(α=0.1, β=0.9) + 0.2 * weighted_MSE + 0.3 * core_MSE`
- Tversky: heavily penalizes FN → aggressive ROI expansion
- weighted_MSE: 70% weight on foreground (gt > 0.05)
- core_MSE: extra 2× penalty on core region (gt > `core_threshold`, default 0.6) to push peak predictions higher

**Uniform** (`loss.type: "uniform"`, default):
`0.7 * Tversky(α=0.1, β=0.9) + 0.3 * weighted_MSE`

### Evaluation Metrics

Metrics computed per-batch in `du2vox/evaluation/metrics.py`:
- Dice (soft), Dice_bin@0.1/0.3/0.5/0.6
- Recall/Precision @0.1, @0.3, @0.6 (gt binarized at 0.05)
- Location Error (mm): Euclidean distance between predicted/GT centroids
- MSE, pred stats (max, mean, std, fraction above thresholds)

When `dataset_manifest.json` is present, `eval_stage1.py` also groups results by:
- **Foci count**: 1-foci, 2-foci, 3-foci
- **Depth tier**: shallow, medium, deep
- **Cross**: foci × depth

### Key Config Notes

Two experiment configs for different source types:
- `gaussian_1000.yaml` — Gaussian source; `normalize_gt: true`, `normalize_gt_mode: global` (99th-percentile normalization across train set), `binarize_gt: false`
- `uniform_1000.yaml` — Uniform/binary source; `binarize_gt: true` at threshold 0.05

Splits are plain text files `splits/train.txt` and `splits/val.txt` listing sample IDs, one per line. The dataset preloads all samples into CPU memory at init.

## Directory Structure

```
du2vox/
├── models/stage1/
│   ├── gcain.py         # GCAIN_full, BasicBlock, GCNMultiScal, GradientProjection
│   ├── blocks.py        # GCNBlock, InputBlock, AdaptiveThreshold, UpdateBlock, SparseUpdate
│   ├── msgc.py          # MultiScaleKNNGraphAttention, KNNGraphCrossAttention, ScaleGate, SensitivityWeighting
│   ├── gcb.py           # (auxiliary GCN building blocks)
│   └── reference/       # Original MS-GDUN reference implementations (MSGDUN.py, variants) for comparison
├── models/stage2/       # Residual INR (WIP)
├── models/baselines/    # Comparison methods (WIP)
├── data/dataset.py      # FMTSimGenDataset, load_npz_as_torch_sparse
├── losses/tversky.py    # TverskyLoss, weighted_mse_loss, core_weighted_loss, criterion, criterion_gaussian
├── evaluation/
│   ├── metrics.py       # evaluate_batch, summarize_metrics
│   └── per_foci.py      # Grouped evaluation by foci count and depth
└── visualization/       # 3D Tecplot-style rendering
scripts/
├── train_stage1.py      # Training; outputs [CSV]-prefixed lines for metric extraction
├── eval_stage1.py       # Single-model evaluation with optional grouped breakdown
├── evaluate_stage1.py   # Comprehensive eval: tables + figures for both source types
├── render_stage1_figures.py
├── export_nifti.py
└── visualize.py
configs/stage1/
├── gaussian_1000.yaml   # Gaussian source: 800 train / 200 val
└── uniform_1000.yaml    # Uniform/binary source: 800 train / 200 val
runs/                    # Experiment outputs (checkpoints, logs, config copy)
results/stage1/          # Evaluation outputs (tables, figures, NIfTIs)
```

## Dependencies

Uses `.venv` (Python 3.12). Key packages: PyTorch, NumPy, SciPy, PyYAML, matplotlib.
