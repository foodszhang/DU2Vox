# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DU2Vox is a two-stage framework for Fluorescence Molecular Tomography (FMT) reconstruction:

- **Stage 1** (active): GCAIN — 6-stage unrolled network with multi-scale graph attention, producing a coarse distribution `coarse_d` over FEM mesh nodes
- **Stage 2** (WIP): Residual INR — FEM-consistent prior-guided voxel distribution generation, powered by the Bridge

The **Bridge** (`scripts/bridge_stage1_to_stage2.py`) is the Stage 1→2 connector: it runs frozen Stage 1 inference, derives per-sample ROIs via tetrahedral dilation, and extracts 8D FEM prior features (`[d_v0, d_v1, d_v2, d_v3, λ0, λ1, λ2, λ3]`) at query points.

## Common Commands

All commands use `uv run` from the project root.

```bash
# Stage 1 training
uv run python scripts/train_stage1.py --config configs/stage1/gaussian_1000.yaml
uv run python scripts/train_stage1.py --config configs/stage1/gaussian_1000.yaml --resume runs/gcain_gaussian_1000/checkpoints/latest.pth
uv run python scripts/train_stage1.py --config configs/stage1/uniform_1000.yaml

# Stage 1 evaluation
uv run python scripts/eval_stage1.py --config configs/stage1/gaussian_1000.yaml --checkpoint runs/gcain_gaussian_1000/checkpoints/best.pth
uv run python scripts/eval_stage1.py --config configs/stage1/uniform_1000.yaml --checkpoint runs/gcain_uniform_1000/checkpoints/best.pth

# Comprehensive eval: tables + figures for both source types
uv run python scripts/evaluate_stage1.py \
    --checkpoint_g runs/gcain_gaussian_1000/checkpoints/best.pth \
    --checkpoint_u runs/gcain_uniform_1000/checkpoints/best.pth \
    --config_g configs/stage1/gaussian_1000.yaml \
    --config_u configs/stage1/uniform_1000.yaml \
    --shared_dir /home/foods/pro/FMT-SimGen/output/shared \
    --output_dir results/stage1/

# Visualize GT vs prediction
uv run python scripts/visualize.py --config configs/stage1/gaussian_1000.yaml --checkpoint runs/gcain_gaussian_1000/checkpoints/best.pth --n_samples 4

# Stage 1→2 Bridge pipeline
uv run python scripts/bridge_stage1_to_stage2.py \
    --config configs/stage1/gaussian_1000.yaml \
    --checkpoint runs/gcain_gaussian_1000/checkpoints/best.pth \
    --split_file /home/foods/pro/FMT-SimGen/data/gaussian_1000/splits/val.txt \
    --output_dir bridge_output/ \
    --tau 0.5 --dilate_layers 1

# Bridge: also compute 8D prior features (slow, for Stage 2 training)
uv run python scripts/bridge_stage1_to_stage2.py \
    --config configs/stage1/gaussian_1000.yaml \
    --checkpoint runs/gcain_gaussian_1000/checkpoints/best.pth \
    --split_file /home/foods/pro/FMT-SimGen/data/gaussian_1000/splits/val.txt \
    --output_dir bridge_output/ --compute_prior_cache
```

**Bash execution rules (mandatory):**
- Never use `python -c "..."` for multi-line code — write to a `.py` file first, then run with `uv run python <file>.py`
- Single-line `python -c "..."` is allowed only for trivial one-liners without newlines

## Data Dependencies

Training data comes from [FMT-SimGen](https://github.com/foodszhang/FMT-SimGen). Two datasets are used:

| Dataset | Source type | GT normalization | Expected use |
|---|---|---|---|
| `gaussian_1000` | Gaussian | `normalize_gt=true`, `global` 99th-pct | Training + evaluation |
| `uniform_1000` | Uniform/binary | `binarize_gt=true` at 0.05 | ROI detection experiments |

Shared FEM assets (from `FMT-SimGen/output/shared/`):
- `mesh.npz` — node coords `[N, 3]`, surface indices, elements `[N_tets, 4]`
- `system_matrix.A.npz` — forward matrix `[S, N]`
- `graph_laplacian_full.*.npz` — L, L0, L1, L2, L3 `[N, N]`
- `knn_idx_full.npy` — kNN indices `[N, 32]`
- `visible_mask.npy` — (optional) if present, `A` and `b` are cropped to visible-only subset `[V, N]` / `[V, 1]`

Per-sample (from `FMT-SimGen/data/{experiment}/samples/{id}/`):
- `measurement_b.npy` — surface measurement `[S, 1]` or `[V, 1]`
- `gt_nodes.npy` — ground truth at FEM nodes `[N, 1]`
- `tumor_params.json` — tumor parameters (center, radius, depth, num_foci)

## Architecture

### Stage 1 Model (GCAIN_full)

6 `BasicBlock` stages run sequentially (cold start X₀=zeros):

```
X₀=zeros → Block₁ → Block₂ → Block₃ → Block₄ → Block₅ → Block₆ → X₆
```

Each `BasicBlock`:
1. **InputBlock**: concat(`[x, LᵀLx, AᵀAx − Aᵀb]`) → `[B, N, 3]`
2. **GCN sequence**: 3→8→16→8→`feat_dim` (spectral GCN: `LeakyReLU(L @ X @ W + b)`)
3. **MultiScale GCN** (`GCNMultiScal`): 4 parallel branches on L0/L1/L2/L3, each with 2 GCN layers — outputs `x_l0, x_l1, x_l2, x_l3` independently
4. **MultiScaleKNNGraphAttention** (`msgc.py`): L0/L1/L2 are queries, L3 is shared key/value — 3 cross-attention heads fused by learned `ScaleGate` (softmax α over 3 scales)
5. **GradientProjection**: `feat_dim → feat_dim//2 → 1` → scalar gradient
6. **UpdateBlock**: `u = x − grad`; `AdaptiveThreshold`: `u * sigmoid(k * (|u| − λ))` where λ is predicted from features (gradient-friendly)

`sens_w = ||A[:, i]||₂` (node sensitivity) weights K and V in attention to account for heterogeneous surface measurement coverage.

### Bridge Pipeline

```
b [S, 1]  →  Stage1Inference  →  coarse_d [N, 1]
                                    ↓
                              derive_roi(coarse_d)  →  roi_tet_indices [M_roi]
                                    ↓
                              FEMBridge.locate_points_batch(query_pts)
                                    ↓
                              get_prior_features  →  prior_8d [M, 8]
                                                         [d_v0,d_v1,d_v2,d_v3,λ0,λ1,λ2,λ3]
```

- `derive_roi`: threshold `coarse_d > τ` → collect all tets containing active nodes → dilate by `dilate_layers` (adds tets sharing vertices with existing ROI tets)
- `FEMBridge`: KDTree on tet centroids (k=16 candidates); validates containment with exact barycentric coordinates via `np.linalg.solve`
- `prior_cache.npz`: contains `prior_8d`, `valid_mask`, `query_node_indices` — for **verification only**, not used by Stage 2 training

### Loss Functions

**Gaussian** (`loss.type: "gaussian"`):
`0.5 * Tversky(α=0.1, β=0.9) + 0.2 * weighted_MSE + 0.3 * core_MSE`
- Tversky heavily penalizes FN (β=0.9) → aggressive ROI expansion
- weighted_MSE: 70% weight on foreground (gt > 0.05)
- core_MSE: 2× extra penalty on core region (gt > 0.6) to push peak values higher

**Uniform** (`loss.type: "uniform"`, default):
`0.7 * Tversky(α=0.1, β=0.9) + 0.3 * weighted_MSE`

### Evaluation Metrics

From `du2vox/evaluation/metrics.py`: Dice (soft), Dice_bin@0.1/0.3/0.5/0.6, Recall/Precision @0.1/0.3/0.6, Location Error (mm) — centroid Euclidean distance, MSE.

When `dataset_manifest.json` is present, `eval_stage1.py` groups results by foci count (1/2/3) and depth tier (shallow/medium/deep).

## Directory Structure

```
du2vox/
├── models/stage1/
│   ├── gcain.py      # GCAIN_full, BasicBlock, GCNMultiScal, GradientProjection
│   ├── blocks.py     # GCNBlock, InputBlock, AdaptiveThreshold, UpdateBlock
│   ├── msgc.py       # MultiScaleKNNGraphAttention, KNNGraphCrossAttention,
│   │                 # ScaleGate, SensitivityWeighting
│   └── reference/    # Reference MS-GDUN implementations for comparison
├── bridge/
│   ├── stage1_inference.py  # Frozen Stage 1 batch inference → coarse_d.npy
│   ├── roi_derivation.py     # derive_roi: tau threshold + tet dilation
│   └── fem_bridging.py       # FEMBridge: KDTree + barycentric coords + 8D prior
├── data/dataset.py   # FMTSimGenDataset (with visible_mask backward compat)
├── losses/tversky.py # TverskyLoss, weighted_mse_loss, core_weighted_loss
├── evaluation/
│   ├── metrics.py    # evaluate_batch, summarize_metrics
│   └── per_foci.py   # Grouped evaluation by foci count and depth
└── visualization/    # 3D rendering scripts
scripts/
├── train_stage1.py
├── eval_stage1.py
├── evaluate_stage1.py   # Comprehensive eval: both source types
├── visualize.py
└── bridge_stage1_to_stage2.py
configs/
├── stage1/
│   ├── gaussian_1000.yaml   # Gaussian: global 99th-pct norm, continuous GT
│   └── uniform_1000.yaml    # Uniform: binarized GT at 0.05
└── bridge/
    └── stage1_to_stage2.yaml
```

## Coordinate System

DU2Vox uses the **FMT-SimGen physical coordinate system** (mm):
- X: left(−19mm) → right(+19mm)
- Y: anterior/head → posterior/tail (within cropped trunk ROI)
- Z: ventral/belly(−Z) → dorsal/back(+Z)

Mesh node coordinates and all query points are in mm. The voxel grid (for Stage 2) uses 0.1mm spacing.

## Key Implementation Notes

- `AdaptiveThreshold` uses `sigmoid(k * (|u| − λ))` (k=10) instead of hard sign/softplus — gradients always flow, never zeroed
- `barycentric_coords_batch` (defined but unused): LinAlgError on degenerate tet → returns `lam123=[-1,-1,-1]` so `inside=False`
- `FMTSimGenDataset` handles backward compatibility: if `visible_mask.npy` is absent, uses full `[S=7465]` surface nodes; if present, crops to `[V=6226]`
- Checkpoints: `runs/{exp}/checkpoints/{best,latest,epoch_NNN}.pth`; config is copied to `runs/{exp}/config.yaml` at training start
