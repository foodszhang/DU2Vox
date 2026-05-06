# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DU2Vox is a two-stage framework for Fluorescence Molecular Tomography (FMT) reconstruction:

- **Stage 1** (active): GCAIN — 6-stage unrolled network with multi-scale graph attention, producing a coarse distribution `coarse_d` over FEM mesh nodes
- **Stage 2** (active, WIP): Residual INR — FEM-consistent prior-guided voxel distribution generation, powered by the Bridge

The **Bridge** (`scripts/bridge_stage1_to_stage2.py`) is the Stage 1→2 connector. **Stage 2** (`scripts/train_stage2.py`) trains a Residual INR that refines FEM-interpolated predictions using learned residual correction.

## Development Commands

```bash
# Linting
uv run ruff check .
uv run ruff check --fix .

# Formatting
uv run black .

# Run tests
uv run pytest

# Single test
uv run pytest tests/test_some_feature.py -v
```

## Common Commands

All commands use `uv run` from the project root.

```bash
# ── Stage 1 ──────────────────────────────────────────────────────────────
uv run python scripts/train_stage1.py --config configs/stage1/gaussian_1000.yaml
uv run python scripts/train_stage1.py --config configs/stage1/uniform_1000.yaml
uv run python scripts/train_stage1.py --config configs/stage1/uniform_1000_v2.yaml --resume runs/gcain_uniform_v2/checkpoints/latest.pth

uv run python scripts/eval_stage1.py --config configs/stage1/gaussian_1000.yaml --checkpoint runs/gcain_gaussian_1000/checkpoints/best.pth
uv run python scripts/eval_stage1.py --config configs/stage1/uniform_1000.yaml --checkpoint runs/gcain_uniform_1000/checkpoints/best.pth

# ── Bridge (Stage 1→2) ─────────────────────────────────────────────────
# Run frozen Stage 1 inference + ROI derivation + FEM prior extraction
uv run python scripts/bridge_stage1_to_stage2.py \
    --config configs/stage1/uniform_1000_v2.yaml \
    --checkpoint runs/gcain_uniform_v2/checkpoints/best.pth \
    --split_file /home/foods/pro/FMT-SimGen/data/uniform_1000_v2/splits/val.txt \
    --output_dir output/bridge_val/

# ── Precomputation (Stage 2, offline) ─────────────────────────────────
# Generate precomputed/*.npz grid files for fast training (~3h total for 1000 samples)
uv run python scripts/precompute_stage2_data.py \
    --config configs/stage2/uniform_1000_v2.yaml \
    --split train --grid_spacing 0.2 \
    --output_dir precomputed/train/

uv run python scripts/precompute_stage2_data.py \
    --config configs/stage2/uniform_1000_v2.yaml \
    --split val --grid_spacing 0.2 \
    --output_dir precomputed/val/

# ── Stage 2 Training ──────────────────────────────────────────────────
# Requires precomputed/ directory to exist (num_workers>0, fork-safe)
uv run python scripts/train_stage2.py \
    --config configs/stage2/uniform_1000_v2.yaml \
    --experiment_name baseline_de_only

# Quick smoke test
uv run python scripts/train_stage2.py \
    --config configs/stage2/uniform_1000_v2.yaml \
    --experiment_name baseline_de_only --max_epochs 5 --max_samples 50

# ── Stage 2 Evaluation ────────────────────────────────────────────────
uv run python scripts/eval_stage2.py \
    --config configs/stage2/uniform_1000_v2.yaml \
    --checkpoint checkpoints/stage2/baseline_de_only/best.pth \
    --output results/stage2_eval.json
```

**Bash execution rules (mandatory):**
- Never write inline `python -c "..."` for multi-line code — write to a `.py` temp file first, then run with `uv run python <file>.py`, and delete after
- Single-line `python -c "..."` only for trivial one-liners without newlines

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
- `roi_tet_indices`, `coarse_d.npy`, `roi_info.json` saved per sample under `output/bridge_{train,val}/`

### Stage 2 Residual INR

**Architecture:**
```
Input:   PE(q_norm) [pe_dim] + prior_8d [8] = in_dim
         q_norm = 2 * (q - bbox_min) / (bbox_max - bbox_min) - 1  → [-1, 1]
proj:    Linear(in_dim → hidden)
h[0..n-1]:  n_hidden_layers × Linear(hidden → hidden) + ReLU
   middle layer (i = n_hidden_layers // 2): concat([h_out, proj(x_in)]) → hidden (2×hidden_dim wide)
out:     Linear(hidden → 1), **zero-init** → step-0 residual = 0 (identity, d_hat = FEM interp)
```

**Key design decisions:**
- Zero-init output layer: at initialization, residual=0, so `d_hat = fem_interp = Σλi·d_vi` (pure FEM interpolation). Training learns to add residual corrections.
- Skip connection at middle layer: `proj(x_in)` projected through separate Linear then ReLU, concatenated with hidden state → wider layer (2×hidden_dim)
- PE normalization: coords pre-normalized to [-1,1] at precomputation time (stored as `grid_coords_norm` in .npz). This prevents aliasing at high PE frequencies (2^9 × 40mm coords would cause sin(20480) without normalization).

**Two dataset modes:**

| Mode | Class | Speed | `num_workers` | When to use |
|------|-------|-------|---------------|-------------|
| Precomputed | `Stage2DatasetPrecomputed` | Fast (~4ms/__getitem__) | >0 (fork-safe) | Training (requires precomputed/ .npz files) |
| On-demand | `Stage2Dataset` | Slow (~100ms/__getitem__) | 0 (FEMBridge not fork-safe) | Debugging, legacy |

**Precomputed .npz format** (`precomputed/{split}/sample_XXXX.npz`):
| Field | Shape | Description |
|-------|-------|-------------|
| `grid_coords` | `[G, 3]` float32 | Raw coords in mm (for FEM eval) |
| `grid_coords_norm` | `[G, 3]` float32 | Normalized to [-1,1] (for training) |
| `prior_8d` | `[G, 8]` float32 | 8D FEM prior features |
| `gt_values` | `[G]` float32 | FEM-interpolated GT values |
| `valid_mask` | `[G]` bool | True if point inside ROI tet |
| `grid_shape` | `[3]` int | `(nx, ny, nz)` grid dimensions |
| `bbox_min/max` | `[3]` float32 | ROI bbox bounds (padded) |

### Stage 2 View Encoder (MCX multiview)

`du2vox/models/stage2/view_encoder.py` provides `ViewEncoderModule` for encoding MCX multi-view projection images (used in `full_multiview` ablation):

- **`ViewEncoder`**: 2D U-Net encoder — input `[B, 1, 256, 256]` MCX fluence projection → output `[B, 32, 64, 64]` feature map. 4× downsampled to capture global context.
- **`ProjectAndSample`**: Projects 3D query points onto 7 view feature maps via orthographic projection. Two modes:
  - `voxel_space=True`: centers by physical volume center `(0, 20, 0)`mm, normalizes by half-extents `(19.0, 20.0, 10.4)` — preserves aspect ratio (Phase 3+)
  - `voxel_space=False`: FOV-based normalization (`±40mm` → `±1`)
  - Views at `ANGLES = [-90, -60, -30, 0, 30, 60, 90]` degrees around Y axis
- **`MultiViewFusion`**: fuses per-view features via `"mean"` or learned `"attn"` (single attention head per query)

`MCX_PHYSICAL_CENTER = (0, 20, 0)`mm and `MCX_HALF_EXTENTS = (19.0, 20.0, 10.4)`mm define voxel-space normalization in `project_3d_to_2d`.

### Ablation Naming

| Experiment | Stage 2 Input | Purpose |
|---|---|---|
| `fem_interp_only` | FEM barycentric interpolation (no MLP) | Lower bound baseline |
| `baseline_de_only` | PE(q_norm) + 8D prior (DE channel coarse_d) | Current: DE-only ablation |
| `full_multiview` | PE(q_norm) + 8D prior + ViewEncoderModule (7-view MCX) | DE+MCX fusion |

## Loss Functions

**Gaussian** (`loss.type: "gaussian"`):
`0.5 * Tversky(α=0.1, β=0.9) + 0.2 * weighted_MSE + 0.3 * core_MSE`
- Tversky heavily penalizes FN (β=0.9) → aggressive ROI expansion
- weighted_MSE: 70% weight on foreground (gt > 0.05)
- core_MSE: 2× extra penalty on core region (gt > 0.6) to push peak values higher

**Uniform** (`loss.type: "uniform"`, default):
`0.7 * Tversky(α=0.1, β=0.9) + 0.3 * weighted_MSE`

**Stage 2** (`loss.type: "mse"`):
`MSE(d_hat[valid], gt[valid])` — MSE on valid ROI points only.

## Evaluation Metrics

**Stage 1** (`du2vox/evaluation/metrics.py`): Dice (soft), Dice_bin@0.1/0.3/0.5/0.6, Recall/Precision @0.1/0.3/0.6, Location Error (mm) — centroid Euclidean distance, MSE. When `dataset_manifest.json` is present, results are grouped by foci count (1/2/3) and depth tier (shallow/medium/deep).

**Stage 2** (per-sample averaged, reported in `train_log.json`):
| Metric | Description |
|--------|-------------|
| `stage2_dice_05` | Dice@0.5 of full Stage2 output vs GT |
| `fem_dice_05` | Dice@0.5 of pure FEM interpolation vs GT |
| `delta_dice_05` | `stage2_dice_05 - fem_dice_05` — **core metric**: MLP residual contribution |
| `stage2_mse` | MSE of Stage2 output vs GT |
| `fem_mse` | MSE of pure FEM vs GT |
| `residual_norm` | Mean absolute residual — monitors MLP correction magnitude |

## Data Dependencies

Training data comes from [FMT-SimGen](https://github.com/foodszhang/FMT-SimGen). Two datasets are used:

| Dataset | Source type | GT normalization | Expected use |
|---|---|---|---|
| `gaussian_1000` | Gaussian | `normalize_gt=true`, `global` 99th-pct | Training + evaluation |
| `uniform_1000` / `uniform_1000_v2` | Uniform/binary | `binarize_gt=true` at 0.05 | ROI detection experiments |

Shared FEM assets (from `FMT-SimGen/output/shared/`):
- `mesh.npz` — node coords `[N, 3]`, surface indices, elements `[N_tets, 4]`
- `system_matrix.A.npz` — forward matrix `[S, N]`
- `graph_laplacian_full.*.npz` — L, L0, L1, L2, L3 `[N, N]`
- `knn_idx_full.npy` — kNN indices `[N, 32]`
- `visible_mask.npy` — (optional) if present, `A` and `b` are cropped to visible-only subset `[V=6226, N]` / `[V, 1]`

Per-sample (from `FMT-SimGen/data/{experiment}/samples/{id}/`):
- `measurement_b.npy` — surface measurement `[S, 1]` or `[V, 1]`
- `gt_nodes.npy` — ground truth at FEM nodes `[N, 1]` — values in [0, 1] for both Gaussian and Uniform
- `tumor_params.json` — tumor parameters (center, radius, depth, num_foci)

Bridge output (from `output/bridge_{train,val}/{id}/`):
- `coarse_d.npy` — Stage 1 output `[N]`
- `roi_tet_indices.npy` — ROI tet indices
- `roi_info.json` — ROI metadata including `roi_bbox_mm`

## Coordinate System

DU2Vox uses the **FMT-SimGen physical coordinate system** (mm):
- X: left(−19mm) → right(+19mm)
- Y: anterior/head → posterior/tail (within cropped trunk ROI)
- Z: ventral/belly(−Z) → dorsal/back(+Z)

Mesh node coordinates and all query points are in mm.

## Key Implementation Notes

- `AdaptiveThreshold` uses `sigmoid(k * (|u| − λ))` (k=10) instead of hard sign/softplus — gradients always flow, never zeroed
- `_barycentric_batch` (numba njit, NOT parallel=True): LinAlgError on degenerate tet → `inside=False`. **Never use `parallel=True`** in numba 0.65 — causes object-mode fallback and wrong results; use `numba.prange` (works in non-parallel njit)
- `numba.boolean` not valid in numba 0.65 — use `np.bool_` (in `bridge/fem_bridging.py` `_barycentric_batch` return type)
- `FMTSimGenDataset` handles backward compatibility: if `visible_mask.npy` is absent, uses full `[S=7465]` surface nodes; if present, crops to `[V=6226]`
- `Stage1Inference._load_shared_assets`: must apply `visible_mask` cropping to `A` matrix when `visible_mask.npy` exists — checkpoint and inference A shape must match; the Bridge config (`--split_file`) must use the same dataset split that was used during Stage 1 training to ensure `visible_mask` presence/absence is consistent
- Checkpoints: `runs/{exp}/checkpoints/{best,latest,epoch_NNN}.pth` (Stage 1); `checkpoints/stage2/{exp}/best.pth` (Stage 2); config is copied to `runs/{exp}/config.yaml` at training start
- Stage 2 val uses `stage2_dice_05` for best model selection and early stopping (not `val_loss`)
- `FEMBridge._roi_set` membership check is dead code — KDTree is already built on active tets only, the check always passes

## Directory Structure

```
du2vox/
├── models/stage1/
│   ├── gcain.py         # GCAIN_full, BasicBlock, GCNMultiScal, GradientProjection
│   ├── blocks.py         # GCNBlock, InputBlock, AdaptiveThreshold, UpdateBlock
│   ├── msgc.py           # MultiScaleKNNGraphAttention, KNNGraphCrossAttention,
│   │                     # ScaleGate, SensitivityWeighting
│   └── reference/        # Reference MS-GDUN implementations
├── models/stage2/
│   ├── residual_inr.py  # ResidualINR + PositionalEncoding (Stage 2)
│   ├── stage2_dataset.py # Stage2Dataset (on-demand) + Stage2DatasetPrecomputed
│   └── view_encoder.py   # ViewEncoderModule: 2D U-Net + ProjectAndSample + MultiViewFusion
├── bridge/
│   ├── stage1_inference.py  # Frozen Stage 1 inference
│   ├── roi_derivation.py     # derive_roi: tau threshold + tet dilation
│   └── fem_bridging.py       # FEMBridge: KDTree + barycentric coords + 8D prior
├── data/dataset.py          # FMTSimGenDataset
├── losses/tversky.py        # TverskyLoss, weighted_mse_loss, core_weighted_loss
├── evaluation/
│   ├── metrics.py           # evaluate_batch, summarize_metrics
│   └── per_foci.py          # Grouped evaluation by foci count and depth
└── visualization/           # 3D rendering scripts

scripts/
├── train_stage1.py
├── eval_stage1.py             # Single-model eval
├── evaluate_stage1.py       # Comprehensive eval: tables + figures for both sources
├── render_stage1_figures.py  # Render Stage 1 evaluation figures
├── export_nifti.py           # NIfTI export of reconstruction results
├── train_stage2.py           # Stage 2 training (precomputed or on-demand)
├── eval_stage2.py           # Stage 2 full-grid evaluation
├── bridge_stage1_to_stage2.py   # Stage 1→2 bridge pipeline
├── precompute_stage2_data.py    # Offline precomputation for Stage 2 training
└── diagnose_projection_alignment.py  # MCX projection geometry debugging

configs/
├── stage1/
│   ├── gaussian_1000.yaml
│   ├── uniform_1000.yaml
│   └── uniform_1000_v2.yaml   # Current active config
└── stage2/
    ├── uniform_1000_v2.yaml    # DE-only baseline (baseline_de_only)
    ├── full_multiview_v6.yaml  # MCX 7-view fusion (full_multiview)
    └── baseline_de_only_v6.yaml
```
