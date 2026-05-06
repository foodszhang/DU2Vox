# AGENTS.md

This file gives coding agents the project-specific context needed to work safely in
DU2Vox. It is derived from `CLAUDE.md`, the current code, and the coordinate-system
documentation.

## Project Snapshot

DU2Vox is a two-stage Fluorescence Molecular Tomography (FMT) reconstruction
project.

- Stage 1 is active: `GCAIN_full`, a 6-stage unrolled graph network that predicts a
  coarse FEM-node distribution `coarse_d`.
- The Bridge is active: frozen Stage 1 inference plus ROI derivation and FEM
  barycentric prior extraction.
- Stage 2 is active and experimental: a Residual INR that refines FEM-interpolated
  priors on voxel/query grids. It also has an optional MCX multiview projection
  encoder.

The repository depends on FMT-SimGen data and shared FEM/MCX assets, normally under
`/home/foods/pro/FMT-SimGen/output/shared`.

## Commands

Run commands from the repository root. Prefer `uv run`.

```bash
uv run ruff check .
uv run ruff check --fix .
uv run black .
uv run pytest
```

Stage 1:

```bash
uv run python scripts/train_stage1.py --config configs/stage1/uniform_1000_v2.yaml
uv run python scripts/eval_stage1.py --config configs/stage1/uniform_1000_v2.yaml --checkpoint runs/gcain_uniform_v2/checkpoints/best.pth
```

Bridge:

```bash
uv run python scripts/bridge_stage1_to_stage2.py \
  --config configs/stage1/uniform_1000_v2.yaml \
  --checkpoint runs/gcain_uniform_v2/checkpoints/best.pth \
  --split_file /home/foods/pro/FMT-SimGen/data/uniform_1000_v2/splits/val.txt \
  --output_dir output/bridge_val/
```

Stage 2 precompute and training:

```bash
uv run python scripts/precompute_stage2_data.py \
  --config configs/stage2/uniform_1000_v2.yaml \
  --split train --grid_spacing 0.2 \
  --output_dir precomputed/train/

uv run python scripts/train_stage2.py \
  --config configs/stage2/uniform_1000_v2.yaml \
  --experiment_name baseline_de_only
```

For quick validation, use small limits when supported:

```bash
uv run python scripts/train_stage2.py \
  --config configs/stage2/uniform_1000_v2.yaml \
  --experiment_name smoke --max_epochs 5 --max_samples 50
```

Do not write multi-line `python -c` commands. Put temporary multi-line code in a
small `.py` file, run it with `uv run python`, then remove it.

## Architecture Map

Important package paths:

- `du2vox/data/dataset.py`: `FMTSimGenDataset` for Stage 1 data loading.
- `du2vox/models/stage1/`: GCAIN blocks, multi-scale graph attention, and reference
  implementations.
- `du2vox/bridge/`: Stage 1 inference, ROI derivation, and FEM point location.
- `du2vox/models/stage2/residual_inr.py`: Residual INR and positional encoding.
- `du2vox/models/stage2/stage2_dataset.py`: on-demand and precomputed Stage 2
  datasets.
- `du2vox/models/stage2/view_encoder.py`: optional MCX 7-view encoder and 3D-to-2D
  projection.
- `du2vox/utils/frame.py`: frame manifest loading and world/voxel transforms.
- `du2vox/evaluation/`: metrics and grouped evaluation.
- `scripts/`: training, evaluation, bridge, precompute, export, visualization, and
  diagnosis entry points.

## Data Contracts

FMT-SimGen shared assets provide:

- `mesh.npz`: FEM `nodes`, `surface_node_indices`, `elements`.
- `system_matrix.A.npz`: forward matrix.
- `graph_laplacian_full.*.npz`: graph Laplacians.
- `knn_idx_full.npy`: kNN graph indices.
- Optional `visible_mask.npy`: when used, the measurement vector and A matrix must
  follow the same visible-surface convention as the checkpoint/config.
- `frame_manifest.json`: authoritative coordinate and voxel-grid metadata.

Per-sample assets usually include:

- `measurement_b.npy`
- `gt_nodes.npy`
- `gt_voxels.npy`
- `tumor_params.json`
- optional MCX projection files for multiview Stage 2.

Bridge output contains one directory per sample with:

- `coarse_d.npy`
- `roi_tet_indices.npy`
- `roi_info.json`

Precomputed Stage 2 `.npz` files contain:

- `grid_coords`: raw world coordinates in mm.
- `grid_coords_norm`: coordinates normalized to `[-1, 1]` for the INR.
- `prior_8d`: `[d_v0, d_v1, d_v2, d_v3, lambda0, lambda1, lambda2, lambda3]`.
- `gt_values`: GT lookup from `gt_voxels.npy`.
- `valid_mask`, `grid_shape`, `bbox_min`, `bbox_max`.

## Coordinate-System Rules

DU2Vox uses FMT-SimGen's `mcx_trunk_local_mm` world frame. Coordinates are in mm.
Read `docs/COORDINATE_SYSTEM.md` before changing geometry, projection, precompute,
or visualization code.

The authoritative source is `frame_manifest.json`; avoid hardcoded volume bounds,
offsets, centers, or voxel-grid metadata. Use `du2vox.utils.frame.FrameManifest` or
`get_frame_constants()` instead.

Current intended frame:

- `world_frame`: `mcx_trunk_local_mm`
- MCX volume bbox: approximately `[0, 0, 0]` to `[38.0, 40.0, 20.8]`
- voxel size: `0.2` mm
- Stage 2 coordinate normalization: `2 * (world - bbox_min) / (bbox_max - bbox_min) - 1`

Be careful with `du2vox/models/stage2/view_encoder.py`: it exposes module-level
constants such as `MCX_VOLUME_CENTER_WORLD`, but they are derived from
`frame_manifest.json` via `get_frame_constants()`. Do not replace these with
literal constants.

## Stage 1 Notes

`GCAIN_full` is a 6-block unrolled network. Each block builds input features from
`[x, L^T L x, A^T A x - A^T b]`, runs GCN/multi-scale graph attention, predicts a
gradient, and applies adaptive thresholding.

Important implementation details:

- `sens_w = ||A[:, i]||_2` is used by attention to encode node sensitivity.
- `AdaptiveThreshold` uses a sigmoid gate to preserve gradients.
- Stage 1 checkpoints are usually under `runs/{experiment}/checkpoints/`.
- Configs are copied into `runs/{experiment}/config.yaml` at training start.
- Keep the `visible_mask` convention consistent across data loading, checkpoints,
  Stage 1 inference, and bridge generation.

## Bridge Notes

The bridge pipeline is:

`measurement_b -> Stage1Inference -> coarse_d -> derive_roi -> FEMBridge -> prior_8d`

`derive_roi` thresholds active nodes, collects tets containing those nodes, then can
dilate ROI tets by vertex adjacency.

`FEMBridge` uses a KDTree on active tet centroids, then validates candidate tets with
exact barycentric coordinates.

Numba constraints in `du2vox/bridge/fem_bridging.py` matter:

- Do not change `_barycentric_batch` to `parallel=True`.
- Keep boolean arrays as `np.bool_`, not `numba.boolean`.
- Degenerate tets should return `inside=False`.

## Stage 2 Notes

Use `Stage2DatasetPrecomputed` for training when precomputed `.npz` files exist. It
is fast and fork-safe, so `num_workers > 0` is acceptable.

Use `Stage2Dataset` only for debugging or legacy paths. It constructs `FEMBridge`
objects on demand and should use `num_workers=0`.

`ResidualINR` input is `PE(q_norm) + prior_8d`, optionally plus multiview features.
Its output layer is zero-initialized, so initial predictions equal pure FEM
interpolation:

`d_hat = sum(lambda_i * d_vi) + residual`

Do not remove the zero initialization unless intentionally changing the Stage 2
training contract.

Stage 2 validation tracks the residual model against the FEM-only baseline. Important
metrics include:

- `stage2_dice_05`
- `fem_dice_05`
- `delta_dice_05`
- `stage2_mse`
- `fem_mse`
- `residual_norm`

## Editing Guidance

- Prefer existing patterns over new abstractions.
- Keep edits scoped to the requested behavior.
- Use structured parsers/APIs for YAML, JSON, NumPy, and sparse matrices.
- Do not regenerate large data, checkpoints, precomputed grids, or figures unless the
  task explicitly requires it.
- Do not alter files under `runs/`, `logs/`, `output/`, `precomputed/`, `results/`,
  or `checkpoints/` unless the task is about generated artifacts.
- Use ASCII for new text unless the surrounding file already uses non-ASCII.
- If changing training behavior, update the relevant config, script, and this file if
  the workflow changes.
- If changing coordinate logic, update or cross-check `docs/COORDINATE_SYSTEM.md`.

## Verification

Choose verification proportional to the change:

- Formatting/lint-only changes: `uv run ruff check .`
- Core Python changes: targeted script or `uv run pytest` if tests exist.
- Stage 1 data/model changes: a small Stage 1 train/eval smoke run.
- Bridge changes: run bridge on a small split or one sample and inspect saved
  `coarse_d.npy`, `roi_tet_indices.npy`, and `roi_info.json`.
- Stage 2 dataset/precompute changes: run `precompute_stage2_data.py` with
  `--max_samples 1` and inspect the generated `.npz` fields.
- Stage 2 training changes: run a short smoke training with `--max_epochs` and
  `--max_samples`.
- Projection/multiview changes: run or update `scripts/diagnose_projection_alignment.py`.

If a command depends on external FMT-SimGen data that is missing locally, report that
clearly instead of fabricating results.
