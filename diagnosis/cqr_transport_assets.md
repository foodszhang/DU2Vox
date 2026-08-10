# CQR Transport Asset Audit

## Git and baseline

- Branch: `feat/cqr-transport-observability`
- HEAD: `673a899fd5579189c8bd50ef6e6443a892600fff`
- Dirty files: 25
- Resolved Stage 1 checkpoint: `/home/foods/pro/DU2Vox/runs/stage1_fmt_simgen_v2_3k_20k_balanced_v2_eval/checkpoints/best.pth`
- Resolved Stage 2 baseline: `/home/foods/pro/DU2Vox/checkpoints/stage2/cqr_v2_3k_rgl_main_sparse/best.pth`

## Dataset

- train: 2400
- val: 300
- test: 300
- Total: 3000
- Dataset root contract valid: True

## Shared physics

- M: shape=[19990, 19990], nnz=273208, dtype=float64
- F: shape=[19990, 19990], nnz=273208, dtype=float64
- A: shape=[7413, 19990], nnz=148185870, dtype=float32
- Surface nodes: 7413
- Visible nodes: 7413
- Full-surface convention confirmed: **True**

## Existing CQR candidate pools

- train: dir_exists=True, files=100, audited=5, blocked=False
- val: dir_exists=True, files=25, audited=5, blocked=False
- test: dir_exists=True, files=5, audited=5, blocked=False

## Blockers

- None

The JSON companion contains per-array shapes, dtypes, ranges, file metadata, role/band/source checks, and checkpoint keys.
