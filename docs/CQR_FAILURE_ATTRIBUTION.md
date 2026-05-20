# CQR Failure Attribution Notes

This document summarizes the CQR failure-attribution work on branch `feat/cqr-v1`.

## Goal

The goal was to separate four possible causes of the CQR gap against ROI-only Stage 2:

- `residual_scale=0.1` limiting CQR residual capacity.
- `prior_ext` / `CQRResidualINR` adding or removing useful conditioning.
- CQR query-domain training differing from common-domain evaluation.
- CQR role definitions merging weak ROI support into core.

## Code Changes

- `scripts/train_stage2.py` now supports `model.residual_scale` for `CQRResidualINR` and `model.prior_source` with `prior_ext | prior_8d`.
- `scripts/train_stage2.py` also supports `--resume_checkpoint` for continuing from saved model weights after an interrupted run.
- `scripts/eval_stage2_unified.py` and `scripts/eval_stage2_common_domain.py` use the same model construction and prior-source selection rules as training.
- `du2vox/bridge/coverage_field.py` supports `core_from_roi_weak`; disabling it keeps high-confidence core separate from weak-support halo.
- `scripts/diagnose_cqr_npz.py` can write aggregate role summaries with `--out_csv`.

## Key Configs

- `configs/stage2/cqr_balanced_v2_rolesplit_priorext_scale1_multiview_mse.yaml`: main CQR setting, using rolesplit, `prior_ext`, and `residual_scale=1.0`.
- `configs/stage2/cqr_balanced_v2_rolesplit_prior8_scale1_multiview_mse.yaml`: prior ablation, same rolesplit and scale, but using `prior_8d`.
- `configs/stage2/cqr_balanced_v2_multiview_mse_scale01.yaml`: conservative residual-scale ablation.
- `configs/stage2/cqr_balanced_v2_query_resinr_prior8_mv_mse.yaml`: CQR query domain with original `ResidualINR` and `prior_8d`.

## Evaluation Protocol

The main numbers use common-domain evaluation:

- Common grid: `output/bridge_20k_val`
- Evaluated bridge/prior: `output/bridge_20k_val_balanced_v2`
- Evaluator: `scripts/eval_stage2_common_domain.py`
- Samples: 200 validation samples
- Grid spacing: `1.0` mm

Training-time validation is own-query-domain validation over the precomputed CQR query cloud. Its FEM value is not directly comparable to common-domain FEM. For example, rolesplit CQR validation reports FEM around `0.5460`, while common-domain balanced_v2 FEM is `0.6051`.

## Main Results

Common-domain results:

| Method | S2 Dice | FEM Dice | Delta |
| --- | ---: | ---: | ---: |
| ROI high-recall MV | 0.6566 | 0.5977 | +0.0589 |
| CQR high-recall MV | 0.6335 | 0.5977 | +0.0359 |
| ROI balanced_v2 MV | 0.6516 | 0.6051 | +0.0465 |
| CQR balanced_v2 old MV | 0.6218 | 0.6051 | +0.0166 |
| CQR rolesplit + prior_ext + scale1 MV | 0.6429 | 0.6051 | +0.0378 |
| CQR rolesplit + prior8 + scale1 MV | 0.6404 | 0.6051 | +0.0352 |

Result files:

- `results/common_domain_s1_op_cqr_rolesplit_prior_ablation.csv`
- `results/common_eval_cqr_balanced_v2_rolesplit_priorext_scale1_mv_mse.json`
- `results/common_eval_cqr_balanced_v2_rolesplit_prior8_scale1_mv_mse.json`

## Prior Ablation

- `rolesplit + prior_ext + scale1`: S2 `0.6429`, FEM `0.6051`, Delta `+0.0378`.
- `rolesplit + prior8 + scale1`: S2 `0.6404`, FEM `0.6051`, Delta `+0.0352`.
- `prior_ext` contribution: `+0.0025` Dice over `prior_8d`, positive but below the `0.005` threshold for a strong independent contribution.

Interpretation: most of the recovery comes from rolesplit plus removing the residual-scale bottleneck. `prior_ext` is helpful but should be described as auxiliary conditioning rather than the main driver.

## Role-Subset Evaluation

Role-subset results for `rolesplit + prior_ext + scale1` on its CQR query domain:

| Subset | S2 Dice | FEM Dice | Delta |
| --- | ---: | ---: | ---: |
| all | 0.5774 | 0.5474 | +0.0300 |
| core | 0.5936 | 0.5600 | +0.0336 |
| core_halo | 0.5778 | 0.5477 | +0.0301 |
| halo | 0.0014 | 0.0062 | -0.0048 |
| sentinel | 0.0000 | 0.0000 | +0.0000 |
| bg | 0.0000 | 0.0000 | +0.0000 |
| non_bg | 0.5774 | 0.5474 | +0.0300 |

Result file:

- `results/cqr_balanced_v2_rolesplit_priorext_scale1_role_subset_comparison.csv`

Interpretation: rolesplit improves core and core-halo refinement, but direct halo recovery remains limited. Sentinel remains diagnostic-only and should not be framed as discovered missed lesions.

## Final Positioning

CQR should be described as confidence-stratified or support-aware residual refinement. The stable claim is:

> CQR separates high-confidence FEM core from weak-support halo and improves residual correction stability.

Avoid claiming:

> CQR discovers missed lesions through sentinel.

The current best CQR setting closes much of the gap to ROI-only: the balanced_v2 CQR gap improved from `0.0298` Dice to `0.0088` Dice, but ROI-only balanced_v2 still remains slightly stronger.
