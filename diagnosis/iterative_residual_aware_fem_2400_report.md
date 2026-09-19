# Iterative Residual-Aware FEM Corrector — 2400/300/300 Report

## Outcome

The FEM-only iterative corrector is the strongest tested reconstruction model by
held-out Dice, but its intended inverse-state/residual mechanism is only partially
supported.

- Reconstruction result: **promising**.
- Iterative spatial-context result: **supported**.
- Accurate recovery of the canonical inverse target: **not established**.
- Measurement-residual consistency: **failed as an emergent property**.

No model, loss, or checkpoint was changed after test evaluation.

## Frozen protocol

- Train/validation/test: 2400/300/300, one seed.
- Stage 1: unchanged balanced-v2 FEM state.
- Model: 383,905-parameter shared FEM corrector plus the 483,297-parameter
  multiview encoder; no voxel head or learned transfer.
- Iterations: three shared-weight updates, each recomputing `b - A x` and
  `A.T @ (b - A x)`.
- Target: canonical `Pi_h(rho_gt)` in binary-support semantics.
- Checkpoint selection: 300-case dense validation Dice only.
- Selected checkpoint: epoch 15, validation Dice `0.700174`.
- The 300-test split was evaluated once after checkpoint selection.

## Dense validation curve

| Epoch | Final Dice |
| ---: | ---: |
| 5 | 0.67558 |
| 10 | 0.68111 |
| 15 | **0.70017** |
| 20 | 0.66160 |
| 25 | 0.66548 |

Training target MSE continued to fall after epoch 15 while dense validation Dice
collapsed. Dense checkpointing was therefore essential.

## Dense 300-test comparison

| Model | Dice | Precision | Recall | Weak Recall | HD95 | Localization | MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage-1 FEM | 0.61738 | 0.59440 | 0.75086 | 0.70976 | 2.60466 | 1.68992 | 0.004131 |
| Plain target-first | 0.63849 | 0.66281 | 0.70484 | 0.67002 | 2.55047 | 1.62329 | 0.003415 |
| Previous ESCB | 0.66683 | 0.74869 | 0.68259 | 0.64294 | 2.52508 | **1.45769** | **0.002863** |
| Iterative FEM corrector | **0.67957** | 0.68783 | **0.76182** | **0.70877** | **2.33519** | 1.47877 | 0.003803 |

Relative to previous ESCB, the new model gains `+0.01274` Dice, `+0.07924` recall,
`+0.06583` weak-source recall, and improves HD95 by `0.18989`. Precision is lower by
`0.06086`, localization error is higher by `0.02108`, and MSE is worse by `0.000940`.

## Iterative effects on 300-test

| State | Dice | Increment |
| --- | ---: | ---: |
| Stage 1 | 0.61738 | — |
| Step 1 | 0.65614 | +0.03876 |
| Step 2 | 0.67827 | +0.02213 |
| Step 3 | 0.67957 | +0.00130 |

The first two corrections account for nearly all reconstruction gain. The third step
is positive under the frozen test evaluation, but only marginally so. This result must
not be used to redesign the current tested model; iteration-count selection belongs to
a future validation-only experiment.

## Mechanism diagnostics

| Diagnostic | 300-test value |
| --- | ---: |
| Inverse-target cosine | 0.41849 |
| Inverse relative L1 | 6.14194 |
| Inverse relative L2 | 1.00355 |
| Initial measurement residual RMS | 0.07131 |
| Final measurement residual RMS | 0.10290 |
| Measurement residual ratio | 2.08873 |

The network uses measurement residual and global context effectively as predictive
features, but it does not behave as a residual-decreasing inverse solver. Its predicted
nodal correction is also only weakly aligned with the canonical inverse target despite
direct supervision. Consequently, the result supports a **FEM-domain iterative
contextual refinement architecture**, but not yet the stronger claim of a faithful
inverse-state correction algorithm.

## Interpretation

Removing representation completion was beneficial: the FEM-only model surpasses the
previous sequential ESCB while recovering recall and weak-source recall. The next
scientific issue is not capacity or voxel representation. It is the mismatch between
support-Dice improvement and the intended nodal/measurement semantics. Any follow-up
must be selected on validation data and should address that mismatch explicitly rather
than adding another voxel completion network.
