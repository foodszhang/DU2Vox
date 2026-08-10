# Target-first observability-partition pilot

This round ran on `feat/cqr-transport-observability`, based on commit
`9f743ab182cb42a61a6b0d35b2917082e2c74497`, using 100 train, 25 val, and 25
test samples from `fmt_simgen_v2_3k_20k`. No 2400-sample training was started.

## Physical control experiments

### Regularization sweep

For the rank-64 CQR correction operator, changing `mu_relative` barely changes
the result:

| mu relative | effective DoF | observable GT correction energy |
| ---: | ---: | ---: |
| `1e-5` | 63.99 | 1.262% |
| `3e-5` | 63.98 | 1.261% |
| `1e-4` | 63.93 | 1.257% |
| `3e-4` | 63.78 | 1.248% |
| `1e-3` | 63.31 | 1.218% |
| `3e-3` | 62.11 | 1.152% |

The low observable energy is therefore not caused by `mu_relative=1e-3`
filtering away weak directions. It is principally a subspace limitation.

### Full 7413-measurement sanity check

Five validation samples were evaluated with all 7413 surface measurement
directions. The computation used the equivalent 2048-by-2048 query Gram
projector; no 7413-by-7413 inverse was formed.

| mu relative | effective DoF | observable GT correction energy |
| ---: | ---: | ---: |
| `1e-5` | 502.65 | 10.59% |
| `1e-3` | 376.14 | 6.37% |
| `3e-3` | 343.92 | 5.67% |

Thus rank-64 understates the observable component, but the non-compressed result
still places roughly 89%--94% of high-resolution morphology correction in weakly
observable directions. This supports the term **observability-partitioned
morphology reconstruction**.

## Training changes

The implementation now supports:

- `observable_pretrain`, `ambiguous_pretrain`, `joint`, and `full` phases;
- relative L1 and cosine supervision for both operator-defined branch targets;
- GT correction forward-effect supervision;
- relative measurement energy before/after correction;
- exact squared-ratio and robust log-ratio one-sided non-worsening metrics;
- a piecewise-linear measurement-consistency schedule;
- per-epoch branch norms, projection ratios, energy fractions, target errors,
  target cosine, alpha anchor, and measurement diagnostics.

The least-squares Stage 1 amplitude scale is applied consistently to both the
Stage 1 forward prediction and the correction forward effect.

## 100/25 target-first pilot

Observable-only pretraining increased target cosine to about `0.28`, but did not
fit target amplitude across 100 resampled samples: relative L1 stayed near or
above 1. This branch is active but remains the weaker mechanism.

Ambiguous-only pretraining produced the first clear performance movement. At
epoch 7, sampled validation Dice improved by `+0.0165` over FEM and approximately
99.9% of predicted correction energy was in the ambiguous branch. Continuing
ambiguous-only training overfit and later degraded validation, so the epoch-7
checkpoint was used for joint training.

Joint target-first training, without measurement-residual consistency, reached a
sampled validation Delta Dice of `+0.0197` at epoch 9. Its unified evaluation is:

| split | samples | FEM Dice | Stage 2 Dice | Delta |
| --- | ---: | ---: | ---: | ---: |
| val | 25 | 0.5334 | 0.5497 | +0.0163 |
| test | 25 | 0.5786 | 0.5824 | +0.0038 |

This is the first pilot in this line with positive val and test deltas under the
same unified evaluator. The test gain is nevertheless much smaller than the val
gain.

## Measurement non-worsening result

The exact proposed loss,
`relu(E_after / E_before - 1)^2`, remains extremely heavy-tailed because some
samples have small residual energy. Even a schedule beginning at `0.001`
dominated branch supervision and reduced Dice.

A robust one-sided alternative,
`log(1 + relu(E_after / E_before - 1))^2`, kept the loss numerically comparable.
At epoch 20 (`lambda_nonworse=0.02`) it retained positive unified deltas:

| split | FEM Dice | Stage 2 Dice | Delta |
| --- | ---: | ---: | ---: |
| val | 0.5334 | 0.5504 | +0.0169 |
| test | 0.5786 | 0.5800 | +0.0014 |

However, mean training `E_after/E_before` was still about 41.2. The robust loss
therefore prevents numerical domination but does not establish better measurement
agreement. The best scientific checkpoint remains the joint target-first model
before residual consistency.

## Decision

Do not start the 2400-sample run yet. The morphology mechanism has crossed its
first effectiveness gate and the observability-partition interpretation is
supported, but two issues remain:

1. observable target amplitude is not learned reliably across samples;
2. measurement non-worsening conflicts with morphology correction and does not
   reduce `E_after/E_before` at the proposed weight.

The next bounded experiment should focus on calibration of the correction
forward-effect and measurement-scale model, while retaining the successful
ambiguous-first/joint curriculum and the existing network architecture.

Detailed artifacts:

- `diagnosis/cqr_mu_sweep_rank64_val25.md`
- `diagnosis/cqr_full_measurement_observability_val5.md`
- `diagnosis/cqr_targetfirst_joint100_val25_unified.json`
- `diagnosis/cqr_targetfirst_joint100_test25_unified.json`
- `diagnosis/cqr_targetfirst_full_robust100_val25_unified.json`
- `diagnosis/cqr_targetfirst_full_robust100_test25_unified.json`
