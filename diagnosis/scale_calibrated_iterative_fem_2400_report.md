# Scale-Calibrated Iterative FEM Corrector: 2400/300/300 Report

## Result

The best single V3 corrector reached 0.710335 validation Dice and 0.698186 test
Dice at the fixed 0.5 operating point. A convex FEM-state ensemble, whose weight was
selected on validation only, reached **0.702730 test Dice**.

This clears the requested 0.7 engineering target for the frozen reconstruction
system. It does **not** establish that the V3 single model alone exceeds 0.7, and the
current 300-test split is no longer a pristine paper confirmation set because it was
previously evaluated during V1 development.

## Method change

The V1 corrector directly used `b - A x`, even though `x` is a binary-support state
and every measurement vector is independently max-normalized. V3 profiles out a
nonnegative nuisance amplitude before computing physics evidence:

\[
\alpha^*(x)=\frac{\max(\langle Ax,b\rangle,0)}{\|Ax\|_2^2+\epsilon},\qquad
r(x)=b-\alpha^*(x)Ax.
\]

Every iteration uses the current FEM state, normalized `A^T r`/Jacobi evidence,
multiview features, local mesh aggregation, and global FEM context. Neural and DC
proposals are bounded. Data consistency is a soft profiled-residual penalty with a
margin rather than a hard rejection gate.

The final ensemble is also a FEM-state method:

\[
x_h^{ens}=0.45x_h^{V1}+0.55x_h^{V3},\qquad
\hat\rho=I_hx_h^{ens}.
\]

Because canonical P1 transfer is linear, averaging the two cached P1 predictions is
exactly equivalent to averaging corrected FEM nodal states first and applying one
fixed analytic P1 transfer. There is no voxel residual/completion network, learned
lifting, observability split, CQR, or RGL.

## Paper basis

- Repeated use of the forward operator and adjoint follows the model-aware pattern
  of [Learned Primal-Dual](https://arxiv.org/abs/1707.06474).
- Separating a learned regularizer/proximal proposal from explicit data consistency
  follows [MoDL](https://arxiv.org/abs/1712.02862).
- Positive bounded step parameters and unrolled optimization structure are motivated
  by [FISTA-Net](https://arxiv.org/abs/2008.02683).
- Direct updates on an irregular mesh are consistent with the graph-convolutional
  inverse formulation in [GCNM](https://arxiv.org/abs/2103.15138).

These papers motivate the architecture. They do not supply evidence for the DU2Vox
performance claim; all numbers below are from the canonical dense common domain.

## Dense test results

| Model | Dice | Precision | Recall | Weak recall | HD95 | Localization | MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage-1 FEM | 0.617379 | 0.594399 | 0.750855 | 0.709763 | 2.604663 | 1.689924 | 0.004131 |
| V1 residual-aware FEM | 0.679571 | 0.687833 | 0.761823 | 0.708770 | 2.335192 | 1.478768 | 0.003803 |
| V3 scale-calibrated proximal FEM | 0.698186 | 0.714061 | 0.752588 | 0.707284 | 2.221322 | 1.389900 | 0.005420 |
| **Frozen 0.45 V1 + 0.55 V3 FEM ensemble** | **0.702730** | **0.714412** | **0.765755** | **0.717153** | **2.248606** | **1.454913** | **0.003664** |

The ensemble improves over Stage 1 by +0.085351 Dice, over V1 by +0.023160,
and over the V3 single model by +0.004544.

## V3 incremental effects

| State | Test Dice | Increment |
| --- | ---: | ---: |
| Stage 1 | 0.617379 | - |
| Iteration 1 | 0.642203 | +0.024824 |
| Iteration 2 | 0.676870 | +0.034667 |
| Iteration 3 | 0.698186 | +0.021316 |

Unlike V1, all three V3 iterations make a material positive contribution on test.

## Validation-only selection

At threshold 0.5:

| V3 weight | V1 weight | Validation Dice |
| ---: | ---: | ---: |
| 0.00 | 1.00 | 0.700174 |
| 0.40 | 0.60 | 0.716624 |
| 0.50 | 0.50 | 0.717700 |
| **0.55** | **0.45** | **0.717931** |
| 0.60 | 0.40 | 0.717860 |
| 1.00 | 0.00 | 0.710335 |

The broad plateau around 0.50--0.65 supports genuine error complementarity rather
than a sharp weight-search accident. The test evaluation used only the frozen 0.55
weight.

## Failed alternatives retained as evidence

### Hard profiled-residual monotonicity

The first raw-Jacobi version reduced residual RMS to 0.56x but caused FEM target MSE
near 265 and dense Dice 0.270. Bounding the update fixed the numerical explosion, but
formal training drove the mean accepted step scale from 0.00285 to exactly zero by
epoch 2. This established that strict residual monotonicity is incompatible with the
binary-support target under the present normalized forward model. The hard gate was
rejected.

### Threshold calibration

Validation selected threshold 0.465 (val Dice 0.710896 versus 0.710335 at 0.5), but
the frozen threshold reduced test Dice from 0.698186 to 0.696115. The reported result
therefore keeps threshold 0.5; threshold tuning is not used to claim success.

## Mechanistic diagnostics and limitations

| Diagnostic | V1 | V3 |
| --- | ---: | ---: |
| Inverse target cosine | 0.418494 | 0.345918 |
| Inverse relative L1 | 6.141940 | 7.760210 |
| Inverse relative L2 | 1.003552 | 1.175234 |
| Measurement residual ratio | 2.088729 (uncalibrated) | 1.068180 (amplitude-profiled) |

The two residual ratios are not numerically interchangeable because V3 changes the
residual definition. V3 is much better for segmentation and its physics residual is
well controlled, but it is less aligned with the exact `Pi_h rho_gt - x_h` target.
Therefore the defensible paper claim is a **scale-calibrated support-reconstruction
corrector**, not a faithful optimizer of the projected nodal MSE.

The ensemble uses two 0.867M-parameter correctors at inference (1.734M parameters
stored in total). This is still small, but it is not the requested single 1.0--1.3M
model. A next paper-quality step is validation-only distillation of the frozen FEM
ensemble into one student corrector, followed by evaluation on a newly generated,
unopened confirmation seed.

## Reproducibility artifacts

- V3 config: `configs/stage2/scale_calibrated_proximal_fem_v3_2400.yaml`
- V3 checkpoint: `runs/scale_calibrated_proximal_fem_v3_2400/checkpoints/best_dense_val_delta_dice.pth`
- Single-model test: `diagnosis/scale_calibrated_proximal_fem_v3_test300.json`
- Val ensemble curve: `diagnosis/iterative_fem_v1_v3_val_ensemble.json`
- Frozen ensemble full test: `diagnosis/iterative_fem_v1_v3_test_ensemble_full.json`
- Threshold falsification: `diagnosis/scale_calibrated_proximal_fem_v3_threshold_calibration.json`

## Conclusion

**Engineering outcome:** the frozen dense test system exceeds 0.7 Dice (0.702730).

**Scientific outcome:** scale-calibrated soft data consistency improves the FEM
corrector, and V1/V3 errors are sufficiently complementary for a validation-selected
FEM-state ensemble to pass 0.7. The single V3 model remains at 0.698186, so a claim
that one scale-calibrated corrector alone exceeds 0.7 would be false. A paper-level
generalization claim still requires a new unopened confirmation set.
