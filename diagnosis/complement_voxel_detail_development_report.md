# Complement-Constrained Voxel Detail: Development Report

## Outcome

**Outcome B — fine detail is valid and the gain is modest on validation.** The
validation-selected constrained model improves frozen V4 by `+0.009332` Dice on
val300 and `+0.010634` on development-test300. It preserves the V4 FEM state to
numerical precision, improves HD95 and localization on both splits, and slightly
improves weak recall. The validation gain is below the predeclared `+0.01` strong
threshold, so this is a fine-detail enhancement rather than a new primary performance
source.

Confirmation remained sealed. It was not inferred, visualized, swept, or used for
checkpoint selection.

## Operator and oracle gates

- Domain: all 1,677,645 valid centers of the canonical 0.2-mm grid, shape
  `(190, 200, 104)`; binary support is `gt_voxels > 0.05`.
- `Pi_h`: the decomposition experiment's unchanged sparse-LU sampled-L2 projection.
- `I_h`: the decomposition experiment's unchanged cached analytic P1 CSR matrix.
- Maximum seeded `||Pi_h I_h c-c||/||c||`: `1.50e-15`.
- Maximum seeded idempotence error of `I_h Pi_h`: `1.33e-15`.
- Mean GT complement leakage on val+development-test: `1.30e-29` in FP64.
- V4-specific oracle Dice: val `0.840032` (`+0.107520`), development-test
  `0.824390` (`+0.107861`). The validation oracle alone authorized training.

The fixed sampled operator is self-adjoint and idempotent under the uniform voxel
inner product, so `Q = I - I_h Pi_h` is an orthogonal complement projector on this
specific sampled domain. This is not a claim about a continuous-domain inner product.

## Frozen experiment

- Frozen V4: epoch 15, K=3, unchanged weights/architecture/loss/optimizer.
- Frozen V4 multiview encoder: 7 views, base channels 32, unchanged checkpoint state.
- Trainable branch: width 160, three residual MLP blocks, GELU, zero-initialized
  signed scalar proposal head.
- Inputs: 6-frequency normalized coordinate encoding, V4 P1 value, exact local P1
  gradient, four local FEM values, four barycentric weights, local mean/std/range,
  V4 0.5-boundary proximity, and frozen multiview optical features.
- Full canonical exact Q: FP64 CPU sparse solve/products with an exact self-adjoint
  custom backward; Q output remains FP32 after a BF16 proposal network.
- Loss: SmoothL1 detail regression + `0.1` detail MSE + `0.1` Tversky support.
- Training: train2400, one seed (`20260901`), batch 1, at most five epochs, fixed
  threshold 0.5. Checkpoints were selected only by maximum dense val300 Dice.
- Full-resolution feasibility gate: 1.071 s and 15.85 GiB for one complete
  forward/backward/optimizer step; no 0.4-mm fallback was used.

## Validation-selected checkpoints

| Model | Epoch | Val Dice | Delta vs V4 | Coarse leakage |
| --- | ---: | ---: | ---: | ---: |
| Unconstrained `rho_c + z` | 2 | 0.743784 | +0.011272 | 0.616094 |
| Constrained `rho_c + Qz` | 3 | 0.741844 | +0.009332 | 4.47e-18 |

The control's gain is not a fine-space result: 61.6% of its residual energy projects
back into FEM space. Its mean relative FEM-state change is 0.2587 on validation. The
constrained model's relative coarse-preservation error is `8.50e-9` (FP32 output
roundoff), with no residual bypass or post-addition clamp.

## Final development comparison

| Method | Dice | Precision | Recall | Weak Recall | HD95 | Localization | MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage1 FEM | 0.617379 | 0.594399 | 0.750855 | 0.709763 | 2.604663 | 1.689924 | 0.004131 |
| V4 coarse only | 0.716529 | 0.706280 | 0.790795 | 0.745791 | 2.173528 | 1.352132 | 0.006638 |
| V4 + unconstrained voxel residual | 0.726636 | 0.713661 | 0.805155 | 0.757568 | 2.022328 | 1.272334 | 0.008129 |
| **V4 + constrained voxel detail** | **0.727163** | **0.722357** | **0.795747** | **0.750303** | **2.004499** | **1.267871** | **0.006789** |
| V4 + oracle detail | 0.824390 | 0.816910 | 0.881746 | 0.844096 | 0.876313 | 0.572605 | 0.005731 |

The principal increment is

`Delta_voxel = Dice(V4 + Qz) - Dice(V4) = +0.010634` on development-test,
and `+0.009332` on validation.

The oracle recovery fraction is `R_oracle = 0.08679` on validation and `0.09859`
on development-test. Thus the learned branch recovers roughly 9–10% of the available
V4-specific fine-detail Dice headroom.

## Detail diagnostics

| Diagnostic | Constrained val | Constrained dev-test | Unconstrained dev-test |
| --- | ---: | ---: | ---: |
| Detail cosine | 0.15715 | 0.16935 | 0.09315 |
| Relative L1 | 1.23544 | 1.24747 | 6.56457 |
| Relative L2 | 1.09844 | 1.09546 | 1.27870 |
| Detail energy ratio | 0.41432 | 0.42380 | 0.80926 |
| Predicted detail norm | 24.735 | 23.916 | 32.363 |
| GT detail norm | 39.813 | 37.992 | 37.992 |
| Positive / negative voxel fraction | 0.5004 / 0.4996 | 0.5003 / 0.4997 | 0.0050 / 0.9950 |
| Energy within 0.2 mm boundary | 0.34679 | 0.34355 | 0.18569 |
| Energy within 0.6 mm boundary | 0.67768 | 0.67501 | 0.37705 |
| Energy outside 0.6 mm | 0.32232 | 0.32499 | 0.62295 |
| Coarse leakage | 4.47e-18 | 4.57e-18 | 0.62679 |

The constrained prediction has positive detail cosine, balanced signed corrections,
and substantially stronger short-range boundary concentration than the control. It
does not reproduce the GT oracle concentration perfectly (historically about 69.3%
within 0.2 mm and 89.9% within 0.6 mm), explaining the small oracle recovery fraction.

## Scientific conclusion

The experiment answers the central question positively but narrowly: after V4 coarse
correction, provably FEM-exclusive degrees of freedom provide a real, learnable
improvement. The exact constraint is valuable because it produces slightly better
development-test Dice, precision, HD95, localization, MSE, and detail cosine than the
ordinary residual while eliminating its large coarse leakage. The validation gain is
modest, so no larger voxel model, extra loss family, or joint fine-tuning is authorized
in this round.

