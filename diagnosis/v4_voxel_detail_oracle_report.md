# V4-Specific Voxel Detail Oracle Report

## Decision: PROCEED

The validation-only oracle gain is `+0.107520` Dice. The development-test oracle gain is reported only as a development comparison and did not determine this decision.

| Method | Val Dice | Dev-Test Dice |
| --- | ---: | ---: |
| Stage1 FEM | 0.63265 | 0.61738 |
| V4 | 0.73251 | 0.71653 |
| V4 + oracle detail | 0.84003 | 0.82439 |

`Delta_oracle-detail = Dice(V4 + w*) - Dice(V4)`:

- validation: `+0.107520`
- development-test: `+0.107861`

Here `w* = rho_GT - I_h Pi_h rho_GT`, and `rho_V4+oracle-detail = I_h x_V4 + w*`. Predictions are raw (unclamped), with the fixed 0.5 support threshold.

## Full metrics

| Split | Method | Dice | Precision | Recall | Weak Recall | HD95 | Localization | MSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val | Stage1 FEM | 0.632646 | 0.612634 | 0.753014 | 0.721189 | 2.228181 | 1.535646 | 0.004327 |
| val | V4 coarse only | 0.732512 | 0.729812 | 0.796022 | 0.744403 | 1.844324 | 1.207367 | 0.006663 |
| val | V4 + oracle detail | 0.840032 | 0.831897 | 0.893842 | 0.857419 | 0.854918 | 0.583330 | 0.005661 |
| development-test | Stage1 FEM | 0.617379 | 0.594399 | 0.750855 | 0.709763 | 2.604663 | 1.689924 | 0.004131 |
| development-test | V4 coarse only | 0.716529 | 0.706280 | 0.790795 | 0.745791 | 2.173528 | 1.352132 | 0.006638 |
| development-test | V4 + oracle detail | 0.824390 | 0.816910 | 0.881746 | 0.844096 | 0.876313 | 0.572605 | 0.005731 |

The operator and oracle use only the existing val300 and development-test300 splits. Confirmation remains sealed and was not inspected.
