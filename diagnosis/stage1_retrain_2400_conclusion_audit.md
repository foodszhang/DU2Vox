# Stage-1 Train2400 Retraining Audit

The checkpoint was selected on val300 before development-test evaluation. Sealed confirmation was not accessed.

## Canonical Dice

| Pipeline state | Val300 | Development-test300 |
| --- | ---: | ---: |
| Historical Stage 1 | 0.632646 | 0.617379 |
| Retrained Stage 1 | 0.640167 | 0.631170 |
| Historical Stage 1 + frozen V4 | 0.732512 | 0.716529 |
| Retrained Stage 1 + frozen V4 | 0.734820 | 0.715630 |
| Historical final hard-Q candidate | 0.741844 | 0.727163 |
| Retrained Stage 1 + frozen V4 + frozen hard-Q | 0.742269 | 0.724546 |

## Paired Dice differences (10,000-case bootstrap 95% CI)

| Contrast | Val300 | Development-test300 |
| --- | ---: | ---: |
| new_stage1_minus_old_stage1 | +0.007520 [-0.000065, +0.015113] | +0.013791 [+0.005587, +0.022056] |
| new_v4_minus_old_v4 | +0.002308 [-0.003606, +0.008276] | -0.000899 [-0.006941, +0.005338] |
| new_final_minus_old_final | +0.000425 [-0.005586, +0.006342] | -0.002617 [-0.008659, +0.003359] |
| new_v4_minus_new_stage1 | +0.094653 [+0.086186, +0.102972] | +0.084461 [+0.075427, +0.092988] |
| new_final_minus_new_v4 | +0.007449 [+0.005569, +0.009384] | +0.008915 [+0.006962, +0.010925] |

## Recomputed decomposition and oracle

- Mean inverse-discrepancy energy fraction: 0.719376.
- Mean representation energy fraction: 0.280624.
- Retrained Stage 1 + oracle hard-Q detail Dice: val 0.742846, development-test 0.735340.
- In the observed new development pipeline, V4 accounts for 90.5% of the gain and hard-Q detail for 9.5%.

## Decision audit

- Confirmed: the historical Stage 1 was undertrained for the current cohort; exact-architecture train2400 retraining improves its canonical Dice.
- Not overturned: a learned hard-Q-only model has not demonstrated 0.73 without FEM correction. The oracle feasibility statement is updated separately above.
- Not overturned: frozen V4 still supplies the dominant observed gain, although its role is scientifically better described as continuation of Stage 1 FEM inversion.
- Not overturned: hard-Q coarse preservation and zero-leakage contracts are independent of the upstream Stage-1 checkpoint.
- Not superseded: the retrained upstream gives no robust final-pipeline advantage over the historical freeze candidate.
- Scoped: A0-A3 remain valid for the historical frozen-V4 backbone. Their exact numeric gates are not automatically transferable to a newly retrained backbone.
