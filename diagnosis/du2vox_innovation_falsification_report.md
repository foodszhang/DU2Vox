# DU2Vox Innovation Falsification Report

## A. Experimental fairness

- same data: yes
- same view encoder: yes
- same CQR/prior: yes
- same optimizer/training budget: yes
- parameter counts: A=711,946, B=731,147, C=731,404
- total trainable parameter spread: 2.66% (target < 5%)
- all reported values use all three preregistered seeds; no best-seed selection
- all 18 main/physics runs stopped at epoch 7 under the same patience rule

| Model | Total | INR | Lifter | View encoder | Projector trainable |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 711,946 | 228,649 | 0 | 483,297 | 0 |
| B | 731,147 | 228,649 | 19,201 | 483,297 | 0 |
| C | 731,404 | 228,906 | 19,201 | 483,297 | 0 |

## B. Main table

| Model | Lifting | Partition | Val Dice | Test Dice | Delta over FEM | Delta over A | FP comp | Comp recall | Localization (mm) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | P1 | no | 0.5333 ± 0.0000 | 0.5786 ± 0.0000 | -0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 6.56 ± 0.00 | 0.9200 ± 0.0000 | 0.824 ± 0.000 |
| B | TC | no | 0.5333 ± 0.0000 | 0.5786 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 6.56 ± 0.00 | 0.9200 ± 0.0000 | 0.824 ± 0.000 |
| C | TC | yes | 0.5333 ± 0.0000 | 0.5786 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 6.56 ± 0.00 | 0.9200 ± 0.0000 | 0.824 ± 0.000 |

### Source-level grouped results

| Model | Group | N | Dice | Component recall | Weak-source recall | FP components | Separation success |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 1-focus | 6 | 0.6281 ± 0.0000 | 1.0000 ± 0.0000 | 0.8472 ± 0.0000 | 7.83 ± 0.00 | 1.0000 ± 0.0000 |
| A | 2-focus | 9 | 0.5544 ± 0.0000 | 0.8889 ± 0.0000 | 0.6997 ± 0.0000 | 2.89 ± 0.00 | 0.7778 ± 0.0000 |
| A | 3-focus | 10 | 0.5706 ± 0.0000 | 0.9000 ± 0.0000 | 0.4895 ± 0.0000 | 9.10 ± 0.00 | 0.7000 ± 0.0000 |
| A | depth=shallow | 7 | 0.5810 ± 0.0000 | 0.9286 ± 0.0000 | 0.7549 ± 0.0000 | 6.00 ± 0.00 | 0.8571 ± 0.0000 |
| A | depth=medium | 9 | 0.5459 ± 0.0000 | 0.9074 ± 0.0000 | 0.6340 ± 0.0000 | 5.78 ± 0.00 | 0.7778 ± 0.0000 |
| A | depth=deep | 9 | 0.6094 ± 0.0000 | 0.9259 ± 0.0000 | 0.5872 ± 0.0000 | 7.78 ± 0.00 | 0.7778 ± 0.0000 |
| B | 1-focus | 6 | 0.6281 ± 0.0000 | 1.0000 ± 0.0000 | 0.8472 ± 0.0000 | 7.83 ± 0.00 | 1.0000 ± 0.0000 |
| B | 2-focus | 9 | 0.5544 ± 0.0000 | 0.8889 ± 0.0000 | 0.6997 ± 0.0000 | 2.89 ± 0.00 | 0.7778 ± 0.0000 |
| B | 3-focus | 10 | 0.5706 ± 0.0000 | 0.9000 ± 0.0000 | 0.4895 ± 0.0000 | 9.10 ± 0.00 | 0.7000 ± 0.0000 |
| B | depth=shallow | 7 | 0.5810 ± 0.0000 | 0.9286 ± 0.0000 | 0.7550 ± 0.0000 | 6.00 ± 0.00 | 0.8571 ± 0.0000 |
| B | depth=medium | 9 | 0.5459 ± 0.0000 | 0.9074 ± 0.0000 | 0.6340 ± 0.0000 | 5.78 ± 0.00 | 0.7778 ± 0.0000 |
| B | depth=deep | 9 | 0.6094 ± 0.0000 | 0.9259 ± 0.0000 | 0.5872 ± 0.0000 | 7.78 ± 0.00 | 0.7778 ± 0.0000 |
| C | 1-focus | 6 | 0.6281 ± 0.0000 | 1.0000 ± 0.0000 | 0.8472 ± 0.0000 | 7.83 ± 0.00 | 1.0000 ± 0.0000 |
| C | 2-focus | 9 | 0.5544 ± 0.0000 | 0.8889 ± 0.0000 | 0.6997 ± 0.0000 | 2.89 ± 0.00 | 0.7778 ± 0.0000 |
| C | 3-focus | 10 | 0.5706 ± 0.0000 | 0.9000 ± 0.0000 | 0.4894 ± 0.0000 | 9.10 ± 0.00 | 0.7000 ± 0.0000 |
| C | depth=shallow | 7 | 0.5810 ± 0.0000 | 0.9286 ± 0.0000 | 0.7549 ± 0.0000 | 6.00 ± 0.00 | 0.8571 ± 0.0000 |
| C | depth=medium | 9 | 0.5459 ± 0.0000 | 0.9074 ± 0.0000 | 0.6340 ± 0.0000 | 5.78 ± 0.00 | 0.7778 ± 0.0000 |
| C | depth=deep | 9 | 0.6094 ± 0.0000 | 0.9259 ± 0.0000 | 0.5872 ± 0.0000 | 7.78 ± 0.00 | 0.7778 ± 0.0000 |

## C. Incremental contribution

- Delta_lift = B-A Dice: 0.0000 ± 0.0000
- Delta_part = C-B Dice: -0.0000 ± 0.0000
- H1 preregistered conditions passed: 0/5; no-harm=True
- H2 strong conditions passed: 1/7; medium conditions passed: 1/7

## D. Physics / mechanism

- Model A: alpha_entropy=1.0840 ± 0.0000, alpha_lambda_l1=0.0000 ± 0.0000, correction_forward_error=1.1292 ± 0.0020, measurement_error=1.3331 ± 0.0065, rho_tc_p1_l1=0.0000 ± 0.0000, rho_tc_p1_l2=0.0000 ± 0.0000, transport_error=0.0000 ± 0.0000
- Model B: alpha_entropy=1.0840 ± 0.0000, alpha_lambda_l1=0.0000 ± 0.0000, correction_forward_error=1.1292 ± 0.0020, measurement_error=1.3332 ± 0.0064, rho_tc_p1_l1=0.0000 ± 0.0000, rho_tc_p1_l2=0.0001 ± 0.0000, transport_error=0.0000 ± 0.0000
- Model C: alpha_entropy=1.0840 ± 0.0000, alpha_lambda_l1=0.0000 ± 0.0000, ambiguous_energy_fraction=0.9558 ± 0.0151, ambiguous_projection_ratio=0.9305 ± 0.0002, correction_forward_error=1.0784 ± 0.0163, measurement_error=1.1673 ± 0.0509, observable_energy_fraction=0.0441 ± 0.0151, observable_projection_ratio=0.1004 ± 0.0001, projected_ambiguous_norm=0.0011 ± 0.0000, projected_observable_norm=0.0002 ± 0.0000, raw_ambiguous_norm=0.0011 ± 0.0000, raw_observable_norm=0.0007 ± 0.0001, rho_tc_p1_l1=0.0000 ± 0.0000, rho_tc_p1_l2=0.0001 ± 0.0000, transport_error=0.0000 ± 0.0000
- partition/plain plain_partition_cosine: 0.9937 ± 0.0046
- partition/plain correction_support_iou: 0.0000 ± 0.0000
- partition/plain spatial_correlation: 0.2922 ± 0.1148
- partition/plain spectral_roughness_ratio_c_over_b: 65.2362 ± 62.6035

Main-round physics interpretation:

- C vs B measurement-error relative reduction: 0.1243 ± 0.0420; same direction in 3/3 seeds.
- C vs B correction-forward relative reduction: 0.0450 ± 0.0159.
- Despite the relative reduction, all main-round C measurement and correction-forward errors remain above the no-correction reference 1.0.
- Correction support IoU at |correction| >= 0.05 is zero because neither model activates correction at that absolute threshold; it is not evidence of disjoint supports.

## D2. Auxiliary shared-physics supervision

All models use the same lambda_correction_physics=0.1; no branch targets are enabled.

| Model | Test Dice | FP comp | Localization | Measurement error | Correction-forward error |
| --- | ---: | ---: | ---: | ---: | ---: |
| A-phys | 0.5786 ± 0.0000 | 6.48 ± 0.00 | 0.823 ± 0.000 | 1.0433 ± 0.0171 | 1.0353 ± 0.0070 |
| B-phys | 0.5786 ± 0.0000 | 6.49 ± 0.02 | 0.823 ± 0.000 | 1.0433 ± 0.0172 | 1.0353 ± 0.0070 |
| C-phys | 0.5786 ± 0.0000 | 6.56 ± 0.00 | 0.824 ± 0.000 | 1.0415 ± 0.0280 | 0.9813 ± 0.0086 |

- Auxiliary C-B Dice: -0.0000 ± 0.0000.
- Auxiliary C-B correction-forward relative reduction: 0.0521 ± 0.0141.
- The auxiliary result improves forward effect but does not produce a reconstruction or source-level advantage.

## E. Final verdict

**Verdict 3: Observability partition is useful; transport lifting is not necessary.**

H1 independent value: no.

H2 reconstruction necessity: yes.

The H2 pass is narrow: it is triggered only by the preregistered main-round measurement-error criterion. Dice, FP components, component recall, localization, separation, and weak-source reconstruction do not improve. Accordingly, the evidence supports retaining observability as a physics constraint/analysis direction, not claiming a demonstrated morphology reconstruction gain.

Thresholds were encoded before reading experiment results; all comparisons are paired by seed.
