# Error-Structured Cross-Discretization 2400/300/300 Report

## Decision

**Outcome C — METHOD CONTRACT FAILED.**

The implemented computation graph is structurally correct and the FEM correction is
useful, but the learned voxel completion does not preserve the intended
inverse/representation separation strongly enough. On the held-out 300-test set its
coarse-space leakage is `0.4525`, versus `2.22e-15` for the canonical representation
target. Its cosine with the representation target is only `0.1084`, while its cosine
with the inverse target is `0.1009`. Therefore the Dice improvement cannot be claimed
as validation of ESCB's intended complement-space mechanism.

Per the preregistered rule, performance is reported below for completeness, but
performance interpretation stops at the contract failure. No projector, observability
split, learned lifting, or additional loss was added after seeing test results.

## Frozen experiment protocol

- Data: one fixed seed, train/val/test = 2400/300/300.
- Semantics: binary support, strict `gt_voxels > 0.05`; metrics threshold at 0.5.
- Operators: the certified decomposition experiment's exact sparse `P` and exact
  `MassProjector`; no replacement Pi_h or I_h.
- ESCB parameters: 1,093,699 total: multiview encoder 483,297, contextual FEM
  corrector 318,817, voxel completion 291,585.
- FEM corrector: three fixed-kNN contextual residual blocks with learned residual
  gates and LayerNorm.
- Training: Phase A 10 epochs, Phase B 10 epochs, Phase C 15 epochs. All required
  target losses remained enabled in Phase C.
- Selection: dense 300-val Dice only. Plain best was epoch 5 (`0.65136`); ESCB best
  was Phase C epoch 5 (`0.68227`). Test was run once after architecture, losses, and
  checkpoints were frozen.

## Dense 300-test reconstruction

| Model | FEM Dice | Corrected FEM Dice | Final Dice | Precision | Recall | Weak Recall | HD95 | Localization | MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FEM baseline | 0.61738 | — | 0.61738 | 0.59440 | 0.75086 | 0.70976 | 2.60466 | 1.68992 | 0.004131 |
| Plain target-first | 0.61738 | — | 0.63849 | 0.66281 | 0.70484 | 0.67002 | 2.55047 | 1.62329 | 0.003415 |
| ESCB | 0.61738 | 0.66268 | 0.66683 | 0.74869 | 0.68259 | 0.64294 | 2.52508 | 1.45769 | 0.002863 |

Numerically, ESCB exceeds plain target-first by `+0.02835` Dice, `+0.08588`
precision, `-0.02539` HD95, `-0.16560` localization error, and `-0.000552` MSE.
Recall is `-0.02225` and weak-source recall is `-0.02708` lower. These differences
do not override the failed semantic contract.

## Component learning

| Diagnostic | 300-test value |
| --- | ---: |
| Inverse target cosine | 0.59221 |
| Inverse relative L1 | 1.29855 |
| Inverse relative L2 | 0.79149 |
| Representation target cosine | 0.10839 |
| Representation relative L1 | 1.19024 |
| Representation relative L2 | 0.99553 |
| Predicted representation leakage | 0.45252 |
| GT representation-target leakage | 2.2243e-15 |

The inverse corrector has a meaningful target alignment and materially improves the
coarse reconstruction. The representation network, however, explains very little of
the certified representation target and retains a large coarse-space component.

### Cross-compensation cosine matrix

| predicted / target | inverse target | representation target |
| --- | ---: | ---: |
| predicted inverse | 0.57801 | 7.21e-11 |
| predicted representation | 0.10090 | 0.10839 |

The analytic inverse prediction is cleanly separated from the representation target.
The representation prediction is not: its intended-target alignment is only slightly
higher than its inverse-target alignment. This is the decisive contract failure.

## Incremental effects

### Stage 1 FEM to corrected FEM

| Metric | Stage 1 FEM | Corrected FEM | Delta |
| --- | ---: | ---: | ---: |
| Dice | 0.61738 | 0.66268 | +0.04531 |
| Precision | 0.59440 | 0.75370 | +0.15930 |
| Recall | 0.75086 | 0.67125 | -0.07961 |
| Weak recall | 0.70976 | 0.63194 | -0.07782 |
| HD95 | 2.60466 | 2.53538 | -0.06928 |
| Localization | 1.68992 | 1.45542 | -0.23450 |
| MSE | 0.004131 | 0.002895 | -0.001236 |

### Corrected FEM to representation-completed output

| Metric | Corrected FEM | Final | Delta |
| --- | ---: | ---: | ---: |
| Dice | 0.66268 | 0.66683 | +0.00415 |
| Precision | 0.75370 | 0.74869 | -0.00501 |
| Recall | 0.67125 | 0.68259 | +0.01134 |
| Weak recall | 0.63194 | 0.64294 | +0.01100 |
| HD95 | 2.53538 | 2.52508 | -0.01030 |
| Localization | 1.45542 | 1.45769 | +0.00227 |
| MSE | 0.002895 | 0.002863 | -0.000032 |

Most of ESCB's reconstruction benefit comes from the contextual FEM correction. The
voxel completion adds only `+0.00415` Dice and has the failed representation-space
diagnostics above.

## Structural and runtime certification

- Oracle decomposition closure remains below the `1e-12` test tolerance.
- Representation target is constructed only as `rho_gt - I_h(Pi_h(rho_gt))` and is
  bit-identical under different inverse predictions.
- Corrected FEM reaches voxels only through the parameter-free certified P1 transfer.
- The forward identity `final = corrected_fem + representation_prediction` is tested
  and was also checked on a saved test prediction (`5.96e-8` max float32 error).
- Phase A/B/C gradient routing passed in the real multiview smoke and formal run.
- Nine targeted semantic/vectorization tests pass.
- Formal training used about 8.15 GiB VRAM with approximately 94% GPU utilization;
  low parameter count was not a utilization bottleneck after multiview vectorization.
- All 300 ESCB test cases have compressed intermediate predictions under
  `output/error_structured_escb_2400_test_predictions/` (about 7.7 GiB).

## Final interpretation

The evidence supports the narrower claim that a stronger contextual correction in
FEM space improves reconstruction before cross-discretization. It does **not** support
the full ESCB claim that the subsequent voxel network learns only the irreducible
representation residual. The correct conclusion is therefore Outcome C, not strong
support, despite ESCB's higher final Dice than the plain residual baseline.
