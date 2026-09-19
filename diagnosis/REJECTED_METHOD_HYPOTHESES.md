# Rejected Method Hypotheses

## Purpose

This file prevents rejected or unsupported historical ideas from being reintroduced
under new names. Rejection here means "not part of the final reconstruction
candidate"; some mechanisms may remain useful as diagnostics or secondary analysis.

## 1. Learned transport-consistent FEM-to-voxel lifting

**Rejected as a necessary reconstruction mechanism.** Canonical analytic P1 already
satisfies the discrete transport-consistency comparison used in the falsification
study. The matched learned TC lifting produced no independent Dice, localization, or
source-level gain over P1.

Final rule: FEM-to-voxel transfer is fixed analytic P1. Do not add a learned lifting
module without a new, preregistered failure that P1 cannot satisfy.

## 2. Observability `P/(I-P)` split as the final morphology architecture

**Rejected as a demonstrated reconstruction contribution.** The partition changed
some forward/measurement diagnostics, but the matched main comparison produced no
Dice, localization, component, separation, or weak-source gain. Predicted partitioned
and plain corrections had cosine `0.9937` in the reported mechanism audit, indicating
that much of the construction acted as a residual reparameterization.

Observability may remain a physics diagnostic. Do not present it as the final
morphology reconstruction mechanism, and do not confuse it with `ker(Pi_h)`.

## 3. CQR/RGL/proposal/observability stack as the headline innovation

**Rejected for the final candidate.** The preregistered A/B/C falsification did not
show a practical reconstruction gain attributable to learned lifting or partitioning;
the broader CQR/RGL/proposal line did not clear the required practical reconstruction
gates. It added mechanism complexity without supporting the final coarse-to-fine
claim.

Final rule: do not restore CQR, RGL, proposal, or observability modules merely to make
the architecture appear more novel.

## 4. Plain unconstrained voxel residual

**Rejected as genuine fine-domain reconstruction.** The matched current control
improved development Dice from `0.716529` to `0.726636`, but its mean coarse leakage
was `0.62679` and its relative FEM-state change was `0.2607`. It therefore rewrote the
coarse V4 solution.

Final rule: the only admissible voxel contribution is `Q z_theta`; no free residual
bypass may be added.

## 5. Unconstrained representation head

**Rejected by its semantic contract.** The 2400/300/300 ESCB experiment produced
representation leakage `0.45252`, compared with approximately `2.22e-15` for the GT
representation target. Representation-target cosine was only `0.10839`.

Loss supervision alone did not keep the prediction in the intended complement.
Final rule: complement membership must be enforced by the fixed operator `Q`, not
requested through a target or leakage penalty.

## 6. Parallel inverse/representation semantic heads

**Rejected as a clean failure-mode decomposition.** The 3000-sample oracle study
certified a valid energy decomposition and different spatial statistics, but the two
oracle corrections improved substantially overlapping reconstruction metrics. The
inverse oracle dominated most metrics, and source-property stratification did not
support a broad semantic separation.

Final rule: describe coarse/detail allocation by approximation spaces, not by claims
that the branches solve disjoint clinical or morphological failure modes.

## 7. Representation-complement ESCB without an architectural projector

**Rejected: Outcome C, method contract failed.** Most benefit came from its FEM
corrector; voxel completion added only `+0.00415` development-test Dice and failed the
coarse-space contract. Do not revive the A/B/C phase scheme, learned representation
head, or direct final residual path as the final two-stage method.

## 8. Threshold calibration

**Rejected for final selection.** Validation selected threshold `0.465` for V3, but
the frozen threshold reduced development-test Dice from `0.698186` at `0.5` to
`0.696115`. The small validation improvement did not transfer.

Final rule: use raw, unclamped predictions with the fixed `0.5` threshold. Do not run
a new threshold sweep during development or confirmation.

## 9. Joint fine-tuning as evidence for Stage 2B

**Not performed and not part of the candidate.** The frozen-V4 constrained branch
already passed the minimum `+0.005` gate, but joint fine-tuning was intentionally
omitted to preserve causal isolation of genuine voxel-only improvement.

Final rule: do not silently joint-tune Stage 2A and Stage 2B. Any future joint study
must retain `w=Qz`, use a preregistered low V4 learning rate, and remain separate from
the frozen candidate result.

## Historical evidence sources

- `diagnosis/du2vox_innovation_falsification_report.md`
- `diagnosis/cqr_targetfirst_observability_partition_report.md`
- `diagnosis/error_structured_bridge_contract.md`
- `diagnosis/error_structured_cross_discretization_2400_report.md`
- `diagnosis/scale_calibrated_iterative_fem_2400_report.md`
- `experiments/cross_discretization_decomposition/artifacts/REPORT.md`

