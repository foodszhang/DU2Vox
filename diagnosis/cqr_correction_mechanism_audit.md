# CQR correction mechanism audit

This audit was run on branch `feat/cqr-transport-observability`, based on commit
`f995bdd32955b791e09dfc15481e8b27384dd4ef`, using
`fmt_simgen_v2_3k_20k`. It addresses only the four requested mechanism checks; it
does not start full training.

## 1. Residual information

The hypothesis that Stage 1 leaves only a 1%--3% measurement residual is not
supported by the current normalization and compressed operator.

| set | rank | mean `||r||/||y||` | mean `||delta_GT||/||GT||` | observable GT energy | mean adjoint cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| train 100 | 64 | 0.1698 | 0.8184 | 0.0234 | 0.3453 |
| val 25 | 64 | 0.3027 | 0.8956 | 0.0122 | 0.2933 |

The residual has nontrivial magnitude, but its regularized adjoint is only
moderately aligned with the observable GT correction. The residual is therefore
useful auxiliary evidence and a consistency constraint, not a sufficient
morphology target.

## 2. Rank sweep

| rank | raw operator energy | observable GT correction energy | adjoint cosine |
| ---: | ---: | ---: | ---: |
| 64 | 11.76% | 1.22% | 0.2933 |
| 128 | 21.10% | 2.23% | 0.2443 |
| 256 | 35.57% | 3.14% | 0.2352 |
| 512 | 54.36% | 3.81% | 0.2406 |

Increasing rank recovers matrix energy, but even rank 512 assigns only 3.81% of
the sampled GT correction energy to the observable component. Rank 64 is narrow,
but rank alone is not the main explanation for the inactive correction branch.
The present candidate-supported projector defines a morphology subspace that is
small at every tested rank.

During this audit, `adjoint_evidence` was corrected to compute
`A_Q^T (A_Q A_Q^T + mu I)^-1 r`; it previously returned only `A_Q^T r`.

## 3. Branch activity and four-sample fitting

Explicit targets are now formed from detached GT correction:

- `t_obs = P_mu (GT - rho0)`
- `t_amb = (I - P_mu) (GT - rho0)`

Training records raw/projected activity and their ratios every epoch. The lifter
can be frozen after Phase A.

For an isolated observable-target capacity run, the absolute target L1 fell from
about `1.073e-2` to a best `9.198e-4` (about 91% reduction). At the best point,
the projected/raw L1 activity ratio was `0.118`. The observable head can learn,
but the projector discards most raw activity.

For the isolated ambiguous-target run, target L1 fell from `0.20915` to `0.12383`.
Projected activity was approximately the raw activity. A one-sample smoke
validation reached a transient best Delta Dice of `+0.13353` at epoch 15, but
later epochs regressed; this is a capacity/activity diagnostic, not a generalization
result.

The decisive objective-scale finding is that the normalized measurement
consistency loss can rise to hundreds when a branch starts moving. With weight
`0.02`, it still dominates target losses of order `1e-3`--`1e-1`, recreating the
zero-correction optimum. The next curriculum should first fit explicit branch
targets with the data term disabled or ramped from zero, then introduce the data
constraint after activity is established.

## 4. Full-tet physics certification

Twenty-five validation samples were evaluated with both stratified 1024-tet and
all relevant tetrahedra quadrature.

| metric | result |
| --- | ---: |
| P1/local-mass maximum relative error | `3.234e-8` |
| P1 compressed-measurement maximum relative error | `2.920e-8` |
| learned lifter transport error, stratified mean | `3.436e-8` |
| learned lifter transport error, all-tet mean | `4.022e-8` |
| absolute difference, mean | `7.469e-9` |
| absolute difference, maximum | `2.259e-8` |

This supports stochastic/stratified physical quadrature for training and
exhaustive quadrature for validation.

The unchanged `cqr_residual_inr` mainline was also run for one epoch on one
sample. It completed successfully after pinning the already-audited stale frame
manifest by SHA-256; sampled Stage 2 and FEM Dice were both `0.3955`, as expected
from zero initialization.

## Decision

Do not start the 2400-sample run yet. The physics and branch capacity checks pass,
but the joint objective is not calibrated. The next bounded experiment should be
a target-first Phase B/C curriculum on 100 train / 25 val, with the consistency
weight ramped only after branch target loss and activity stabilize. Dual-basis
work remains a follow-up hypothesis and was intentionally not implemented here.

Detailed machine-readable results are in:

- `diagnosis/cqr_residual_information_train100.json`
- `diagnosis/cqr_residual_rank_sweep_val25.json`
- `diagnosis/cqr_lifter_transport_stratified_vs_all_val25.json`
- `diagnosis/cqr_correction_mechanism_audit.json`
