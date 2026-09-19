# Residual-Consistent Iterative FEM Corrector V2

## Decision

V1 reached 0.700174 dense validation Dice but only 0.679570 on the already-opened
300-sample test split. Its profiled inverse correction was useful for segmentation,
but the uncalibrated measurement residual increased to 2.02x on validation and the
third iteration contributed only +0.00344 Dice. V2 therefore changes the inverse
update mechanism, not the FEM-to-voxel bridge and not the model width.

The V2 computation graph is:

```text
x_h -> profile nonnegative measurement amplitude
    -> scale-invariant b - alpha A x
    -> normalized A^T residual / Jacobi evidence
    -> bounded contextual FEM proposal
    -> profiled-residual trust-region line search
    -> corrected FEM state
    -> fixed canonical P1 transfer
```

There is no voxel completion head, learned lifting, observability split, CQR, RGL,
or representation-complement projector.

## Why the V1 residual was not a valid physical constraint

The dataset supplies a binary-support FEM state, while `measurement_b` is normalized
independently by its sample maximum. Consequently, directly evaluating `b - A x_h`
silently assumes a known concentration/amplitude scale that the learning target does
not contain. V2 profiles out the nonnegative nuisance amplitude analytically:

\[
\alpha^*(x)=\frac{\max(\langle Ax,b\rangle,0)}{\|Ax\|_2^2+\epsilon},\qquad
r(x)=b-\alpha^*(x)Ax.
\]

This residual is invariant to positive rescaling of the FEM support state. The
analytic amplitude and fixed `A`/`A^T` products have no trainable parameters.

## Literature-to-implementation mapping

- Learned Primal-Dual repeatedly exposes learned blocks to the forward operator and
  its adjoint. V2 likewise recomputes `A x`, the profiled residual, and `A^T r` after
  every FEM update, but stays in the certified FEM node space. See
  [Adler and Oktem, 2018](https://arxiv.org/abs/1707.06474).
- MoDL alternates learned regularization with explicit model-based data consistency.
  V2 separates its contextual learned proposal from a fixed analytic residual check,
  instead of asking one MLP to learn both. See
  [Aggarwal, Mani, and Jacob, 2018](https://arxiv.org/abs/1712.02862).
- FISTA-Net makes optimization structure and constrained step parameters part of the
  unrolled architecture. V2 uses bounded positive DC steps and a finite trust-region
  line search. See [Xiang et al., 2021](https://arxiv.org/abs/2008.02683).
- Graph convolutional inverse methods motivate operating directly on an irregular
  FEM mesh rather than rasterizing the inverse state before correction. See
  [Herzberg et al., 2021](https://arxiv.org/abs/2103.15138).

These references motivate the mechanism; they do not establish a DU2Vox performance
claim. That claim is determined only by dense common-domain validation and an unseen
confirmation set.

## Numerical safeguards and scientific contracts

1. The neural proposal is bounded by `max_neural_update * tanh(delta)`.
2. The raw Jacobi vector is never applied directly. It is RMS-normalized, passed
   through `tanh`, and bounded by a learned positive `max_dc_update` coefficient.
3. The forward pass selects the largest candidate scale in
   `[1, 0.5, 0.25, 0.125, 0]` that does not increase the profiled residual.
4. A rejected proposal is exactly zero in the forward computation. A straight-through
   task gradient lets it learn an admissible proposal instead of receiving zero
   gradient indefinitely.
5. The final voxel value is exactly canonical analytic P1 interpolation of the final
   FEM state.
6. Dense evaluation records per-step Dice, accepted scale, profiled amplitude, and
   measurement-residual ratio. A low accepted-scale rate is reported as a failure
   mode, not hidden.

## Smoke findings

The first implementation applied raw Jacobi updates. It reduced the measurement
residual to 0.56x but caused FEM target MSE near 265 and dense Dice 0.270. This is the
classic failure `better data residual != better reconstruction` under an ill-scaled,
ill-conditioned inverse operator, so that implementation was rejected.

The bounded implementation passed the end-to-end 4/4, two-epoch smoke:

- FEM target MSE: about 0.00314;
- measurement residual ratio: 0.99--1.00;
- dense output remained within about 0.0003 Dice of Stage 1;
- checkpointing, backward, multiview evidence, dense evaluation, and analytic P1 all
  completed successfully;
- 17 combined bridge/iterative tests passed before the straight-through test was
  added; the final iterative suite has 11 passing tests.

The smoke is a correctness test only and is not performance evidence.

## Evaluation protocol for a 0.7 claim

The existing 300-test result has already been inspected, so it cannot be reused for
repeated architecture selection. V2 is selected solely by 300-val dense Dice. The
old test may be reported as exploratory continuity, but a paper-level statement that
test Dice exceeds 0.7 requires a newly generated, frozen simulation seed/split (or an
external dataset) that remains unopened until architecture, loss, threshold, and
checkpoint are fixed. There are exactly 2400/300/300 IDs in the current 3000-sample
partition, so no untouched confirmation subset remains inside it.

Success requires both:

1. final dense Dice above the V1 baseline with useful per-step increments; and
2. non-degenerate trust-region behavior (not all-zero accepted steps) with stable
   profiled residuals.

If V2 does not beat V1 on validation, it is rejected without another test evaluation.
