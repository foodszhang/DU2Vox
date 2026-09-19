# Unified Iterative FEM V4 Development Report

## Method position

V4 implements Scale-Calibrated Cross-Stage Inverse-State Adaptation (SCIA):

```text
measurement y -> frozen Stage-1 FEM x0
              -> shared iterative FEM adaptation
                 [Ax, analytic alpha, raw/SI residuals, A^T evidence,
                  mesh context, multiview evidence, bounded update]
              -> corrected FEM xK
              -> canonical analytic P1
              -> dense voxel prediction
```

There is no voxel decoder, representation-complement head, learned lifting,
observability partition, CQR, RGL, or post-P1 residual.

## V1, V3, and V4

V1 uses the scale-sensitive conventional mismatch

\[
r_{raw}=y-Ax,\qquad g_{raw}=A^T r_{raw}.
\]

It is scale-sensitive because the measured vector is independently normalized to
unit positive maximum while the binary-support FEM state has no corresponding
physical amplitude. Consequently, `y - Ax` retains the arbitrary relative scale of
the normalized measurement and forward response.

V3 analytically profiles the nonnegative nuisance amplitude

\[
\alpha^*=\frac{\max(\langle Ax,y\rangle,0)}{\|Ax\|_2^2+\epsilon},\qquad
r_{SI}=y-\alpha^*Ax.
\]

Its state-gradient evidence is the existing verified implementation
`alpha * A.T @ r_si`; V4 reuses that exact code path rather than redefining the
operator.

V4-A computes both channels from the current state at each of the same three
iterations. A 97-parameter sigmoid mixer starts at 80% SI evidence and learns a
node-wise blend, while the main shared contextual cell receives both unblended
channels. Neural and data-consistency proposals retain the V3 bounds. The V3 best
checkpoint initializes all shape-compatible contextual and view-encoder weights;
the expanded input layers receive an explicit SI-preserving mapping, while new raw
columns start at zero.

## Capacity control

The encoder remains at base width 32 and the FEM hidden width remains 144. V4 is
therefore directly capacity-matched to V3 except for the tiny evidence mixer and a
small number of added input weights. This is a stronger control than creating a new
large generic network: any material gain cannot be explained by a parameter-scale
increase.

## Training and selection

- Development train/validation/test: the unchanged 2400/300/300 split.
- Optimization: AdamW, LR `1e-4`, weight decay `1e-5`, BF16, batch size 4.
- Iterations: 3 shared-weight updates.
- Dense validation: every 5 epochs.
- Checkpoint: maximum 300-sample dense validation Dice.
- Threshold: fixed at 0.5; no calibration.
- V4 variants executed: V4-A only unless explicitly stated below.

## Results

Development results and physics diagnostics are inserted only after validation-based
checkpoint selection is complete. The old 300-sample test is explicitly a
previously observed development test, not an untouched confirmation cohort.
