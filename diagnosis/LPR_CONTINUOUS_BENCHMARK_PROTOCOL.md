# LPR Continuous-Field Benchmark Protocol

Operational status, restart instructions, remaining gates, and completion checklist
are maintained in `../CONTINUOUS_LPR_EXECUTION_HANDOFF.md`. This protocol remains
the scientific freeze and must not be edited merely to match observed results.

## Freeze status

This protocol was fixed before validation comparison and before any
development-test reconstruction was evaluated. Sealed confirmation data are outside
scope and remain unopened.

## Task and cohort

The sole task variable is the fluorophore field. Anatomy, optical coefficients, DE
operator, noise settings, MCX detector geometry, seven views, and photon budget are
inherited from the established 20k-mesh cohort.

The field is an additive, positive mixture of 1--3 oriented anisotropic Gaussians:

```text
K probabilities:       0.34 / 0.33 / 0.33
component amplitude:   Uniform[0.50, 1.50]
minor sigma:           Uniform[0.35, 0.80] mm
other-axis multiplier: independent Uniform[1.10, 1.80]
orientation:           Haar-uniform SO(3)
truncation:            Mahalanobis radius 3.5
minimum center gap:    4.0 mm
seed:                  20260915
```

The realized 3000-case cohort has 1013/992/995 cases with K=1/2/3. Realized
principal-axis FWHM spans 0.825--3.376 mm. Split sizes are 2400/300/300, with the
600-case holdout pool split deterministically by `(K, depth tier)`.

Raw voxel GT, raw nodal GT, and raw DE measurements are retained. No per-case peak
normalization or independent prediction/GT normalization is allowed for training or
quantitative evaluation. MCX view normalization remains the established
per-view-max view-encoder contract; absolute yield remains available through raw DE
measurements and the FEM state.

## Naming: approximation-space reference state

The FEM nodal target used throughout Stage 2 is the L2 projection of the
continuous ground-truth field onto the P1 approximation space,

```text
x_h* = Pi_h rho* = argmin_c || I_h c - rho* ||^2,
```

obtained by solving the mass-matrix system `(P^T P) x = P^T rho*`. Call it the
**approximation-space reference state** on first use and the **reference state**
thereafter; `projected reference field` is acceptable when the P1-lifted voxel
field `I_h x_h*` is meant.

Three points motivate the wording.

1. `x_h*` is a mathematical reference defining the component of `rho*` that the
   coarse space can represent. It is not a measured or physically guaranteed
   fluorophore distribution.
2. Because `(P^T P)^-1` has negative off-diagonals, `x_h*` carries roughly 50%
   negative nodal coefficients even though the voxel GT, the measurement and the
   prolongation are all non-negative. The P1 basis cannot represent a sharp
   source, so the least-squares optimum overshoots at the source and undershoots
   on its neighbours. Measured on val300 in `diagnosis/lpr_negativity_audit_val300.json`.
3. Scoring `I_h x_h*` therefore reports what the approximation space can reach,
   not what a deployable method achieves.

Do not write `FEM ground truth`, `physical FEM ground truth`, `FEM truth`, or
`true FEM field`: `x_h*` is signed, so presenting it as physical truth is
indefensible.

Where earlier material says `P1 oracle`, prefer **approximation-space upper bound**
in prose and **Projection reference (upper bound)** in tables. "Oracle" implies a
deployable method; this is instead the ceiling of a fixed, non-learned
approximation space.

## Generator identity checks

- voxel and nodal GT call one additive analytic mixture evaluator;
- MCX source generation calls that same evaluator on the identical float32 voxel
  centers;
- `measurement_b` is checked against the saved `A @ gt_nodes` contract;
- rotations, placement, valid-domain mass, amplitudes, and fixed data range are
  checked across all 3000 cases;
- split, generator, case, mesh, system-matrix, frame, source, and projection files
  are cryptographically recorded in the dataset receipt.

## Trainable methods

- **M0 Initial FEM:** continuous-trained Stage 1, `I_h x_h^(0)`.
- **M1 V4 coarse:** continuous-trained three-step V4, `I_h x_h^c`.
- **M2 proposed:** frozen M1 state and terminal latent, residual MLP, exact hard-Q,
  `I_h x_h^c + Q z`.
- **M3 matched unconstrained:** identical decoder inputs, parameter count,
  initialization seed, sample order, epochs, optimizer, and loss as M2, but output
  is `I_h x_h^c + z`.
- **Traditional baseline:** nonnegative graph-Tikhonov FEM inversion; relative
  regularization is selected from `{1e-4, 1e-3, 1e-2, 1e-1}` on validation CCC.

The transferred binary-trained Stage1+V4 stack is evaluated on validation as an
internal development comparison. The formal table uses the continuous-trained
coarse stack.

## Final architecture contract

The M2/M3 decoder input is exactly:

```text
[PE(q), I_h x_h^c(q), lambda(q), G_e, I_h H_h^c(q)]
```

`G_e` is the flattened analytic 4x3 P1 basis-gradient matrix. The terminal latent is
concatenated last. There is no CST, delta-H route, one-ring, direct voxel-side view,
learned interpolation, or voxel-loss gradient path into the frozen coarse state.
The M2 output is always passed through `Q = I - I_h Pi_h`.

## Explicit training objectives

Stage 1 uses per-case source MSE + global nodal MSE + relative L2. V4 draws 50% of
queries uniformly over the complete valid domain and 50% uniformly over nonzero GT
support, but the sampler does not define a loss implicitly:

```text
L_voxel = 1.0 * mean_case MSE(global stratum)
        + 1.0 * mean_case MSE(source stratum)
```

The historical mixed-sample `lambda_projected` term is zero. V4 additionally uses
uniform FEM coefficient supervision and the fixed data-consistency term. M2/M3 run
on the complete canonical domain, so their detail and final-field losses require no
sampling correction.

## Validation selection and test governance

- Stage 1 checkpoint: minimum validation continuous loss.
- V4 checkpoint: maximum dense full-domain validation CCC.
- M2/M3 checkpoint: maximum dense full-domain validation CCC.
- Tikhonov alpha: maximum full-domain validation CCC.

All val300 artifacts and their config/checkpoint hashes must be written to
`diagnosis/lpr_continuous_validation_freeze.json` before generating test bridge
states or evaluating any development-test prediction. Test evaluation scripts reject
unfrozen config/checkpoint pairs.

## Frozen metrics

Headline metrics use all canonical valid voxels and the raw common amplitude scale:

- true masked 3D SSIM, Gaussian 11x11x11 window, sigma 1.5 voxels, fixed data range
  2.0, K1=0.01, K2=0.03;
- PSNR with the same fixed data range 2.0;
- full-domain CCC;
- full-domain relative L2 and MSE.

Auxiliary amplitude/localization metrics score every known GT source inside fixed
GT-defined Mahalanobis regions. A missed source is never removed. They include local
peak relative error, source-integrated relative error, localization error, detection
recall, and all-pair log-contrast error.

Morphology is secondary: per-source Dice@50%-isosurface, corresponding HD95, and
principal-axis FWHM relative error. Global half-max Dice is not used for unequal
multi-source cases.

M2 and M3 additionally report `||Pi_h detail||/(||detail||+eps)` and relative
coarse-state preservation. Q idempotence and annihilation are checked on the same
continuous test predictions.

Primary paired comparisons are M1 vs M0, M2 vs M1, and M2 vs M3 with 10,000 paired
bootstrap draws and 95% confidence intervals. Positive directional differences are
defined to favor the first method.

## Figure selection

Cases are chosen mechanically from proposed-method development-test SSIM ranks:
nearest 25th percentile, median, and 75th percentile, plus the multi-source case
nearest the multi-source median. No visual inspection enters selection. Profiles use
the stored major principal axis through each known center and a fixed 0.35-mm-radius
cylinder.
