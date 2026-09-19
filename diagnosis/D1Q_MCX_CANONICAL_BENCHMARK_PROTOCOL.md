# D1-Q MCX-Canonical Continuous-Field Benchmark Protocol

Freeze date: 2026-09-15, before derived-cohort validation training/evaluation.

## Decision

This is the active first-pass continuous-field benchmark. It intentionally prioritizes
a fast, internally consistent reconstruction experiment over the more ambitious
additive randomly oriented Gaussian-mixture distribution. The latter cohort and its
paused MCX job are retained but are not the active experiment.

Historical D1-Q is never modified. A new derived cohort is built at
`data/d1q_mcx_canonical_3k_20k`.

## Canonical field and forward contract

For each case, the exact `source-sample_XXXX.bin` that generated the existing MCX
`.jnii`/`proj.npz` is embedded into the full `(190, 200, 104)` XYZ volume using the
stored MCX JSON `Pos` and pattern dimensions. This saved MCX source is the canonical
voxel GT `rho*`.

Between voxel centers, `rho*` is defined by fixed trilinear interpolation. Nodal GT is
that nonnegative interpolant sampled at the frozen FEM nodes:

```text
x_h* = S_voxel_to_nodes(rho*)
Y = max(A x_h*, 0)
```

> Naming note: in this cohort `x_h*` is the **non-negative point sample**
> `S_voxel_to_nodes(rho*)`. That is deliberately NOT the signed L2 projection
> `Pi_h rho*`, so the two cohorts use different nodal targets on purpose. Call
> this one the *sampled* nodal GT to keep it distinct. Terminological conventions
> for the projected variant are fixed in `LPR_CONTINUOUS_BENCHMARK_PROTOCOL.md`
> under "Naming: approximation-space reference state".

The final clipping matches FMT-SimGen `FEMSolver.forward`. It only removes signed
numerical surface responses from the stored matrix representation.

The source-to-node map `S_voxel_to_nodes` is deliberately not the global sampled-L2
`Pi_h`. A smoke test of direct `Pi_h rho*` produced 50.2% negative nodal coefficients,
negative mass equal to 31.3% of positive nodal mass, and negative DE measurements.
That is unsuitable for a physical fluorophore source. The method's exact hard-Q
operator remains the certified sampled-L2 projection:

```text
Q = I - I_h Pi_h
```

There is no conflict: source discretization and approximation-space projection serve
different purposes.

The historical placement rule did not require the complete Gaussian tail to lie
inside the certified tetrahedral domain. In the derived cohort, outside-domain source
mass has median `8.22e-5`, mean `0.0152`, 95th percentile `0.0836`, and maximum
`0.2920` of total MCX source mass; 104/3000 cases exceed 10%. This first-pass protocol
retains all cases and treats this as inherited forward-domain mismatch rather than
post-hoc excluding difficult cases. It must be reported as a limitation and checked
as a stratification variable. The richer paused cohort was designed to remove this
limitation.

## Reused and regenerated assets

Reused exactly from historical D1-Q:

- case IDs and 2400/300/300 splits;
- source parameters and the historical max-composed, axis-aligned source family;
- actual MCX source binary and JSON;
- existing MCX `.jnii` and seven-view `proj.npz` noise realizations;
- anatomy, mesh, optical coefficients, detector geometry, and system matrix.

Regenerated cheaply in the derived root:

- `gt_voxels.npz` from the embedded MCX source binary;
- `gt_nodes.npy` from fixed positive trilinear sampling;
- `measurement_b.npy` from the frozen forward matrix and nodal GT;
- per-case derivation records and a dataset manifest.

No MCX simulation is performed. Reused large assets are hardlinked, not copied, and
their hashes are frozen in the derived receipt. The historical dataset is read-only.

## Source distribution and allowed claims

The distribution is the historical D1-Q family:

- 1--3 foci;
- continuous Gaussian-derived sphere, ellipsoid, and irregular fields;
- unequal within-case source intensity;
- pointwise-maximum composition;
- no random 3-D orientation.

Do not call it an additive randomly oriented Gaussian mixture. Results answer the
method question on this unified continuous-field distribution only. A later richer
distribution is optional follow-up, not required for this first pass.

## Normalization

Per-case normalization is intentional and requested. Raw derived arrays are retained
on disk. Training loaders apply the frozen normalization explicitly:

- `measurement_b` divided by its own observable per-case maximum;
- voxel/nodal target divided by the same saved `gt_scale.npy`, equal to the
  authoritative voxel GT peak. Stage1 must not independently divide by
  `gt_nodes.max()`, because the coarse mesh may not sample the voxel peak;
- prediction is not independently peak-normalized during evaluation.

The two divisors mean that the normalized arrays are not connected by the unscaled
shared operator. Algebraically, if `b_scale = max(measurement_b)` and `gt_scale` is
the saved voxel peak, the matching operator would be

```text
y_norm = measurement_b / b_scale
x_norm = x / gt_scale
A_norm(case) = (gt_scale / b_scale) A.
y_norm = A_norm(case) x_norm
```

However, `gt_scale / b_scale` contains the target peak and is unavailable for a real
inference case. It may be used only to audit the saved simulation contract, never as
a formal model input or operator multiplier. Formal Stage1 physics evidence profiles
out an unknown positive scale from current `Ax` and `y`, normalizes the adjoint
direction, and weights it by the dimensionless relative residual. V4 must likewise
use a leakage-free observable or profiled formulation. Both using the unscaled
shared `A` as though the normalized equation were exact and injecting the GT-derived
matching multiplier into a claimed inference model are prohibited.

The scale file is part of the frozen data contract, not an optional loader
convenience. Projection-target metadata must record
`normalization_scale_filename: gt_scale.npy`; loaders reject targets whose metadata
omits or disagrees with it. An earlier train/validation projection-target cache used
the peak restricted to the FEM valid domain and is invalid for this protocol. It
must be regenerated from the same saved `gt_scale.npy` before any training resumes.

Metrics therefore describe reconstruction on a normalized within-case concentration
scale. They support relative source-amplitude, spatial, morphology, localization, and
normalized integrated-field claims, but not recovery of cross-case absolute physical
yield. Fixed PSNR/SSIM data range is `1.0` under this protocol.

## Training objective

Signal-aware sampling must use explicit losses rather than letting the sampler hide
the objective:

```text
L_voxel = lambda_global * MSE(global stratum)
        + lambda_source * MSE(source stratum)
```

Start with `lambda_global = lambda_source = 1`. Any change is selected on validation
only and frozen before development-test. Dense validation remains independent of the
query sampler.

## Matched decoder mechanism comparison

After continuous V4 validation selection and train/validation state caching, use
`scripts/materialize_d1q_decoder_pair.py` to create M2/M3 configs. Its gate permits
differences only in `experiment.name` and `model.mode`; all inputs, terminal latent,
decoder dimensions, initialization seed, optimizer, losses, training budget, and
validation selection remain identical. The hard-Q arm declares `model.mode: hard`;
the control declares `model.mode: unconstrained`. Both use the
`approximation_space_separated` input contract, which forbids direct voxel-side views
and requires terminal `Hc` concatenation.

## Methods

- M0: continuously retrained Stage1 initial FEM reconstruction.
- M1: continuously retrained V4 coarse FEM correction.
- M2: proposed terminal-Hc residual decoder with exact hard-Q.
- M3: matched unconstrained decoder, identical except for final Q.
- Traditional baseline: nonnegative graph-Tikhonov FEM inversion.

All trainable methods are retrained or fairly adapted on this derived cohort. Old
binary/normalized checkpoints are transfer diagnostics only.

The first full Stage1 attempt is excluded: hard output clamp made all-zero an
absorbing state and it collapsed by epoch 2. Formal Stage1 uses
`leaky_relu_unbounded` during training and clips only the exported normalized bridge
state to `[0, 1]`.

The final decoder remains exactly:

```text
[PE(q), I_h x_h^c(q), lambda(q), G_e, I_h H_h^c(q)] -> z(q)
rho_hat = I_h x_h^c + Qz
```

No CST, delta-H routing, one-ring, direct voxel-side views, learned interpolation, or
voxel-loss gradient into the frozen coarse state.

## Evaluation and governance

Use the same full valid domain, true fixed-window 3-D SSIM, CCC, relative L2, MSE,
all-GT-source auxiliary metrics, source-resolved morphology, hard-Q leakage, paired
10,000-draw bootstrap, and fixed figure selection defined in
`diagnosis/LPR_CONTINUOUS_BENCHMARK_PROTOCOL.md`, except fixed data range is `1.0`
because the active target scale is per-case normalized.

Do not create or evaluate development-test predictions until all val300 method
artifacts and selected checkpoints/alpha have been written to a new validation freeze
receipt specific to this derived cohort. Sealed confirmation remains prohibited.
