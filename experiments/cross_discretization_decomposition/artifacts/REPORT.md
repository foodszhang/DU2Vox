# DU2Vox cross-discretization support-space decomposition

## Decision

**Outcome B — PARTIAL support.** The decomposition is numerically exact and the two
components are both non-negligible. Their spatial correlation and boundary
concentration differ strongly. However, the oracle corrections do not cleanly split
into different reconstruction failure modes: both improve essentially the same metric
families, and the inverse oracle dominates almost every metric. This is insufficient
evidence to justify a new dual-branch Stage-2 architecture now. No Stage-2 model was
designed, modified, trained, or retrained.

The conclusion is strictly about **binary source-support space**. Raw-intensity results
are sensitivity analysis only because Stage 1 was trained against binary nodal support,
whereas raw voxel GT contains source amplitudes above one.

## Exact experiment definition

- Dataset: unchanged DU2Vox/FMT-SimGen 3000-case split (2400/300/300).
- Target: `rho_GT = (gt_voxels > 0.05).astype(float64)`.
- Fixed domain: all 0.2 mm GT voxel centers contained in the complete FEM tetrahedral
  mesh; it is independent of GT, Stage-1 output, ROI, and CQR.
- Domain size: 1,677,645 of 3,952,000 GT centers.
- FEM space: complete 19,990-node P1 mesh. Sparse `P` has 6,710,580 entries.
- Inner product: uniform voxel quadrature, `M = 0.2^3 I = 0.008 I mm^3`.
- Projection: sparse operator form. `(P.T M P)c = P.T M rho`; all 19,990 columns
  are active, sparse LU succeeds with no ridge (`epsilon = 0`). The dense projector
  is never formed.
- Stage-1 input: existing frozen balanced-v2 `coarse_d` for every split.

Runtime was about 75.9 minutes with roughly 1.3 GB resident memory. The cached sparse
operator is reusable.

## Numerical sanity checks

Across all 3000 cases, worst absolute checks were:

- relative decomposition closure: 4.94e-17;
- M-orthogonality cosine: 1.59e-15;
- relative Pythagorean energy error: 5.42e-15.

The decomposition therefore passes the numerical gate by a wide margin.

## Experiment A: energy decomposition

| Quantity | Mean | Median | P05 | P25 | P75 | P95 |
|---|---:|---:|---:|---:|---:|---:|
| inverse fraction | 0.728 | 0.736 | 0.532 | 0.655 | 0.808 | 0.902 |
| representation fraction | 0.272 | 0.264 | 0.098 | 0.192 | 0.345 | 0.468 |

Representation error exceeds 10% of total energy in
94.8% of cases and exceeds 20% in
72.4%. Inverse error exceeds 50% in
97.0%. Thus neither term is generally negligible,
although inverse error is the larger term.

Raw-intensity sensitivity gives a very similar mean representation fraction
(0.270) and median (0.268); this supports robustness
of the energy ratio, but does not authorize an amplitude-space claim.

## Experiments B and D: spatial/statistical roles

| Statistic (mean) | inverse error | representation error |
|---|---:|---:|
| autocorrelation at 0.6 mm | 0.821 | 0.092 |
| absolute gradient mean | 0.0087 | 0.0111 |
| energy within 0.2 mm of GT boundary | 0.218 | 0.693 |
| energy within 0.6 mm of GT boundary | 0.523 | 0.899 |
| fraction near zero (`abs <= 0.01`) | 0.963 | 0.965 |
| excess kurtosis | 207.1 | 230.1 |

The representation component is markedly shorter-range and boundary concentrated:
69.3% of its energy lies within 0.2 mm and 89.9% within 0.6 mm of the true boundary,
versus 21.8% and 52.3% for inverse error. Conversely, inverse error remains strongly
autocorrelated at 0.6 mm (0.821 versus 0.092). This is the strongest empirical support
for distinct roles.

Both components are zero-inflated and extremely heavy-tailed. Generalized-Gaussian
maximum-likelihood fits collapse toward very small shape/scale values because the
sample includes a dominant near-zero mass; the nominal AIC/BIC preference is therefore
not a trustworthy basis for choosing an L1/L2 prior. A spike-and-slab or conditional
model would need separate validation. The current analysis does **not** justify hard-
coding L1 for one branch and L2 for the other.

## Experiment C: oracle consequences

| Reconstruction | Dice | Precision | Recall | Weak recall | Localization mm | HD95 mm | ASSD mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage1 P1 | 0.634 | 0.615 | 0.744 | 0.699 | 1.661 | 2.428 | 0.635 |
| + oracle inverse | 0.938 | 0.965 | 0.913 | 0.905 | 0.090 | 0.208 | 0.093 |
| + oracle representation | 0.740 | 0.709 | 0.853 | 0.812 | 0.885 | 1.314 | 0.242 |
| GT | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |

Inverse correction improves Dice by 0.304, weak recall by
0.205, and HD95 by
2.220 mm. Representation correction also improves those same
metrics (Dice +0.106, weak recall
+0.113, HD95
1.114 mm), rather than isolating a boundary-only failure mode.
It also creates thresholded small components in some cases (mean FP component count
3.456 versus 0.065), reflecting oscillatory
complement corrections. This weakens a clean two-branch interpretation.

## Experiment E: dependence on source properties

- Depth has essentially no association with the fractions (Spearman magnitude 0.011,
  p=0.54), so the proposed “deeper means more inverse fraction” hypothesis is not
  supported here.
- Representation fraction increases modestly with total volume (Spearman 0.275) and
  minimum source volume (0.272). The proposed “smaller means more representation
  fraction” direction is not supported.
- More foci modestly increase inverse fraction (Spearman 0.144).
- Greater inter-source distance increases representation fraction (Spearman 0.194),
  equivalently closer sources increase inverse fraction, consistent with merging or
  inverse ill-conditioning, but the effect is small.
- Intensity association is negligible (Spearman magnitude 0.055).

These effects are statistically detectable at N=3000 but mostly weak. They do not
provide the physically broad separation expected under Outcome A.

## Historical CQR/ROI sensitivity and Experiment F

Historical comparison was computed only where existing CQR precomputes were present
(100 train, 25 validation, 25 test cases). It restricts metrics to those old points but
does not redefine the projection space. It is secondary because that domain depends on
Stage-1 proposals.

Experiment F is skipped. Available alternative mesh directories are not certified as
matched coarse/default/fine discretizations of this exact dataset and lack matched
Stage-1 reconstructions. A valid test requires regenerated shared physics, GT sampling,
forward matrices, and frozen Stage-1 outputs for at least three nested or carefully
comparable meshes while preserving source cases and measurement convention.

## Failure cases and limitations

- `Pi_h rho_GT` is an unconstrained L2 projection and can overshoot or become negative;
  thresholded oracle metrics therefore need interpretation alongside energy metrics.
- Fixed voxel quadrature is appropriate for the uniform center grid, but the projection
  is a sampled voxel-space projection, not the exact continuous FEM mass projection.
- Component matching uses 6-connectivity and a 3 mm matching radius; conclusions rely
  more heavily on Dice/HD95/ASSD and energy identities than on component counts.
- Representative slices are qualitative and cannot replace the 3000-case statistics.
- No mesh-resolution intervention was available, so causality from discretization was
  not experimentally manipulated.

## Final scientific answer

**PARTIAL — interesting but insufficient.** The existing dataset provides strong
evidence that the binary-support error admits a numerically sound decomposition into a
dominant FEM-representable inverse term and a substantial, shorter-range,
boundary-concentrated FEM-complement term. It does **not** yet show that the two oracle
corrections have sufficiently different reconstruction consequences: both improve the
same failure modes and the inverse oracle dominates. Source-property stratification is
also weaker and partly opposite to the proposed hypotheses.

Recommended next step: review these oracle results, then obtain a controlled matched
mesh-resolution experiment and validate constrained/nonnegative oracle variants. Do
not redesign Stage 2 or start a long training run yet.

## Reproducible outputs

- `run_analysis.py`: fixed-domain operator, decomposition, validation, oracle metrics,
  statistics, and base figures.
- `build_report.py`: representative selection, comparison figure, and this report.
- `artifacts/operator_cache/`: sparse P and fixed-domain arrays.
- `artifacts/tables/`: all per-case and summary CSVs.
- `artifacts/figures/`: energy, spatial, oracle, statistical, and dependence figures.
