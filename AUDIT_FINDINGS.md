# Independent Forensic Audit of the Historical Continuous-Source Experiment

Date: 2026-09-15  
Auditor: Codex, independent re-analysis from code, checkpoints, logs, split files,
per-case predictions, and sample assets.  
Scope: historical `fmt_simgen_v2_quantitative_gaussian_3k_20k` (D1-Q), Exp A,
and Exp B. The sealed confirmation set was not accessed.

## Executive verdict

The historical D1-Q experiment is **not an admissible final continuous-field LPR
benchmark**. Some infrastructure is reusable, but the experiment does not implement
the claimed final architecture and its main quantitative metrics do not use the full
valid reconstruction domain.

The strongest provenance finding is that Exp B retrained the V4 FEM corrector and
view encoder; it did not train a voxel/Q decoder. The saved model inventory explicitly
contains `voxel_head: 0`, and the per-case artifacts contain only FEM-interpolated
Stage 1 and V4 iteration outputs. Consequently, neither Exp A nor Exp B is an
experiment of the complete approximation-space-separated sequential-refinement
framework.

The D1-Q source generator also violates its own configuration comments: nodal/voxel
GT uses a nominal 4-sigma support, MCX uses 3 sigma plus a global 1% threshold, and
the ellipsoid nodal/voxel cutoff mixes dimensionless and millimetre quantities.
Multiple foci are composed by pointwise maximum and have no random 3-D orientation;
these two properties are intentional in the D1-Q config, so they are limitations
relative to the later additive oriented-mixture benchmark rather than implementation
errors. Per-case measurement/GT normalization was also an intentional requested D1-Q
task definition. It is not a generator error, but it changes the estimand from
cross-case absolute concentration to normalized within-case field reconstruction.

Disposition for the later raw-amplitude additive-mixture benchmark: preserve D1-Q as
historical forensic evidence and use one audited, additive, randomly oriented
anisotropic Gaussian-mixture cohort. This disposition follows the new benchmark
contract; it is not justified by the intentional D1-Q normalization alone.

## Evidence identity

- Dataset: `/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_quantitative_gaussian_3k_20k`
- Mesh SHA256: `718cb70f0b12c3c8f5e7f10d9e216295efc79ff3221602b934ba04c137f34c64`
- Current frame-manifest SHA256: `a89d36a52cc0d33e4764f6a65056a2dfa7fbe9fa55ed3458c0beb84d5ba31ee3`
- Stage 1 checkpoint SHA256: `f3c8fb07007b822312ada67a8f3325ff49a4f2344fa8ebf110e954ba9a6e73d3`
- Exp B checkpoint SHA256: `9a3333b0437f27161eeef3cdcd197a458484f48ebe8bbbda89234e98d8411992`
- Train/val/test split SHA256: `870ec2c7...`, `7587bdbf...`, `a35ad349...`
- Split sizes are 2400/300/300, IDs are unique, and all three pairwise intersections
  are empty.
- All 3000 declared cases contain `measurement_b.npy`, `gt_nodes.npy`,
  `gt_voxels.npz`, `tumor_params.json`, and `proj.npz`.
- All 3000 cases have bridge outputs and continuous projection targets in their
  declared split mappings.
- The generator and DU2Vox working trees are dirty and the dataset contains no
  immutable per-case manifest of source/GT/measurement/view hashes. Therefore exact
  regeneration provenance is **not certified**, despite current asset completeness.

Machine-readable raw re-analysis is in
`diagnosis/d1q_independent_forensic_audit.json`; the audit program is
`scripts/audit_d1q_forensics.py`.

## 1. What Exp A and Exp B actually trained

| Claim or implementation fact | Verdict | Direct evidence |
| --- | --- | --- |
| Initial Stage 1 was trained on binary D0 and frozen for D1-Q | **VALID** | `logs/d1q_bridge_val_train.log:9` loads `runs/stage1_fmt_simgen_v2_3k_20k_balanced_v2_eval/checkpoints/best.pth` (epoch 71). `configs/stage1/fmt_simgen_v2_quantitative_gaussian_3k_20k_bridge.yaml` is inference-only and retains `binarize_gt: true`. |
| Exp A transferred the binary-trained V4 without D1-Q retraining | **VALID** | `diagnosis/d1q_expA_d0trained_test300_contgt.json` identifies the D0 V4 checkpoint at epoch 15. Its checkpoint embeds D0 dataset and projection-target paths. |
| Exp A and B used the same frozen coarse output and differed only in Q | **INVALID** | Exp B checkpoint is `unified_dual_evidence_fem_v4`, epoch 25. `runs/d1q_expB_continuous/model_info.json:3-8` records 868,166 trained V4/view parameters and `voxel_head: 0`. The raw artifacts contain `stage1_fem`, `step1_fem`, `step2_fem`, and `step3_fem`, with no voxel proposal or `Qz`. |
| Exp B retrained only the voxel/Q decoder | **INVALID** | No voxel decoder is instantiated by `scripts/train_iterative_fem_corrector.py`; `runs/d1q_expB_continuous/model_info.json:8` is conclusive (`voxel_head: 0`). |
| The coarse FEM branch was retrained on continuous data | **VALID for Exp B; INVALID for Exp A** | Exp B trained all V4 corrector and view-encoder parameters on 2400 D1-Q cases, initialized from the binary D0 V3 checkpoint (`model_info.json:13-16`). Stage 1 remained binary-trained in both experiments. |
| Exp B terminal latent came from a continuous V4 checkpoint | **PARTIALLY VALID** | V4 internally forms a terminal hidden state (`iterative_fem_corrector.py:748,868-869`), but Exp B evaluation did not export or consume it in a voxel decoder. It is therefore only an internal V4 state, not evidence for terminal-latent voxel completion. |
| Historical D1-Q is the full framework trained for continuous reconstruction | **INVALID** | It is a coarse-FEM adaptation experiment only. The formal hard-Q decoder, terminal latent concatenation, and coarse-preservation measurement are absent. |

The accurate historical description is:

> Exp A transfers binary-trained Stage 1 and binary-trained V4 to D1-Q. Exp B keeps
> the same binary-trained Stage 1 coarse input but retrains the V4 FEM corrector and
> view encoder under a continuous, per-sample-normalized supervision contract. No
> complementary voxel branch is trained or evaluated.

## 2. `signal_query_fraction = 0.5`

Verdict: **PARTIALLY VALID as an implementation of stratified sampling; INVALID as
an implicit scientific objective.**

The sampler draws 4096 of 8192 queries uniformly from nonzero GT positions and 4096
uniformly from the complete valid domain (`error_structured_dataset.py:170-180`). If
the signal occupies fraction `f`, a signal voxel has sampling weight relative to a
background voxel of `1 + 1/f`. On historical test300, the audited mean `f` is
0.005868, so the average relative weighting is about 171.4:1, not 1:1.

There is no inverse-probability/importance weight in the batch. Voxel MSE is the
ordinary sample mean (`train_iterative_fem_corrector.py:207`), and the other query
losses use the same biased sample. Thus the voxel objective changed from a uniform
valid-domain objective to a mixture objective:

```text
0.5 * E_uniform(signal positions) + 0.5 * E_uniform(valid domain)
```

This deliberately emphasizes peak/profile fitting and downweights background. It can
alter peak, integrated mass, support width, and background-pedestal bias. It is not a
neutral data-loader optimization.

Dense V4 prediction is independent of the sampled query set: dense evaluation applies
P1 to all 1,677,645 valid voxel centers. However, the later quantitative scorer then
restricts most metrics to a GT-defined signal mask, so the reported evaluation is not
a full-domain evaluation.

Required fix: use an explicit two-term source/global objective with separately
computed reductions and frozen validation-selected weights. This makes the intended
objective auditable and avoids fragile importance weights for several nonlinear
losses.

## 3. Continuous GT and forward generation

| Item | Verdict | Evidence |
| --- | --- | --- |
| Positive continuous values and 1-3 unequal-amplitude foci | **VALID** | Raw audit: K counts 832/1072/1096; 6264 amplitudes span 0.60009-1.79978. |
| Isotropic and anisotropic source sizes | **PARTIALLY VALID** | Sphere and axis-aligned ellipsoid/irregular parameters exist, with base sigma 0.26732-0.79987 mm. There is no rotation/orientation field in any of 6264 foci. |
| Random anisotropic Gaussian mixture | **INVALID** | `TumorSample.evaluate` combines foci using `np.maximum` (`tumor_generator.py:270`), not the additive mixture in the target formulation. |
| Same analytic field for nodal GT, voxel GT, and MCX views | **INVALID** | Nodal/voxel sphere support is 4 sigma (`tumor_generator.py:152`); MCX bounding/evaluation uses 3 sigma (`mcx_source.py:163,271`) and then applies a global 1% threshold (`mcx_source.py:109-113`). Sample 0000 voxel GT contains values down to 0.0306% of its sampled peak, proving it was not subjected to the MCX 1% truncation. |
| Ellipsoid Gaussian cutoff is dimensionally correct | **INVALID** | `tumor_generator.py:176,186` calculates dimensionless `dist2` but compares it with `(4 * max(sigma_mm))**2`. MCX instead compares dimensionless distance against a cutoff expressed in mm (`mcx_source.py:281`). Both are dimensionally inconsistent unless the maximum sigma happens to equal 1 mm. |
| FEM forward measurement matches the approximation-space formulation | **VALID in form** | Builder samples the analytic field at FEM nodes and calls the FEM solver; `FEMSolver.forward` computes the fixed linear FEM forward response. This is a legitimate `rho -> nodal P1 coefficients -> Y` simulation contract, but it is not identical to the differently truncated MCX view field. |
| Requested per-case normalization is implemented | **VALID, with a scope limitation** | `normalize_b: true` divides each measurement by its own maximum (`error_structured_dataset.py:234-236`) and `per_sample_peak` divides each GT by its sampled peak (`gt_io.py:207-211`). This matches the requested normalized D1-Q task and does not require data regeneration. It does mean Exp B cannot be used to claim recovery of cross-case absolute physical yield; multiplying an output by held-out GT peak would be oracle rescaling. Relative amplitudes within a case remain represented. |
| Split leakage | **VALID (none detected)** | 2400/300/300 split IDs are unique and pairwise disjoint. No duplicate ID leakage was found. |
| Immutable case matching/provenance | **PARTIALLY VALID** | IDs and asset coverage match, but no per-case hash ledger binds tumor parameters, nodal/voxel GT, DE measurement, MCX views, bridge state, and projection target. Dirty uncommitted generator code prevents a certified regeneration claim. |

Because the nodal/voxel field and MCX view field differ, a view-enabled method receives
evidence generated from a different source support than its DE measurement and GT.
This is a data-contract error, not just a documentation discrepancy.

## 4. Metric audit

| Metric | Verdict | Actual historical domain/definition and issue |
| --- | --- | --- |
| PSNR | **INVALID as a main quantitative metric** | `eval_quantitative_metrics.py:136,159-160` computes it only where GT is above 1% of its peak. The range is the current case's GT peak, hence case-dependent unless GT was oracle-normalized. Prediction is not clamped, which is acceptable, but the range and domain are not fixed. |
| 3-D SSIM | **INVALID / absent** | No SSIM implementation exists in the historical evaluator. |
| Pearson | **INVALID as reported main metric** | Computed only on the GT signal mask (`p_sig/g_sig`, lines 136,161), not the full valid domain. |
| CCC | **INVALID as reported main metric** | Same GT signal-only restriction (`lines 136,162`). The CCC formula itself is standard. |
| Relative L2 | **INVALID as reported main metric** | Same signal-only restriction (`lines 163-166`). Independent min-max normalization is not used, but oracle per-sample GT peak normalization is used for Exp B. |
| MSE | **INVALID as reported main metric** | Same signal-only restriction (`lines 163-167`), so background artifacts are omitted. |
| Integrated fluorescence | **INVALIDly labelled** | `integrated_ratio` sums only the GT >1%-peak mask (`lines 169-170`), not the full valid volume. It is a GT-support-restricted signed ratio. |
| Peak recovery | **PARTIALLY VALID** | A local per-focus maximum avoids a remote global artifact, but focus masks use nearest-focus 3-times-largest-axis spheres, not oriented Mahalanobis regions. Negative predictions and overlapping attribution are not fully specified. |
| Contrast | **INVALID** | When the predicted weak-source peak is below 1% of the predicted scale, the pair is counted and then excluded from the error (`lines 217-230`). The resulting error is conditioned on detection and can reverse method rankings. |
| Dice at 50%-isosurface | **PARTIALLY VALID as morphology only** | Prediction and GT are each thresholded at half of their own global peak (`lines 145-150`), already erasing amplitude. In unequal multi-source cases, a source below half the global peak can disappear from GT and/or prediction. It is unsuitable as a headline metric. |
| Centroid/localization | **PARTIALLY VALID** | Uses clipped positive intensities but only within the GT signal mask, so remote false-positive mass is invisible. |

Required replacement protocol: full canonical valid-domain PSNR/CCC/Pearson/relative
L2/MSE; fixed-range masked true 3-D SSIM with one frozen Gaussian window; full-domain
integrated fluorescence; all-GT-source local amplitude errors plus explicit detection
recall; and Dice/HD95/FWHM only as secondary morphology metrics. Prediction and GT
must remain on the original shared intensity scale.

## 5. Raw test300 re-analysis of the mass claim

The published `integrated_ratio = 5.9536` is reproducible, but only under the
historical convention: per-sample-peak-normalized GT, unchanged binary-trained
prediction, and a GT >1%-peak signal mask. It is not a full-volume physical mass
ratio.

Directly from the saved Exp A `step3_fem` arrays:

- signal-mask signed ratio: 5.9536;
- positive background mass / total GT mass: 5.9335;
- full-domain nonnegative predicted mass / GT mass: 11.7944;
- full-domain signed ratio: -89.7558;
- negative mass magnitude / GT mass is very large (the raw output is unclamped);
- mean global peak ratio: 1.6856;
- mean signal occupancy: 0.5868% of the valid domain.

Therefore the value 5.95 does not isolate a binary-shape mechanism. Peak overshoot,
in-source broadening/support dilation, positive background pedestal, and extensive
negative background all coexist. The signed and nonnegative full-domain answers even
have opposite qualitative interpretations.

For Exp B on its normalized scale, the historical signal-mask ratio is 1.5474, but
the full-domain nonnegative mass ratio is 10.7157 and positive background alone is
9.2015 times total GT mass. Its mean full-domain relative L2 is 1.1148, whereas the
historical signal-only metric is much smaller. This demonstrates that the previous
metric domain concealed the dominant background error.

## 6. Profile and FWHM audit

Verdict: **PARTIALLY VALID**.

The revised implementation correctly replaced whole-slab averaging with a fixed
1-mm-radius cylinder (`gaussian_profile_analysis.py:40,51-54`), which removes the
earlier slab-dilution bug. It also uses known source centers.

However, it evaluates only global x/y/z axes (`line 31`), not the source's principal
axis; the generator has no orientation, and irregular sources do not have a single
analytic principal-axis Gaussian profile. FWHM is estimated from occupied bin centers
without crossing interpolation (`lines 70-84`). Figures are simply the first cases
encountered (`lines 170-171`), not a preregistered percentile rule.

The quoted 1.55x value is the median of 1894 per-axis ratios and is numerically present
(`1.54545`); the mean is 2.1841. Thus **the arithmetic is valid under the old extraction
rule, but the scientific generalization is not**.

## 7. Hard-Q contract

Verdict for historical D1-Q: **INVALID / not measured because the branch is absent**.

Neither Exp A nor Exp B produced `z`, `Qz`, coarse leakage, or preservation error.
Consequently no historical continuous test300 result can support `Q^2 ~= Q`,
`Pi_h Q ~= 0`, or `Pi_h rho_hat ~= x_h^c`. D0 algebra tests cannot substitute for a
formal end-to-end continuous test measurement. These diagnostics must be computed for
both hard-Q and matched unconstrained models in Phase II.

The existing binary hard-Q implementation is also not the user-specified final model:
`scripts/train_complement_voxel_detail.py:245-250` hard-thresholds GT, and its decoder
features do not include terminal `H_h^c`. It can supply the exact sparse-Q operator and
matched decoder skeleton, but not the final continuous training contract.

## 8. Verdict on prior scientific claims

1. **“Integrated ratio 5.95 proves the model treats Gaussian as binary.” — INVALID.**
   The number is a GT-mask-restricted, oracle-normalized signed ratio. Raw outputs show
   simultaneous peak overshoot, dilation/broadening, positive background, and much
   larger negative background. It does not identify one cause.

2. **“Mass overestimation is caused by binary-like broadening.” — PARTIALLY VALID as
   a hypothesis, INVALID as an attribution.** Broadening is observed, but peak
   overshoot, support dilation, and pedestal are independently large. A causal
   decomposition was not performed.

3. **“The reversed contrast result reflects better contrast recovery.” — INVALID.**
   Missed weak-source pairs were excluded. The conditional metric is subject to
   detection censoring and cannot support the comparison.

4. **“FWHM is 1.55x.” — PARTIALLY VALID.** It is the old median; the old mean is
   2.18x. The extraction is fixed-axis, binned, and not a principal-axis protocol.

5. **“A/B differ only through the Q decoder.” — INVALID.** There is no Q decoder.
   Exp B retrained the V4 FEM corrector and view encoder; Exp A did not.

6. **“The observed A/B improvement is final voxel completion over a fixed coarse
   state.” — INVALID.** It is predominantly a comparison of transferred versus
   continuous-trained coarse FEM correction. Coarse and final attribution were
   conflated.

## 9. Reusable versus rejected artifacts

Reusable after tests and explicit reconfiguration:

- canonical analytic P1 matrix and sampled-L2 `Pi_h` cache, pinned by mesh hash;
- exact FP64 sparse hard-Q implementation and matched unconstrained decoder skeleton;
- V4 iterative FEM corrector implementation and optional terminal-hidden export;
- split size convention and non-sealed development-test governance;
- sparse lossless voxel GT storage.

Must be replaced or materially corrected:

- D1-Q source distribution and all generated D1-Q data;
- separate GT/MCX Gaussian evaluators and their inconsistent cutoffs;
- max composition and absent 3-D orientation;
- per-sample measurement/GT peak normalization for quantitative amplitude recovery;
- implicit sampler-defined loss;
- Dice-based V4 checkpoint selection for the continuous task;
- signal-only quantitative evaluator, contrast censoring, and fixed-axis profile
  selection;
- binary-only hard-Q target path and decoder without terminal `H_h^c`.

## 10. Phase-II freeze decision

The historical dataset fails the requested continuous benchmark contract and will not
be reused for the LPR main table. Phase II will use one 2400/300/300 dataset with:

- additive `K=1..3` anisotropic Gaussian mixtures;
- uniformly sampled SO(3) orientation stored per focus;
- one shared evaluator for FEM nodes, voxel centers, and MCX pattern values;
- an explicit, dimensionless Mahalanobis truncation rule;
- frozen amplitude, sigma/FWHM, separation, anatomy, optics, noise, and view settings;
- unnormalized measurement and GT amplitude, with a fixed evaluation data range;
- per-case cryptographic hashes binding parameters, GT, measurement, and views;
- an explicit source/global loss rather than an implicit sampling objective.

Formal M0/M1/M2/M3 and baselines will be selected on validation continuous-field
metrics and evaluated once on the existing development-test role. No sealed
confirmation data is authorized or accessed by this audit.

## 11. Phase-II user-selected D1-Q first pass and Stage1 forward audit

This is a post-audit execution addendum, not a revision of the Phase-I verdict. The
user subsequently selected a cheaper first pass that reuses the saved D1-Q MCX
source distribution and requires per-case normalization. It is not the richer
additive/oriented distribution proposed above and must not be described as such.

- **System matrix / raw saved forward contract — VALID.** On val300,
  `max(A x_raw, 0)` matches saved `measurement_b.npy` with median relative L2
  `9.85e-8` and maximum `3.03e-7`. Case alignment and the raw matrix are not the
  cause of poor Stage1 reconstruction.
- **Fixed-A equation after independent per-case normalization — INVALID as an exact
  physics equation.** `b/b_max` versus `A(x/gt_scale)` has median relative L2
  `0.6812` and maximum `0.9994`. The matching algebraic multiplier
  `gt_scale/b_max` restores the saved equation but contains GT information and is
  prohibited as a formal inference input.
- **Legacy raw Stage1 third channel — PARTIALLY VALID.** It is leakage-free and can
  function as a learned normalized-adjoint feature, but it cannot be called a
  calibrated data-consistency gradient under this normalization.
- **Leakage-free profiled Stage1 evidence — VALID implementation, empirically worse
  at the bounded epoch10 comparison.** Its val300 CCC/relative L2 were
  `0.4076/0.8545`, versus `0.4355/0.8072` for raw-feature v2. It improved the
  high-intensity-ratio multi-source contrast log error (`5.6720 -> 4.2050`) and
  detection recall (`0.1865 -> 0.2030`) but worsened the overall field and local
  integrated error.
- **Stage1 handles unequal source strength adequately — INVALID at the current
  checkpoints.** In the 101 val cases with multi-source parameter intensity ratio
  above 1.5, raw-feature v2 had CCC `0.4001`, relative L2 `0.8451`, all-source peak
  relative error `0.7632`, integrated relative error `0.7406`, and detection recall
  `0.1865`. All GT sources were included; there was no detection censoring.
- **Poor Stage1 source recovery is mainly an unavoidable FEM approximation limit —
  INVALID.** The matched nodal-GT P1 oracle reaches CCC `0.8983`, relative L2
  `0.3797`, mass ratio `1.0036`, per-source detection recall `0.8206`, and contrast
  log error `0.6067`. For the same 101 high-intensity-ratio cases its CCC is
  `0.8841` and recall `0.7904`. P1 does incur a mean source-peak error of `0.3395`,
  but the much larger learned Stage1 error is predominantly inverse reconstruction,
  not representation alone.

Evidence files:

- `diagnosis/d1q_mcx_canonical_stage1_forward_contract_val300.json`
- `diagnosis/d1q_mcx_canonical_stage1_v2_epoch10_val300_continuous_sources.json`
- `diagnosis/d1q_mcx_canonical_stage1_profiled_epoch10_val300_continuous_sources.json`
- `diagnosis/d1q_mcx_canonical_gt_nodes_p1_oracle_val300_continuous_sources.json`

## 12. Lumped-volume Stage1 val300 audit (epoch29 versus epoch30)

This audit reused the already exported physical Stage1 states and did not change or
train the network. It replaced equal-node aggregation with the P1 lumped mass
diagonal `m_i = sum_{T contains i} |T|/4` on the certified 19,990-node mesh.

- **“Epoch30 is the better continuous-field checkpoint because it has lower
  validation loss.” — INVALID.** Epoch30 has a slightly lower median weighted
  relative L2 (`0.7629` versus `0.7725`) but a severe under-amplitude bias (median
  mass ratio `0.5092` versus `1.0244`). Epoch29 has higher weighted CCC in `85.7%`
  of paired cases, lower source-composition error in `91.3%`, and lower weak-source
  error in `65.3%`.
- **“Current Stage1 failure is mainly a single scalar amplitude mismatch.” —
  INVALID.** After the best nonnegative per-case global rescaling, median shape
  error remains `0.7195` at epoch29, only `6.9%` below its median wRelL2. Median COM
  error is `2.2305 mm`. Correcting one global gain cannot repair the dominant
  spatial error.
- **“The main spatial failure can be attributed only to close-source overlap.” —
  INVALID.** Distance and depth strata are non-monotonic. The more defensible result
  is that source count, morphology/localization, and source composition matter:
  epoch29 median shape error rises from `0.565` for one source to `0.802` for three,
  and COM error rises from `1.608` to `3.344 mm`.
- **“Weak-source recovery is adequate once FEM representation is accounted for.” —
  INVALID.** Across all 219 multi-source cases, epoch29 median source-composition
  error is `0.2367` and weak-source error is `0.7808`. The matched P1-oracle
  source-attribution floors are only `0.0072` and `0.0215`, respectively.
- **P1 oracle under the lumped-node protocol — VALID only as an identity control.**
  Its whole-field errors are exactly zero because its coefficients are normalized
  `gt_nodes`. This is not the earlier cross-discretization P1-versus-voxel oracle.

Verdict: current Stage1 primarily fails in spatial field reconstruction—localization,
morphology, multi-source composition, and weak-source allocation—with an additional
checkpoint-dependent global amplitude problem. Epoch29 is the safer current
development checkpoint, but neither checkpoint is adequate as a final continuous
Stage1 result.

Complete evidence: `diagnosis/stage1_continuous_fem_volume_audit/REPORT.md`, with a
300-row case CSV, full stratified summaries, protocol JSON, and four mechanically
selected common-scale MIP figures.
