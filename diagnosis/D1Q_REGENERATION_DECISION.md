# D1-Q Reuse versus Regeneration Decision

Date: 2026-09-15

## Correction to the audit language

Per-case normalization was explicitly requested for D1-Q. It is therefore **not an
implementation error**. The raw arrays remain on disk; normalization happens in the
DU2Vox data loader. Consequently, changing or retaining this normalization never
requires regenerating the 3000 source cases.

The scientific limitation is narrower: a model trained against per-case peak-
normalized GT estimates a normalized field. It can evaluate morphology, spatial
distribution, and relative source amplitudes within a case, but it cannot support a
claim of cross-case absolute fluorescence-yield recovery. Prediction and GT still
must not be independently normalized after reconstruction.

## Issue-by-issue regeneration analysis

| Finding | Actual status | What is invalidated | Minimum remedy | Full GT regeneration required? |
| --- | --- | --- | --- | --- |
| Per-case `b` and GT-peak normalization | Intentional preprocessing | Only absolute cross-case amplitude claims | Keep and describe normalized task, or disable at training time using existing raw arrays | **No** |
| `signal_query_fraction=0.5` without weights | Training-objective ambiguity | Historical objective attribution | Use explicit source/global loss or importance weights | **No** |
| Signal-only/case-range metrics and censored contrast | Evaluation error | Historical headline metrics | Re-evaluate saved aligned predictions on the correct domain where scale permits | **No** |
| Exp A/B parameter attribution | Experiment-description error | Claims that only Q changed or full framework was trained | Relabel old experiments; retrain the intended modules | **No** |
| Pointwise `max` composition | Explicit D1-Q design choice | Compatibility with the later additive-mixture equation | Keep if envelope/max is accepted; otherwise recompute GT, nodes, DE and views | **Yes only if additive sum is required** |
| No random 3-D source orientation | Distribution limitation, not a coding error | Claims about orientation-general anisotropic morphology | Keep if axis-aligned sources are accepted; otherwise sample orientations and recompute | **Yes only if random orientation is required** |
| Minimum separation 1.4 mm | Explicit old distribution choice | Broad non-overlapping multi-source coverage | Keep if overlap is desired; otherwise replace placement distribution | **Yes only if the new separation contract is required** |
| Ellipsoid cutoff compares a dimensionless radius with a value carrying mm | Generator implementation error | Exact source support/size interpretation for ellipsoid-like GT | Accept the saved field as an empirical field, or correct and recompute its GT/DE | **Not logically required for reconstruction; required for a correctly parameterized Gaussian claim** |
| GT/DE field versus MCX field uses different cutoffs plus an MCX-global 1% threshold | Cross-modal data-contract error | Any view-enabled training/evaluation claiming all inputs correspond to one fluorophore field | Regenerate MCX source patterns and projections from the saved GT field, or remove MCX views | **No for GT/DE; yes for MCX views** |
| MCX voxel-center float precision | Boundary-level numerical mismatch | Exact source identity at truncation boundary | Rebuild source pattern; measured effect in the corrected cohort is around float32 epsilon | **No** |
| MCX volume relative path for a dataset outside FMT-SimGen | Execution-path bug affecting the new DU2Vox-resident cohort | MCX launch, not source statistics | Correct path and rerun failed/missing MCX cases | **No** |

## Reuse options

### Option 1: reuse historical D1-Q without MCX views

Use the existing raw GT, nodal coefficients, DE measurements, splits, and source
parameters. Apply the requested per-case normalization explicitly and train a
DE/FEM-only V4. This needs no case regeneration. It cannot test random orientation or
the additive-mixture formulation, and it changes the present view-enabled V4
information contract.

### Option 2: reuse historical D1-Q with repaired views

Keep the saved D1-Q field exactly as the empirical target and rebuild each MCX source
pattern from that saved voxel field (or an exact legacy evaluator), then rerun all
MCX simulations/projections. This avoids resampling GT and DE, but costs essentially
the same MCX wall time as the active new cohort. It still studies max-composed,
axis-aligned legacy fields rather than the requested additive oriented mixture.

### Option 2b: make the saved MCX source the canonical D1-Q GT

This is the cheapest internally consistent reuse path and is preferable to
regenerating MCX if the historical D1-Q source family is accepted.

1. Treat each saved `source-sample_XXXX.bin`, embedded into the full volume using
   `sample_XXXX.json` `Pos` and `Pattern` dimensions, as the authoritative voxel
   fluorescence field. This is the exact source that produced the existing `.jnii`
   and `proj.npz`; JSON tumor parameters alone are less authoritative because the
   historical generator code was not committed/frozen at generation time.
2. Preserve the historical dataset unchanged as forensic evidence. Write a new
   derived dataset root rather than overwriting old GT or measurements.
3. Define the field between MCX voxel centers by fixed trilinear interpolation and
   sample that positive field at FEM nodes. A direct global sampled-L2 `Pi_h` was
   tested and rejected for forward generation: on the smoke case it made 50.2% of
   nodal coefficients negative, negative nodal mass was 31.3% of positive mass, and
   2.17% of DE measurements became negative. The hard-Q method still uses the exact
   certified `Pi_h`; only the physical source-to-FEM generation map differs.
4. Recompute `measurement_b = A @ gt_nodes` using the frozen system matrix.
5. Retain the existing MCX `.jnii`/`proj.npz`, source binary, source JSON, case IDs,
   split, anatomy, optics, views, and noise realization.
6. Audit all 3000 identities and freeze hashes before training. Then retrain every
   formal trainable method under the requested explicit per-case normalization.

This avoids all new MCX simulations. It changes the field definition from an
untruncated analytic Gaussian to the voxel-sampled, cutoff/thresholded field actually
emitted into MCX. It also remains a pointwise-max, axis-aligned legacy distribution,
not the later additive randomly oriented mixture. Paper terminology and FWHM/source
parameter claims must reflect that distinction.

Direct saved-artifact evidence is in
`diagnosis/d1q_gt_mcx_source_contract3000.json`, produced by
`scripts/audit_d1q_gt_mcx_source_contract.py`. Across all 3000 cases:

- exact GT/source support matches: 0;
- exact value matches: 0;
- fraction of GT support absent from MCX source: median 58.43%, mean 61.28%;
- MCX-only support: exactly 0 for every case;
- full-field relative L2 difference: median 2.26%, mean 4.82%, maximum 43.25%;
- common-support relative L2: median `1.51e-6`.

The combination of zero MCX-only support, substantial GT-only low-amplitude support,
and near-identical common-support values is direct evidence of extra MCX-side
cutoff/thresholding, independent of historical source-code reconstruction.

### Option 3: continue the new final cohort

The new GT/DE cohort has already been generated, audited, split, used to train
continuous Stage1, and bridged for train/validation. The active process is not
resampling those 3000 cases; it is producing the missing consistent MCX views. This
option directly realizes K=1--3 additive, unequal-amplitude, randomly oriented
anisotropic mixtures with a frozen separation and source-size distribution.

## Recommendation

Continue Option 3 for the final LPR experiment. The reason is **not** the requested
per-case normalization. The reason is that the user's final benchmark specification
asks what approximation-space-separated refinement contributes on one rich additive,
randomly oriented continuous-field distribution, while view-enabled V4 requires GT,
DE, and MCX evidence from the same field.

Historical D1-Q can be reused for a supplementary normalized-field result or a
val-only transfer diagnostic. It should not be regenerated merely to change
normalization.

If the intended final paper task is instead explicitly per-case normalized and does
not require additive/randomly oriented sources, Option 1 is scientifically viable
and the active MCX workload can be avoided by freezing a DE/FEM-only V4 protocol.
That would be a material change to the already written final benchmark protocol and
must be decided before V4 training, not after validation results are seen.
