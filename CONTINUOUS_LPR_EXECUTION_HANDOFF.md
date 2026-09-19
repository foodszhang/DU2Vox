# Continuous-Field LPR Experiment: Execution Handoff

Last updated: 2026-09-16 Asia/Hong_Kong

> **Active decision update:** the user selected the cheaper MCX-canonical D1-Q
> first-pass benchmark. Its frozen protocol is
> `diagnosis/D1Q_MCX_CANONICAL_BENCHMARK_PROTOCOL.md`. The new additive/oriented
> cohort's `lpr_mcx` process remains deliberately stopped. A separate tmux session
> `d1q_mcx_canonical` is building the derived dataset without running MCX; its log is
> `logs/d1q_mcx_canonical_build.log`. After that build, downstream configs and gates
> must target `data/d1q_mcx_canonical_3k_20k`, not the paused richer cohort.
> The derived 3000-case build completed. Its manifest records inherited MCX source
> mass outside the certified FEM domain (median 0.0082%, mean 1.52%, 104 cases above
> 10%); the active first-pass protocol retains and stratifies these cases rather than
> cherry-picking them away.
>
> The derived build and its independent 3000-case audit are complete:
> `diagnosis/d1q_mcx_canonical_dataset_audit3000.json`. Dataset identity is frozen in
> `diagnosis/d1q_mcx_canonical_dataset_freeze.json`. A shared per-case
> `gt_scale.npy` equal to the canonical voxel peak is used across Stage1 and voxel
> targets; Stage1 must not renormalize by its lower nodal peak.
>
> **2026-09-16 execution status:** the user resumed D1-Q validation experiments. The
> richer-cohort MCX process and its children have since been **terminated outright**
> (the old `PID 2203122` / MCX wrapper `PID 2251469` no longer exist; see "Live
> long-running process" for what replaced them). Development-test and sealed
> confirmation remain closed.
>
> The 2700 existing train/val projection targets under
> `precomputed/d1q_mcx_canonical_projection_targets` are now marked **invalid for
> reuse**. They divided by the peak restricted to the FEM valid domain, whereas the
> frozen benchmark requires the saved full-volume `gt_scale.npy`. Code now treats
> that filename as an explicit metadata contract and rejects this old cache. Do not
> delete it; regenerate it with `--normalization-scale-filename gt_scale.npy` only
> after experiments are resumed. Test targets remain deliberately absent.
>
> Stage1 smoke `runs/stage1_d1q_mcx_canonical_smoke` completed 2 epochs and reduced
> val loss from 10.6782 to 4.6764. It is diagnostic only, never a formal candidate.
> The first full Stage1 attempt used hard `clamp`, collapsed to all-zero output by
> epoch 2, and early-stopped at epoch 21 (`val_loss=0.287756`). It is invalid and
> must never be selected. The config now uses `leaky_relu_unbounded`, and the train
> script rejects continuous-field training with hard clamp/ReLU. Per the user's
> 2026-09-16 instruction, no replacement experiment had been launched during the
> code-only pause. On resumption, a first launch was stopped before epoch 1 because
> it reused the contaminated v1 run directory. The formal replacement uses the new
> clean experiment name `stage1_d1q_mcx_canonical_2400_v2`; the old directory is
> retained as failed evidence and must never be selected.

When the user resumes preprocessing, regenerate the invalid train/val cache into a
new directory (or explicitly overwrite only after preserving the old evidence) with:

```bash
rtk uv run python scripts/precompute_error_structured_targets.py \
  --dataset-root data/d1q_mcx_canonical_3k_20k \
  --shared-dir /home/foods/pro/FMT-SimGen/output/shared_mesh_20k \
  --operator-cache experiments/cross_discretization_decomposition/artifacts/operator_cache \
  --output-dir precomputed/d1q_mcx_canonical_projection_targets_v2 \
  --splits train val \
  --gt-mode continuous \
  --normalize-gt per_sample_peak \
  --normalization-scale-filename gt_scale.npy \
  --allow-stale-frame-manifest
```

The D1-Q V4 config points to the `_v2` directory. All 2700 train/validation targets
now exist and passed full recomputation audit. Do not generate test targets before
validation freeze.

The independent full target audit is implemented in
`scripts/audit_projection_target_contract.py`. Run it on `_v2` after precomputation;
it recomputes every requested `Pi_h` coefficient array from GT using the saved scale,
checks metadata scales and shapes, and writes a machine-readable receipt. It has not
been run during this pause.

After V4 is selected on validation and its train/val coarse states plus terminal
latents are cached, materialize the matched hard-Q/unconstrained decoder pair with
`scripts/materialize_d1q_decoder_pair.py`. The script validates V4 checkpoint type,
epoch, embedded config, hashes, and projection-target scale metadata, then writes two
configs that may differ only by experiment name and the hard-Q mode switch. It does
not train or evaluate anything.

Code-only verification completed on 2026-09-16: targeted Ruff checks passed and 43
unit/contract tests passed, covering explicit normalization scale, continuous-field
metrics, dataset/freeze governance, iterative V4 contracts, exact hard-Q, and matched
decoder-pair equality. `black` is not installed in the environment; `ruff format`
was used successfully instead. No training, inference, evaluation, target
precomputation, or data generation was launched during this code pass.

On experiment resumption, an additional normalization issue was found:
`measurement_b / b_max` and `x / gt_scale` cannot be connected by the unscaled
shared `A`. Supplying the exact multiplier `gt_scale / b_max` repairs the equation
numerically but leaks a GT-derived quantity at inference, so that path is diagnostic
only and is not authorized for the formal model. Stage1 now has a leakage-free
`profiled_normalized` evidence mode that analytically eliminates the unknown positive
relative scale from the current prediction and measurement. V4 must receive the
same leakage audit before training; do not use its currently implemented GT-scale
operator path as a paper result.

Experiment execution resumed on 2026-09-16. The `_v2` projection cache now contains
all 2700 train/val targets and passed independent full recomputation audit:
`diagnosis/d1q_mcx_canonical_projection_targets_v2_audit2700.json` (`status=passed`,
maximum relative L2 `5.03e-8`, maximum absolute error `5.96e-8`). The fixed-A v2
Stage1 run was stopped after epoch 12. Its third channel violates the normalized
forward equation if interpreted as an exact residual, but remains leakage-free as a
learned normalized-adjoint feature. It is therefore being continued only as a
bounded empirical Stage1 candidate, with that interpretation explicitly restricted.

Two 50-case profiled-evidence smoke runs exposed an all-zero training attractor. The
revised evidence now uses an amplitude-invariant adjoint direction multiplied by the
dimensionless relative measurement residual and fixed node RMS 0.05. The scale was
chosen to match the measured legacy cold-start range (val300 median RMS 0.063), not
by test-set tuning. The next bounded experiment is 10 epochs on train2400/val300 in
`runs/stage1_d1q_mcx_canonical_2400_profiled_v1`. At epoch 10 it reached val loss
0.2407 and underperformed the raw-feature v2 epoch10 value 0.2224. Full val300
continuous evaluation also favored v2 (CCC 0.4355 vs 0.4076; relative L2 0.8072 vs
0.8545). The profiled arm is retained as a failed leakage-free control.

Source-resolved val300 evaluation includes every GT source in fixed GT-defined ROIs.
For the 101 multi-source cases with parameter intensity ratio above 1.5, raw-feature
v2 achieved CCC 0.4001, relative L2 0.8451, per-source detection recall 0.1865, and
contrast log error 5.6720. Profiled evidence achieved 0.3729, 0.8888, 0.2030, and
4.2050 respectively. Stage1 therefore has a confirmed weak/unequal-source amplitude
failure despite modest profiled contrast improvement. The raw-feature v2 has resumed
from epoch 12 to a bounded epoch30 in tmux `d1q_stage1_raw_resume`; do not extend it
beyond that bound without reviewing validation results.

The normalized nodal-GT approximation-space upper bound (the projection reference,
formerly called the "P1 oracle"; see the naming section of
`diagnosis/LPR_CONTINUOUS_BENCHMARK_PROTOCOL.md`) was evaluated with the identical
val300 protocol.
It reaches CCC 0.8983, relative L2 0.3797, mass ratio 1.0036, all-source peak error
0.3395, detection recall 0.8206, and contrast log error 0.6067. In the same 101
high-intensity-ratio cases it reaches CCC 0.8841 and recall 0.7904. Thus P1/FEM
representation explains part of peak attenuation, but most current Stage1 failure is
inverse reconstruction rather than an unavoidable approximation-space ceiling.

### Stage1 pause for user decision (2026-09-16)

The raw-feature run resumed from epoch 12 and completed the user-approved bounded
endpoint at epoch 30. It is now stopped; do not resume until the user decides the
Stage1 policy. No V4, decoder, test300, or sealed-confirmation action was launched.

Two val300 checkpoints were exported and evaluated with all-source metrics:

| checkpoint | PSNR | CCC | RelL2 | mass ratio | source peak err | source recall | contrast log err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| epoch30 best val loss | 39.013 | 0.4570 | 0.7838 | 0.6255 | 0.7250 | 0.2444 | 6.6690 |
| epoch29 best Dice@0.5 | 39.218 | 0.5374 | 0.7775 | 1.3373 (median 1.0275) | 0.6698 | 0.3428 | 4.4858 |

Epoch29 is better on nearly every continuous/source metric but has a heavy-tailed
mass-overestimation mean. Epoch30 has lower training/validation loss yet is strongly
under-amplitude. Therefore `val_loss` is not a scientifically adequate Stage1
checkpoint selector for this task. The run was still improving near the bound
(epoch30 set a new val-loss minimum), so epoch30 is not evidence of convergence.

Before any continuation, agree one of these paths with the user:

1. recommended: add leakage-free validation logging/checkpointing for CCC, relative
   L2, mass error, and source recall; retain raw evidence explicitly as a learned
   normalized-adjoint feature; then resume only to the original early-stop contract;
2. freeze epoch29 as a development checkpoint, accepting poor weak-source recall and
   leaving amplitude correction to V4;
3. make one scoped Stage1 loss correction for amplitude/source recovery and retrain,
   which changes the method and requires a matched validation comparison.

Do not choose between these paths automatically. Per-case normalization also removes
cross-case absolute intensity, so Stage1 can only be judged on normalized field and
within-case unequal-source recovery under the current user-required protocol.

The active Stage1 process loaded the pre-fix auxiliary metric code, so its logged
location error can be nonsensical when signed leaky outputs have near-zero total
mass. This does not affect its primary `val_loss` checkpoint selection. The metric
implementation is fixed for subsequent evaluation to use the same clipped physical
state exported by the bridge; final localization must be recomputed by the canonical
world-coordinate evaluator.

### Stage1 lumped-volume audit completed while training remains paused

The requested no-retraining val300 audit is complete at
`diagnosis/stage1_continuous_fem_volume_audit/`. It uses tetrahedral lumped nodal
volumes rather than equal node weights and contains one row per case, grouped CSV/JSON
statistics, and four fixed-rule GT/prediction/error MIP figures with common scales.

| checkpoint | wRelL2 | mass ratio | weighted CCC | COM mm | shape error | composition error | weak error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| epoch29 | 0.7725 | 1.0244 | 0.6119 | 2.2305 | 0.7195 | 0.2367 | 0.7808 |
| epoch30 | 0.7629 | 0.5092 | 0.5333 | 2.1643 | 0.7047 | 0.3226 | 0.9048 |
| nodal P1 identity control | 0 | 1 | 1 | 0 | 0 | 0.0072 | 0.0215 |

The audit does not justify resuming Stage1 unchanged. Epoch29 is scientifically safer
than epoch30: epoch30's slightly lower wRelL2 comes with systematic under-amplitude
and substantially worse source composition. The main remaining failure is spatial
shape/localization and multi-source/weak-source allocation, not merely a scalar gain.
Training, V4, decoder, development-test, and sealed confirmation remain paused.

This is the operational continuation document for the continuous-field LPR
experiment. Read it together with, in this order:

1. `diagnosis/FINAL_METHOD_CANON.md`
2. `diagnosis/FINAL_RESULT_LEDGER.md`
3. `diagnosis/REJECTED_METHOD_HYPOTHESES.md`
4. `AUDIT_FINDINGS.md`
5. `diagnosis/LPR_CONTINUOUS_BENCHMARK_PROTOCOL.md`

The protocol file is the scientific freeze. This file records mutable execution
status and the exact remaining order of work. Do not reinterpret historical D0/D1Q
results as formal continuous-field results.

## Non-negotiable scope and governance

- Final method: fixed analytic P1 interpolation, corrected scalar FEM state,
  terminal `Hc` concatenated last, residual MLP, and exact hard `Q`.
- No CST, delta-H routing, one-ring, direct voxel-side views, learned interpolation,
  Transformer, or Gaussian-primitive decoder.
- The active first-pass task is the single 3000-case MCX-canonical historical D1-Q
  continuous-field distribution described in the active frozen protocol. The richer
  additive/oriented Gaussian-mixture cohort is paused follow-up work.
- `test.txt` is a development-test split. Do not create its Stage1 bridge, V4 cache,
  decoder predictions, or metrics until the val300 selection receipt exists.
- Never access a sealed confirmation set. This workflow does not authorize it.
- Do not replace raw amplitude-sensitive evaluation with separately normalized GT
  and prediction.
- Run repository commands through `rtk`, normally as `rtk uv run ...`.
- Both DU2Vox and FMT-SimGen worktrees contain intentional uncommitted work. Do not
  reset, clean, or overwrite unrelated changes.

## Why there is another 3000-case data operation

There are two different 3000-case cohorts/workflows:

1. The historical DeepSeek continuous/D1Q cohort is not the final benchmark. Its
   generator uses axis-aligned sphere/ellipsoid/irregular sources and intentional
   non-additive `max` composition, while its saved MCX views have a GT/MCX cutoff
   mismatch. Its requested per-case normalization is not an implementation error;
   it defines normalized within-case reconstruction and does not by itself require
   regeneration. D1Q remains forensic evidence because it does not match the later
   additive, randomly oriented benchmark contract and because its view/GT fields are
   inconsistent.
2. The new final cohort is
   `data/lpr_continuous_gaussian_mixture_3k_20k`. Its 3000 GT fields, source
   parameters, DE measurements, and 2400/300/300 split have already been generated.
   They are **not being sampled again**. The active long job rebuilds/produces MCX
   source configs, `.jnii`, and seven-view `proj.npz` for those fixed cases.

The MCX rebuild is necessary because the audit found two concrete implementation
errors:

- MCX source evaluation used voxel centers at a different floating-point precision
  from the GT generator, changing membership at the 3.5-Mahalanobis truncation
  boundary. It was fixed by mirroring DualSampler float32 centers exactly.
- `fmt_simgen/mcx_config.py` wrote a volume path relative to the wrong working
  directory when the dataset lived under DU2Vox. It now derives the relative path
  from each sample output directory.

After rebuilding all 3000 source files, the audited source/GT contract has maximum
absolute error `1.49e-8`, maximum relative L2 `8.38e-10`, and zero GT mass outside
the MCX source bounding boxes. Evidence:
`diagnosis/lpr_continuous_mcx_source_contract3000.json`.

## Completed work

### Phase I forensic audit

- Human-readable verdicts: `AUDIT_FINDINGS.md`.
- Machine-readable recomputation:
  `diagnosis/d1q_independent_forensic_audit.json`.
- The old A/B claim was invalid: B retrained a continuous V4 corrector/view encoder,
  not only the Q decoder, and both arms inherited binary-trained Stage1 state.
- The old `signal_query_fraction=0.5` silently changed the objective without
  importance weights; the new protocol has explicit global and source losses.
- Old amplitude, PSNR, mass, contrast, half-max, and profile claims were audited and
  are not to be copied into the final result narrative.

### Final continuous dataset

- Dataset: `data/lpr_continuous_gaussian_mixture_3k_20k`.
- Splits: train 2400, val 300, development-test 300; pairwise disjoint.
- Fixed generator seed: `20260915`.
- Realized K counts: 1013/992/995 for K=1/2/3.
- Source family: additive positive, oriented anisotropic Gaussian mixtures.
- Generator config:
  `/home/foods/pro/FMT-SimGen/config/lpr_continuous_gaussian_mixture_3k_20k.yaml`.
- Full sanity audit: `diagnosis/lpr_continuous_dataset_audit3000.json`.
- Corrected source contract audit:
  `diagnosis/lpr_continuous_mcx_source_contract3000.json`.
- FEM projection targets: 3000 `.npy` files plus metadata in
  `precomputed/lpr_continuous_projection_targets`.

Do not regenerate source parameters, GT arrays, DE measurements, or splits unless a
newly demonstrated correctness failure invalidates the frozen cohort.

### Continuous Stage1 and bridge

- Config: `configs/stage1/lpr_continuous_gaussian_mixture_2400.yaml`.
- Formal training completed 150 epochs.
- Selected checkpoint:
  `runs/stage1_lpr_continuous_gaussian_mixture_2400/checkpoints/best.pth`.
- Checkpoint SHA-256:
  `7e93fd9c2f84ee54b6a405d84c53b82b729c2b2cfc63f4a968e5d0a018f51f40`.
- Selection: minimum continuous validation loss, epoch 147, val loss approximately
  `0.1395471`.
- Train bridge: `output/bridge_lpr_continuous_train`, 2400 `coarse_d.npy` files.
- Validation bridge: `output/bridge_lpr_continuous_val`, 300 `coarse_d.npy` files.
- Development-test bridge intentionally does not exist yet. Do not create it before
  validation freeze.

### Code already added or repaired

- Raw continuous metrics and true 3D SSIM:
  `du2vox/evaluation/continuous_field.py`.
- Metric tests: `tests/test_continuous_field_metrics.py`.
- Dataset/source audits: `scripts/audit_lpr_dataset_sanity.py`,
  `scripts/audit_lpr_mcx_source_contract.py`.
- Full benchmark evaluator: `scripts/eval_lpr_continuous_benchmark.py`.
- Paired bootstrap: `scripts/bootstrap_lpr_continuous.py`.
- Dataset and selection receipts: `scripts/freeze_lpr_continuous_dataset.py` and
  `scripts/freeze_lpr_continuous_selection.py`.
- Traditional baseline: `scripts/eval_nonnegative_tikhonov.py` and
  `configs/baselines/lpr_continuous_tikhonov.yaml`.
- V4 and complement training/evaluation scripts contain the explicit continuous
  loss and test-freeze gates.

## Live long-running process

Two MCX launches have occurred for this cohort; only the second is live.

**Superseded first launch (1e7 photons).** Started 2026-09-15 16:06 in tmux
`lpr_mcx`, paused 16:26 at `sample_0022`. It was terminated outright on
2026-09-16 so the photon budget could be halved and aligned with the d1q
canonical cohort. Its log is preserved as `logs/lpr_mcx_1e7_paused_partial.log`.
The 23 `.jnii` it had produced were deleted and their MCX JSONs rewritten to
5e6, so **no 1e7 artefact remains in this cohort**.

**Current launch (5e6 photons), RUNNING.** Runs in tmux `lpr_mcx_5e6`, log
`/home/foods/pro/DU2Vox/logs/lpr_mcx_5e6.log`. It was `SIGSTOP`ped at the user's
request on 2026-09-17 20:07 and resumed on 2026-09-18 01:15 via
`kill -CONT` on the MCX wrapper followed by the pipeline Python. Parent
`rtk`/`uv` processes sleep while they wait. PIDs at the pause were pipeline Python
`2442336` and MCX wrapper `2528225` (ephemeral: resolve them again with
`rtk pgrep -af` before any future signal). Launch command:

```bash
cd /home/foods/pro/FMT-SimGen
rtk uv run python scripts/run_mcx_pipeline.py \
  --samples_dir /home/foods/pro/DU2Vox/data/lpr_continuous_gaussian_mixture_3k_20k/samples \
  --shared-dir output/shared_mesh_20k \
  --config config/lpr_continuous_gaussian_mixture_3k_20k.yaml
```

No `--force_sources`: phase 2m sources are already complete (3000). The config
sets `mcx.photons: 5000000`, matching the historical D1-Q views reused by the
d1q canonical cohort so the two cohorts' MCX view noise is comparable.

**Throughput correction.** An early estimate of 34.9 s/case from the first 33
samples was optimistic. Measured over 1260 completed cases (2026-09-16 16:30 to
2026-09-17 11:51) the true average is **55.3 s/case**, with 47.4 s/case over the
most recent 200. The early samples were atypically fast. The 1e7-to-5e6 halving
therefore bought almost nothing (the 1e7 run measured 49 s/case on its 23
samples): **MCX wall time is dominated by volume-loading and photon-path
geometry, not photon count, so lowering the photon budget further would only add
view noise without meaningful speedup.**

Phase 3m is sequential MCX simulation: **1872/3000** `.jnii` completed
(`sample_0000`–`sample_1871`) as of 2026-09-18 01:16, running again after the
01:15 resume. `proj.npz` is still 0 because phase 4m projection runs only after
phase 3m finishes entirely, so the Stage 2 multiview branch remains unavailable
until then. Remaining 1128 cases at the measured 47.4 s/case is roughly 14.9 h.

**Known false FAILs (do not react to them).** `sample_0444` (2026-09-17 00:47)
and `sample_1870` (2026-09-18 01:15) are both logged as
`FAIL — MCX timed out after 600s`. These are artefacts of pausing, not bad
samples. `mcx.exe` is a Windows binary behind WSL interop, so `SIGSTOP` on the WSL
side does not stop the simulation; each sample finished shortly after its pause
was issued, while the frozen pipeline's wall-clock subprocess timeout kept
ticking. The failure is recorded at resume time. Both are cosmetic: the pipeline
processes its fixed startup sample list in order and moves straight to the next
sample, so neither was rerun, and phase 4m selects by `has_proj` rather than by
this log.

Both artefacts were independently verified complete and valid:

| sample | bytes | at pause | `load_jnii_volume` | finite |
| --- | ---: | --- | --- | --- |
| `sample_0444` | 8,830,044 | ran ~1 min past pause | (190, 200, 104) float32, min 0.0 | yes |
| `sample_1870` | 8,630,748 | ran ~17 s past pause | (190, 200, 104) float32, min 0.0 | yes |

Both byte counts fall inside the cohort range (min 8,300,992 / median 8,780,268 /
max 8,859,620) and their magnitudes match their neighbours. **Do not rerun
either.** Phase 3m/4m select samples by file existence (`has_jnii`, `has_proj`),
not by the failure log, so both reach projection normally.

Consequence for Gate 1: the log will contain exactly these two `FAIL` lines. Gate
1's "zero `MCX ... FAIL`" check must be judged against artefact validity, not the
raw grep count.

**Pausing is not clean on this path.** Because the Windows MCX process ignores
WSL signals, any `SIGSTOP` pause takes effect only at the current sample's
completion and always records that sample as a timeout FAIL. Either let a sample
finish before pausing, or accept one false FAIL per pause.

Monitor without disturbing the process:

```bash
rtk tmux ls
rtk pgrep -af 'run_mcx_pipeline.py|mcx.exe'
rtk tail -n 40 /home/foods/pro/DU2Vox/logs/lpr_mcx_5e6.log
rtk find /home/foods/pro/DU2Vox/data/lpr_continuous_gaussian_mixture_3k_20k/samples -name '*.jnii'
rtk find /home/foods/pro/DU2Vox/data/lpr_continuous_gaussian_mixture_3k_20k/samples -name proj.npz
rtk df -h /home/foods/pro/DU2Vox
```

`rtk find` prints a summary count. Search the log for both MCX and projection
failures.

If the run must be resumed after a pause, first discover the current Python and
MCX PIDs with `rtk pgrep -af`, verify they are the stopped members of this
pipeline, then resume the MCX child followed by the pipeline Python:

```bash
rtk proxy /bin/kill -CONT <mcx_wrapper_pid>
rtk proxy /bin/kill -CONT <pipeline_python_pid>
```

PIDs are ephemeral: never copy these numbers after a reboot or process restart
without resolving and checking them again. If the stopped session no longer exists,
record the last log error and rerun the same pipeline without `--force_sources`;
existing completed `.jnii`/projection files are skipped. Never use `--force_mcx` or
`--no_skip` unless a correctness issue specifically requires overwriting valid
results.

## Remaining execution plan

The order below is mandatory. A failed verification must be resolved before moving
to the next gate.

### Gate 1: finish and certify MCX views — **COMPLETE (2026-09-18)**

Status: passed. Receipt at `diagnosis/lpr_continuous_dataset_freeze.json`.

1. Phase 3m finished 2026-09-18 17:14: **2998 succeeded, 2 failed (175500.8 s)**.
   Phase 4m finished 17:49: **3000 succeeded, 0 failed (2096.8 s)**. The pipeline
   exited non-zero only because of the two known pause artefacts; phase 4m
   projected both successfully.
2. Exactly 3000 `.jnii` and 3000 `proj.npz` exist.
3. View certification is implemented in
   `scripts/audit_lpr_mcx_view_certification.py` (notes: the existing
   `audit_lpr_dataset_sanity.py` covers GT/source/measurement consistency but not
   projections). Full-cohort result, `diagnosis/lpr_mcx_view_certification.json`,
   status `passed`: 3000/3000 with the seven expected angle keys, 256x256 float32,
   zero non-finite, zero negative, and zero degenerate views; view non-zero
   fraction 0.112--0.161 (median 0.137), consistent with the pipeline's reported
   avg 0.138.
4. `MCX ... FAIL`: exactly two, `sample_0444` and `sample_1870`, both explained as
   pause artefacts and both verified valid. `Projection ... FAIL`: zero. The raw
   grep count is therefore 2, not 0; the criterion is met on artefact validity.
5. Dataset sanity audit:
   `diagnosis/lpr_continuous_dataset_sanity_post_mcx.json`, **10/10 acceptance
   criteria pass**. Both `node_max_abs_error` and `voxel_probe_max_abs_error` are
   **exactly 0.0 for all 3000 cases**.

**Audit convention bug found and fixed here.** The first post-MCX run failed
`analytic_voxel_probe_max_abs_error_le_5e-6`: 2994/3000 cases shifted at the 1e-7
level and `sample_2877` reached 2.366e-3. Root cause was the audit script, not the
data. It recomputed the mixture at **float64** voxel centres while the stored
volume was generated at **float32** centres (FMT-SimGen
`fmt_simgen/sampling/dual_sampler.py`, and the Sep-15 precision fix that mirrored
those centres exactly for the MCX sources). At the 3.5-sigma truncation surface a
centre shift of ~1e-7 mm flips membership, and the resulting disagreement equals
the Gaussian value at 3.5 sigma (~2.2e-3), far above the 5e-6 bound. Confirmed on
the single worst voxel of `sample_2877`, index (129, 169, 40), stored
2.365794964e-03: analytic at float64 centres returned 0, analytic at float32
centres returned 2.365794964e-03, i.e. **exactly the stored value**. Only 1 of 3000
cases exceeded the threshold, by exactly one voxel, at Mahalanobis 3.5000. The
audit now mirrors the float32 convention, after which both probe and node errors
are exactly zero everywhere. The dataset was never wrong; the comparison was.

6. The immutable receipt was created only after the above passed:

```bash
cd /home/foods/pro/DU2Vox
rtk uv run python scripts/freeze_lpr_continuous_dataset.py \
  --dataset-root data/lpr_continuous_gaussian_mixture_3k_20k \
  --shared-dir /home/foods/pro/FMT-SimGen/output/shared_mesh_20k \
  --generator-config /home/foods/pro/FMT-SimGen/config/lpr_continuous_gaussian_mixture_3k_20k.yaml \
  --generator-root /home/foods/pro/FMT-SimGen \
  --require-projections \
  --output diagnosis/lpr_continuous_dataset_freeze.json
```

Receipt: 3000 cases, splits 2400/300/300, `projections_required: true`, five
hashed files per case (`tumor_params.json`, `measurement_b.npy`, `gt_nodes.npy`,
`gt_voxels.npz`, `proj.npz`), six hashed shared assets including
`frame_manifest.json` = `a89d36a5` (the re-certified manifest). Spot-checked
sample_0000/0001/0444/1500/1870/2999 and all shared assets: every recorded hash
matches disk.

Do not edit dataset/generator files after this receipt. If anything changes, the
receipt must fail its hash checks and the change must be scientifically justified.
Note that the receipt records the generator config **as modified** (it pins
`mcx.photons: 5000000`), which is the config that actually produced these views.

### Gate 2: smoke and train continuous V4 (M1)

Config: `configs/stage2/lpr_continuous_v4_2400.yaml`.

1. Run a bounded smoke first and inspect finite loss, correct raw GT range, view
   shapes, explicit global/source strata, and checkpoint writing:

```bash
rtk uv run python scripts/train_iterative_fem_corrector.py \
  --config configs/stage2/lpr_continuous_v4_2400.yaml \
  --experiment-name lpr_continuous_v4_smoke \
  --max-samples 20 --max-val-samples 8 --max-epochs 2 --smoke
```

2. Run the formal training without overrides:

```bash
rtk uv run python scripts/train_iterative_fem_corrector.py \
  --config configs/stage2/lpr_continuous_v4_2400.yaml
```

3. Select only by dense full-domain val CCC as frozen in the protocol. Record epoch,
   config hash, checkpoint hash, and complete val300 artifact. Do not select by Dice.
4. Evaluate the continuous-trained V4 on val300 and save aligned M0 and M1
   predictions. Do not use `--split test` yet.

### Gate 3: binary-to-continuous transfer diagnostic on validation only

This is an internal architecture decision, not a formal test method.

1. Build a val-only bridge for the historical binary-trained Stage1 using the new
   continuous samples and the exact checkpoint/config it originally used.
2. Evaluate the historical binary-trained V4 stack on the same val300 with raw
   continuous metrics.
3. Compare transferred binary Stage1+V4 with continuous-trained Stage1+V4, casewise.
4. The final main table must use continuous-trained M1 regardless; this diagnostic
   determines how strongly the transfer failure/success is discussed.
5. Do not generate a transferred-stack development-test result unless separately
   justified by the already frozen protocol; it is not needed for the main claim.

### Gate 4: cache frozen M1 state and terminal latent for train/val

Use the selected continuous V4 checkpoint and its exact epoch. Cache train and val
only with `scripts/precompute_frozen_v4_states.py`, including
`--cache-terminal-hidden` and an explicit `--latent-output-dir`. Verify sample IDs,
shape/dtype, finite values, checkpoint epoch, and FP16 round-trip tolerance.

Do not cache test yet. The formal decoder config must point to these caches and must
state the selected V4 checkpoint plus `expected_epoch`.

### Gate 5: materialize and train matched M2/M3 configs

Create the formal configs only after the M1 checkpoint is selected, so they cannot
silently reference the old binary V4. Both configs must have:

- `input_contract: approximation_space_separated`;
- exact input `[PE(q), I_h x_h^c(q), lambda(q), G_e, I_h H_h^c(q)]`;
- `use_terminal_latent: true`, terminal latent concatenated last;
- the same residual MLP, parameter count, seed `20260915`, initialization, sample
  order, optimizer, epochs, and explicit final-field loss;
- frozen M1 state and no voxel-loss gradient into the coarse corrector;
- no view features or delta-H at the voxel decoder.

Only the output contract differs:

- M2: `I_h x_h^c + Qz`, exact hard-Q.
- M3: `I_h x_h^c + z`, unconstrained.

Run a two-epoch/small-sample smoke for each, then formal training. Verify logged
parameter counts are identical. Select each independently by full-domain val CCC.
Evaluate both on val300, save predictions, and record leakage/preservation. Hard-Q
should be numerically near zero, but do not assume this without measurement.

### Gate 6: traditional baseline validation

Smoke the solver on a few validation cases first. Then evaluate all four frozen
relative alphas `{1e-4, 1e-3, 1e-2, 1e-1}` on the same val300 using
`scripts/eval_nonnegative_tikhonov.py`. Select the alpha with maximum full-domain raw
CCC. Do not add a post-hoc grid after seeing development-test results.

M0 is also the learned FEM reconstruction baseline, so the required baseline set is
continuous-trained M0 plus traditional graph-Tikhonov. Do not import published D0
numbers as if they used this cohort.

### Gate 7: unified val300 evaluation and selection freeze

1. Run `scripts/eval_lpr_continuous_benchmark.py` over aligned M0, M1, M2, M3, and
   selected Tikhonov prediction directories.
2. Require 300 identical case IDs for every method.
3. Verify raw common amplitude scale and the fixed metric definitions from the
   protocol.
4. Run targeted tests, at minimum:

```bash
rtk uv run pytest -q tests/test_continuous_field_metrics.py tests/test_voxel_complement_projection.py tests/test_confirmation_governance.py
rtk uv run ruff check du2vox/evaluation/continuous_field.py scripts/eval_lpr_continuous_benchmark.py scripts/bootstrap_lpr_continuous.py scripts/eval_nonnegative_tikhonov.py scripts/freeze_lpr_continuous_dataset.py scripts/freeze_lpr_continuous_selection.py
```

5. Create `diagnosis/lpr_continuous_validation_freeze.json` with
   `scripts/freeze_lpr_continuous_selection.py`. Include V4, M2, M3, and selected
   Tikhonov candidate entries. Every validation JSON must state `split=val` and
   `n_samples=300`. The dataset receipt is mandatory.
6. Open the receipt and manually verify hashes and
   `development_test_accessed_before_freeze: false`.

No test action is allowed before this file exists and passes inspection.

### Gate 8: development-test300, exactly after freeze

After Gate 7 only:

1. Generate `output/bridge_lpr_continuous_test` using the frozen Stage1 checkpoint.
2. Evaluate frozen V4 on test300 with `--freeze-receipt` and save M0/M1 predictions.
3. Cache test M1 state and terminal `Hc` using the frozen V4 checkpoint.
4. Evaluate frozen M2 and M3 on test300 with the same receipt and save predictions.
5. Evaluate only the selected Tikhonov alpha on test300 with the receipt.
6. Run the unified benchmark evaluator on all aligned test300 predictions.
7. Do not tune, rerun selection, change thresholds, or modify methods based on these
   results.

### Gate 9: statistics, hard-Q audit, tables, and figures

1. Run `scripts/bootstrap_lpr_continuous.py` with 10,000 paired draws and seed
   `20260915` for M1-M0, M2-M1, and M2-M3 on SSIM, PSNR, CCC, and relative L2.
2. Report means and paired 95% CIs, not means alone.
3. On the same test300, compute and aggregate:
   `||Pi_h Qz||/(||Qz||+eps)`, coarse preservation error, `Q^2-Q`, and `Pi_h Q`.
4. Report all-GT-source amplitude, mass, localization, detection/recall, contrast,
   source-resolved Dice@50%, HD95, and principal-axis cylindrical FWHM. Do not censor
   missed sources.
5. Build the headline method table and hard-Q mechanism table specified by the user.
6. Select figure cases mechanically from M2 test SSIM ranks: nearest 25th percentile,
   median, 75th percentile, plus the multi-source case nearest the multi-source
   median. Do not inspect images before selection.
7. Plot GT/M0/M1/M2 axial-coronal-sagittal maps and known-center/principal-axis
   profiles using the fixed 0.35-mm cylinder. Visualization copies may be normalized;
   metric inputs may not.
8. Write the final scientific result around M0 -> M1 -> M2 and M2 vs M3. The claim
   must concern approximation-space-separated refinement on one continuous task,
   not continuous-vs-binary performance.

## Known risks and checks before expensive runs

- MCX is the current critical path. Check log failures, disk, file counts, and view
  finiteness before training V4.
- The first attempted MCX launch had immediate bad-volume-path failures and was
  stopped. The corrected log was started afresh. Never count those failed attempts
  as generated data.
- `diagnosis/lpr_continuous_dataset_audit3000.json` preserves the initial pre-fix
  source mismatch evidence; use the later dedicated source-contract JSON for the
  corrected result. Do not rewrite history by deleting the initial audit.
- Runtime-smoke the graph-Tikhonov implementation before its val300 sweep.
- Directly test source-resolved morphology/profile code on controlled synthetic
  cases before using it in the final table.
- Full masked 3D SSIM is expensive. First verify one case and memory use, then run
  the fixed full evaluation without changing its window/data range.
- Check `git status` before every edit. The repository was already highly dirty;
  preserve all pre-existing work.
- Any missing external FMT-SimGen asset is a real blocker to report, never a reason
  to fabricate an output.

## Completion checklist

- [x] Phase I independent audit and verdict document
- [x] New continuous GT/DE cohort and fixed splits
- [x] Generator and MCX-source contract repair/audit
- [x] Continuous Stage1 training (150 epochs; selected epoch 147)
- [x] Train/val Stage1 bridge for the selected continuous checkpoint
- [x] Continuous projection targets
- [x] Reused 3000 MCX simulations and MCX-canonical GT certified
- [x] Dataset identity receipt
- [x] Continuous V4 smoke, train, val300 selection
- [x] Binary-transfer diagnostic on val300
- [ ] Train/val frozen M1 and terminal-Hc cache
- [ ] Matched M2/M3 smoke, train, val300 selection
- [ ] Tikhonov validation selection
- [ ] Unified val300 results
- [ ] Validation selection freeze receipt
- [ ] Development-test bridge and all frozen test300 evaluations
- [ ] 10,000-draw paired bootstrap
- [ ] Continuous test300 hard-Q contract audit
- [ ] Final tables and fixed-rule figures
- [ ] Final scientific conclusion and ledger update

## 2026-09-18 LPR V4 validation checkpoint

LPR Stage1 is trained and is not the current failure point. Its selected checkpoint
is `runs/stage1_lpr_continuous_gaussian_mixture_2400/checkpoints/best.pth`; the
aligned val300 field metrics are CCC 0.761871, relative L2 0.565274, PSNR 46.232939
dB, and signed integrated ratio 0.985707.

Two V4 attempts were stopped at their first formal dense validation (epoch 5):

- Historical continuous objective, with `max_dc_update` reduced from 0.1 to 0.01:
  Dice improved 0.233588 -> 0.367778, but CCC fell to 0.757122, relative L2 rose
  to 0.64901, PSNR fell to 45.0529 dB, and integrated ratio rose to 3.5644.
- Lumped-FEM-volume mass-stable objective (`lambda_fem_mass=0.001`), run directory
  `runs/lpr_continuous_v4_massstable_2400`: Dice improved to 0.339680, CCC to
  0.770661, and peak relative error 0.441532 -> 0.372578. However, relative L2
  worsened to 0.597996, PSNR to 45.728101 dB, and integrated ratio to 1.450845.
  The mass drift is systematic (median ratio 1.333238, 75th percentile 1.712906),
  not a few outliers. Only 25.7% of cases improve integrated relative error.

The second run was deliberately interrupted after the epoch-5 validation gate. Do
not present either checkpoint as a selected M1, do not cache terminal Hc from it,
and do not advance to M2/M3 or development-test300. The next task is to verify that
the V4 update/DC route rather than assume more training will repair it. The mass
integration contract has now been checked on all val300 cases: lifting the saved
`Pi_h GT` targets back through the certified P1 operator gives dense mass ratio
mean/median 1.000000005/1.000000005 (range 0.999999877--1.000000134), while the
lumped-nodal target integral divided by the raw valid-voxel integral gives mean
1.000002451 (range 0.999094752--1.001506878). Thus the observed 1.450845 prediction
ratio is not a projection-target or integration-domain mismatch. Hard-Q cannot be
expected to repair this systematic coarse-space mass error while preserving the
corrected FEM state.

## 2026-09-19 volume-centered V4 recovery

The mass pedestal was localized to the neural update, not the analytic DC route:
turning DC off on the old epoch-5 checkpoint increased signed mass ratio from 1.451
to 1.534 with essentially unchanged morphology, while DC-only stayed near Stage1
(mass 0.972, CCC 0.76194). The P1 target/mass contract remained exact.

V4 now optionally subtracts the lumped-volume mean from every neural update. This
leaves spatial FEM redistribution trainable while assigning global-mass change to
the explicit physics/DC route. A scratch run collapsed to DC-only, so the useful
epoch-5 LPR spatial solution was fine-tuned under the new constraint. At epoch 20,
an oracle-relative negative-mass excess term (`0.01`) also limits diffuse negative
side lobes without requiring zero negativity, which the sampled-L2 P1 oracle itself
does not satisfy (val300 mean negative-mass ratio 0.07893).

The balanced val300 candidate is:

`runs/lpr_continuous_v4_centered_finetune_2400/checkpoints/candidate_balanced_val_epoch20.pth`

It improves Stage1 CCC 0.76187 -> 0.80165, PSNR 46.23294 -> 46.60700 dB,
relative L2 0.56527 -> 0.54240, MSE 1.40954e-4 -> 1.28745e-4, integrated
relative error 0.19166 -> 0.18696, peak error 0.44153 -> 0.33191, and Dice@50%
0.23359 -> 0.38246. Signed mass ratio is 1.00128 and negative-mass ratio is
0.10868. Paired 10,000-draw val bootstrap CIs exclude zero for CCC, PSNR,
relative-L2 reduction, and MSE reduction. Details and hashes are in
`diagnosis/lpr_continuous_v4_candidate_selection.json`.

Do not use `best_dense_val_delta_dice.pth`: it still points to epoch 15 because the
historical selector maximizes CCC alone. Epoch 15 gains only 0.00061 CCC over epoch
20 but has negative-mass ratio 0.31576 and worse PSNR/relative L2/MSE. Epoch 20 is a
validation candidate, not the final multi-method freeze. Test300 and sealed
confirmation remain untouched.

## 2026-09-19 why the legacy continuous Dice is low

The iterative-FEM evaluator's field named `dice` is the historical binary metric
from `experiments/cross_discretization_decomposition/decomposition.py`: both raw
prediction and raw GT are thresholded at the same absolute value `0.5`. It is not
per-case half maximum and it is not source-resolved. On the LPR val300 cohort the
588 nominal source amplitudes range from 0.500154 to 1.484666, so no nominal GT
source is strictly below 0.5. Nevertheless, an attenuated prediction can readily
fail to cross 0.5, and the absolute metric then assigns an empty support. Stage1
has 81/300 zero-Dice cases and epoch20 V4 has 38/300; 137/300 and 64/300,
respectively, score below 0.1. This metric therefore entangles amplitude recovery
with morphology and remains auxiliary only.

The epoch20 predictions were cached under
`precomputed/lpr_continuous_v4_balanced_epoch20/val`, and the all-GT-source audit is
`diagnosis/lpr_v4_balanced_epoch20_source_morphology_val300.json`. It uses disjoint
GT-defined Mahalanobis ROIs and each source's own GT/predicted local peak, with no
detection censoring. Source-resolved Dice@50 improves only 0.514547 -> 0.525530,
while Dice@20 improves 0.599958 -> 0.624887. The larger V4 effect is amplitude and
detection recovery: mean local predicted peak rises 0.472539 -> 0.593337 versus
GT 0.974448, and detection recall rises 0.506803 -> 0.688776. Shape remains broad:
50%-support volume ratio falls from 1.900923 to 1.595909 but is still well above
one. Weak-source Dice@50 is essentially unchanged (0.484628 -> 0.484828), although
weak-source detection recall improves 0.669492 -> 0.754237. Thus low legacy Dice is
partly a threshold artifact, but there is real remaining broadening and weak-source
shape/separation headroom. Do not claim the issue is purely metric-induced.

`scripts/eval_lpr_continuous_morphology.py` now accepts repeatable aligned dense
inputs as `--prediction NAME=DIRECTORY:NPZ_KEY`, allowing M0/M1/M2/M3 to use the
same source-resolved protocol without adding expensive 3D SSIM to every diagnostic
run. Ruff passes for this script. This is evaluation plumbing only; no network was
changed or retrained in this audit.
