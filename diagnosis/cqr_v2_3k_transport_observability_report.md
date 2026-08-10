# CQR v2 3k Transport-Observability Report

## 1. Git and data context

- Working branch: `feat/cqr-transport-observability`.
- Implementation commit: `673a899fd5579189c8bd50ef6e6443a892600fff`.
- Base branch/commit: `feat/cqr-v1` at
  `0ab376f1870793ba8c404186ff8d6d60941c7e41`.
- Dataset: `/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k`.
- Actual split counts: train `2400`, val `300`, test `300` (total `3000`).
- Stage 1 checkpoint:
  `runs/stage1_fmt_simgen_v2_3k_20k_balanced_v2_eval/checkpoints/best.pth`.
- Resolved sparse CQR baseline checkpoint:
  `checkpoints/stage2/cqr_v2_3k_rgl_main_sparse/best.pth`.

No commit was pushed and no full 3k training was started.

## 2. Shared physics and manifest certification

- Shared root: `/home/foods/pro/FMT-SimGen/output/shared_mesh_20k`.
- Mesh: `19990` nodes, `99209` tetrahedra, `7413` surface nodes.
- `M/F`: `[19990,19990]`, `273208` nonzeros each.
- `A`: `[7413,19990]`; measurements and A therefore use the full-surface convention.
- `visible_mask.npy` contains `7413/7413` visible entries and Stage 1 has
  `use_visible_mask: false`.
- Certified `frame_manifest.json` SHA256:
  `25b77f5d11430ebf4d9735f19534782afb56f7fe5cab525a569028b0f3c02e2b`.

The manifest mtime is old, but its coordinate contents, mesh bounds, voxel grid, surface
convention and shared matrices were re-audited. The new config pins the SHA256; training and
unified/grouped evaluation now abort if the file contents change. This is a content
certification, not an mtime refresh or a rebuilt FMT-SimGen asset.

## 3. Pilot assets and CQR coverage

The existing CQR mainline builder was used without overwriting existing pools. Current pilot
coverage is:

| split | CQR pools | physics sidecars | fixed quadratures |
| --- | ---: | ---: | ---: |
| train | 100 | 100 | 100 |
| val | 25 | 25 | 25 |
| test | 5 | 5 | 5 |

Every pool contains `32768` candidates and training samples `2048` points per epoch. The
core/halo/bg/proposal roles, `prior_lift[15]`, RGL indicators, correction bands and independent
measurement proposal are retained. The three-split audit now has at least five files per split.

Most 32768-candidate pools represented more than 99% of their target tet volume, but
`sample_0058` represented only `95.08%`. Random 2048 subsets cover much less physical volume.
Consequently candidate weights are used only for the candidate-supported correction operator;
fixed quadrature remains separate. Stratified pilot quadrature uses `1024` tets / `4096` points
per sample and represented about `16%` to `54%` of target volume depending on the sample.
Strict full-volume validation still requires `physics_tet_mode: all`.

## 4. Compressed Green operator

Cache: `physics_cache/v2_3k_20k_transport_rank64.npz`.

The cache was rebuilt with rank `64`, oversampling `16`, seed `20260722`, and two power
iterations. Results:

- operator identity relative error: `7.765260e-09`;
- maximum absolute error: `2.592020e-08`;
- rank-64 cache retained energy: `0.11760960`;
- optical modes finite: yes;
- rank-128 Gram condition number: `2.63568`.

Power-iteration spectral diagnostic:

| rank | retained energy |
| ---: | ---: |
| 16 | 0.03726 |
| 32 | 0.06947 |
| 64 | 0.12498 |
| 128 | 0.21099 |

Rank 64 is therefore a deliberately compressed observable subspace, not a representation of
all forward-matrix Frobenius energy. It is adequate for the pilot contract but rank adequacy for
formal morphology inversion remains unproved.

## 5. P1, lifter and projector checks

P1 consistency was rerun on 20 train samples:

- mean/max source-load relative error: `3.126e-08` / `3.188e-08`;
- mean/max compressed-measurement relative error: `2.670e-08` / `2.954e-08`;
- pass threshold: yes.

Thus P1 already preserves the FEM transport action to numerical precision. The lifter performs
morphology-sensitive convex redistribution under a transport anchor.

- zero-init lifter gives `alpha == lambda` and `rho0 == FEM P1`;
- `alpha >= 0` and rows sum to one;
- proposal `tet_id == -1` is masked and never gathered as tet 0;
- observable/ambiguous solves run in float32 with autocast disabled;
- backward is finite under bf16 autocast;
- observable plus ambiguous projection reconstructs the raw correction within float32 tolerance.

## 6. Three-phase pilot

Pilot size: 100 train, 25 val, 2048 sampled queries, 5 epochs per phase. All phases use
`proj.npz`, per-view-max normalization, multiscale attention fusion, bf16 AMP and the fixed
stratified quadrature.

| phase | checkpoint | observed result |
| --- | --- | --- |
| A: lifter | `cqr_tc_phase_a_pilot100_r1/best.pth` | finite; logged peak val delta `+0.00056`; transport error at epoch 5 `1.17e-05` |
| B: observable | `cqr_tc_phase_b_pilot100_r2/best.pth` | finite; observable residual ratio was not stably reduced (`~1.01-1.26`) |
| C: full | `cqr_tc_phase_c_pilot100_r1/best.pth` | finite; ambiguous leakage `2.30e-06` to `2.65e-06`; late val delta fell to `-0.0005` |

The pilot exposed and fixed a phase-transition bug: a resumed phase previously produced no
checkpoint unless it improved delta Dice by more than `0.0005`. Each resumed experiment now
saves a phase-local epoch-0 `best.pth` before training, so B to C continuation is guaranteed.

GPU peaks were approximately `652 MiB` (A), `678 MiB` (B), and `685 MiB` (C), with at most
`706 MiB` reserved. Host memory at final verification was 19 GiB total and 18 GiB available.

## 7. Matched unified val evaluation

The following is a 25-sample candidate-domain val pilot, not a formal full val/test result:

| model | Dice | FEM Dice | delta | precision | recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| historical sparse CQR | 0.34951 | 0.53344 | -0.18393 | 0.23396 | 0.93907 |
| Phase C phase-local best | 0.53345 | 0.53344 | +0.00001 | 0.49922 | 0.73836 |

The historical checkpoint remains loadable, including the compatible 4-to-5 band-embedding
migration. The transport model preserves the FEM baseline but does not yet show a meaningful
improvement. The Phase C best is intentionally the safe phase-local epoch-0 baseline because
later full-phase epochs did not exceed the checkpoint threshold.

## 8. Verification and unresolved issues

- Targeted Ruff for all changed Python files: pass.
- Pytest: `10 passed`, including manifest hash certification and old sparse
  checkpoint/config compatibility.
- Full-repository Ruff: still fails on 90 unrelated pre-existing legacy errors.
- Complete 2400/300/300 CQR pools, sidecars and quadratures have not been generated.
- `physics_tet_mode: all` validation has not been run.
- Common-domain/full-grid observability weights remain undefined; candidate weights must not be
  reused for that grid.
- No formal full test-set result exists; historical val-up/test-down risk remains unresolved.
- Observable residual reduction was unstable and the full branch slightly degraded late-pilot
  validation, so loss weights/curriculum require another pilot before formal training.

Recommendation: **do not start full Phase A/B/C training yet**. The mechanics, numerics,
manifest certification, phase checkpoint chain and 100/25/5 pilot assets are ready, but the
observable objective needs a focused tuning pilot and an `all`-tet validation before spending
the full 3k compute budget.

## 9. Required A-I answers

- **A. Working on `feat/cqr-v1` lineage?** Yes. Work started from verified `feat/cqr-v1` and
  now lives on independent branch `feat/cqr-transport-observability`.
- **B. Using `fmt_simgen_v2_3k_20k`?** Yes.
- **C. Reused 32768 candidate pool / 2048 training sample?** Yes; pilot coverage is
  train/val/test `100/25/5`.
- **D. Compressed Green operator passed?** Yes, relative identity error `7.765e-09`.
- **E. P1 local-mass consistency passed?** Yes, maximum error `3.188e-08`.
- **F. Lifter initializes to P1?** Yes.
- **G. Observable/ambiguous projection differentiable?** Yes, including bf16-autocast fallback.
- **H. Old sparse CQR baseline runnable?** Yes; load, smoke and 25-sample unified val all run.
- **I. Start full training?** No. Run focused objective tuning and all-tet validation first.
