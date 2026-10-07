# Project state

Updated: 2026-10-07
Commit: d18b345f49a8d9189d7149cfe09f3bafbea58869
Branch: exp/lpr-continuous-stage2 (working tree has uncommitted changes; runs/ is gitignored)

## Current handoff

- Branch: exp/lpr-continuous-stage2
- Current HEAD: d18b345f49a8d9189d7149cfe09f3bafbea58869 (dirty working tree;
  runs/ is gitignored)
- Active experiment: physics-conditioned encoder pair M_image_only / P_physics
  (seed 20260928, 30 epochs, train2400/val300), run serially per
  `diagnosis/rte_baselines_v2/PHYSICS_ENCODER_PROTOCOL_20261007.md`; queue
  status `training`, arm `M_image_only`, PID 14102 at audit time; both-arm
  fitchecks complete; no test or confirmation access.
- Latest completed experiment/result: RTE paper A-D factorial seed 20260928,
  all four arms with val300 summaries (final arm D completed 2026-10-07
  16:04); seeds 20261007/20261008 deferred by the authorized focus change.
- Current scientific question: does operator-derived physics conditioning
  (sensitivity / local response-similarity descriptors) in the learned
  optical spatial encoder improve reconstruction over the parameter-matched
  image-only arm (the M/P mechanism test of the protocol above)?
- Next decision required: after both arms complete, scientific review of the
  single-seed val-selected M/P comparison (development evidence, not a
  confirmed superiority claim) - report as-is versus authorize further seeds,
  plus the disposition of the deferred A-D seeds.
- Last scientifically reviewed commit: d18b345f49a8d9189d7149cfe09f3bafbea58869,
  the recorded git HEAD of the 2026-09-30 user-authorized DA/RTE paper freeze
  (`diagnosis/rte_mismatch/paper_freeze_20260930.json`; development-candidate
  freeze with recorded reproduction deviation; no development-test or sealed
  confirmation authorized). This freeze does not rewrite the canon line.

## Objective

Two-stage coarse-to-fine Fluorescence Molecular Tomography (FMT)
reconstruction, defined by `diagnosis/FINAL_METHOD_CANON.md` (highest-priority
scientific definition; entry point `diagnosis/README.md`):

- Stage 1: coarse FEM inverse reconstruction `y -> x_h^(0)`.
- Stage 2A: physics-guided iterative FEM refinement (frozen V4, K = 3).
- Transfer: fixed canonical analytic P1 (`I_h`); not learned.
- Stage 2B: complement-constrained voxel detail; only `Q z_theta` with
  `Q = I - I_h Pi_h` enters `rho_final = I_h x_h^(3) + Q z_theta`.

The final method does not learn FEM-to-voxel interpolation.

## Current status

- Method status: development freeze candidate. Sealed confirmation remains
  unopened (`diagnosis/FINAL_METHOD_CANON.md` section 11;
  `diagnosis/FINAL_RESULT_LEDGER.md`).
- Data identity (`diagnosis/FINAL_RESULT_LEDGER.md`): dataset
  `fmt_simgen_v2_3k_20k`; train 2400 / validation 300 / development-test 300
  (already observed) / confirmation sealed. Canonical 0.2 mm grid
  `(190, 200, 104)`; 1,677,645 valid voxels; 19,990 FEM nodes; GT semantics
  `gt_voxels > 0.05`; fixed threshold 0.5.
- Recorded development-test Dice: Stage 1 P1 0.617379; V4 coarse 0.716529;
  V4 + constrained `Qz` 0.727163 (validation 0.741844). The 300-sample test is
  a development-test, never final unseen generalization.
- Recorded structural result (development-test): constrained detail coarse
  leakage 4.57e-18; relative coarse-state preservation error 8.72e-9.
- The parallel LPR continuous-field line is validation-frozen with
  development-test results in `diagnosis/LPR_CONTINUOUS_FINAL_RESULTS.md`;
  its sealed confirmation is also unopened.
- `diagnosis/v4_frozen_protocol.md` does not exist; confirmation evaluation
  stays blocked by default (`AGENTS.md`).

## Active work

Verified at audit time; queue receipts and logs are authoritative.

- RTE paper A-D factorial (`diagnosis/rte_baselines_v2/E2_PAPER_ABLATION_PROTOCOL_20261006.md`):
  seed 20260928 completed; seeds 20261007/20261008 deferred by user focus
  change (`diagnosis/rte_baselines_v2/paper_ablation_20261006/queue_status.json`).
- Physics-conditioned encoder study (`diagnosis/rte_baselines_v2/PHYSICS_ENCODER_PROTOCOL_20261007.md`):
  both-arm fitcheck receipt complete
  (`diagnosis/rte_baselines_v2/physics_encoder_20261007/fitcheck_receipt.json`);
  queue status `training`, arm `M_image_only`, seed 20260928
  (`.../physics_encoder_20261007/queue_status.json`; PID 14102 live at audit).
- Execution constraints: serial GPU jobs under the shared lock, 6144 MiB
  allocator cap, 8192 MiB free reserve; train/val only, no test or
  confirmation access (`AGENTS.md`).

## Open questions

- Frozen-candidate checkpoints referenced by `diagnosis/FINAL_RESULT_LEDGER.md`
  (`runs/unified_dual_evidence_fem_v4_2400/checkpoints/best_dense_val_delta_dice.pth`,
  `runs/complement_voxel_detail_v1_2400_constrained/checkpoints/best_dense_val_dice.pth`,
  and the matched unconstrained control) are absent from this working tree at
  audit time; `runs/` is gitignored. Locate or regenerate them before relying
  on, or reproducing, the frozen rows.
- Stage 2B magnitude: the learned voxel branch recovers only 8.68%
  (validation) / 9.86% (development-test) of the oracle detail Dice headroom;
  recorded as an Outcome B fine-detail enhancement, not the dominant mechanism
  (`diagnosis/FINAL_METHOD_CANON.md` sections 8-9).
- Freeze identifier: no clean freeze commit or
  `du2vox-two-stage-freeze-candidate-v1` tag exists yet
  (`diagnosis/FINAL_RESULT_LEDGER.md`).
- RTE baseline/encoder learning and formal-training gates are unresolved; an
  implementation-ready receipt does not assert that GPU gates passed
  (`AGENTS.md`). Open diagnostics: GCGM reconstruction failure, GAICN useful
  signal recovery, PAH2T negative-mass reporting (`AGENTS.md`).
