# Canonical method specification

Status: mirrors `diagnosis/FINAL_METHOD_CANON.md` (development freeze
candidate; sealed confirmation unopened). The canon is the highest-priority
scientific definition; if this file and the canon disagree, the canon wins.
This file adds no new scientific claims.

## Inputs

- Surface measurement `measurement_b.npy` (`y`) and FEM forward matrix
  `system_matrix.A.npz` (`A`), with `mesh.npz`, graph Laplacians, kNN indices,
  optional `visible_mask.npy`, and the authoritative `frame_manifest.json`
  (`AGENTS.md` data contracts; `docs/COORDINATE_SYSTEM.md`).
- World frame `mcx_trunk_local_mm`, mm coordinates. Stage 1 network
  implementation is described in `AGENTS.md` (6-block unrolled GCAIN).
- Canonical evaluation domain: 0.2 mm grid `(190, 200, 104)`; 1,677,645 valid
  FEM-domain voxels; 19,990 FEM nodes; GT semantics `gt_voxels > 0.05`;
  reconstruction threshold 0.5 (`diagnosis/FINAL_RESULT_LEDGER.md`).

## Pipeline

1. Stage 1 (`balanced-v2` FEM reconstruction, canon section 11) produces the
   coarse inverse state `y -> x_h^(0)`; reference development-test Dice
   0.617379 (`FINAL_METHOD_CANON.md` section 2).
2. Stage 2A: frozen V4, `K = 3` shared-weight iterations (canon section 3).
   At iteration k:

   ```text
   p^(k) = A x^(k)
   r_raw^(k) = y - A x^(k)
   alpha_k = argmin_{alpha >= 0} ||y - alpha A x^(k)||^2
   r_SI^(k) = y - alpha_k A x^(k)
   g_raw^(k) = A^T r_raw^(k)
   g_SI^(k)  = alpha_k A^T r_SI^(k)
   x^(k+1) = x^(k) + Delta x_neural^(k) + Delta x_DC^(k)
   ```

   Forward predictions, residuals, adjoints, and normalized evidence are
   recomputed at every iteration. The shared update cell combines raw and
   scale-invariant physics evidence, adjoint and Jacobi/sensitivity-normalized
   directions, FEM mesh context, frozen seven-view optical features, current /
   initial / historical FEM states, and lightweight global conditioning.
3. Transfer: fixed canonical analytic P1 `I_h` evaluates the corrected
   continuous P1 FEM field at canonical 0.2 mm voxel centers. Not learned; the
   cached sparse P1 matrix and sampled-L2 projection from the 3000-sample
   decomposition experiment are reused, and no second projection definition is
   permitted (canon section 4).
4. Stage 2B: with `rho_c = I_h x_h^(3)` and `Pi_h` the canonical sampled
   voxel-to-FEM projection (`Pi_h I_h ~= I`, measured maximum relative
   left-identity error 1.50e-15; `I_h Pi_h` numerically self-adjoint and
   idempotent on the fixed uniformly weighted sampled domain), define
   `Q = I - I_h Pi_h`. The network predicts an unconstrained signed proposal
   `z_theta`; only `w_theta = Q z_theta` may enter:

   ```text
   rho_final = I_h x_h^(3) + Q z_theta
   Pi_h rho_final ~= x_h^(3)
   ```

   The voxel branch cannot overwrite the physics-refined FEM state; it adds
   only FEM-invariant voxel degrees of freedom (canon section 5).

## Training

- Stage 2B candidate (canon section 5): full 0.2 mm canonical domain without
  the 0.4 mm fallback; width 160 and three residual GELU MLP blocks;
  six-frequency coordinate encoding; V4 P1 value and exact local P1 gradient;
  four local FEM values and barycentric weights; local mean, standard
  deviation, and range; V4 support-boundary proximity; frozen multiview
  optical features; exact FP64 sparse `Q` forward/backward with FP32
  constrained output; SmoothL1 detail loss + `0.1` MSE + `0.1` Tversky
  support loss; raw unclamped output.
- Checkpoint choices are made from validation Dice only; no threshold sweep
  (`diagnosis/FINAL_RESULT_LEDGER.md`).
- Frozen candidate artifacts recorded in the ledger (paths as written there):
  Stage 2A config `configs/stage2/unified_dual_evidence_fem_v4_2400.yaml`,
  checkpoint `runs/unified_dual_evidence_fem_v4_2400/checkpoints/best_dense_val_delta_dice.pth`
  (epoch 15); Stage 2B config
  `configs/stage2/complement_voxel_detail_v1_2400.yaml`, constrained
  checkpoint
  `runs/complement_voxel_detail_v1_2400_constrained/checkpoints/best_dense_val_dice.pth`
  (epoch 3, validation Dice 0.7418439308802287). See `research/STATE.md` open
  questions: these checkpoint paths are currently absent from this working
  tree.
- No joint Stage 2A/2B fine-tuning in the candidate (canon section 11;
  `REJECTED_METHOD_HYPOTHESES.md` section 9).

## Inference

- Raw, unclamped model output with the fixed threshold `0.5` (canon
  section 11; `REJECTED_METHOD_HYPOTHESES.md` section 8).
- Confirmation inference, Dice, visualization, threshold sweep, and checkpoint
  selection have never been run; development-test only
  (`FINAL_RESULT_LEDGER.md`, confirmation governance).

## Frozen assumptions

- FEM-to-voxel transfer is fixed analytic P1; a learned lifting module requires
  a new preregistered failure that P1 cannot satisfy
  (`REJECTED_METHOD_HYPOTHESES.md` section 1).
- Only `Q z_theta` may contribute voxel detail; no free residual bypass may be
  added (`REJECTED_METHOD_HYPOTHESES.md` section 4).
- Complement membership must be enforced by the fixed operator `Q`, not
  requested through a target or leakage penalty
  (`REJECTED_METHOD_HYPOTHESES.md` section 5).
- `ker(Pi_h)` is not the measurement null space `ker(A)`. On the fixed
  uniformly sampled domain the verified self-adjoint/idempotent operator
  supports the orthogonal-complement term; do not extend that claim to an
  uncertified continuous-domain inner product (canon section 7). Preferred
  terms: FEM-invariant voxel detail; coarse-space-invariant fine component;
  complement-constrained voxel detail.
- CQR/RGL/proposal/observability stack is excluded from the headline method
  (`REJECTED_METHOD_HYPOTHESES.md` section 3).

## Known ambiguities

- The learned voxel branch recovers only 8.68% (validation) / 9.86%
  (development-test) of the V4 detail oracle Dice headroom (+0.107520 /
  +0.107861); it is an Outcome B fine-detail enhancement, not the dominant
  performance mechanism (canon sections 8-9).
- The oracle study did not establish cleanly disjoint reconstruction failure
  modes; do not claim a stronger semantic separation than the evidence
  (canon section 8).
- Development-test is observed data; sealed confirmation is unopened and no
  freeze commit/tag exists yet (`FINAL_RESULT_LEDGER.md`).
- Active RTE comparison and encoder branches are candidate implementations
  pending numerical/training gates; they are not part of the frozen candidate
  and are train/val only (`AGENTS.md`).
