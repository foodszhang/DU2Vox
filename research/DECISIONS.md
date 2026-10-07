# Accepted decisions

Record decisions only after they are explicitly accepted.

These entries capture rules that the repository's scientific governance
documents state as final. Source documents do not record decision dates;
entries were captured during the 2026-10-07 Research OS audit. The architecture
freeze itself is still pending; see `research/STATE.md` open questions.

## Entry format

### D-XXX — Title

- Date:
- Status: ACCEPTED | SUPERSEDED
- Decision:
- Evidence:
- Consequences:

## Entries

### D-001 — Fixed analytic P1 is the only FEM-to-voxel transfer

- Date: audit capture 2026-10-07 (source date not recorded)
- Status: ACCEPTED
- Decision: FEM-to-voxel transfer is the fixed canonical analytic P1 operator.
  Do not add a learned transport-consistent lifting module without a new,
  preregistered failure that P1 cannot satisfy.
- Evidence: `diagnosis/REJECTED_METHOD_HYPOTHESES.md` section 1;
  `diagnosis/FINAL_METHOD_CANON.md` sections 4, 11.
- Consequences: transfer is not trainable; no learned lifting in the frozen
  candidate.

### D-002 — Voxel detail enters only as `Q z_theta`

- Date: audit capture 2026-10-07 (source date not recorded)
- Status: ACCEPTED
- Decision: Only `w_theta = Q z_theta` with `Q = I - I_h Pi_h` may enter the
  reconstruction: `rho_final = I_h x_h^(3) + Q z_theta`. No free residual
  bypass or unconstrained representation head may be added; complement
  membership is enforced by the fixed operator `Q`, not by a loss or target.
- Evidence: `diagnosis/FINAL_METHOD_CANON.md` sections 5-6;
  `diagnosis/REJECTED_METHOD_HYPOTHESES.md` sections 4-5.
- Consequences: matched unconstrained control (`0.62679` coarse leakage on
  development-test) is control-only and contract-invalid as the method.

### D-003 — Fixed threshold 0.5, no threshold calibration

- Date: audit capture 2026-10-07 (source date not recorded)
- Status: ACCEPTED
- Decision: Use raw, unclamped predictions with the fixed `0.5` threshold. Do
  not run a new threshold sweep during development or confirmation.
- Evidence: `diagnosis/FINAL_METHOD_CANON.md` sections 4, 11;
  `diagnosis/REJECTED_METHOD_HYPOTHESES.md` section 8 (V3 calibration at 0.465
  did not transfer to development-test).
- Consequences: threshold is frozen; validation-tuned thresholds are excluded.

### D-004 — No joint Stage 2A/2B fine-tuning in the frozen candidate

- Date: audit capture 2026-10-07 (source date not recorded)
- Status: ACCEPTED
- Decision: Do not silently joint-tune V4 and the voxel branch. Any future
  joint study must retain `w = Qz`, use a preregistered low V4 learning rate,
  and remain separate from the frozen candidate result.
- Evidence: `diagnosis/REJECTED_METHOD_HYPOTHESES.md` section 9;
  `diagnosis/FINAL_METHOD_CANON.md` section 11 (excluded from the candidate).
- Consequences: causal isolation of the voxel-only improvement is preserved.

### D-005 — Sealed confirmation stays closed until explicit freeze

- Date: audit capture 2026-10-07 (source date not recorded)
- Status: ACCEPTED
- Decision: Do not open confirmation data or use `--allow-confirmation-eval`
  before `diagnosis/v4_frozen_protocol.md` exists. The final comparison must be
  launched once through `scripts/run_sealed_confirmation_suite.py`; its receipt
  prevents a rerun. Never quote development-test performance as final unseen
  generalization.
- Evidence: `diagnosis/FINAL_METHOD_CANON.md` section 11;
  `diagnosis/FINAL_RESULT_LEDGER.md` (confirmation governance); `AGENTS.md`.
- Consequences: confirmation inference/Dice/visualization/threshold/checkpoint
  selection remain recorded as not run; the 300-sample test is a
  development-test only.

### D-006 — Terminology and claim discipline

- Date: audit capture 2026-10-07 (source date not recorded)
- Status: ACCEPTED
- Decision: Do not call `ker(Pi_h)` the measurement null space `ker(A)`, and do
  not claim a continuous orthogonal complement without an inner-product proof.
  Do not claim learned FEM-to-voxel transfer, voxel super-resolution created
  by P1, the first learned iterative FEM method, measurement-null-space
  reconstruction, that every implementation component is an innovation, that
  the learned voxel branch recovers all representation discrepancy, or final
  unseen generalization from the observed development-test.
- Evidence: `diagnosis/FINAL_METHOD_CANON.md` sections 7, 10.
- Consequences: preferred language is FEM-invariant voxel detail /
  coarse-space-invariant fine component / complement-constrained voxel detail.
