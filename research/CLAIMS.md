# Scientific claims ledger

Do not promote a claim from OPEN to SUPPORTED without explicit evidence review.

All rows were captured during the 2026-10-07 Research OS audit from
`diagnosis/FINAL_METHOD_CANON.md` and `diagnosis/FINAL_RESULT_LEDGER.md`. They
remain OPEN: the supporting evidence is development-test scope, and the sealed
confirmation set is unopened. The Evidence column cites repository documents,
not external review.

| ID | Claim | Evidence | Status | Paper/Figure use |
|---|---|---|---|---|
| C-001 | DU2Vox performs hierarchical reconstruction across heterogeneous discretizations: a physics-guided iterative FEM stage first corrects the coarse inverse component while retaining the explicit forward model; after fixed analytic P1 transfer, a structurally constrained voxel branch adds only fine degrees of freedom that leave the corrected FEM state invariant. | `FINAL_METHOD_CANON.md` sections 9-11; `FINAL_RESULT_LEDGER.md` | OPEN (development-test only; sealed confirmation not run) | headline method claim (pending confirmation) |
| C-002 | The constrained voxel branch preserves the corrected FEM state (`Pi_h rho_final ~= x_h^(3)`): measured coarse leakage 4.57e-18 (development-test) / 4.47e-18 (validation), versus 0.62679 for the matched unconstrained control. | `FINAL_METHOD_CANON.md` sections 5-6; `FINAL_RESULT_LEDGER.md` structural ledger | OPEN (fixed sampled-domain operator check; no continuous-domain proof; confirmation not run) | architecture / invariance claim |
| C-003 | The gain over Stage 1 is dominated by FEM refinement: Stage 1 -> V4 +0.099151 Dice (approximately 90.3%), versus V4 -> constrained `Qz` +0.010634 development-test (+0.009332 validation, approximately 9.7%). | `FINAL_METHOD_CANON.md` section 9; `FINAL_RESULT_LEDGER.md` increment ledger | OPEN (development-test only; confirmation not run) | attribution / ablation reporting |
