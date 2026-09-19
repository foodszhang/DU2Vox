# Error-Structured Cross-Discretization Bridge Contract

Status before full training: **STRUCTURAL CONTRACT PASSED; smoke leakage requires
full-training monitoring.**

Post-training audit: **METHOD CONTRACT FAILED.** The 300-test predicted
representation leakage is `0.4525188127`, compared with target leakage
`2.2242911861e-15`. Predicted representation cosine is `0.1083858997` against its
intended target and `0.1009001358` against the inverse target. The runtime computation
and fixed targets remain structurally correct, but the learned separation is not
strong enough. See `diagnosis/error_structured_cross_discretization_2400_report.md`.

This certification is restricted to binary source-support space. It does not make a
raw-intensity reconstruction claim.

## Certified operator and data identity

- Canonical experiment: `experiments/cross_discretization_decomposition`.
- Canonical fixed domain: every 0.2 mm GT voxel center contained in the complete FEM
  tetrahedral mesh; 1,677,645 of 3,952,000 centers.
- FEM mesh: complete 19,990-node P1 mesh from
  `/home/foods/pro/FMT-SimGen/output/shared_mesh_20k/mesh.npz`.
- Mesh SHA256: `718cb70f0b12c3c8f5e7f10d9e216295efc79ff3221602b934ba04c137f34c64`.
- Frame-manifest SHA256:
  `25b77f5d11430ebf4d9735f19534782afb56f7fe5cab525a569028b0f3c02e2b`.
- Sparse P cache:
  `experiments/cross_discretization_decomposition/artifacts/operator_cache/P_full_fem_gt_centers.npz`;
  shape 1,677,645 x 19,990 with 6,710,580 entries.
- Valid mask / mapping: the cache's `domain_arrays.npz::valid_flat_indices`, in the
  same C-order used by the decomposition experiment. ESCB samples rows of this fixed
  array and never derives a GT-, ROI-, CQR-, or prediction-dependent domain.
- Split mapping is unchanged: 2400/300/300. Split SHA256 values are train
  `870ec2c...6c3a66c`, val `43c7549...b48598`, and test
  `17bdd83...9073e3`.

## Required questions

1. **Which exact code defines Pi_h?**

   `MassProjector.coefficients()` in
   `experiments/cross_discretization_decomposition/decomposition.py`. It solves
   `(P.T @ P)c = P.T @ rho` on all 19,990 active columns with sparse LU and no ridge.
   Production code imports that exact class through
   `du2vox/bridge/canonical_cross_discretization.py`; it does not contain a second
   projection implementation.

2. **Which exact code defines I_h?**

   The certified sparse `DomainOperator.p`, built by `build_domain_operator()` in the
   same experiment and loaded from the certified P cache. Dense evaluation applies
   `P @ coefficients`. Training takes the same cached CSR row's four node indices and
   barycentric values and calls the parameter-free `canonical_p1_torch()`.

3. **Is it identical to the 3000-sample decomposition experiment?**

   **YES.** The experiment's `load_operator()` and `MassProjector` are imported
   directly. Runtime checks pin the domain definition, P shape/nnz, mesh hash, frame
   hash, and active-column count.

4. **What exactly is x_h?**

   The frozen balanced-v2 Stage-1 `coarse_d.npy` for the corresponding unchanged
   split/sample. It is a 19,990-element sigmoid support prediction. No Stage-1
   retraining or conversion is performed. In inspected `sample_0000`, its range is
   `[9.69e-14, 0.999937]`.

5. **What exactly is Pi_h rho_gt?**

   The unconstrained sampled-L2 P1 coefficient vector obtained by applying the exact
   `MassProjector` to `(gt_voxels.ravel()[valid_flat_indices] > 0.05).astype(float64)`.
   It is not `gt_nodes.npy`. As in the decomposition report, it can be negative or
   exceed one; inspected `sample_0000` ranges from -0.27247 to 1.36656.

6. **What exactly is inverse target?**

   `Pi_h(rho_gt) - x_h`, in the 19,990-dimensional FEM nodal space. Phase A and C
   supervise this target (equivalently the corrected state against `Pi_h(rho_gt)`).

7. **What exactly is representation target?**

   `(gt_voxels > 0.05) - P @ Pi_h(gt_voxels > 0.05)` on the canonical valid centers.
   Code constructs it through `fixed_representation_target(gt, pi_gt_voxel)`; the
   function has no inverse prediction argument.

8. **Can representation target change when inverse prediction changes?**

   **NO.** Unit test `test_representation_target_is_independent_of_inverse_prediction`
   enforces the function signature and bit-identical target under different inverse
   corrections. Training also compares the dataset target to a fresh fixed-target
   construction before every loss.

9. **Is FEM-to-voxel transfer learnable?**

   **NO.** It is a four-term barycentric sum using rows from the certified sparse P.
   There is no lifting module or transfer parameter.

10. **Can the representation network directly alter FEM nodal state?**

    **NO.** It receives the already corrected P1 context and returns only a query-space
    scalar. Final output is exactly `corrected_fem_voxel + representation_prediction`.

11. **Are Phase A/B/C gradient routes correct?**

    **YES.** Real 4-train/4-val multiview smoke results:

    - A: inverse grad L1 `0.558752`; representation grad L1 `0`.
    - B: inverse grad L1 `0`; representation grad L1 `0.000244851`.
    - C: inverse grad L1 `4.81522`; representation grad L1 `0.114125`.

    The shared view encoder is trainable in A, frozen in B (so corrected FEM cannot
    drift while representation pretrains), and trainable in C.

12. **What is GT representation-target coarse-space leakage?**

    Mean across the four dense smoke validation cases:
    `2.4836697100145384e-15`. This uses a fresh canonical Pi_h solve followed by P.

13. **What is predicted representation leakage after smoke?**

    Mean `0.999992457999231` after only one four-sample epoch per phase. This is
    high and is explicitly **not** evidence of learned complement structure: the
    zero-initialized head has only moved to a tiny near-constant field. It is a required
    warning for full training. If leakage remains high after meaningful convergence,
    the final method contract must be marked failed; no projector will be added to hide
    it.

14. **Does oracle decomposition still close numerically?**

    **YES.** The certified 3000-case worst relative closure is `4.94e-17`. The new
    real-sample unit test independently reconstructs
    `P x_h + P(Pi_h rho_gt - x_h) + rho_gt - P Pi_h rho_gt` and passes a `1e-12`
    relative tolerance.

15. **Are binary-support semantics identical to the decomposition experiment?**

    **YES.** Both use the strict threshold `gt_voxels > 0.05`, float support targets,
    the same valid indices, and Stage-1 balanced-v2 outputs trained with
    `binarize_gt=true`, `binarize_threshold=0.05`. Reconstruction metrics threshold
    predictions at 0.5. ESCB applies no output clipping or clamp. Pi_h is deliberately
    unconstrained, matching the experiment. The only clipping occurred historically
    while certifying tiny numerical barycentric tolerance in P construction; the
    cached P itself is reused unchanged.

## Smoke and dependency gates

- Seven semantic tests pass in `tests/test_error_structured_bridge_contract.py`,
  including the upgraded inverse-corrector capacity gate. Two additional tests verify
  that vectorized multiview encoding and projection sampling preserve eval semantics.
- Sequential forward identity is exact under `torch.equal`.
- No import or parameter dependency on `CQRObservabilityProjector`, `P_mu`,
  `I-P_mu`, `TransportConsistentCQRLifter`, CQR, RGL, or learned lifting.
- The FEM inverse corrector is a three-block lightweight contextual network. Each
  block aggregates fixed-mesh kNN hidden features, applies a learned contextual
  update, and combines it with the nodal state through a learned residual gate and
  LayerNorm. It has 318,817 parameters, compared with 483,297 in the shared multiview
  encoder and 291,585 in voxel representation completion.
- Dense smoke checkpoints and JSON diagnostics for the certified upgraded model were
  written under `runs/escb_contextual318k_b4_contract_smoke/`.
- ESCB has 1,093,699 parameters; the strong plain target-first control has 774,882.
  Both use the same query rows, PE, multiview encoder design, projection normalization,
  Stage-1 inputs, data, and dense validation domain. The additional ESCB capacity is
  confined to the FEM-domain corrector rather than a learned transfer or voxel head.

The three mandatory answers (8, 9, 10) are all **NO**. Full target precomputation and
the single-seed 2400/300 validation runs are therefore authorized by the method gate.
