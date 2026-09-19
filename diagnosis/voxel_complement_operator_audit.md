# Voxel Complement Operator Audit

## Result

The canonical operator contract passes on the unchanged 0.2-mm decomposition domain. `Pi_h I_h` is a numerical left identity, and `I_h Pi_h` is an idempotent, self-adjoint sampled-L2 projector under the uniform voxel inner product. Therefore `Q = I - I_h Pi_h` may be described as the orthogonal complement projector for this fixed sampled domain (not as a continuous-domain claim).

## Required audit

1. Exact code path for `Pi_h`: `experiments/cross_discretization_decomposition/decomposition.py::MassProjector.coefficients`. It uses sparse LU to solve `(P.T @ P)c = P.T @ rho`, with zero ridge for this certified cache.
2. Exact code path for `I_h`: `experiments/cross_discretization_decomposition/decomposition.py::DomainOperator.p`, loaded by `du2vox/bridge/canonical_cross_discretization.py::CanonicalCrossDiscretization`; application is the fixed CSR product `P @ c`.
3. Identical to decomposition experiment: **yes**. Production imports the experiment's `load_operator` and `MassProjector` directly.
4. `||Pi_h I_h c - c||_2 / ||c||_2` (maximum of three seeded random vectors): `1.4998521e-15`.
5. Decomposition closure error `||rho - I_h Pi_h rho - w*||_2 / ||rho||_2`: mean `0`, maximum `0` across val300 + development-test300.
6. GT complement leakage `||I_h Pi_h w*||_2^2 / ||w*||_2^2`: mean `1.303281e-29`, maximum `4.0277173e-29`.
7. Domain/grid: canonical GT-center grid, shape `(190, 200, 104)`, isotropic `0.2` mm, C-order, binary support `gt_voxels > 0.05`.
8. Valid voxel count: `1,677,645` of `3,952,000`; FEM nodes: `19,990`.

Additional numerical checks: relative idempotence error of `I_h Pi_h` is at most `1.3307969e-15`; seeded bilinear self-adjointness error is at most `2.8964287e-15`.

No confirmation data or confirmation inference was used.
