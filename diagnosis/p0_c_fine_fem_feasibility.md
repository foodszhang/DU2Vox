# P0-C matched fine-FEM feasibility result

Date: 2026-09-07

## Verdict

**Feasibility failure / no-go under the preregistered operator.** The full 54-case
residual-interrogation audit was not started. The frozen V4 and hard-`Q`,
state-conditioned complementary prior remain unchanged.

This is a numerical operator-contract result, not evidence that fine detail is
physically unobservable.

## Operator and provenance

- Definition: `B_H = C_h M_H^-1 P_H^T W_v`, applied matrix-free.
- Coarse mesh SHA256:
  `718cb70f0b12c3c8f5e7f10d9e216295efc79ff3221602b934ba04c137f34c64`.
- Frame manifest SHA256:
  `25b77f5d11430ebf4d9735f19534782afb56f7fe5cab525a569028b0f3c02e2b`.
- Fine mesh: 146,599 nodes and 793,672 tetrahedra.
- Detector nodes: the unchanged 7,413 coarse surface nodes.
- Canonical valid domain: 1,677,645 voxel centers; `W_v = 0.008 I`.
- `M_H` action hash:
  `aa9454192d77db6a2a71d86c2786b1b0433ad5dcc37414a2128ff389dc30062a`.
- `P_H` action hash:
  `37a9365155804d3cb4bcaa1ce44edd6dd3df57441cafa9a82643f49b8ae0f405`.

The actual saved coarse `K` and `C` matrices were used to infer the piecewise
constant diffusion and absorption coefficients. This avoids relying on a nominal
dataset-manifest copy that does not reproduce the saved system matrix exactly.

## Numerical checks

| Check | Result | Gate | Status |
| --- | ---: | ---: | --- |
| forward/adjoint dot-product relative error | `9.45e-10` | `<= 1e-7` | pass |
| FP64 CG relative residual | `7.79e-9` to `9.95e-9` in 22 recorded solves | `<= 1e-8` | pass |
| `B_H I_h` vs `A`, 20-action mean | `15.455%` | `<= 5%` | **fail** |
| `B_H I_h` vs `A`, 20-action P95 | `15.814%` | `<= 10%` | **fail** |

The observed mismatch persists after matching the saved coarse optical operator. It
is therefore attributable to the proposed canonical-voxel quadrature extension
versus the original FEM source-mass action, rather than CG convergence or a stale
optical-parameter file.

Per the preregistration, the failure occurred before the five-case/full-54 mechanism
study, randomized probes/spectrum, frozen reconstruction reruns, or MCX secondary
study. No network was trained, no reconstruction checkpoint/config was changed, and
sealed confirmation remained unopened.

## D0 interpretation remains unchanged

For the canonical D0 extension `B_0=A Pi_h`, `B_0 I_h=A` and `B_0 Q=0` hold to
numerical precision. The implementation-level generator contract is
`measurement_b = maximum(A @ gt_nodes, 0)` (the maximum only clips small negative
numerical surface values), while `gt_voxels` is an independent sample of the same
analytic source. Original D0 residuals may have shared-source statistical
association with voxel detail, but cannot establish direct measurement support for
`Q rho*`.
