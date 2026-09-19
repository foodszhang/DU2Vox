# DU2Vox Final Method Canon

## Status and authority

This document is the highest-priority scientific definition of the current DU2Vox
method. Use it before historical branches, experiment reports, or chat summaries.
Only confirmed method facts belong here.

Status: **development freeze candidate; sealed confirmation remains unopened**.

## 1. Core problem

DU2Vox is a two-stage coarse-to-fine Fluorescence Molecular Tomography (FMT)
reconstruction framework.

Stage 1:

```text
measurement -> coarse FEM inverse reconstruction x_h^(0)
```

Stage 2 contains two sequential levels:

1. physics-guided iterative refinement of the FEM inverse state;
2. coarse-invariant voxel fine-detail enrichment.

The final method does not learn FEM-to-voxel interpolation.

## 2. Stage 1: coarse FEM reconstruction

Stage 1 provides the coarse physical inverse solution:

```text
y -> x_h^(0)
```

The current reference development-test Dice after fixed canonical P1 evaluation is
`0.617379`. Stage 1 is the coarse FEM inverse reconstruction, not a voxel decoder.

## 3. Stage 2A: physics-guided FEM inverse refinement

Starting from `x_h^(0)`, V4 performs `K=3` shared-weight iterative updates. At
iteration `k`:

```text
p^(k) = A x^(k)

r_raw^(k) = y - A x^(k)

alpha_k = argmin_{alpha >= 0} ||y - alpha A x^(k)||^2
r_SI^(k) = y - alpha_k A x^(k)

g_raw^(k) = A^T r_raw^(k)
g_SI^(k)  = alpha_k A^T r_SI^(k)
```

The shared update cell combines:

- raw and scale-invariant physics evidence;
- adjoint and Jacobi/sensitivity-normalized directions;
- FEM mesh context;
- frozen seven-view optical features;
- current, initial, and historical FEM states;
- lightweight global conditioning.

The update is:

```text
x^(k+1) = x^(k) + Delta x_neural^(k) + Delta x_DC^(k)
```

Forward predictions, residuals, adjoints, and normalized evidence are recomputed at
every iteration.

Development-test progression:

| State | Dice | Increment |
| --- | ---: | ---: |
| `x_h^(0)` | 0.617379 | - |
| `x_h^(1)` | 0.649459 | +0.032081 |
| `x_h^(2)` | 0.693379 | +0.043920 |
| `x_h^(3)` | 0.716529 | +0.023150 |

Stage 2A therefore contributes `+0.099151` absolute Dice over Stage 1.

## 4. Canonical FEM-to-voxel transfer

`I_h` is the fixed canonical analytic P1 FEM-to-voxel operator. It is not learned.
It evaluates the corrected continuous P1 FEM field at canonical voxel centers; it
does not create new fine-scale information.

The certified domain is the complete set of 0.2-mm GT voxel centers contained in the
FEM tetrahedral mesh:

- grid shape: `(190, 200, 104)`;
- full grid voxels: 3,952,000;
- valid voxels: 1,677,645;
- FEM nodes: 19,990;
- support semantics: `gt_voxels > 0.05`;
- reconstruction threshold: `0.5`.

The cached sparse P1 matrix and the corresponding sampled-L2 projection are reused
directly from the 3000-sample decomposition experiment. No second projection
definition is permitted.

## 5. Stage 2B: complement-constrained voxel detail

Let:

```text
rho_c = I_h x_h^(3)
```

`Pi_h` is the canonical sampled voxel-to-FEM projection. The certified operator
contract is:

```text
Pi_h I_h ~= I
```

The measured maximum relative left-identity error is `1.50e-15`. On this fixed,
uniformly weighted sampled domain, `I_h Pi_h` is numerically self-adjoint and
idempotent. Define:

```text
Q = I - I_h Pi_h
```

The voxel network predicts an unconstrained signed proposal `z_theta`, but only

```text
w_theta = Q z_theta
```

may enter the reconstruction. The final output is exactly:

```text
rho_final = I_h x_h^(3) + Q z_theta
```

Therefore:

```text
Pi_h rho_final ~= x_h^(3)
```

The voxel branch cannot overwrite the physics-refined FEM state. It introduces only
FEM-invariant voxel degrees of freedom.

The current Stage 2B candidate uses:

- full 0.2-mm canonical domain, without the 0.4-mm fallback;
- width 160 and three residual GELU MLP blocks;
- six-frequency coordinate encoding;
- V4 P1 value and exact local P1 gradient;
- four local FEM values and barycentric weights;
- local mean, standard deviation, and range;
- V4 support-boundary proximity;
- frozen multiview optical features;
- exact FP64 sparse `Q` forward/backward with FP32 constrained output;
- SmoothL1 detail loss + `0.1` MSE + `0.1` Tversky support loss;
- raw unclamped output and fixed threshold `0.5`.

Development results:

```text
V4 coarse:             0.716529
V4 + constrained Qz:  0.727163
Delta voxel:          +0.010634
```

Mean detail coarse leakage is `4.57e-18`, and mean relative coarse-state
preservation error is `8.72e-9` on development-test.

## 6. Why the architectural constraint matters

The matched unconstrained residual control also improves Dice:

```text
0.716529 -> 0.726636
```

However, `62.68%` of its development-test residual energy projects back into the FEM
coarse space. Its relative coarse-state change is `0.2607`. It therefore substantially
rewrites the already refined FEM solution and is not a genuine fine-domain branch.

The constrained model gives comparable or slightly better development reconstruction
while reducing coarse leakage to numerical zero. The complement constraint is
architectural, not merely encouraged by a loss.

## 7. Mathematical interpretation

The core principle is not merely "FEM handles low frequency and voxels handle high
frequency." The precise statement is:

- the FEM stages reconstruct the component representable in the FEM approximation
  space;
- the voxel stage adds only a component in `ker(Pi_h)`, leaving the corrected FEM
  state invariant.

Under the verified `Pi_h I_h ~= I` contract:

```text
V_voxel = Range(I_h) direct-sum ker(Pi_h)
```

On the fixed uniformly sampled domain, the verified self-adjoint/idempotent operator
also supports the term orthogonal complement. Do not extend that claim to an
uncertified continuous-domain inner product.

`ker(Pi_h)` is not the measurement null space `ker(A)`. Never call the voxel detail:

- a physical null-space component;
- an unobservable component;
- a measurement-null component.

Preferred terminology:

- FEM-invariant voxel detail;
- coarse-space-invariant fine component;
- complement-constrained voxel detail.

## 8. Empirical support for the decomposition

The 3000-sample cross-discretization analysis found:

```text
inverse-discrepancy energy:          72.8%
representation/detail energy:       27.2%

detail energy within 0.2 mm:        69.3%
detail energy within 0.6 mm:        89.9%
detail autocorrelation at 0.6 mm:   0.092
inverse autocorrelation at 0.6 mm:  0.821
```

Thus the decomposition is algebraically certified and the two components have
different spatial statistics. This does not mean the learned voxel branch recovers
all 27.2% representation energy.

The V4-specific detail oracle gain is `+0.107520` on validation and `+0.107861` on
development-test. The learned branch recovers only `8.68%` and `9.86%` of that Dice
headroom, respectively.

The earlier oracle study did not establish cleanly disjoint reconstruction failure
modes: inverse and representation oracles improved overlapping metric families.
Do not claim a semantic failure-mode separation stronger than the evidence.

## 9. Current development performance

| Method | Val Dice | Development-test Dice |
| --- | ---: | ---: |
| Stage 1 P1 | 0.632646 | 0.617379 |
| V4 FEM refinement | 0.732512 | 0.716529 |
| V4 + constrained voxel detail | 0.741844 | 0.727163 |

From Stage 1 to the final candidate:

- absolute Dice improvement: `+0.109784`;
- relative Dice improvement: approximately `17.8%`;
- fraction of the original Dice error removed: approximately `28.7%`;
- FEM refinement contribution: `+0.099151`, approximately `90.3%` of the gain;
- voxel detail contribution: `+0.010634`, approximately `9.7%` of the gain;
- HD95: `2.6047 -> 2.0045` mm;
- localization: `1.6899 -> 1.2679` mm;
- weak recall: `0.7098 -> 0.7503`.

Because validation gain from Stage 2B is `+0.009332`, the voxel branch is currently
an Outcome B fine-detail enhancement, not the dominant performance mechanism.

## 10. Final methodological claim

Do not claim:

- learned FEM-to-voxel transfer;
- voxel super-resolution created by P1;
- the first learned iterative FEM method;
- measurement-null-space reconstruction;
- a continuous orthogonal complement without an inner-product proof;
- that every implementation component is an innovation;
- that the learned voxel branch recovers all representation discrepancy;
- final unseen generalization from the observed development-test.

The central claim is:

> DU2Vox performs hierarchical reconstruction across heterogeneous discretizations.
> A physics-guided iterative FEM stage first corrects the coarse inverse component
> while retaining the explicit forward model. After fixed analytic P1 transfer, a
> structurally constrained voxel branch adds only fine degrees of freedom that leave
> the corrected FEM state invariant.

In concise form:

```text
correct what the coarse physical space can represent,
then add only what that coarse space cannot represent.
```

## 11. Frozen architecture candidate

```text
Stage 1:
    balanced-v2 FEM reconstruction

Stage 2A:
    frozen V4 epoch 15
    K = 3
    raw + scale-invariant evidence
    shared FEM update cell
    multiview encoder
    FEM mesh and global context

Transfer:
    fixed canonical analytic P1

Stage 2B:
    validation-selected epoch 3 voxel-detail checkpoint
    exact Q = I - I_h Pi_h
    full canonical 0.2-mm domain

Threshold:
    0.5

Sealed confirmation:
    NOT OPENED
```

Excluded from the candidate:

- learned lifting;
- observability partition as a reconstruction branch;
- CQR/RGL as a headline mechanism;
- unconstrained voxel residual bypass;
- joint fine-tuning;
- threshold calibration;
- FEM ensembles as the primary method.

## 12. Source artifacts

- `diagnosis/voxel_complement_operator_audit.md`
- `diagnosis/v4_voxel_detail_oracle_report.md`
- `diagnosis/complement_voxel_detail_development_report.md`
- `diagnosis/final_two_stage_method_freeze_candidate.md`
- `experiments/cross_discretization_decomposition/artifacts/REPORT.md`

