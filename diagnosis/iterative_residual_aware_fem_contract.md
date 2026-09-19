# Iterative Residual-Aware FEM Corrector Contract

Status: **IMPLEMENTED; ENGINEERING SMOKE PASSED; NO FULL-SCALE PERFORMANCE CLAIM.**

Post-training audit: the formal epoch-15 checkpoint reaches validation/test Dice
`0.70017/0.67957`, but inverse-target cosine is only `0.41849` and the 300-test
measurement-residual ratio is `2.08873`. The structural computation contract remains
valid; the stronger interpretation as an accurate, residual-decreasing inverse-state
solver is not established. See
`diagnosis/iterative_residual_aware_fem_2400_report.md`.

## Computation graph

For the frozen balanced-v2 Stage-1 state `x_h`, the model performs three shared-weight
updates entirely in the 19,990-node FEM space:

```text
x_h^(0) = x_h
measurement residual r_b^(t) = b - A x_h^(t)
node evidence = A^T r_b^(t) + local kNN + global FEM context + multiview
x_h^(t+1) = x_h^(t) + Corrector(node evidence, hidden^(t))
final = canonical_P1(x_h^(3))
```

There is no representation-complement target, voxel residual network, learned
lifting, observability split, or post-P1 learnable module.

## Data and physics semantics

- Stage 1: unchanged balanced-v2 `coarse_d.npy` support state.
- Inverse target: the certified `Pi_h(rho_gt) - x_h` from the 3000-sample
  decomposition cache.
- Binary GT: strict `gt_voxels > 0.05` on the same fixed valid domain.
- Forward system: full `A[7413, 19990]` from
  `shared_mesh_20k/system_matrix.A.npz`.
- Measurement convention: `use_visible_mask=false`, with per-sample max-normalized
  `measurement_b`, identical to balanced-v2 Stage 1.
- Residual sign: `b - A x`; the node-space descent evidence is `A.T @ (b - A x)`.
- P1 transfer: the certified sparse canonical operator only; it has no parameters.
- FEM coefficients are not clipped or clamped.

## Corrector architecture

- Three correction iterations with one shared cell.
- Two fixed-kNN contextual residual blocks per iteration.
- Local features include current/initial/history state, neighbor state statistics,
  normalized adjoint residual, diagonal-preconditioned residual, and neighbor
  residual statistics.
- Mesh-global mean/max hidden pooling and residual/state statistics generate FiLM
  scale/shift for every iteration.
- Multiview images are encoded once per forward and sampled at FEM node positions.
- Each intermediate FEM state is supervised against `Pi_h(rho_gt)`.
- Sampled projection supervision targets `I_h(Pi_h(rho_gt))`, not raw voxel residual.

## Capacity and smoke certification

```text
multiview encoder     483,297
FEM corrector         383,905
total                 867,202
learned transfer            0
voxel head                  0
```

The real 4-train/4-val, two-epoch smoke completed forward, backward, checkpointing,
dense evaluation, and independent checkpoint reload. Initial corrector gradient L1
was `1.53509` after disabling cosine loss at the zero-output initialization. The
smoke is an engineering check only and is not a performance pilot.

Six focused unit tests verify the fixed forward/adjoint residual, iterative state
updates and residual recomputation, zero-initialized Stage-1 identity, analytic P1
output, absence of voxel/lifting modules, gradient flow, and the 200k--500k corrector
capacity contract. A reloaded dense prediction satisfied
`final_prediction == step3_fem` exactly in float32.
