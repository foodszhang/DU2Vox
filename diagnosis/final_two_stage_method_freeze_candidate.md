# Final Two-Stage Method: Freeze Candidate

## Recommendation

Freeze the following architecture as an **Outcome B candidate** for review:

`Stage1 FEM -> frozen V4 Stage2A -> fixed P1 -> constrained voxel Stage2B`.

Do not run sealed confirmation yet. Confirmation requires an explicit user decision
that this architecture and protocol are final.

## Frozen computation

```text
measurement y
      -> Stage1 FEM inverse reconstruction x_h^(0)
      -> frozen V4, K=3: x_h^(1) -> x_h^(2) -> x_h^(3)
      -> fixed canonical P1: rho_c = I_h x_h^(3)
      -> local boundary-sensitive voxel MLP: z_theta
      -> fixed exact Q = I - I_h Pi_h: w_theta = Q z_theta
      -> rho_final = rho_c + w_theta
```

The voxel branch has no learned lifting, FEM-node output, direct final decoder, free
residual bypass, detail ReLU, or final clamp. V4 and its multiview encoder remain
frozen. The exact structural contract is

`Pi_h rho_final ~= x_h^(3)`.

Measured mean relative preservation error is `8.50e-9` on validation and `8.72e-9`
on development-test; detail coarse leakage is `4.47e-18` and `4.57e-18` respectively.

## Evidence supporting the candidate

| Evidence | Validation | Development-test |
| --- | ---: | ---: |
| V4 Dice | 0.732512 | 0.716529 |
| Constrained final Dice | 0.741844 | 0.727163 |
| `Delta_voxel` | +0.009332 | +0.010634 |
| Oracle detail Dice | 0.840032 | 0.824390 |
| Oracle headroom | +0.107520 | +0.107861 |
| Oracle recovery fraction | 0.08679 | 0.09859 |
| Detail cosine | 0.15715 | 0.16935 |

The matched unconstrained control reaches 0.743784 validation and 0.726636
development-test Dice, but changes the projected coarse state materially (leakage
0.616/0.627). It is therefore rejected as the final hierarchical method.

## Freeze interpretation

This candidate supports the paper description **hierarchical error allocation across
discretizations**:

- FEM space handles inverse localization, large-scale support, and physics-guided
  coarse correction.
- The voxel branch handles only the sampled FEM-null fine component.

Because validation improvement is between +0.005 and +0.01, describe Stage2B as a
fine-detail enhancement, not the dominant performance mechanism. No joint fine-tuning
was performed, despite passing its minimum authorization gate, in order to preserve
the clean frozen-V4 causal result.

## Protocol to freeze before confirmation

- full canonical 0.2-mm domain; no 0.4-mm fallback;
- binary support semantics and fixed threshold 0.5;
- width 160, three residual GELU MLP blocks, coordinate frequencies 6;
- frozen V4 epoch 15 and frozen shared multiview encoder;
- exact FP64 sparse Q forward/backward with FP32 constrained output;
- SmoothL1 + 0.1 MSE detail loss + 0.1 Tversky support loss;
- validation-selected epoch 3 checkpoint;
- raw, unclamped final prediction;
- no threshold sweep and no architecture/loss changes after freezing.

Only after explicit approval should `scripts/run_sealed_confirmation_suite.py` be
extended/used for the single permitted confirmation launch and receipt.

