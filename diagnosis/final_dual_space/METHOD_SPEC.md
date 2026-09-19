# Correction-State Cross-Space Transfer (CST)

## Frozen approximation-space contract

Stage I maps measurements to an initial FEM state `x0`. Stage II first applies the
frozen three-iteration V4 corrector and then reconstructs only the complementary
voxel component:

```text
x0, Y, G, coarse-side views -> V4 -> xc, H0, Hc
xc - x0 = Delta x
Hc - H0 = Delta H
rho_hat = I_h xc + Q z_theta
Q = I - I_h Pi_h
```

`I_h`, `Pi_h`, and `Q` are the certified cached analytic operators. They are not
learned or approximated. There is no voxel residual bypass around `Q`.

## Query-local correction state

For every canonical voxel query, the containing tetrahedron and cached barycentric
weights define:

```text
s(q) = [x0(q), xc(q), Delta x(q),
        grad x0|e, grad xc|e, grad Delta x|e,
        lambda(q), G_e]
```

`G_e` is the flattened analytic P1 basis-gradient matrix for the tetrahedron. The
descriptor has 28 channels and contains no learned interpolation or oracle input.

The four nodal hidden values are interpolated by the same barycentric weights:

```text
h0(q) = sum_i lambda_i H0_i
hc(q) = sum_i lambda_i Hc_i
Delta h(q) = hc(q) - h0(q)
```

`H0` is the frozen V4 shared-cell hidden after its first invocation and before the
first scalar state update is applied. `Hc` is the terminal hidden after the third
invocation. Both therefore have the same 144-D representation and need no semantic
adapter before subtraction. Their FP16 caches are identity- and error-audited.

## Bounded structured transfer

```text
a = phi_s(s)
b = phi_b(hc)
u = phi_u(Delta h)                 # selected transition representation may vary
m = tanh(W_m a)                    # 8 or 16 group gates
alpha = sigmoid(alpha_logit)       # strictly in (0, 1)
u_tilde = u * (1 + alpha * expand_groups(m))
t = b + u_tilde
```

The ordinary residual MLP decoder receives coordinate PE, `s`, `t`, and only those
conditional features that pass validation gates. Its zero-initialized output head
starts at pure analytic P1. Modulation is local, bounded, and occurs before hard-Q.

The optional one-ring variant uses one residual message-passing layer over shared
tetrahedral faces (at most four neighbors). The optional direct-view variant reuses
the frozen 32-D V4 view features. Neither is retained without its preregistered
paired validation gain.

## Backward responsibility

The main cached-backbone experiment enforces:

```text
d(detail/final loss) / d(V4 coarse parameters) = 0
```

Thus the voxel objective cannot rewrite `xc` or `Hc`. The V4 continuation record and
the matched unrestricted joint control are used to decide whether independent
coarse-side continuation adds value; if it does not, the validation-selected V4
epoch remains frozen. In all cases the forward identity is exact:

```text
Pi_h rho_hat ~= xc
```

## Fixed training and evaluation contract

- train/validation/development-test: 2400/300/300 fixed samples;
- seed: 20260901;
- five epochs for each matched voxel candidate;
- AdamW, LR `1e-4`, weight decay `1e-5`;
- SmoothL1 detail + `0.1` detail MSE + `0.1` final Tversky;
- six-frequency coordinate encoding;
- raw outputs, GT support `>0.05`, reconstruction threshold `0.5`;
- exact FP64 sparse hard-Q forward/backward;
- architecture and checkpoint selection from validation300 only;
- no sealed-confirmation access.

## Final validation outcome

Eight formal candidates were trained on the fixed 2400/300 split and every one was
evaluated casewise on the complete validation300 set before any development-test
access. Decoder widths were matched so that all CST arms sit within `0.5%` of the
193,121-parameter sequential-concat baseline B4.

Mechanism ablation (paired 10,000-draw bootstrap on validation300, fixed threshold
`0.5`):

| Comparison | Mean Dice difference | 95% CI | Decision |
| --- | ---: | --- | --- |
| Best CST (`cst_d128_g16_l3_one_ring`) vs B4 | +0.000390 | `[-0.000327, +0.001125]` | claim **not** supported |
| one-ring vs plain `Delta h` reference | +0.001378 | `[+0.000733, +0.002023]` | retained |
| direct views vs reference | +0.000268 | `[-0.000166, +0.000718]` | rejected (below `+0.002` and CI spans 0) |
| Hc + `Delta H` vs Hc-only | +0.000581 | `[-0.000024, +0.001174]` | not significant |
| `[Hc, Delta H]` adapter vs plain `Delta h` | -0.000194 | `[-0.000588, +0.000189]` | no gain |

Validation-selected CST: `cst_d128_g16_l3_one_ring`, epoch 5, validation300 Dice
`0.7477080374447118`.

Because the best CST candidate's paired validation confidence interval against B4
includes zero, the validation-frozen final method is the strong sequential-concat
baseline **B4**, not CST. This decision was recorded in `VALIDATION_FREEZE.json`
before the single development-test300 evaluation.

The frozen CST candidate was still evaluated once on development-test300 as a
mechanism arm: it reached `+0.000988` paired Dice over B4 with CI
`[+0.000200, +0.001812]` (Holm-adjusted `p = 0.0157`). This same-direction but
non-transferring result does not reopen model selection; per protocol the final
model stays B4 and the "correction-state-aware transfer" claim is downgraded to a
reported mechanism study rather than the headline contribution.

Constrained-output structural checks on the evaluated CST candidate remain at
numerical zero: mean coarse leakage `4.50e-18` and mean relative coarse-state
preservation error `8.77e-09`.
