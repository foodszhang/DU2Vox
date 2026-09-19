# P0-1 Weighted L2 Projection Contract

## Result

**Passed.** The unchanged production projection is the weighted sampled-L2 projection

```text
Pi_h = (P^T W P)^-1 P^T W
I_h  = P
P_h  = I_h Pi_h
Q    = I - P_h
```

On the certified uniform 0.2-mm domain, `W = 0.0080000000000000019 I`.
The scalar weight cancels exactly, so the existing production solve
`(P^T P)^-1 P^T` is the same operator. No second projection was introduced.

## Certified domain and inner product

- Scope: fixed uniform 0.2-mm GT-center samples inside the certified FEM domain.
- Grid shape: `(190, 200, 104)`.
- Valid voxel centers: `1677645`; FEM nodes: `19990`.
- Inner product: `<u,v>_W = u^T W v` with voxel volume in mm^3.

This contract is limited to the fixed sampled domain. It is not a continuous-domain
orthogonality claim, and a nonuniform sampled domain would require explicit `W`.

## Numerical audit

| Identity | Relative error |
| --- | ---: |
| `pi_h_i_h_relative_l2` | `1.524310e-15` |
| `p_h_idempotence_relative_l2` | `1.332608e-15` |
| `q_idempotence_relative_l2` | `1.583716e-16` |
| `p_h_weighted_self_adjoint_relative` | `8.372793e-15` |
| `p_h_q_relative_l2` | `1.543139e-16` |
| `pi_h_q_relative_l2` | `3.223848e-17` |
| `weighted_pythagorean_relative` | `1.356292e-16` |
| `final_coarse_preservation_relative_l2` | `1.544339e-15` |

All FP64 core errors are below `1e-12`.

## Cache identity

- `P_full_fem_gt_centers.npz`: `e70c135c7e67a58427a4c956e35cd5e115884f7eecc64bb58081f20ffcfbc742`
- `domain_arrays.npz`: `1fc5bed7ca1708f42e9c4786b94583232d1c0305f4fc11715dd68f8bc584494b`
- `operator_metadata.json`: `9258b9dcec6ba2f65dd72a4679378a0b2d49b858ff3a2c6a895551d5cb9a4899`

No confirmation data or confirmation inference was used.
