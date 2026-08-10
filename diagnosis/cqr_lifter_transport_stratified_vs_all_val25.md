# CQR P1 Transport Consistency

The four-point degree-2 rule integrates the product `N_i * rho_P1` exactly. The reference local matrix is the generator's source mass, `V/20 * (ones(4,4) + eye(4))`.

- Samples: 25
- Source-load max relative error: `3.233840e-08`
- Compressed-measurement max relative error: `2.920195e-08`
- P1/local-mass gate passed: **True**

## Per sample

| sample | tets | depth | foci | source error | compressed error |
| --- | ---: | --- | ---: | ---: | ---: |
| sample_0001 | 1024 | medium | 2 | 3.108e-08 | 2.657e-08 |
| sample_0008 | 1024 | medium | 2 | 3.093e-08 | 2.698e-08 |
| sample_0013 | 1024 | deep | 2 | 3.142e-08 | 2.768e-08 |
| sample_0014 | 1024 | shallow | 2 | 3.100e-08 | 2.788e-08 |
| sample_0026 | 1024 | medium | 2 | 3.102e-08 | 2.858e-08 |
| sample_0029 | 1024 | deep | 1 | 3.065e-08 | 2.719e-08 |
| sample_0039 | 1024 | deep | 1 | 3.149e-08 | 2.677e-08 |
| sample_0047 | 1024 | shallow | 1 | 3.130e-08 | 2.789e-08 |
| sample_0060 | 1024 | medium | 1 | 3.089e-08 | 2.462e-08 |
| sample_0086 | 1024 | deep | 1 | 3.066e-08 | 2.651e-08 |
| sample_0101 | 1024 | deep | 2 | 3.111e-08 | 2.606e-08 |
| sample_0102 | 1024 | shallow | 2 | 3.168e-08 | 2.837e-08 |
| sample_0131 | 1024 | medium | 3 | 3.132e-08 | 2.715e-08 |
| sample_0147 | 1024 | medium | 1 | 3.164e-08 | 2.725e-08 |
| sample_0156 | 1024 | shallow | 2 | 3.102e-08 | 2.641e-08 |
| sample_0162 | 1024 | medium | 2 | 3.183e-08 | 2.792e-08 |
| sample_0171 | 1024 | shallow | 3 | 3.131e-08 | 2.576e-08 |
| sample_0177 | 1024 | shallow | 3 | 3.145e-08 | 2.663e-08 |
| sample_0181 | 1024 | medium | 3 | 3.234e-08 | 2.522e-08 |
| sample_0205 | 1024 | medium | 2 | 3.137e-08 | 2.920e-08 |
| sample_0208 | 1024 | medium | 3 | 3.132e-08 | 2.701e-08 |
| sample_0221 | 1024 | deep | 1 | 3.188e-08 | 2.493e-08 |
| sample_0229 | 1024 | deep | 2 | 3.143e-08 | 2.639e-08 |
| sample_0232 | 1024 | medium | 1 | 3.050e-08 | 2.739e-08 |
| sample_0237 | 1024 | shallow | 3 | 3.093e-08 | 2.748e-08 |

## Learned lifter: primary versus comparison quadrature

- Primary transport-error mean: `3.436077e-08`
- Comparison transport-error mean: `4.021696e-08`
- Absolute delta mean: `7.469406e-09`
- Absolute delta maximum: `2.258720e-08`

Conclusion: P1 already preserves the FEM transport action to numerical precision. The learned lifter must therefore be described as morphology-sensitive support redistribution under a FEM transport-action constraint.
