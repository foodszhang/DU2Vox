# CQR P1 Transport Consistency

The four-point degree-2 rule integrates the product `N_i * rho_P1` exactly. The reference local matrix is the generator's source mass, `V/20 * (ones(4,4) + eye(4))`.

- Samples: 20
- Source-load max relative error: `3.187679e-08`
- Compressed-measurement max relative error: `2.954166e-08`
- P1/local-mass gate passed: **True**

## Per sample

| sample | tets | depth | foci | source error | compressed error |
| --- | ---: | --- | ---: | ---: | ---: |
| sample_0000 | 1024 | deep | 2 | 3.116e-08 | 2.734e-08 |
| sample_0003 | 1024 | deep | 2 | 3.105e-08 | 2.682e-08 |
| sample_0004 | 1024 | deep | 2 | 3.082e-08 | 2.862e-08 |
| sample_0005 | 1024 | shallow | 1 | 3.134e-08 | 2.605e-08 |
| sample_0007 | 1024 | medium | 1 | 3.100e-08 | 2.584e-08 |
| sample_0009 | 1024 | medium | 1 | 3.047e-08 | 2.615e-08 |
| sample_0010 | 1024 | shallow | 3 | 3.120e-08 | 2.841e-08 |
| sample_0011 | 1024 | medium | 1 | 3.188e-08 | 2.627e-08 |
| sample_0012 | 1024 | deep | 2 | 3.094e-08 | 2.647e-08 |
| sample_0015 | 1024 | shallow | 1 | 3.152e-08 | 2.536e-08 |
| sample_0016 | 1024 | shallow | 3 | 3.080e-08 | 2.954e-08 |
| sample_0017 | 1024 | deep | 2 | 3.147e-08 | 2.789e-08 |
| sample_0018 | 1024 | shallow | 2 | 3.142e-08 | 2.600e-08 |
| sample_0019 | 1024 | deep | 3 | 3.133e-08 | 2.785e-08 |
| sample_0020 | 1024 | deep | 3 | 3.121e-08 | 2.607e-08 |
| sample_0021 | 1024 | shallow | 3 | 3.160e-08 | 2.437e-08 |
| sample_0023 | 1024 | medium | 1 | 3.162e-08 | 2.762e-08 |
| sample_0024 | 1024 | deep | 3 | 3.186e-08 | 2.694e-08 |
| sample_0025 | 1024 | shallow | 1 | 3.102e-08 | 2.560e-08 |
| sample_0027 | 1024 | shallow | 1 | 3.144e-08 | 2.476e-08 |

Conclusion: P1 already preserves the FEM transport action to numerical precision. The learned lifter must therefore be described as morphology-sensitive support redistribution under a FEM transport-action constraint.
