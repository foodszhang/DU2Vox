# Stage-1 continuous-field FEM-volume audit

This audit changes no network and performs no training. Epoch 29 and epoch 30 are
the already exported physical `coarse_d.npy` states. All scalar FEM metrics use
the P1 lumped mass diagonal `m_i = sum_{T contains i} |T|/4`; nodes are not
weighted equally.

## Metric contract

- `wRelL2 = sqrt(sum m_i (p_i-g_i)^2 / sum m_i g_i^2)`.
- mass ratio uses `sum m_i p_i / sum m_i g_i`.
- CCC uses lumped-volume-normalized weighted moments over the full FEM domain.
- COM uses the nonnegative nodal fields and lumped-volume mass.
- shape error is wRelL2 after the best nonnegative global rescaling of prediction;
  it removes overall amplitude but retains displacement, separation, and composition error.
- source composition is total-variation distance between predicted attributed mass
  fractions and the declared component-template mass fractions. Prediction mass is
  assigned by intensity-independent normalized-shape territories over the FEM mesh;
  therefore every GT source participates and background artifacts are not discarded.
- weak-source error is the relative attributed mass error for the lowest-parameter-
  intensity component; it is undefined for single-source cases.

## Overall medians

| Method | wRelL2 | mass ratio | CCC | COM mm | shape error | composition error | weak error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| epoch29 | 0.7725 | 1.0244 | 0.6119 | 2.2305 | 0.7195 | 0.2367 | 0.7808 |
| epoch30 | 0.7629 | 0.5092 | 0.5333 | 2.1643 | 0.7047 | 0.3226 | 0.9048 |
| p1_oracle | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0072 | 0.0215 |

## Stratified medians

### Source count

| Group | n | e29 wRelL2 | e30 wRelL2 | e29 mass | e30 mass | e29 composition | e30 composition | e29 weak | e30 weak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 81 | 0.663 | 0.639 | 1.349 | 0.619 | N/A | N/A | N/A | N/A |
| 2 | 109 | 0.769 | 0.764 | 0.999 | 0.524 | 0.169 | 0.250 | 0.776 | 0.981 |
| 3 | 110 | 0.824 | 0.831 | 0.925 | 0.455 | 0.308 | 0.363 | 0.784 | 0.734 |

### Strong/weak parameter-intensity ratio

| Group | n | e29 wRelL2 | e30 wRelL2 | e29 mass | e30 mass | e29 composition | e30 composition | e29 weak | e30 weak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| multi_>=2 | 37 | 0.810 | 0.829 | 1.181 | 0.563 | 0.205 | 0.305 | 0.804 | 0.995 |
| multi_[1,1.25) | 64 | 0.776 | 0.779 | 0.797 | 0.374 | 0.261 | 0.328 | 0.694 | 0.846 |
| multi_[1.25,1.5) | 54 | 0.680 | 0.694 | 0.922 | 0.547 | 0.235 | 0.317 | 0.773 | 0.902 |
| multi_[1.5,2) | 64 | 0.871 | 0.853 | 1.081 | 0.491 | 0.259 | 0.342 | 0.841 | 0.805 |
| single | 81 | 0.663 | 0.639 | 1.349 | 0.619 | N/A | N/A | N/A | N/A |

### Minimum source-center distance

| Group | n | e29 wRelL2 | e30 wRelL2 | e29 mass | e30 mass | e29 composition | e30 composition | e29 weak | e30 weak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| multi_<5mm | 34 | 0.730 | 0.778 | 0.977 | 0.424 | 0.192 | 0.206 | 0.785 | 0.604 |
| multi_>=20mm | 39 | 0.719 | 0.687 | 1.006 | 0.533 | 0.162 | 0.211 | 0.676 | 0.858 |
| multi_[10,20)mm | 68 | 0.766 | 0.752 | 0.970 | 0.527 | 0.205 | 0.323 | 0.773 | 0.988 |
| multi_[5,10)mm | 78 | 0.882 | 0.868 | 0.888 | 0.455 | 0.326 | 0.381 | 0.802 | 0.967 |
| single | 81 | 0.663 | 0.639 | 1.349 | 0.619 | N/A | N/A | N/A | N/A |

### Depth tier

| Group | n | e29 wRelL2 | e30 wRelL2 | e29 mass | e30 mass | e29 composition | e30 composition | e29 weak | e30 weak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| deep | 86 | 0.777 | 0.758 | 0.951 | 0.504 | 0.240 | 0.333 | 0.767 | 0.938 |
| medium | 123 | 0.734 | 0.753 | 1.009 | 0.513 | 0.211 | 0.251 | 0.790 | 0.976 |
| shallow | 91 | 0.809 | 0.812 | 1.121 | 0.523 | 0.258 | 0.333 | 0.789 | 0.784 |

## Failure-mode diagnosis

1. **A single global amplitude correction is not sufficient.** Epoch29 has a near-unity median mass ratio (1.024) but its median wRelL2 remains 0.773. Optimally rescaling every prediction reduces the median error by only approximately 6.9% (epoch30: 7.6%). The dominant residual is therefore spatial/shape error, not only global gain.
2. **Amplitude calibration is nevertheless unstable.** The median absolute mass error is 0.392 at epoch29. Epoch30 is systematically under-amplitude (median mass ratio 0.509, median absolute mass error 0.519). Epoch29 has lower mass error in 66.3% of paired cases.
3. **Localization and multi-source structure are major failures.** Epoch29 median COM error is 2.231 mm. Its median shape error rises from 0.565 for one source to 0.802 for three sources, while COM error rises from 1.608 to 3.344 mm.
4. **Weak-source relative recovery is poor beyond the attribution floor.** On all 219 multi-source cases, epoch29 median composition error is 0.237 and weak-source error is 0.781; the P1-oracle floors are only 0.007 and 0.021. Epoch29 beats epoch30 in 91.3% of cases for composition and 65.3% for weak-source error.
5. **Source distance, intensity ratio, and depth are modifiers, not a single causal explanation.** The stratified medians are not monotonic across distance or depth bins. Three-source cases are consistently harder, but the current audit cannot reduce the failure to close-source overlap alone.
6. **Epoch29 is the scientifically safer of the two checkpoints despite the similar wRelL2.** Epoch30 has slightly lower median wRelL2 (0.763 versus 0.773), but epoch29 has higher CCC in 85.7% of cases and materially better amplitude, source composition, and weak-source recovery. Epoch30's lower validation loss is achieved with a pronounced low-amplitude/multi-source-collapse bias.

**Audit verdict:** current Stage-1 failure is primarily spatial field reconstruction--localization, morphology, and multi-source composition/weak-source allocation--with an additional checkpoint-dependent global amplitude problem. It is not adequately explained by a single scalar amplitude mismatch, and the non-monotonic distance/depth results do not support claiming that source separation alone is the cause.

## Interpretation guardrail

Under this node-domain protocol, the P1 oracle is exactly the normalized nodal GT.
It is therefore an identity sanity control and must score zero on whole-field errors.
Its source-composition scores need not be zero: those compare the historical
maximum-composed nodal field against the declared latent component-template fractions,
and thus expose the source-attribution floor of the FEM/GT contract. It is not the
cross-discretization P1 approximation oracle previously evaluated against raw voxel GT.
The latter answers a different question and must not be compared numerically as if it
used this lumped-node metric contract.

See `per_case.csv`, `grouped_summary.csv`, `summary.json`, and `figures/` for the
complete casewise and stratified evidence.
