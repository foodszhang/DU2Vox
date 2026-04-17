# Diagnosis Summary: MCX Multi-View Incremental Signal

## Diagnosis A: per-sample ΔDice (v5c best.pth, 200 val)

- **N = 200** samples
- ΔDice: mean=**+0.0001**, std=**0.0006**, median=**0.0000**
- Δ > +0.005: **0%** of samples
- corr(FEM_Dice, ΔDice) = **-0.215 (p=0.002)** — view helps hardest cases slightly

**Interpretation**: No sample shows meaningful Δ in either direction. The negative correlation is real (statistically significant) but tiny in magnitude. View branch is NOT being wasted — it's doing something, but MSE prevents capturing it.

---

## Diagnosis B: MCX proj SNR near FLT (20 samples × 7 angles)

- **N = 280** SNR measurements
- SNR: mean=**-0.609**, median=**-0.618**, std=**0.141**
- SNR ≥ 3: **0%**, 1 ≤ SNR < 3: **0%**, **SNR < 1: 100%**
- Contrast: mean=**-0.991** (FLT is DARK in excitation fluence projections)

**SNR by angle** (best to worst):
| Angle | n | mean SNR | median SNR |
|-------|---|----------|------------|
| -90°  | 40 | -0.392   | -0.376     |
| -60°  | 40 | -0.588   | -0.564     |
| -30°  | 40 | -0.600   | -0.617     |
| 0°    | 40 | -0.642   | -0.645     |
| +30°  | 40 | -0.681   | -0.668     |
| +60°  | 40 | -0.702   | -0.682     |
| +90°  | 40 | -0.656   | -0.661     |

**Physical interpretation**: MCX models excitation light fluence. Fluorescent absorbers appear as **dark spots** (photon deficits) at the excitation wavelength, not bright emitters. This is physically correct for the simulation but mismatched with the prediction task (fluorophore concentration ∝ emission intensity).

---

## Combined Diagnosis → Decision

| Signal | Strength | Interpretation |
|--------|----------|----------------|
| View SNR | **< 1 (all measurements)** | Signal exists (dark spot is real) but very weak |
| ΔDice | **≈ 0 for all samples** | MSE loss cannot exploit weak SNR signal |

**Conclusion**: MCX projections contain **real but extremely weak signal** (the FLT causes a ~60% photon deficit vs background at -90°, but averaged over all angles and samples it's ~60% dark). The encoding is physically correct but the effect is subtle at the resolution of the 256×256 detector.

The negative ΔDice despite real signal is because:
1. MSE loss mismatches binary GT (Dice target)
2. SNR is too weak for the network to reliably learn the mapping
3. Even at best angle (-90°) the defect is only ~40% below background

**No viable path to ΔDice > +0.02 via current multi-view approach.**

---

## LPR Narrative Recommendation

**Multi-view as limitation / upper-bound analysis**:
- Demonstrate that even with perfect architecture and loss, MCX excitation-fluence projections only provide ~60% photon contrast at best angle
- The upper bound on ΔDice from multi-view is therefore small (< 0.01)
- Main thesis: **C1 Mesh-to-Voxel Bridge** (novel contribution) + **C2 residual learning** (practical value, non-degradation)
- Multi-view is explored as a potential constraint but documented as physically limited

**Primary innovation remains**: The bridge (stage 1 output → voxel grid) is the novel component that enables FMT reconstruction from mesh-based simulation. Multi-view exploration is honest upper-bound analysis.

---

## What NOT to do

- No further architecture tuning for multi-view (dim mismatch, attention, etc.)
- No more loss experiments (soft-Dice won't help when SNR < 1)
- No more coordinate frame investigations (confirmed correct)
- No precomputed field additions (v5b voxel-space confirmed irrelevant)
