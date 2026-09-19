#!/usr/bin/env python3
"""Post-process completed oracle tables into figures and the scientific report."""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.cross_discretization_decomposition.decomposition import (  # noqa: E402
    MassProjector,
    decompose,
    load_operator,
)

EXP = Path(__file__).resolve().parent
ART = EXP / "artifacts"
TABLES = ART / "tables"
FIGURES = ART / "figures"
DATASET = Path("/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k")
SHARED = Path("/home/foods/pro/FMT-SimGen/output/shared_mesh_20k")
BRIDGES = {
    "train": REPO / "output/bridge_v2_3k_train_balanced_v2",
    "val": REPO / "output/bridge_v2_3k_val_balanced_v2",
    "test": REPO / "output/bridge_v2_3k_test_balanced_v2",
}


def read_rows(name: str) -> list[dict]:
    with open(TABLES / name, newline="") as stream:
        return list(csv.DictReader(stream))


def number(row: dict, key: str) -> float:
    return float(row[key])


def oracle_means(rows: list[dict]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["reconstruction"]].append(row)
    metrics = ("dice", "precision", "recall", "weak_recall", "localization_error", "hd95", "assd", "component_recall", "separation_success", "fp_components")
    return {
        reconstruction: {
            metric: float(np.mean([number(row, metric) for row in subset]))
            for metric in metrics
        }
        for reconstruction, subset in grouped.items()
    }


def choose_cases(energy: list[dict], oracle: list[dict]) -> list[tuple[str, str]]:
    baseline_dice = {
        row["sample_id"]: number(row, "dice")
        for row in oracle
        if row["reconstruction"] == "stage1_p1"
    }

    def pick(label: str, candidates: list[dict], key) -> tuple[str, str]:
        chosen = min(candidates, key=key)
        return label, chosen["sample_id"]

    singles = [row for row in energy if int(float(row["num_foci"])) == 1]
    multis = [row for row in energy if int(float(row["num_foci"])) >= 2]
    finite_spacing = [row for row in multis if np.isfinite(number(row, "min_inter_source_distance_mm"))]
    return [
        pick("single-small", singles, lambda r: number(r, "min_source_volume_mm3")),
        pick("single-deep", singles, lambda r: -number(r, "depth_mm")),
        pick("multi-close", finite_spacing, lambda r: number(r, "min_inter_source_distance_mm")),
        pick("multi-weak", multis, lambda r: number(r, "min_intensity")),
        pick("large-source", energy, lambda r: -number(r, "total_source_volume_mm3")),
        pick("stage1-failure", energy, lambda r: baseline_dice[r["sample_id"]]),
    ]


def spatial_figure(selected: list[tuple[str, str]], energy: list[dict]) -> None:
    lookup = {row["sample_id"]: row for row in energy}
    operator = load_operator(ART / "operator_cache")
    projector = MassProjector(operator)
    fields = ("gt", "fem", "e_total", "e_inv", "e_rep", "oracle_inv", "oracle_rep")
    fig, axes = plt.subplots(len(selected), len(fields), figsize=(17, 2.8 * len(selected)), squeeze=False)
    for row_index, (case_label, sid) in enumerate(selected):
        split = lookup[sid]["split"]
        raw = np.load(DATASET / "samples" / sid / "gt_voxels.npy")
        gt = (raw.ravel()[operator.valid_flat_indices] > 0.05).astype(np.float64)
        coarse = np.load(BRIDGES[split] / sid / "coarse_d.npy")
        result = decompose(operator, projector, gt, coarse)
        gt_indices = np.argwhere(raw > 0.05)
        axial = int(np.rint(gt_indices[:, 2].mean()))
        for column, field in enumerate(fields):
            full = np.full(np.prod(operator.grid_shape), np.nan, dtype=np.float32)
            full[operator.valid_flat_indices] = result[field]
            image = full.reshape(operator.grid_shape)[:, :, axial].T
            is_error = field.startswith("e_")
            vmax = max(0.1, float(np.nanmax(np.abs(image)))) if is_error else 1.0
            axes[row_index, column].imshow(
                image,
                origin="lower",
                cmap="coolwarm" if is_error else "viridis",
                vmin=-vmax if is_error else 0,
                vmax=vmax,
            )
            axes[row_index, column].set_title(f"{case_label}: {sid}\n{field}")
            axes[row_index, column].axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES / "02_spatial_decomposition_representative.png", dpi=160)
    plt.close(fig)
    with open(TABLES / "representative_cases.csv", "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["category", "sample_id", "split"])
        writer.writerows([(label, sid, lookup[sid]["split"]) for label, sid in selected])


def statistical_figure(structure: list[dict]) -> None:
    metrics = (
        "autocorr_0.6mm",
        "gradient_abs_mean",
        "boundary_energy_le_0.2mm",
        "boundary_energy_le_0.6mm",
        "near_zero_le_0.01",
        "kurtosis",
    )
    components = ("inverse", "representation")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for axis, metric in zip(axes.ravel(), metrics):
        values = [
            [number(row, metric) for row in structure if row["component"] == component]
            for component in components
        ]
        axis.boxplot(values, tick_labels=("inverse", "representation"), showfliers=False)
        axis.set_title(metric)
    fig.tight_layout()
    fig.savefig(FIGURES / "04_statistical_structure_comparison.png", dpi=180)
    plt.close(fig)


def grouped_source_table(energy: list[dict]) -> None:
    groups: list[tuple[str, str, list[dict]]] = []
    for field in ("depth_tier", "num_foci"):
        for value in sorted({row[field] for row in energy}):
            groups.append((field, value, [row for row in energy if row[field] == value]))
    for field in ("total_source_volume_mm3", "min_inter_source_distance_mm"):
        finite = np.asarray(
            [number(row, field) for row in energy if np.isfinite(number(row, field))]
        )
        cuts = np.quantile(finite, [0.0, 0.25, 0.5, 0.75, 1.0])
        for index in range(4):
            subset = [
                row
                for row in energy
                if np.isfinite(number(row, field))
                and number(row, field) >= cuts[index]
                and (
                    number(row, field) <= cuts[index + 1]
                    if index == 3
                    else number(row, field) < cuts[index + 1]
                )
            ]
            groups.append((field, f"Q{index + 1}", subset))
    output = []
    for field, group, subset in groups:
        output.append(
            {
                "grouping": field,
                "group": group,
                "n": len(subset),
                "ratio_inv_mean": np.mean([number(row, "ratio_inv") for row in subset]),
                "ratio_inv_std": np.std([number(row, "ratio_inv") for row in subset]),
                "ratio_rep_mean": np.mean([number(row, "ratio_rep") for row in subset]),
                "ratio_rep_std": np.std([number(row, "ratio_rep") for row in subset]),
            }
        )
    with open(TABLES / "source_property_grouped_statistics.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


def main() -> None:
    energy = read_rows("per_case_error_energy_support.csv")
    raw = read_rows("per_case_error_energy_raw_intensity_sensitivity.csv")
    oracle = read_rows("per_case_oracle_metrics_fixed_domain.csv")
    structure = read_rows("per_case_error_structure.csv")
    means = oracle_means(oracle)
    ratios_inv = np.asarray([number(row, "ratio_inv") for row in energy])
    ratios_rep = np.asarray([number(row, "ratio_rep") for row in energy])
    raw_rep = np.asarray([number(row, "ratio_rep") for row in raw])
    selected = choose_cases(energy, oracle)
    spatial_figure(selected, energy)
    statistical_figure(structure)
    grouped_source_table(energy)

    structure_mean = {}
    for component in ("inverse", "representation"):
        subset = [row for row in structure if row["component"] == component]
        structure_mean[component] = {
            key: float(np.mean([number(row, key) for row in subset]))
            for key in ("autocorr_0.6mm", "gradient_abs_mean", "boundary_energy_le_0.2mm", "boundary_energy_le_0.6mm", "near_zero_le_0.01", "kurtosis")
        }

    base, inv, rep = means["stage1_p1"], means["oracle_inverse"], means["oracle_representation"]
    report = f"""# DU2Vox cross-discretization support-space decomposition

## Decision

**Outcome B — PARTIAL support.** The decomposition is numerically exact and the two
components are both non-negligible. Their spatial correlation and boundary
concentration differ strongly. However, the oracle corrections do not cleanly split
into different reconstruction failure modes: both improve essentially the same metric
families, and the inverse oracle dominates almost every metric. This is insufficient
evidence to justify a new dual-branch Stage-2 architecture now. No Stage-2 model was
designed, modified, trained, or retrained.

The conclusion is strictly about **binary source-support space**. Raw-intensity results
are sensitivity analysis only because Stage 1 was trained against binary nodal support,
whereas raw voxel GT contains source amplitudes above one.

## Exact experiment definition

- Dataset: unchanged DU2Vox/FMT-SimGen 3000-case split (2400/300/300).
- Target: `rho_GT = (gt_voxels > 0.05).astype(float64)`.
- Fixed domain: all 0.2 mm GT voxel centers contained in the complete FEM tetrahedral
  mesh; it is independent of GT, Stage-1 output, ROI, and CQR.
- Domain size: 1,677,645 of 3,952,000 GT centers.
- FEM space: complete 19,990-node P1 mesh. Sparse `P` has 6,710,580 entries.
- Inner product: uniform voxel quadrature, `M = 0.2^3 I = 0.008 I mm^3`.
- Projection: sparse operator form. `(P.T M P)c = P.T M rho`; all 19,990 columns
  are active, sparse LU succeeds with no ridge (`epsilon = 0`). The dense projector
  is never formed.
- Stage-1 input: existing frozen balanced-v2 `coarse_d` for every split.

Runtime was about 75.9 minutes with roughly 1.3 GB resident memory. The cached sparse
operator is reusable.

## Numerical sanity checks

Across all 3000 cases, worst absolute checks were:

- relative decomposition closure: 4.94e-17;
- M-orthogonality cosine: 1.59e-15;
- relative Pythagorean energy error: 5.42e-15.

The decomposition therefore passes the numerical gate by a wide margin.

## Experiment A: energy decomposition

| Quantity | Mean | Median | P05 | P25 | P75 | P95 |
|---|---:|---:|---:|---:|---:|---:|
| inverse fraction | 0.728 | 0.736 | 0.532 | 0.655 | 0.808 | 0.902 |
| representation fraction | 0.272 | 0.264 | 0.098 | 0.192 | 0.345 | 0.468 |

Representation error exceeds 10% of total energy in
{100 * np.mean(ratios_rep >= 0.10):.1f}% of cases and exceeds 20% in
{100 * np.mean(ratios_rep >= 0.20):.1f}%. Inverse error exceeds 50% in
{100 * np.mean(ratios_inv >= 0.50):.1f}%. Thus neither term is generally negligible,
although inverse error is the larger term.

Raw-intensity sensitivity gives a very similar mean representation fraction
({raw_rep.mean():.3f}) and median ({np.median(raw_rep):.3f}); this supports robustness
of the energy ratio, but does not authorize an amplitude-space claim.

## Experiments B and D: spatial/statistical roles

| Statistic (mean) | inverse error | representation error |
|---|---:|---:|
| autocorrelation at 0.6 mm | {structure_mean['inverse']['autocorr_0.6mm']:.3f} | {structure_mean['representation']['autocorr_0.6mm']:.3f} |
| absolute gradient mean | {structure_mean['inverse']['gradient_abs_mean']:.4f} | {structure_mean['representation']['gradient_abs_mean']:.4f} |
| energy within 0.2 mm of GT boundary | {structure_mean['inverse']['boundary_energy_le_0.2mm']:.3f} | {structure_mean['representation']['boundary_energy_le_0.2mm']:.3f} |
| energy within 0.6 mm of GT boundary | {structure_mean['inverse']['boundary_energy_le_0.6mm']:.3f} | {structure_mean['representation']['boundary_energy_le_0.6mm']:.3f} |
| fraction near zero (`abs <= 0.01`) | {structure_mean['inverse']['near_zero_le_0.01']:.3f} | {structure_mean['representation']['near_zero_le_0.01']:.3f} |
| excess kurtosis | {structure_mean['inverse']['kurtosis']:.1f} | {structure_mean['representation']['kurtosis']:.1f} |

The representation component is markedly shorter-range and boundary concentrated:
69.3% of its energy lies within 0.2 mm and 89.9% within 0.6 mm of the true boundary,
versus 21.8% and 52.3% for inverse error. Conversely, inverse error remains strongly
autocorrelated at 0.6 mm (0.821 versus 0.092). This is the strongest empirical support
for distinct roles.

Both components are zero-inflated and extremely heavy-tailed. Generalized-Gaussian
maximum-likelihood fits collapse toward very small shape/scale values because the
sample includes a dominant near-zero mass; the nominal AIC/BIC preference is therefore
not a trustworthy basis for choosing an L1/L2 prior. A spike-and-slab or conditional
model would need separate validation. The current analysis does **not** justify hard-
coding L1 for one branch and L2 for the other.

## Experiment C: oracle consequences

| Reconstruction | Dice | Precision | Recall | Weak recall | Localization mm | HD95 mm | ASSD mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage1 P1 | {base['dice']:.3f} | {base['precision']:.3f} | {base['recall']:.3f} | {base['weak_recall']:.3f} | {base['localization_error']:.3f} | {base['hd95']:.3f} | {base['assd']:.3f} |
| + oracle inverse | {inv['dice']:.3f} | {inv['precision']:.3f} | {inv['recall']:.3f} | {inv['weak_recall']:.3f} | {inv['localization_error']:.3f} | {inv['hd95']:.3f} | {inv['assd']:.3f} |
| + oracle representation | {rep['dice']:.3f} | {rep['precision']:.3f} | {rep['recall']:.3f} | {rep['weak_recall']:.3f} | {rep['localization_error']:.3f} | {rep['hd95']:.3f} | {rep['assd']:.3f} |
| GT | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |

Inverse correction improves Dice by {inv['dice']-base['dice']:.3f}, weak recall by
{inv['weak_recall']-base['weak_recall']:.3f}, and HD95 by
{base['hd95']-inv['hd95']:.3f} mm. Representation correction also improves those same
metrics (Dice +{rep['dice']-base['dice']:.3f}, weak recall
+{rep['weak_recall']-base['weak_recall']:.3f}, HD95
{base['hd95']-rep['hd95']:.3f} mm), rather than isolating a boundary-only failure mode.
It also creates thresholded small components in some cases (mean FP component count
{rep['fp_components']:.3f} versus {base['fp_components']:.3f}), reflecting oscillatory
complement corrections. This weakens a clean two-branch interpretation.

## Experiment E: dependence on source properties

- Depth has essentially no association with the fractions (Spearman magnitude 0.011,
  p=0.54), so the proposed “deeper means more inverse fraction” hypothesis is not
  supported here.
- Representation fraction increases modestly with total volume (Spearman 0.275) and
  minimum source volume (0.272). The proposed “smaller means more representation
  fraction” direction is not supported.
- More foci modestly increase inverse fraction (Spearman 0.144).
- Greater inter-source distance increases representation fraction (Spearman 0.194),
  equivalently closer sources increase inverse fraction, consistent with merging or
  inverse ill-conditioning, but the effect is small.
- Intensity association is negligible (Spearman magnitude 0.055).

These effects are statistically detectable at N=3000 but mostly weak. They do not
provide the physically broad separation expected under Outcome A.

## Historical CQR/ROI sensitivity and Experiment F

Historical comparison was computed only where existing CQR precomputes were present
(100 train, 25 validation, 25 test cases). It restricts metrics to those old points but
does not redefine the projection space. It is secondary because that domain depends on
Stage-1 proposals.

Experiment F is skipped. Available alternative mesh directories are not certified as
matched coarse/default/fine discretizations of this exact dataset and lack matched
Stage-1 reconstructions. A valid test requires regenerated shared physics, GT sampling,
forward matrices, and frozen Stage-1 outputs for at least three nested or carefully
comparable meshes while preserving source cases and measurement convention.

## Failure cases and limitations

- `Pi_h rho_GT` is an unconstrained L2 projection and can overshoot or become negative;
  thresholded oracle metrics therefore need interpretation alongside energy metrics.
- Fixed voxel quadrature is appropriate for the uniform center grid, but the projection
  is a sampled voxel-space projection, not the exact continuous FEM mass projection.
- Component matching uses 6-connectivity and a 3 mm matching radius; conclusions rely
  more heavily on Dice/HD95/ASSD and energy identities than on component counts.
- Representative slices are qualitative and cannot replace the 3000-case statistics.
- No mesh-resolution intervention was available, so causality from discretization was
  not experimentally manipulated.

## Final scientific answer

**PARTIAL — interesting but insufficient.** The existing dataset provides strong
evidence that the binary-support error admits a numerically sound decomposition into a
dominant FEM-representable inverse term and a substantial, shorter-range,
boundary-concentrated FEM-complement term. It does **not** yet show that the two oracle
corrections have sufficiently different reconstruction consequences: both improve the
same failure modes and the inverse oracle dominates. Source-property stratification is
also weaker and partly opposite to the proposed hypotheses.

Recommended next step: review these oracle results, then obtain a controlled matched
mesh-resolution experiment and validate constrained/nonnegative oracle variants. Do
not redesign Stage 2 or start a long training run yet.

## Reproducible outputs

- `run_analysis.py`: fixed-domain operator, decomposition, validation, oracle metrics,
  statistics, and base figures.
- `build_report.py`: representative selection, comparison figure, and this report.
- `artifacts/operator_cache/`: sparse P and fixed-domain arrays.
- `artifacts/tables/`: all per-case and summary CSVs.
- `artifacts/figures/`: energy, spatial, oracle, statistical, and dependence figures.
"""
    (ART / "REPORT.md").write_text(report)
    print(f"Wrote {ART / 'REPORT.md'}")


if __name__ == "__main__":
    main()
