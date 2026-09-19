# DU2Vox Final Result Ledger

> **IMPORTANT**
>
> The existing 300-sample test split was observed during development. It is a
> **development-test**, not the final untouched confirmation set.
>
> The sealed confirmation set remains unopened. Never quote development-test
> performance as final unseen generalization.

## Data identity

```text
dataset: fmt_simgen_v2_3k_20k
train: 2400 existing development samples
validation: 300 existing development samples
development-test: 300 previously observed development samples
confirmation: sealed and unopened

canonical voxel grid: (190, 200, 104), 0.2 mm
full voxel count: 3,952,000
valid FEM-domain voxel count: 1,677,645
FEM node count: 19,990
GT semantics: gt_voxels > 0.05
evaluation threshold: 0.5
```

Operator identity:

```text
mesh SHA256:
718cb70f0b12c3c8f5e7f10d9e216295efc79ff3221602b934ba04c137f34c64

frame manifest SHA256:
25b77f5d11430ebf4d9735f19534782afb56f7fe5cab525a569028b0f3c02e2b

Pi_h I_h maximum relative error: 1.50e-15
I_h Pi_h maximum idempotence error: 1.33e-15
```

## Reconstruction ledger

| Model | Val Dice | Dev-Test Dice | Status |
| --- | ---: | ---: | --- |
| Stage 1 canonical P1 | 0.632646 | 0.617379 | baseline |
| V1 residual-aware FEM | 0.700174 | 0.679571 | historical |
| V3 scale-calibrated FEM | 0.710335 | 0.698186 | historical |
| V1/V3 FEM ensemble | 0.717931 | 0.702730 | secondary historical result |
| V4 unified dual-evidence FEM | 0.732512 | 0.716529 | frozen coarse backbone |
| V4 + unconstrained voxel residual | 0.743784 | 0.726636 | control only; contract invalid |
| **V4 + constrained `Qz`** | **0.741844** | **0.727163** | final freeze candidate |
| V4 + oracle detail | 0.840032 | 0.824390 | oracle only |

All checkpoint choices in the final comparison were made from validation Dice only.
No threshold sweep was used.

## Final candidate development metrics

| Method | Dice | Precision | Recall | Weak Recall | HD95 | Localization | MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage 1 P1 | 0.617379 | 0.594399 | 0.750855 | 0.709763 | 2.604663 | 1.689924 | 0.004131 |
| V4 coarse | 0.716529 | 0.706280 | 0.790795 | 0.745791 | 2.173528 | 1.352132 | 0.006638 |
| V4 + unconstrained residual | 0.726636 | 0.713661 | 0.805155 | 0.757568 | 2.022328 | 1.272334 | 0.008129 |
| **V4 + constrained `Qz`** | **0.727163** | **0.722357** | **0.795747** | **0.750303** | **2.004499** | **1.267871** | **0.006789** |
| V4 + oracle detail | 0.824390 | 0.816910 | 0.881746 | 0.844096 | 0.876313 | 0.572605 | 0.005731 |

## Increment ledger

```text
Stage 1 -> V4:
    +0.099151 Dice

V4 -> constrained Qz:
    validation:       +0.009332 Dice
    development-test: +0.010634 Dice

Stage 1 -> final candidate:
    +0.109784 absolute Dice
    approximately +17.8% relative Dice
    approximately 28.7% of original Dice error removed

gain allocation:
    Stage 2A FEM refinement: approximately 90.3%
    Stage 2B voxel detail:   approximately 9.7%
```

## Structural ledger

| Quantity | Validation | Development-test |
| --- | ---: | ---: |
| Constrained detail coarse leakage | 4.47e-18 | 4.57e-18 |
| Constrained coarse preservation relative L2 | 8.50e-9 | 8.72e-9 |
| Constrained detail cosine | 0.15715 | 0.16935 |
| Unconstrained detail coarse leakage | 0.61609 | 0.62679 |
| Oracle-detail Dice headroom | +0.107520 | +0.107861 |
| Learned oracle recovery fraction | 0.08679 | 0.09859 |

## Frozen artifacts

```text
Stage 2A config:
configs/stage2/unified_dual_evidence_fem_v4_2400.yaml

Stage 2A checkpoint:
runs/unified_dual_evidence_fem_v4_2400/checkpoints/best_dense_val_delta_dice.pth
epoch: 15

Stage 2B config:
configs/stage2/complement_voxel_detail_v1_2400.yaml

Stage 2B constrained checkpoint:
runs/complement_voxel_detail_v1_2400_constrained/checkpoints/best_dense_val_dice.pth
epoch: 3
validation Dice: 0.7418439308802287

matched unconstrained control checkpoint:
runs/complement_voxel_detail_v1_2400_unconstrained/checkpoints/best_dense_val_dice.pth
epoch: 2
validation Dice: 0.7437841518719991
```

## Version ledger

```text
working branch: feat/complement-constrained-voxel-detail
base/current recorded HEAD: 46f650d41d4938aa700c7e637a14e074072fce8d
candidate Git tag: NOT CREATED
```

The recorded HEAD predates the uncommitted candidate implementation and therefore is
not a reproducible freeze identifier by itself. Create a clean, reviewed freeze
commit before tagging `du2vox-two-stage-freeze-candidate-v1`. After the single sealed
confirmation and final approval, use a separate paper-method tag.

## Confirmation governance

```text
confirmation inference: not run
confirmation Dice: not computed
confirmation visualization: not generated
confirmation threshold sweep: not run
confirmation checkpoint selection: not run
```

Do not open confirmation until the user explicitly freezes the architecture and
authorizes the one permitted sealed confirmation suite.

