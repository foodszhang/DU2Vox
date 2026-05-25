# DU2Vox Stage1 Balanced Finetune

## Motivation

The old 20k Stage1 checkpoint was high-recall oriented and selected by `Dice@0.3`. The balanced Stage1 finetune uses sigmoid support output, support-oriented loss, and checkpoint selection by `Dice@0.5`.

The goal was to test whether the balanced Stage1 checkpoint is a better initializer for the CQR rolesplit pipeline, not just whether it improves mesh-level Dice.

## Stage1 Training Result

Checkpoint:

```text
runs/stage1_uniform_1000_20k_balanced_dice05_finetune_norestart/checkpoints/best_dice05.pth
```

Training summary:

- Best epoch: `110`
- Dice@0.5: `0.5666`
- Dice@0.3: `0.5611`
- Soft Dice: `0.5606`
- Best val loss: `0.2536`

## Stage1 Audit

Result files:

- `results/stage1_old_highrecall_audit.json`
- `results/stage1_balanced_dice05_finetune_audit.json`
- `results/stage1_checkpoint_audit_comparison.csv`

Audit comparison:

| Method | Dice@0.5 | Dice@0.3 | Precision@0.5 | Recall@0.5 | Pred pos@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| old high-recall | 0.5939 | 0.5789 | 0.5010 | 0.8672 | 0.00305 |
| balanced finetune | 0.5666 | 0.5586 | 0.5592 | 0.7273 | 0.00245 |

Interpretation: balanced finetune is more compact and has higher precision, but it loses recall and underperforms the old checkpoint on mesh Dice@0.5.

## Bridge Sweep

Result file:

- `results/stage1_balanced_dice05_finetune_bridge_sweep.csv`

Sweep summary:

| Tau | Dilate | Common FEM Dice | GT ROI Coverage | ROI Tet Ratio |
| ---: | ---: | ---: | ---: | ---: |
| 0.3 | 0 | 0.6002 | 0.9240 | 0.0065 |
| 0.4 | 0 | 0.6015 | 0.9214 | 0.0063 |
| 0.5 | 0 | 0.6035 | 0.9193 | 0.0061 |
| 0.6 | 0 | 0.6056 | 0.9159 | 0.0060 |
| 0.3 | 1 | 0.5759 | 0.9471 | 0.0201 |
| 0.4 | 1 | 0.5762 | 0.9467 | 0.0197 |
| 0.5 | 1 | 0.5763 | 0.9461 | 0.0193 |
| 0.6 | 1 | 0.5775 | 0.9429 | 0.0190 |

Selected operating point:

- Tau: `0.6`
- Dilate: `0`
- Common FEM Dice: `0.6056`
- GT ROI coverage: `0.9159`
- ROI tet ratio: `0.0060`

This slightly improves the old balanced_v2 common FEM Dice (`0.6051`) while producing a much more compact ROI. The tradeoff is lower GT ROI coverage than the preferred `0.94` target.

## CQR Precompute Diagnostic

Result file:

- `results/cqr_stage1_balanced_dice05_finetune_rolesplit_val20_summary.csv`

Val20 CQR query-domain diagnostic:

- Overall `gt_pos@0.5`: `0.1697`
- Overall `fem_dice@0.5`: `0.5248`
- Core `gt_pos@0.5`: `0.2911`
- Halo `gt_pos@0.5`: `0.0860`
- Sentinel `gt_pos@0.5`: not used (`sentinel=0.00`)

Interpretation: the new compact Stage1 bridge still leaves meaningful positives in halo, but FEM support in halo is near zero. This is a risky query distribution for pure residual/MSE training.

## Stage2 CQR Result

Result files:

- `results/common_eval_cqr_stage1_balanced_dice05_finetune_rolesplit_small.json`
- `results/cqr_stage1_balanced_finetune_comparison.csv`

Small sanity common-domain result:

| Method | Samples | S2 Dice | FEM Dice | Delta |
| --- | ---: | ---: | ---: | ---: |
| new Stage1 rolesplit small | 20 | 0.5792 | 0.6039 | -0.0247 |

Historical full reference:

| Method | Samples | S2 Dice | FEM Dice | Delta |
| --- | ---: | ---: | ---: | ---: |
| old Stage1 rolesplit prior_ext scale1 | 200 | 0.6429 | 0.6051 | +0.0378 |

The small sanity run improved own-query validation slightly, but common-domain evaluation degraded below FEM. Per the task gate, full Stage2 training and support-head training were not run.

## Conclusion

- Balanced Stage1 improves compactness and precision, but does not improve mesh Dice versus old high-recall Stage1.
- Bridge sweep shows a compact operating point, `tau=0.6,dilate=0`, with common FEM Dice `0.6056`, slightly above old balanced_v2 FEM `0.6051`.
- The compact ROI has lower GT coverage (`0.9159`), and CQR val20 shows positives remain in halo while FEM halo support is near zero.
- Stage2 small sanity fails common-domain gating with `Delta=-0.0247`, so this checkpoint should not yet replace the old Stage1 in the full CQR pipeline.
- Next step should be Stage1/bridge tuning to raise coverage without the dilated-ROI FEM collapse, before support-head full training.
