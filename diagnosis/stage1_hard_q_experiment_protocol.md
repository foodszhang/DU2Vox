# Direct retrained-Stage1 hard-`Q` experiment protocol

Status: implementation ready; none of these commands have been executed.

```text
measurement -> epoch-122 retrained Stage1 -> canonical P1 -> exact FP64 Qz
```

The Stage1 checkpoint SHA256 is
`148ea58a2147d4889646069d1ef65e291ec5affea191cfaf280f71f7376932ec`.
Neither arm references iterative-corrector configs, checkpoints, corrected states,
terminal latents, or pretrained view-encoder weights.

## Baselines and smoke

The configs differ only in name and `model.use_views`. The no-view arm strictly
zero-fills the 32-D view slot; both instantiate the same encoder and decoder.

```bash
uv run python scripts/train_complement_voxel_detail.py \
  --config configs/stage2/stage1_hard_q_seed20260901.yaml --mode hard

uv run python scripts/train_complement_voxel_detail.py \
  --config configs/stage2/stage1_hard_q_views_seed20260901.yaml --mode hard
```

For a full forward/backward smoke, add `--max-epochs 1 --max-samples 1` and a
disposable `--experiment-name`. The recorded `peak_gpu_memory_gib` must be at most 16.

Evaluate both five-epoch baselines on val300 using their run-local `config.yaml` and
write `diagnosis/stage1_hard_q/val_no_views.json` and `val_views.json`.

## Bounded tuning and validation freeze

Only the higher-val-Dice baseline may be tuned. Both trials restart from seed
`20260901`; neither resumes a checkpoint.

```bash
uv run python scripts/train_complement_voxel_detail.py \
  --config <winner-config> --mode hard --max-epochs 10 --lr 3e-5 \
  --disable-early-stopping --experiment-name <winner>_lr3e-5_10ep

uv run python scripts/train_complement_voxel_detail.py \
  --config <winner-config> --mode hard --max-epochs 10 --lr 3e-4 \
  --disable-early-stopping --experiment-name <winner>_lr3e-4_10ep
```

After evaluating both tuning checkpoints on val300, create the receipt:

```bash
uv run python scripts/freeze_stage1_hard_q_selection.py \
  --no-views diagnosis/stage1_hard_q/val_no_views.json \
  --views diagnosis/stage1_hard_q/val_views.json \
  --tuning diagnosis/stage1_hard_q/val_<winner>_lr3e-5_10ep.json \
           diagnosis/stage1_hard_q/val_<winner>_lr3e-4_10ep.json \
  --output diagnosis/stage1_hard_q/validation_freeze.json
```

The selector rejects mismatched initialization, model/loss settings, parameter
inventory, sample order, LR, and epoch budgets. Selection is by val300 final Dice at
the fixed `0.5` threshold only.

## Development-test evaluation

Only after the receipt exists may a frozen candidate be evaluated:

```bash
uv run python scripts/eval_complement_voxel_detail.py \
  --config <run>/config.yaml \
  --checkpoint <run>/checkpoints/best_dense_val_dice.pth \
  --split test \
  --freeze-receipt diagnosis/stage1_hard_q/validation_freeze.json \
  --output diagnosis/stage1_hard_q/development_test300.json
```

The two frozen baselines may also be evaluated for the direct-view comparison; this
cannot change the val-selected final version. Summarize without reselection:

```bash
uv run python scripts/summarize_stage1_hard_q_experiment.py \
  --receipt diagnosis/stage1_hard_q/validation_freeze.json \
  --selected-test diagnosis/stage1_hard_q/development_test300.json \
  --no-views-test diagnosis/stage1_hard_q/development_test300_no_views.json \
  --views-test diagnosis/stage1_hard_q/development_test300_views.json \
  --output diagnosis/stage1_hard_q/development_summary.json
```

Evaluation reports Dice, HD95, localization, MSE, relative L2, detail cosine,
detail-relative L1/L2, boundary energy, coarse leakage, coarse preservation, and
10,000-draw paired 95% CIs. The summary checks Dice `0.73`, oracle reference
`0.73534`, recovered oracle space, and the direct-view `+0.002`/positive-lower-CI
gate. Sealed confirmation is outside this protocol.
