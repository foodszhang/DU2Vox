# A0-A3 matched information-source audit protocol

This protocol is limited to the eight matched arms A0-A3 and VA0-VA3. It does not
authorize D1-D3 generation, V1-V5 development, residual interrogation, V4 training,
or sealed-confirmation access.

All arms use seed `20260901`, five epochs, the same 231-dimensional decoder input,
zero initialization, sample order, loss, exact hard-`Q`, and full 0.2-mm canonical
domain. Missing S/L/V groups are zeros and no presence flag is supplied. A3/VA3
alone use `Pi_h rho*` as both the local state and final analytic-P1 coarse field.

Run the cache once for each split, retaining the existing FP32 state location and
writing FP16 terminal latents separately:

```bash
uv run python scripts/precompute_frozen_v4_states.py \
  --config configs/stage2/unified_dual_evidence_fem_v4_2400.yaml \
  --checkpoint runs/unified_dual_evidence_fem_v4_2400/checkpoints/best_dense_val_delta_dice.pth \
  --split train \
  --output-dir precomputed/complement_voxel_detail/v4_states/train \
  --cache-terminal-hidden \
  --latent-output-dir precomputed/information_audit/v4_terminal_hidden/train
```

Repeat only by changing `train` to `val` and `test`. The three cache manifests record
artifact hashes and a deterministic 24/3/3 stratified allocation of the 30-case
FP16/state-identity audit.

Train all eight arm configs under `configs/stage2/information_audit/`. Each run first
performs a full-domain forward/backward memory smoke with activation checkpointing.
If its peak exceeds 16 GiB, the run stops and the canonical domain must not be
reduced. For example:

```bash
uv run python scripts/train_information_audit.py \
  --config configs/stage2/information_audit/a0.yaml
```

Evaluate all eight validation checkpoints before making any decision:

```bash
uv run python scripts/eval_information_audit.py \
  --config configs/stage2/information_audit/a0.yaml \
  --checkpoint runs/information_audit_a0/checkpoints/best_dense_val_dice.pth \
  --split val --output diagnosis/information_audit/val/a0.json
```

After all eight validation JSON files exist, freeze the architecture decisions and
checkpoint identities:

```bash
uv run python scripts/summarize_information_audit.py \
  --val-dir diagnosis/information_audit/val \
  --output diagnosis/information_audit/validation_freeze.json
```

Only then evaluate development-test, passing that receipt to every arm:

```bash
uv run python scripts/eval_information_audit.py \
  --config configs/stage2/information_audit/a0.yaml \
  --checkpoint runs/information_audit_a0/checkpoints/best_dense_val_dice.pth \
  --split test \
  --validation-freeze diagnosis/information_audit/validation_freeze.json \
  --output diagnosis/information_audit/development_test/a0.json
```

Finally rerun the summarizer with `--development-test-dir`. Gate decisions remain
validation-only; development-test values are appended only as frozen evaluation.
