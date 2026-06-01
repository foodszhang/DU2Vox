#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/foods/pro/DU2Vox"
cd "$ROOT_DIR"

STAGE1_PARENT_PID="${1:-68300}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-21600}"
TARGET_EPOCH="${TARGET_EPOCH:-100}"

STAGE1_CONFIG="configs/stage1/fmt_simgen_v2_3k_20k_balanced_v2_eval.yaml"
STAGE1_RUN_DIR="runs/stage1_fmt_simgen_v2_3k_20k_balanced_v2_eval"
STAGE1_LOG="$STAGE1_RUN_DIR/train_20260601_011748.log"
STAGE1_CKPT="$STAGE1_RUN_DIR/checkpoints/best.pth"

STAGE2_CONFIG="configs/stage2/cqr_v2_3k_rgl_main_multiview.yaml"
TRAIN_SPLIT="/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k/splits/train.txt"
VAL_SPLIT="/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k/splits/val.txt"

TRAIN_BRIDGE="output/bridge_v2_3k_train_balanced_v2"
VAL_BRIDGE="output/bridge_v2_3k_val_balanced_v2"
TRAIN_PRECOMP="precomputed/cqr_v2_3k_rgl_main/train"
VAL_PRECOMP="precomputed/cqr_v2_3k_rgl_main/val"

mkdir -p logs
HANDOFF_LOG="logs/v2_3k_stage1_handoff_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$HANDOFF_LOG") 2>&1

echo "[handoff] started $(date -Is)"
echo "[handoff] watching pid=$STAGE1_PARENT_PID target_epoch=$TARGET_EPOCH max_wait=${MAX_WAIT_SECONDS}s"

start_ts=$(date +%s)
last_epoch=0
while true; do
  now_ts=$(date +%s)
  elapsed=$((now_ts - start_ts))
  if [[ -f "$STAGE1_LOG" ]]; then
    parsed_epoch=$(grep -E '^\[CSV\] [0-9]+,' "$STAGE1_LOG" | tail -n 1 | cut -d' ' -f2 | cut -d',' -f1 || true)
    if [[ -n "${parsed_epoch:-}" ]]; then
      last_epoch="$parsed_epoch"
    fi
  fi

  if [[ "$last_epoch" -ge "$TARGET_EPOCH" ]]; then
    echo "[handoff] target epoch reached: $last_epoch"
    break
  fi
  if [[ "$elapsed" -ge "$MAX_WAIT_SECONDS" ]]; then
    echo "[handoff] max wait reached after ${elapsed}s at epoch $last_epoch"
    break
  fi
  if ! ps -p "$STAGE1_PARENT_PID" >/dev/null 2>&1; then
    echo "[handoff] stage1 process already exited at epoch $last_epoch"
    break
  fi
  sleep 60
done

if ps -p "$STAGE1_PARENT_PID" >/dev/null 2>&1; then
  child_pid=$(pgrep -P "$STAGE1_PARENT_PID" || true)
  if [[ -n "$child_pid" ]]; then
    echo "[handoff] sending SIGINT to child pid=$child_pid"
    kill -INT "$child_pid" || true
  else
    echo "[handoff] sending SIGINT to parent pid=$STAGE1_PARENT_PID"
    kill -INT "$STAGE1_PARENT_PID" || true
  fi
  sleep 30
fi

if [[ ! -f "$STAGE1_CKPT" ]]; then
  echo "[handoff][ERROR] missing best checkpoint: $STAGE1_CKPT"
  exit 1
fi

echo "[handoff] using checkpoint: $STAGE1_CKPT"
echo "[handoff] bridge train"
uv run python scripts/bridge_stage1_to_stage2.py \
  --config "$STAGE1_CONFIG" \
  --checkpoint "$STAGE1_CKPT" \
  --split_file "$TRAIN_SPLIT" \
  --output_dir "$TRAIN_BRIDGE" \
  --tau 0.18 \
  --dilate_layers 0 \
  --min_component_size 3 \
  --batch_size 32

echo "[handoff] bridge val"
uv run python scripts/bridge_stage1_to_stage2.py \
  --config "$STAGE1_CONFIG" \
  --checkpoint "$STAGE1_CKPT" \
  --split_file "$VAL_SPLIT" \
  --output_dir "$VAL_BRIDGE" \
  --tau 0.18 \
  --dilate_layers 0 \
  --min_component_size 3 \
  --batch_size 32

echo "[handoff] precompute train"
uv run python scripts/precompute_stage2_cqr.py \
  --config "$STAGE2_CONFIG" \
  --split train \
  --output_dir "$TRAIN_PRECOMP" \
  --overwrite

echo "[handoff] precompute val"
uv run python scripts/precompute_stage2_cqr.py \
  --config "$STAGE2_CONFIG" \
  --split val \
  --output_dir "$VAL_PRECOMP" \
  --overwrite

echo "[handoff] diagnose val precompute"
uv run python scripts/diagnose_cqr_npz.py \
  --dir "$VAL_PRECOMP" \
  --max_files 20

echo "[handoff] train stage2"
uv run python scripts/train_stage2.py \
  --config "$STAGE2_CONFIG" \
  --experiment_name cqr_v2_3k_rgl_main_multiview

echo "[handoff] completed $(date -Is)"
