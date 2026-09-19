#!/usr/bin/env bash
set -euo pipefail

# Persistent formal rerun after the 2026-09-11 terminal-session interruption.
# New experiment names preserve the incomplete runs as failure-audit evidence.
mkdir -p diagnosis/final_dual_space/logs

run_candidate() {
  local config="$1"
  local experiment="$2"
  uv run python scripts/train_correction_state_transfer.py \
    --config "$config" \
    --experiment-name "$experiment" \
    >"diagnosis/final_dual_space/logs/${experiment}.log" 2>&1
}

run_candidate configs/stage2/cst/cst_d64_g8_l3.yaml cst_d64_g8_l3_formal &
p1=$!
run_candidate configs/stage2/cst/cst_d96_g8_l3.yaml cst_d96_g8_l3_formal &
p2=$!
run_candidate configs/stage2/cst/cst_d128_g16_l3.yaml cst_d128_g16_l3_formal &
p3=$!
run_candidate configs/stage2/cst/cst_d96_g16_l4.yaml cst_d96_g16_l4_formal &
p4=$!

status=0
for pid in "$p1" "$p2" "$p3" "$p4"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
