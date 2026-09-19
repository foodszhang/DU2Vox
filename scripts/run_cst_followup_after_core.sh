#!/usr/bin/env bash
set -euo pipefail

# Wait for the independently persistent core search. The d96 branch is the
# predeclared reference for mechanism tests; if another core wins validation,
# its winning settings will also receive the retained-module comparison.
while tmux has-session -t du2vox_cst_core 2>/dev/null; do
  sleep 60
done

for run in \
  cst_d64_g8_l3_formal \
  cst_d96_g8_l3_formal \
  cst_d128_g16_l3_formal \
  cst_d96_g16_l4_formal; do
  test "$(python -c "import json; print(len(json.load(open('runs/${run}/history.json'))))")" = 5
done

mkdir -p diagnosis/final_dual_space/logs
run_candidate() {
  local config="$1"
  local experiment="$2"
  uv run python scripts/train_correction_state_transfer.py \
    --config "$config" \
    --experiment-name "$experiment" \
    >"diagnosis/final_dual_space/logs/${experiment}.log" 2>&1
}

run_candidate configs/stage2/cst/cst_hc_only_d96_l3.yaml cst_hc_only_d96_l3_formal &
p1=$!
run_candidate configs/stage2/cst/cst_hc_delta_d96_g8_l3.yaml cst_hc_delta_d96_g8_l3_formal &
p2=$!
run_candidate configs/stage2/cst/cst_d96_g8_l3_one_ring.yaml cst_d96_g8_l3_one_ring_formal &
p3=$!
run_candidate configs/stage2/cst/cst_d96_g8_l3_views.yaml cst_d96_g8_l3_views_formal &
p4=$!

status=0
for pid in "$p1" "$p2" "$p3" "$p4"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
