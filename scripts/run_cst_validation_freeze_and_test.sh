#!/usr/bin/env bash
set -euo pipefail

while tmux has-session -t du2vox_cst_followup 2>/dev/null; do
  sleep 60
done

output=diagnosis/final_dual_space
mkdir -p "$output/validation" "$output/development_test" "$output/logs"

evaluate_val() {
  local config="$1"
  local run="$2"
  local name="$3"
  test "$(python -c "import json; print(len(json.load(open('runs/${run}/history.json'))))")" = 5
  uv run python scripts/eval_correction_state_transfer.py \
    --config "$config" \
    --checkpoint "runs/${run}/checkpoints/best_dense_val_dice.pth" \
    --split val \
    --output "$output/validation/${name}.json" \
    >"$output/logs/eval_val_${name}.log" 2>&1
}

# Four formal core candidates (plain delta innovation, matched parameter budget).
evaluate_val configs/stage2/cst/cst_d64_g8_l3.yaml cst_d64_g8_l3_formal cst_d64_g8_l3
evaluate_val configs/stage2/cst/cst_d96_g8_l3.yaml cst_d96_g8_l3_formal cst_d96_g8_l3
evaluate_val configs/stage2/cst/cst_d128_g16_l3.yaml cst_d128_g16_l3_formal cst_d128_g16_l3
evaluate_val configs/stage2/cst/cst_d96_g16_l4.yaml cst_d96_g16_l4_formal cst_d96_g16_l4

# Mechanism follow-ups rebuilt around the validation-winning d128/g16/depth3 shape.
evaluate_val configs/stage2/cst/cst_hc_only_d128_g16_l3.yaml cst_hc_only_d128_g16_l3_formal cst_hc_only_d128_g16_l3
evaluate_val configs/stage2/cst/cst_hc_delta_d128_g16_l3.yaml cst_hc_delta_d128_g16_l3_formal cst_hc_delta_d128_g16_l3
evaluate_val configs/stage2/cst/cst_d128_g16_l3_one_ring.yaml cst_d128_g16_l3_one_ring_formal cst_d128_g16_l3_one_ring
evaluate_val configs/stage2/cst/cst_d128_g16_l3_views.yaml cst_d128_g16_l3_views_formal cst_d128_g16_l3_views

# Candidate roles are classified from each resolved config, not from run names.
uv run python scripts/freeze_cst_selection.py \
  --validation-dir "$output/validation" \
  --output-dir "$output" \
  --expected-candidates 8 \
  --baseline-validation diagnosis/information_audit/validation/a2.json

selected_config=$(python -c "import json; print(json.load(open('$output/VALIDATION_FREEZE.json'))['selected_config'])")
selected_checkpoint=$(python -c "import json; print(json.load(open('$output/VALIDATION_FREEZE.json'))['selected_checkpoint'])")
uv run python scripts/eval_correction_state_transfer.py \
  --config "$selected_config" \
  --checkpoint "$selected_checkpoint" \
  --split test \
  --validation-freeze "$output/VALIDATION_FREEZE.json" \
  --output "$output/development_test/cst_selected.json" \
  >"$output/logs/eval_development_test_cst_selected.log" 2>&1
