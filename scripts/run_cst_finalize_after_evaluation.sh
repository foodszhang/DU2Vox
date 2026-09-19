#!/usr/bin/env bash
set -euo pipefail

while tmux has-session -t du2vox_cst_evaluation 2>/dev/null; do
  sleep 60
done

output=diagnosis/final_dual_space
test -f "$output/VALIDATION_FREEZE.json"
test -f "$output/development_test/cst_selected.json"
selected_name=$(python -c "import json; print(json.load(open('$output/VALIDATION_FREEZE.json'))['selected_cst_candidate'])")

uv run python scripts/finalize_dual_space_results.py \
  --output-dir "$output" \
  --proposed-val "$output/validation/${selected_name}.json" \
  --proposed-test "$output/development_test/cst_selected.json" \
  --b4-val diagnosis/information_audit/validation/a2.json \
  --b4-test diagnosis/information_audit/development_test/a2.json \
  --q-only-no-view-test diagnosis/stage1_hard_q/development_test_no_views.json \
  --q-only-test diagnosis/stage1_hard_q/development_test_selected.json \
  --validation-freeze "$output/VALIDATION_FREEZE.json" \
  --p0-comparison diagnosis/p0_joint_vs_frozen_v4_comparison.json \
  --responsibility-audit "$output/RESPONSIBILITY_TRAINING_AUDIT.json" \
  >"$output/logs/finalize.log" 2>&1

uv run ruff check \
  du2vox/models/stage2/correction_state_transfer.py \
  scripts/train_correction_state_transfer.py \
  scripts/eval_correction_state_transfer.py \
  scripts/freeze_cst_selection.py \
  scripts/finalize_dual_space_results.py \
  >"$output/logs/final_lint.log" 2>&1
uv run pytest -q \
  tests/test_correction_state_transfer.py \
  tests/test_iterative_fem_corrector_contract.py \
  tests/test_voxel_complement_projection.py \
  >"$output/logs/final_tests.log" 2>&1

uv run python scripts/audit_final_dual_space_release.py \
  --output-dir "$output" \
  >"$output/logs/completion_audit.log" 2>&1

date --iso-8601=seconds >"$output/PIPELINE_COMPLETE"
