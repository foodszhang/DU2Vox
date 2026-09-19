from __future__ import annotations

import copy

import pytest

from scripts.materialize_d1q_decoder_pair import assert_matched_pair


def test_matched_pair_allows_only_name_and_hard_q_mode() -> None:
    hard = {
        "experiment": {"name": "hard"},
        "model": {"mode": "hard", "hidden_dim": 160},
        "training": {"epochs": 30, "seed": 7},
    }
    free = copy.deepcopy(hard)
    free["experiment"]["name"] = "free"
    free["model"]["mode"] = "unconstrained"
    assert_matched_pair(hard, free)


def test_matched_pair_rejects_training_budget_difference() -> None:
    hard = {
        "experiment": {"name": "hard"},
        "model": {"mode": "hard"},
        "training": {"epochs": 30},
    }
    free = copy.deepcopy(hard)
    free["experiment"]["name"] = "free"
    free["model"]["mode"] = "unconstrained"
    free["training"]["epochs"] = 31
    with pytest.raises(RuntimeError, match="training.epochs"):
        assert_matched_pair(hard, free)
