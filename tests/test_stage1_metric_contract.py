from __future__ import annotations

import torch

from du2vox.evaluation.metrics import evaluate_batch


def test_stage1_metrics_use_exported_nonnegative_state() -> None:
    nodes = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    target = torch.tensor([[[1.0], [0.0]]])
    signed = torch.tensor([[[0.1], [-0.1]]])
    physical = signed.clamp(0.0, 1.0)
    signed_metrics = evaluate_batch(signed, target, nodes)
    physical_metrics = evaluate_batch(physical, target, nodes)
    assert signed_metrics == physical_metrics
    assert signed_metrics["location_error"] == 0.0
