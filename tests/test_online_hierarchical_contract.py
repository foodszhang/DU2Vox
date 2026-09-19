from __future__ import annotations

import torch
import torch.nn as nn

from scripts.train_online_hierarchical_v4 import configure_trainable_parameters


class _DummyV4(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.Linear(3, 3)
        self.view_encoder = nn.Linear(3, 3)
        self.measurement_residual = nn.Linear(3, 3, bias=False)
        self.dc_step_logits = nn.Parameter(torch.ones(3))


def test_joint_exposes_only_voxel_and_v4_cell() -> None:
    v4 = _DummyV4()
    detail = nn.Linear(3, 1)
    voxel, cell = configure_trainable_parameters(v4, detail, "joint_v4_cell")
    assert voxel and cell
    assert all(parameter.requires_grad for parameter in detail.parameters())
    assert all(parameter.requires_grad for parameter in v4.cell.parameters())
    assert not any(parameter.requires_grad for parameter in v4.view_encoder.parameters())
    assert not any(parameter.requires_grad for parameter in v4.measurement_residual.parameters())
    assert not v4.dc_step_logits.requires_grad


def test_frozen_online_exposes_only_voxel_branch() -> None:
    v4 = _DummyV4()
    detail = nn.Linear(3, 1)
    voxel, cell = configure_trainable_parameters(v4, detail, "frozen_online")
    assert voxel and not cell
    assert not any(parameter.requires_grad for parameter in v4.parameters())
