import torch

from du2vox.models.stage2.plain_cqr_residual_control import PlainCQRResidualControl
from du2vox.models.stage2.transport_observability_cqr_inr import (
    TransportObservabilityCQRINR,
)


def synthetic_case(lifting_mode: str):
    torch.manual_seed(19)
    batch, points, nodes, rank, surface = 1, 17, 8, 4, 5
    values = torch.rand(batch, points, 4)
    barycentric = torch.rand(batch, points, 4)
    barycentric /= barycentric.sum(dim=-1, keepdim=True)
    prior = torch.cat([values, barycentric, torch.rand(batch, points, 7)], dim=-1)
    kwargs = {
        "green_node_modes": torch.randn(rank, nodes),
        "elements": torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]),
        "measurement_basis": torch.randn(surface, rank),
        "projected_forward_modes": torch.randn(rank, nodes),
        "n_freqs": 2,
        "hidden_dim": 32,
        "n_hidden_layers": 2,
        "view_feat_dim": 3,
    }
    inputs = {
        "coords": torch.rand(batch, points, 3) * 2 - 1,
        "prior_lift": prior,
        "view_feat": torch.rand(batch, points, 3),
        "correction_band": torch.randint(0, 5, (batch, points)),
        "tet_ids": torch.randint(0, 2, (batch, points)),
        "role": torch.randint(0, 5, (batch, points)),
        "query_src_tag": torch.randint(0, 4, (batch, points)),
        "candidate_cell_weight": torch.rand(batch, points),
        "n_valid_candidate_pool": torch.tensor([32768]),
        "measurement_b": torch.rand(batch, surface),
        "coarse_d": torch.rand(batch, nodes),
    }
    return PlainCQRResidualControl(lifting_mode=lifting_mode, **kwargs), kwargs, inputs


def test_p1_plain_zero_initialization_and_backward():
    model, _, inputs = synthetic_case("p1")
    output = model(**inputs)
    p1 = (inputs["prior_lift"][..., :4] * inputs["prior_lift"][..., 4:8]).sum(-1)
    torch.testing.assert_close(output["rho0"], p1)
    torch.testing.assert_close(output["d_hat"], p1)
    torch.testing.assert_close(output["plain_correction"], torch.zeros_like(p1))
    assert model.lifter is None
    output["d_hat"].sum().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_tc_plain_and_partition_share_backbone_capacity():
    plain, kwargs, inputs = synthetic_case("transport")
    partition = TransportObservabilityCQRINR(**kwargs)
    p1_plain, _, _ = synthetic_case("p1")
    plain_output = plain(**inputs)
    partition_output = partition(**inputs)
    torch.testing.assert_close(plain_output["rho0"], partition_output["rho0"])
    plain_backbone = sum(parameter.numel() for parameter in plain.backbone.parameters())
    partition_backbone = sum(parameter.numel() for parameter in partition.backbone.parameters())
    assert plain_backbone == partition_backbone
    for p1_parameter, tc_parameter in zip(
        p1_plain.backbone.parameters(), plain.backbone.parameters()
    ):
        torch.testing.assert_close(p1_parameter, tc_parameter)
    for plain_parameter, partition_parameter in zip(
        plain.backbone.parameters(), partition.backbone.parameters()
    ):
        torch.testing.assert_close(plain_parameter, partition_parameter)
    plain_total = sum(parameter.numel() for parameter in plain.parameters())
    partition_total = sum(parameter.numel() for parameter in partition.parameters())
    assert abs(partition_total - plain_total) / partition_total < 0.01
