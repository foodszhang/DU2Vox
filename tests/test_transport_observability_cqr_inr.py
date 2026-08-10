from pathlib import Path

import torch
import yaml

from du2vox.models.stage2.cqr_residual_inr import CQRResidualINR
from du2vox.models.stage2.transport_observability_cqr_inr import (
    TransportObservabilityCQRINR,
)


def synthetic_batch():
    torch.manual_seed(8)
    batch, points, nodes, rank, surface = 1, 13, 8, 4, 5
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
        "hidden_dim": 24,
        "n_hidden_layers": 2,
        "view_feat_dim": 0,
    }
    inputs = {
        "coords": torch.rand(batch, points, 3) * 2 - 1,
        "prior_lift": prior,
        "correction_band": torch.randint(0, 5, (batch, points)),
        "tet_ids": torch.randint(0, 2, (batch, points)),
        "candidate_cell_weight": torch.rand(batch, points),
        "n_valid_candidate_pool": torch.tensor([32768]),
        "measurement_b": torch.rand(batch, surface),
        "coarse_d": torch.rand(batch, nodes),
    }
    return kwargs, inputs


def test_model_zero_init_is_p1_and_backward_finite():
    kwargs, inputs = synthetic_batch()
    model = TransportObservabilityCQRINR(**kwargs)
    output = model(**inputs)
    expected = (inputs["prior_lift"][..., :4] * inputs["prior_lift"][..., 4:8]).sum(-1)
    torch.testing.assert_close(output["rho0"], expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(output["d_hat"], expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(output["data_relative_before"], torch.ones(1))
    torch.testing.assert_close(output["data_relative_after"], torch.ones(1))
    assert torch.isfinite(output["transport_error"]).all()
    output["d_hat"].sum().backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_phase_curriculum_can_freeze_lifter_and_select_branch_heads():
    kwargs, _ = synthetic_batch()
    model = TransportObservabilityCQRINR(**kwargs)

    model.set_phase("observable", freeze_lifter_after_phase_a=True)
    assert not any(parameter.requires_grad for parameter in model.lifter.parameters())
    assert all(parameter.requires_grad for parameter in model.observable_head.parameters())
    assert not any(parameter.requires_grad for parameter in model.ambiguous_head.parameters())

    model.set_phase("full", freeze_lifter_after_phase_a=True)
    assert not any(parameter.requires_grad for parameter in model.lifter.parameters())
    assert all(parameter.requires_grad for parameter in model.observable_head.parameters())
    assert all(parameter.requires_grad for parameter in model.ambiguous_head.parameters())

    model.set_phase("ambiguous_pretrain", freeze_lifter_after_phase_a=True)
    assert not any(parameter.requires_grad for parameter in model.lifter.parameters())
    assert not any(parameter.requires_grad for parameter in model.observable_head.parameters())
    assert all(parameter.requires_grad for parameter in model.ambiguous_head.parameters())

    model.set_phase("joint", freeze_lifter_after_phase_a=True)
    assert not any(parameter.requires_grad for parameter in model.lifter.parameters())
    assert all(parameter.requires_grad for parameter in model.observable_head.parameters())
    assert all(parameter.requires_grad for parameter in model.ambiguous_head.parameters())


def test_old_sparse_cqr_checkpoint_and_config_remain_loadable():
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs/stage2/cqr_v2_3k_rgl_main_multiview_sparse.yaml"
    checkpoint_path = root / "checkpoints/stage2/cqr_v2_3k_rgl_main_sparse/best.pth"
    if not checkpoint_path.exists():
        return
    cfg = yaml.safe_load(config_path.read_text())
    model_cfg = cfg["model"]
    model = CQRResidualINR(
        n_freqs=model_cfg["n_freqs"],
        hidden_dim=model_cfg["hidden_dim"],
        n_hidden_layers=model_cfg["n_hidden_layers"],
        prior_dim=model_cfg["prior_dim"],
        skip_connection=model_cfg["skip_connection"],
        view_feat_dim=model_cfg["view_feat_dim"],
        residual_scale=model_cfg["residual_scale"],
        use_band_embedding=model_cfg["use_band_embedding"],
        band_embed_dim=model_cfg["band_embed_dim"],
        num_bands=model_cfg["num_bands"],
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["residual_inr"], strict=True)
