import torch

from du2vox.models.stage1.blocks import InputBlock


def _operators() -> tuple[torch.Tensor, torch.Tensor]:
    laplacian = torch.eye(3, dtype=torch.float32).to_sparse()
    forward = torch.tensor(
        [[1.0, 0.5, -0.25], [0.0, 2.0, 1.0]],
        dtype=torch.float32,
    )
    return laplacian, forward


def test_raw_physics_evidence_preserves_legacy_adjoint() -> None:
    laplacian, forward = _operators()
    block = InputBlock(laplacian, forward, physics_evidence="raw")
    x = torch.tensor([[[0.2], [0.4], [0.1]]], dtype=torch.float32)
    b = torch.tensor([[[0.7], [0.3]]], dtype=torch.float32)

    result = block(x, b)
    expected = (x.squeeze(-1) @ forward.T - b.squeeze(-1)) @ forward

    torch.testing.assert_close(result[..., 2], expected)


def test_profiled_evidence_is_invariant_to_measurement_amplitude() -> None:
    laplacian, forward = _operators()
    block = InputBlock(
        laplacian,
        forward,
        physics_evidence="profiled_normalized",
    )
    x = torch.tensor([[[0.2], [0.4], [0.1]]], dtype=torch.float32)
    b = torch.tensor([[[0.7], [0.3]]], dtype=torch.float32)

    evidence = block(x, b)[..., 2]
    scaled_evidence = block(x, 37.0 * b)[..., 2]

    torch.testing.assert_close(evidence, scaled_evidence, rtol=1e-5, atol=1e-6)


def test_profiled_evidence_has_finite_nonzero_cold_start() -> None:
    laplacian, forward = _operators()
    block = InputBlock(
        laplacian,
        forward,
        physics_evidence="profiled_normalized",
    )
    x = torch.zeros((1, 3, 1), dtype=torch.float32)
    b = torch.tensor([[[0.7], [0.3]]], dtype=torch.float32)

    evidence = block(x, b)[..., 2]

    assert torch.isfinite(evidence).all()
    assert torch.linalg.vector_norm(evidence) > 0
    torch.testing.assert_close(evidence.square().mean().sqrt(), torch.tensor(0.05))


def test_profiled_evidence_vanishes_at_scaled_data_consistency() -> None:
    laplacian, forward = _operators()
    block = InputBlock(
        laplacian,
        forward,
        physics_evidence="profiled_normalized",
    )
    x = torch.tensor([[[0.2], [0.4], [0.1]]], dtype=torch.float32)
    b = 7.0 * (x.squeeze(-1) @ forward.T).unsqueeze(-1)

    evidence = block(x, b)[..., 2]

    torch.testing.assert_close(evidence, torch.zeros_like(evidence), atol=1e-6, rtol=0)


def test_invalid_physics_evidence_is_rejected() -> None:
    laplacian, forward = _operators()
    try:
        InputBlock(laplacian, forward, physics_evidence="gt_scaled")
    except ValueError as exc:
        assert "physics_evidence" in str(exc)
    else:
        raise AssertionError("invalid physics evidence mode was accepted")
