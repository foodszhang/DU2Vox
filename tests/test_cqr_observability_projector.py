import torch

from du2vox.models.stage2.cqr_observability_projector import CQRObservabilityProjector


def test_observable_plus_ambiguous_recovers_raw_and_backward():
    torch.manual_seed(4)
    projector = CQRObservabilityProjector()
    a_query = torch.randn(2, 5, 17, requires_grad=True)
    raw = torch.randn(2, 17, requires_grad=True)
    observable = projector.project_observable(raw, a_query)
    ambiguous = projector.project_ambiguous(raw, a_query)
    torch.testing.assert_close(observable + ambiguous, raw)
    (observable.square().mean() + ambiguous.square().mean()).backward()
    assert torch.isfinite(raw.grad).all()
    assert torch.isfinite(a_query.grad).all()


def test_bf16_autocast_solve_falls_back_to_float32_without_nan():
    projector = CQRObservabilityProjector()
    a_query = torch.randn(1, 4, 12)
    raw = torch.randn(1, 12)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = projector.project_observable(raw, a_query)
    assert result.dtype == raw.dtype
    assert torch.isfinite(result).all()
