import torch

from du2vox.models.stage2.transport_cqr_lifter import TransportConsistentCQRLifter


def make_prior(batch=2, points=7):
    torch.manual_seed(2)
    values = torch.rand(batch, points, 4)
    barycentric = torch.rand(batch, points, 4)
    barycentric = barycentric / barycentric.sum(dim=-1, keepdim=True)
    extras = torch.rand(batch, points, 7)
    return torch.cat([values, barycentric, extras], dim=-1)


def test_zero_init_lifter_is_p1_and_convex():
    torch.manual_seed(1)
    lifter = TransportConsistentCQRLifter(torch.randn(5, 8), torch.tensor([[0, 1, 2, 3]]))
    prior = make_prior()
    tet_ids = torch.zeros(prior.shape[:2], dtype=torch.long)
    band = torch.ones_like(tet_ids)
    output = lifter(prior, tet_ids, band)
    expected = (prior[..., :4] * prior[..., 4:8]).sum(dim=-1)
    torch.testing.assert_close(output["alpha"], prior[..., 4:8], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(output["rho0"], expected, atol=1e-6, rtol=1e-6)
    assert torch.all(output["alpha"] >= 0)
    torch.testing.assert_close(output["alpha"].sum(dim=-1), torch.ones_like(expected))


def test_lifter_backward_is_finite():
    lifter = TransportConsistentCQRLifter(torch.randn(4, 8), torch.tensor([[0, 1, 2, 3]]))
    prior = make_prior(batch=1)
    output = lifter(prior, torch.zeros(1, 7, dtype=torch.long), torch.ones(1, 7, dtype=torch.long))
    output["rho0"].sum().backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in lifter.parameters())
