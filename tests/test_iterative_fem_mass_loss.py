import torch

from scripts.train_iterative_fem_corrector import fem_mass_relative_loss


def test_fem_mass_relative_loss_uses_nodal_volume_weights():
    target = torch.tensor([[1.0, 2.0, 0.0]])
    weights = torch.tensor([[1.0, 3.0, 10.0]])
    exact_loss, exact_ratio = fem_mass_relative_loss(target, target, weights)
    torch.testing.assert_close(exact_loss, torch.tensor(0.0))
    torch.testing.assert_close(exact_ratio, torch.tensor([1.0]))

    half_loss, half_ratio = fem_mass_relative_loss(0.5 * target, target, weights)
    torch.testing.assert_close(half_loss, torch.tensor(0.25))
    torch.testing.assert_close(half_ratio, torch.tensor([0.5]))
