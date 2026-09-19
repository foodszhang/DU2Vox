import torch

from du2vox.models.stage1.blocks import sparse_left_mm_batched


def _reference(matrix: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.sparse.mm(matrix, item) for item in values])


def test_sparse_left_mm_batched_matches_sample_loop_forward_and_backward() -> None:
    indices = torch.tensor([[0, 0, 1, 2, 3], [0, 2, 1, 0, 3]])
    weights = torch.tensor([1.0, -0.2, 0.7, 0.4, 1.3])
    matrix = torch.sparse_coo_tensor(indices, weights, (4, 4)).coalesce()
    values = torch.randn(3, 4, 5, generator=torch.Generator().manual_seed(7))

    reference_values = values.clone().requires_grad_(True)
    batched_values = values.clone().requires_grad_(True)
    reference = _reference(matrix, reference_values)
    batched = sparse_left_mm_batched(matrix, batched_values)

    torch.testing.assert_close(batched, reference, rtol=1e-6, atol=1e-7)
    reference.square().sum().backward()
    batched.square().sum().backward()
    torch.testing.assert_close(
        batched_values.grad, reference_values.grad, rtol=1e-6, atol=1e-7
    )
