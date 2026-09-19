import numpy as np
import scipy.sparse as sp
import torch

from du2vox.models.stage2.complement_voxel_detail import ExactVoxelComplement
from du2vox.models.stage2.correction_state_transfer import CorrectionStateTransfer


def build_model(innovation: str = "delta") -> CorrectionStateTransfer:
    prolongation = sp.csr_matrix(
        np.asarray([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=np.float64)
    )
    return CorrectionStateTransfer(
        coordinate_dim=5,
        state_dim=7,
        latent_dim=8,
        transfer_dim=16,
        modulation_groups=4,
        decoder_width=24,
        decoder_depth=2,
        complement=ExactVoxelComplement(prolongation),
        innovation=innovation,
    )


def test_cst_zero_initialization_and_bounded_modulation() -> None:
    model = build_model()
    coordinate = torch.randn(3, 5)
    state = torch.randn(3, 7)
    h0 = torch.randn(3, 8)
    hc = torch.randn(3, 8)
    proposal = model.proposal(coordinate, state, h0, hc)
    assert torch.equal(proposal, torch.zeros_like(proposal))
    assert 0.0 < float(model.alpha.detach()) < 1.0


def test_cst_transition_receives_gradient() -> None:
    model = build_model("hc_delta")
    torch.nn.init.normal_(model.decoder_output.weight)
    coordinate = torch.randn(3, 5)
    state = torch.randn(3, 7, requires_grad=True)
    h0 = torch.randn(3, 8, requires_grad=True)
    hc = torch.randn(3, 8, requires_grad=True)
    model.proposal(coordinate, state, h0, hc).sum().backward()
    assert h0.grad is not None and torch.count_nonzero(h0.grad)
    assert hc.grad is not None and torch.count_nonzero(hc.grad)
    assert state.grad is not None and torch.count_nonzero(state.grad)


def test_cst_optional_context_and_view_slots_are_unambiguous() -> None:
    prolongation = sp.csr_matrix(np.eye(3, 2, dtype=np.float64))
    model = CorrectionStateTransfer(
        coordinate_dim=5,
        state_dim=7,
        latent_dim=8,
        transfer_dim=16,
        modulation_groups=4,
        decoder_width=24,
        decoder_depth=2,
        complement=ExactVoxelComplement(prolongation),
        one_ring_context_dim=6,
        direct_view_dim=4,
    )
    coordinate = torch.randn(3, 5)
    state = torch.randn(3, 7)
    h0 = torch.randn(3, 8)
    hc = torch.randn(3, 8)
    one_ring = torch.randn(3, 6)
    views = torch.randn(3, 4)
    output = model.proposal(coordinate, state, h0, hc, one_ring, views)
    assert output.shape == (3,)
    assert torch.equal(output, torch.zeros_like(output))


def test_cst_one_ring_context_shape_and_gradient() -> None:
    prolongation = sp.csr_matrix(np.eye(3, 2, dtype=np.float64))
    model = CorrectionStateTransfer(
        coordinate_dim=5,
        state_dim=7,
        latent_dim=8,
        transfer_dim=16,
        modulation_groups=4,
        decoder_width=24,
        decoder_depth=2,
        complement=ExactVoxelComplement(prolongation),
        one_ring_context_dim=6,
    )
    elements = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    neighbors = torch.tensor([[0, 1, 0, 0], [1, 0, 1, 1]])
    x0 = torch.randn(5)
    xc = torch.randn(5)
    h0 = torch.randn(5, 8, requires_grad=True)
    hc = torch.randn(5, 8, requires_grad=True)
    context = model.build_one_ring_context(x0, xc, h0, hc, elements, neighbors)
    assert context is not None and context.shape == (2, 6)
    context.sum().backward()
    assert h0.grad is not None
    assert hc.grad is not None
