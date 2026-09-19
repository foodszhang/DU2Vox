from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from du2vox.bridge.canonical_cross_discretization import (
    CanonicalCrossDiscretization,
    canonical_p1_torch,
)
from du2vox.models.stage2.error_structured_bridge import (
    ErrorStructuredCrossDiscretizationBridge,
    FEMInverseCorrectionNet,
    VoxelRepresentationCompletionNet,
    fixed_representation_target,
)


REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "experiments/cross_discretization_decomposition/artifacts/operator_cache"
SHARED = Path("/home/foods/pro/FMT-SimGen/output/shared_mesh_20k")
SAMPLES = Path("/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k/samples")
BRIDGE = REPO / "output/bridge_v2_3k_train_balanced_v2"


def tiny_model() -> ErrorStructuredCrossDiscretizationBridge:
    knn = torch.tensor(
        [[0, 1], [0, 1], [1, 2], [2, 3], [3, 4]], dtype=torch.long
    )
    return ErrorStructuredCrossDiscretizationBridge(
        FEMInverseCorrectionNet(knn, hidden_dim=16, n_hidden_layers=2),
        VoxelRepresentationCompletionNet(
            n_freqs=2, hidden_dim=16, n_hidden_layers=2
        ),
    )


def tiny_inputs() -> dict[str, torch.Tensor]:
    return {
        "x_h": torch.tensor([[0.0, 0.2, 0.8, 0.4, 0.1]]),
        "node_coords_norm": torch.linspace(-1, 1, 15).reshape(1, 5, 3),
        "query_coords_norm": torch.tensor(
            [[[-0.5, 0.0, 0.2], [0.4, -0.1, 0.8]]]
        ),
        "query_node_indices": torch.tensor(
            [[[0, 1, 2, 3], [1, 2, 3, 4]]]
        ),
        "query_barycentric": torch.tensor(
            [[[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]]
        ),
    }


@pytest.mark.skipif(
    not (CACHE / "P_full_fem_gt_centers.npz").exists()
    or not (SAMPLES / "sample_0000/gt_voxels.npy").exists()
    or not (BRIDGE / "sample_0000/coarse_d.npy").exists(),
    reason="certified decomposition assets are unavailable",
)
def test_oracle_decomposition_closure_on_certified_real_sample() -> None:
    canonical = CanonicalCrossDiscretization(
        CACHE, shared_dir=SHARED, factorize=True
    )
    gt_volume = np.load(SAMPLES / "sample_0000/gt_voxels.npy", mmap_mode="r")
    gt = (
        np.asarray(gt_volume).ravel()[canonical.operator.valid_flat_indices] > 0.05
    ).astype(np.float64)
    x_h = np.load(BRIDGE / "sample_0000/coarse_d.npy").astype(np.float64)
    pi_gt = canonical.project_coefficients(gt)
    stage1 = canonical.prolong(x_h)
    inverse = canonical.prolong(pi_gt - x_h)
    representation = gt - canonical.prolong(pi_gt)
    closure = stage1 + inverse + representation - gt
    relative = np.linalg.norm(closure) / np.linalg.norm(gt - stage1)
    assert relative < 1e-12


def test_representation_target_is_independent_of_inverse_prediction() -> None:
    gt = torch.tensor([[0.0, 1.0, 1.0]])
    pi_gt_voxel = torch.tensor([[0.1, 0.7, 1.2]])
    delta_x_1 = torch.randn(1, 5)
    delta_x_2 = torch.randn(1, 5)
    target_1 = fixed_representation_target(gt, pi_gt_voxel)
    target_2 = fixed_representation_target(gt, pi_gt_voxel)
    assert not torch.equal(delta_x_1, delta_x_2)
    assert torch.equal(target_1, target_2)
    assert list(inspect.signature(fixed_representation_target).parameters) == [
        "gt_values",
        "projected_gt_values",
    ]


def test_analytic_transfer_matches_direct_barycentric_sum_and_has_no_parameters() -> None:
    inputs = tiny_inputs()
    transferred = canonical_p1_torch(
        inputs["x_h"],
        inputs["query_node_indices"],
        inputs["query_barycentric"],
    )
    expected = torch.tensor([[0.44, 0.41]])
    assert torch.allclose(transferred, expected, atol=1e-7)
    assert "nn." not in inspect.getsource(canonical_p1_torch)
    assert not isinstance(canonical_p1_torch, torch.nn.Module)


def test_sequential_forward_identity() -> None:
    model = tiny_model()
    output = model(**tiny_inputs())
    assert torch.equal(
        output["final_prediction"],
        output["corrected_fem_voxel"]
        + output["representation_prediction"],
    )


def test_phase_freeze_and_gradient_contract() -> None:
    model = tiny_model()
    inputs = tiny_inputs()
    inverse_target = torch.ones_like(inputs["x_h"]) * 0.25
    representation_target = torch.tensor([[0.2, -0.3]])
    for phase, expected in {
        "A": (True, False),
        "B": (False, True),
        "C": (True, True),
    }.items():
        model.zero_grad(set_to_none=True)
        model.set_phase(phase)
        assert any(p.requires_grad for p in model.inverse_net.parameters()) == expected[0]
        assert any(p.requires_grad for p in model.representation_net.parameters()) == expected[1]
        output = model(**inputs)
        inverse_loss = torch.mean(
            (output["inverse_correction"] - inverse_target) ** 2
        )
        representation_loss = torch.mean(
            (output["representation_prediction"] - representation_target) ** 2
        )
        loss = {
            "A": inverse_loss,
            "B": representation_loss,
            "C": inverse_loss + representation_loss,
        }[phase]
        loss.backward()
        inverse_grad = sum(
            p.grad.abs().sum().item()
            for p in model.inverse_net.parameters()
            if p.grad is not None
        )
        representation_grad = sum(
            p.grad.abs().sum().item()
            for p in model.representation_net.parameters()
            if p.grad is not None
        )
        assert (inverse_grad > 0) == expected[0]
        assert (representation_grad > 0) == expected[1]


def test_fem_contextual_corrector_has_intended_lightweight_capacity() -> None:
    knn = torch.arange(32).repeat(19990, 1) % 19990
    corrector = FEMInverseCorrectionNet(
        knn, hidden_dim=144, n_hidden_layers=3, view_feat_dim=32
    )
    parameter_count = sum(parameter.numel() for parameter in corrector.parameters())
    assert 200_000 <= parameter_count <= 500_000
    assert len(corrector.context_blocks) == 3


def test_no_observability_or_learned_lifting_dependency() -> None:
    source = inspect.getsource(
        __import__(
            "du2vox.models.stage2.error_structured_bridge", fromlist=["unused"]
        )
    )
    forbidden = (
        "CQRObservabilityProjector",
        "P_mu",
        "I-P_mu",
        "TransportConsistentCQRLifter",
        "transport_cqr_lifter",
    )
    for name in forbidden:
        assert name not in source
