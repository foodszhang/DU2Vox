from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from du2vox.bridge.canonical_cross_discretization import CanonicalCrossDiscretization
from du2vox.models.stage2.complement_voxel_detail import (
    ComplementConstrainedVoxelDetail,
    ExactVoxelComplement,
    normalize_detail_mode,
)


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "experiments/cross_discretization_decomposition/artifacts/operator_cache"
SAMPLES = Path("/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k/samples")
VAL_SPLIT = Path("/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k/splits/val.txt")


@lru_cache(maxsize=1)
def _operators() -> tuple[CanonicalCrossDiscretization, ExactVoxelComplement]:
    canonical = CanonicalCrossDiscretization(CACHE, factorize=True)
    return canonical, ExactVoxelComplement(
        canonical.p, quadrature_weight=canonical.quadrature_weight
    )


def _relative(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _ground_truth(canonical: CanonicalCrossDiscretization) -> np.ndarray:
    sample_id = VAL_SPLIT.read_text().splitlines()[0]
    volume = np.load(SAMPLES / sample_id / "gt_voxels.npy")
    return (volume.ravel()[canonical.operator.valid_flat_indices] > 0.05).astype(np.float64)


def test_weighted_projection_left_identity() -> None:
    canonical, complement = _operators()
    coefficients = np.random.default_rng(1).standard_normal(complement.n_fem_nodes)
    recovered = canonical.project_coefficients(canonical.prolong(coefficients))
    assert _relative(recovered, coefficients) < 1e-12


def test_coarse_and_complement_projectors_are_idempotent() -> None:
    _, complement = _operators()
    z = np.random.default_rng(2).standard_normal(complement.n_voxels)
    pz = complement.project_numpy(z)
    qz = complement.apply_numpy(z)
    assert _relative(complement.project_numpy(pz), pz) < 1e-12
    assert _relative(complement.apply_numpy(qz), qz) < 1e-12


def test_coarse_projector_is_weighted_self_adjoint() -> None:
    canonical, complement = _operators()
    rng = np.random.default_rng(3)
    first = rng.standard_normal(complement.n_voxels)
    second = rng.standard_normal(complement.n_voxels)
    lhs = canonical.inner_product(first, complement.project_numpy(second))
    rhs = canonical.inner_product(complement.project_numpy(first), second)
    error = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)
    assert error < 1e-12


def test_complement_is_annihilated_by_both_coarse_maps() -> None:
    canonical, complement = _operators()
    qz = complement.apply_numpy(
        np.random.default_rng(4).standard_normal(complement.n_voxels)
    )
    assert np.linalg.norm(complement.project_numpy(qz)) / np.linalg.norm(qz) < 1e-12
    assert np.linalg.norm(canonical.project_coefficients(qz)) / np.linalg.norm(qz) < 1e-12


def _assert_weighted_pythagorean(values: np.ndarray) -> None:
    canonical, complement = _operators()
    coarse = complement.project_numpy(values)
    detail = complement.apply_numpy(values)
    total = canonical.weighted_energy(values)
    pieces = canonical.weighted_energy(coarse) + canonical.weighted_energy(detail)
    assert abs(total - pieces) / max(total, 1e-30) < 1e-12


def test_weighted_pythagorean_decomposition_random_and_gt() -> None:
    canonical, complement = _operators()
    _assert_weighted_pythagorean(
        np.random.default_rng(5).standard_normal(complement.n_voxels)
    )
    _assert_weighted_pythagorean(_ground_truth(canonical))


def test_gt_contract_matches_explicit_definition() -> None:
    canonical, complement = _operators()
    gt = _ground_truth(canonical)
    explicit = gt - canonical.prolong(canonical.project_coefficients(gt))
    assert _relative(complement.apply_numpy(gt), explicit) < 1e-12


def test_exact_custom_backward_is_q_transpose_and_q_for_uniform_w() -> None:
    _, complement = _operators()
    generator = torch.Generator().manual_seed(6)
    values = torch.randn(complement.n_voxels, generator=generator, dtype=torch.float64)
    values.requires_grad_()
    weights = torch.randn(complement.n_voxels, generator=generator, dtype=torch.float64)
    torch.dot(complement(values), weights).backward()
    expected = complement.apply_numpy(weights.numpy())
    assert _relative(values.grad.numpy(), expected) < 1e-12


def test_final_coarse_preservation_fp64_and_fp32() -> None:
    canonical, complement = _operators()
    rng = np.random.default_rng(7)
    coefficients = rng.standard_normal(complement.n_fem_nodes)
    detail = complement.apply_numpy(rng.standard_normal(complement.n_voxels))
    final = canonical.prolong(coefficients) + detail
    assert _relative(canonical.project_coefficients(final), coefficients) < 1e-12
    final_fp32 = (
        canonical.prolong(coefficients).astype(np.float32) + detail.astype(np.float32)
    )
    assert _relative(canonical.project_coefficients(final_fp32), coefficients) < 1e-7


def test_soft_penalty_has_nonzero_proposal_gradient_and_does_not_change_output() -> None:
    _, complement = _operators()
    model = ComplementConstrainedVoxelDetail(
        input_dim=3, complement=complement, hidden_dim=128, n_hidden_layers=3, mode="soft"
    )
    proposal = torch.randn(1, complement.n_voxels, requires_grad=True)
    penalty = model.soft_coarse_penalty(proposal, eps=1e-12)
    penalty.backward()
    assert proposal.grad is not None and float(proposal.grad.abs().sum()) > 0
    assert torch.equal(model.constrain(proposal.detach()), proposal.detach())


def test_legacy_constrained_mode_maps_to_hard() -> None:
    assert normalize_detail_mode("constrained") == "hard"
    assert normalize_detail_mode(True) == "hard"
    assert normalize_detail_mode(False) == "unconstrained"
