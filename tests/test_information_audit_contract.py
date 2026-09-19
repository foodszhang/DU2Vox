from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from du2vox.models.stage2.complement_voxel_detail import (
    ComplementConstrainedVoxelDetail,
    ExactVoxelComplement,
)
from du2vox.models.stage2.information_audit import (
    INFORMATION_ARMS,
    InformationFeatureLayout,
    assemble_information_features,
    select_coarse_state,
)
from scripts.precompute_frozen_v4_states import (
    fp16_relative_l2,
    metadata_stratified_audit_ids,
    sha256,
    stratified_audit_ids,
)
from scripts.train_information_audit import load_audit_config


def _groups(layout: InformationFeatureLayout) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    return {
        "coordinate_pe": torch.randn(5, layout.coordinate_dim, generator=generator),
        "barycentric": torch.randn(5, layout.barycentric_dim, generator=generator),
        "state": torch.randn(5, layout.state_dim, generator=generator),
        "latent": torch.randn(5, layout.latent_dim, generator=generator),
        "views": torch.randn(5, layout.view_dim, generator=generator),
    }


def test_fixed_layout_and_authorized_channel_sensitivity() -> None:
    layout = InformationFeatureLayout()
    groups = _groups(layout)
    assert layout.input_dim == 231
    for name, arm in INFORMATION_ARMS.items():
        baseline = assemble_information_features(arm, layout=layout, **groups)
        assert baseline.shape == (5, 231)
        for group in ("state", "latent", "views"):
            changed = copy.copy(groups)
            changed[group] = groups[group] + 1000.0
            perturbed = assemble_information_features(arm, layout=layout, **changed)
            authorized = {
                "state": arm.use_state,
                "latent": arm.use_latent,
                "views": arm.use_views,
            }[group]
            assert (not torch.equal(baseline, perturbed)) is authorized, (name, group)


def test_a0_is_insensitive_to_every_non_geometry_source() -> None:
    layout = InformationFeatureLayout()
    groups = _groups(layout)
    baseline = assemble_information_features(
        INFORMATION_ARMS["A0"], layout=layout, **groups
    )
    changed = copy.copy(groups)
    for key in ("state", "latent", "views"):
        changed[key] = torch.randn_like(changed[key]) * 1e6
    assert torch.equal(
        baseline,
        assemble_information_features(
            INFORMATION_ARMS["A0"], layout=layout, **changed
        ),
    )


def test_oracle_state_is_confined_to_a3_and_va3() -> None:
    frozen = torch.tensor([1.0])
    oracle = torch.tensor([2.0])
    for name, arm in INFORMATION_ARMS.items():
        selected = select_coarse_state(
            arm, frozen_state=frozen, oracle_state=oracle
        )
        assert torch.equal(selected, oracle if name in {"A3", "VA3"} else frozen)


def test_all_arm_configs_have_identical_decoder_and_training_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    configs = [
        load_audit_config(root / "configs/stage2/information_audit" / f"{name}.yaml")
        for name in ("a0", "a1", "a2", "a3", "va0", "va1", "va2", "va3")
    ]
    reference = configs[0]
    for cfg in configs[1:]:
        assert cfg["model"] == reference["model"]
        assert cfg["training"] == reference["training"]
        assert cfg["loss"] == reference["loss"]
        assert cfg["validation"] == reference["validation"]
    complement = ExactVoxelComplement(sp.eye(4, format="csr"))
    states = []
    for _ in configs:
        torch.manual_seed(20260901)
        model = ComplementConstrainedVoxelDetail(
            input_dim=231,
            complement=complement,
            hidden_dim=160,
            n_hidden_layers=3,
            mode="hard",
        )
        states.append(model.state_dict())
    assert all(
        torch.equal(states[0][key], state[key])
        for state in states[1:]
        for key in states[0]
    )
    assert len({sum(value.numel() for value in state.values()) for state in states}) == 1


def test_checkpointed_proposal_has_backward_and_zero_initial_output() -> None:
    complement = ExactVoxelComplement(sp.eye(4, format="csr"))
    model = ComplementConstrainedVoxelDetail(
        input_dim=7,
        complement=complement,
        hidden_dim=128,
        n_hidden_layers=3,
        mode="hard",
    )
    features = torch.randn(4, 7)
    proposal = model.proposal_checkpointed(features)
    assert torch.equal(proposal, torch.zeros_like(proposal))
    proposal.sum().backward()
    assert model.output.bias.grad is not None


def test_stratified_latent_audit_selection_is_deterministic() -> None:
    ids = [f"sample_{index:04d}" for index in range(300)]
    first = stratified_audit_ids(ids, 30, 20260901)
    second = stratified_audit_ids(ids, 30, 20260901)
    assert first == second
    assert len(first) == 30
    assert all(
        any(sample in first for sample in ids[start : start + 10])
        for start in range(0, 300, 10)
    )


def test_fp16_latent_error_finiteness_shape_and_hash_are_deterministic(
    tmp_path: Path,
) -> None:
    values = np.random.default_rng(23).normal(size=(19, 144)).astype(np.float32)
    quantized, error = fp16_relative_l2(values)
    assert quantized.shape == values.shape
    assert quantized.dtype == np.float16
    assert np.isfinite(quantized).all()
    assert error <= 5e-4
    first = tmp_path / "first.npy"
    second = tmp_path / "second.npy"
    np.save(first, quantized)
    np.save(second, quantized)
    assert sha256(first) == sha256(second)


def test_fp16_latent_rejects_nonfinite_values() -> None:
    values = np.zeros((2, 144), dtype=np.float32)
    values[0, 0] = np.nan
    try:
        fp16_relative_l2(values)
    except ValueError as error:
        assert "non-finite" in str(error)
    else:
        raise AssertionError("Non-finite latent was accepted")


def test_metadata_stratified_selection_is_deterministic(tmp_path: Path) -> None:
    samples = tmp_path / "samples"
    samples.mkdir()
    ids = []
    for index in range(20):
        sid = f"s{index:02d}"
        ids.append(sid)
        (samples / sid).mkdir()
        payload = {
            "foci": [{}] * (1 + index % 2),
            "depth_tier": "deep" if index % 4 < 2 else "shallow",
            "source_type": "multi" if index % 2 else "single",
        }
        (samples / sid / "tumor_params.json").write_text(json.dumps(payload))
    first = metadata_stratified_audit_ids(ids, samples, 8, 7)
    second = metadata_stratified_audit_ids(ids, samples, 8, 7)
    assert first == second
    assert len(first) == 8
