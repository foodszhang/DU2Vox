from __future__ import annotations

import torch

from du2vox.models.stage2.iterative_fem_corrector import (
    DualScaleFEMMeasurementEvidence,
    FEMMeasurementResidual,
    FrozenConvexFEMCorrectorEnsemble,
    IterativeResidualAwareFEMCorrector,
    ResidualConsistentIterativeFEMCorrector,
    ScaleCalibratedFEMMeasurementResidual,
    UnifiedDualEvidenceFEMCorrector,
)


def _model(n_iterations: int = 3) -> IterativeResidualAwareFEMCorrector:
    torch.manual_seed(7)
    n_nodes = 8
    system = torch.randn(5, n_nodes)
    knn = torch.tensor(
        [[(node + offset) % n_nodes for offset in range(3)] for node in range(n_nodes)]
    )
    return IterativeResidualAwareFEMCorrector(
        system_matrix=system,
        knn_indices=knn,
        hidden_dim=16,
        n_context_blocks=2,
        n_iterations=n_iterations,
    )


def _inputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(11)
    return {
        "x_h": torch.randn(2, 8),
        "measurement_b": torch.randn(2, 5),
        "node_coords_norm": torch.randn(2, 8, 3),
        "query_node_indices": torch.tensor(
            [[[0, 1, 2, 3], [4, 5, 6, 7]], [[0, 2, 4, 6], [1, 3, 5, 7]]]
        ),
        "query_barycentric": torch.tensor([[[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]] * 2),
    }


def _residual_consistent_model(
    n_iterations: int = 3,
) -> ResidualConsistentIterativeFEMCorrector:
    torch.manual_seed(19)
    n_nodes = 8
    system = torch.randn(5, n_nodes)
    knn = torch.tensor(
        [[(node + offset) % n_nodes for offset in range(3)] for node in range(n_nodes)]
    )
    return ResidualConsistentIterativeFEMCorrector(
        system_matrix=system,
        knn_indices=knn,
        hidden_dim=16,
        n_context_blocks=1,
        n_iterations=n_iterations,
        max_neural_update=0.5,
        residual_tolerance=0.0,
    )


def _v4_model(n_iterations: int = 3) -> UnifiedDualEvidenceFEMCorrector:
    torch.manual_seed(31)
    n_nodes = 8
    system = torch.randn(5, n_nodes)
    knn = torch.tensor(
        [[(node + offset) % n_nodes for offset in range(3)] for node in range(n_nodes)]
    )
    return UnifiedDualEvidenceFEMCorrector(
        system_matrix=system,
        knn_indices=knn,
        hidden_dim=16,
        n_context_blocks=1,
        n_iterations=n_iterations,
        max_neural_update=0.25,
        max_dc_update=0.1,
    )


def test_v4_volume_centered_neural_update_has_zero_lumped_mass() -> None:
    torch.manual_seed(37)
    n_nodes = 8
    weights = torch.linspace(0.5, 2.0, n_nodes)
    model = UnifiedDualEvidenceFEMCorrector(
        system_matrix=torch.randn(5, n_nodes),
        knn_indices=torch.tensor(
            [[(node + offset) % n_nodes for offset in range(3)] for node in range(n_nodes)]
        ),
        hidden_dim=16,
        n_context_blocks=1,
        n_iterations=3,
        max_neural_update=0.25,
        max_dc_update=0.0,
        nodal_volume_weights=weights,
        volume_center_neural_update=True,
    )
    with torch.no_grad():
        model.cell.delta_head.bias.fill_(0.2)
    inputs = _inputs()
    output = model.correct_nodes(
        inputs["x_h"], inputs["measurement_b"], inputs["node_coords_norm"]
    )
    weighted_mass = (output["step_corrections"] * weights).sum(dim=-1)
    assert torch.max(weighted_mass.abs()) < 1e-6


def test_measurement_residual_is_exact_fixed_forward_adjoint() -> None:
    torch.manual_seed(3)
    system = torch.randn(4, 6)
    state = torch.randn(2, 6)
    measurement = torch.randn(2, 4)
    operator = FEMMeasurementResidual(system)
    output = operator(state, measurement)
    expected_residual = measurement - state @ system.t()
    expected_backprojection = expected_residual @ system
    assert torch.allclose(output["forward_residual"], expected_residual)
    assert torch.allclose(output["backprojection"], expected_backprojection)
    assert sum(parameter.numel() for parameter in operator.parameters()) == 0
    assert "system_matrix" not in operator.state_dict()


def test_measurement_residual_applies_casewise_forward_operator_scale() -> None:
    torch.manual_seed(5)
    system = torch.randn(4, 6)
    state = torch.randn(2, 6)
    scale = torch.tensor([0.25, 3.0])
    measurement = (state @ system.t()) * scale[:, None]
    operator = FEMMeasurementResidual(system)
    output = operator(state, measurement, scale)
    assert torch.max(output["forward_residual"].abs()) < 1e-6
    assert torch.max(output["backprojection"].abs()) < 1e-5


def test_v4_scaled_raw_adjoint_matches_scaled_operator() -> None:
    torch.manual_seed(6)
    system = torch.randn(5, 7)
    state = torch.randn(2, 7)
    measurement = torch.randn(2, 5)
    scale = torch.tensor([0.4, 2.0])
    operator = DualScaleFEMMeasurementEvidence(system)
    output = operator(state, measurement, scale)
    forward = (state @ system.t()) * scale[:, None]
    residual = measurement - forward
    expected_adjoint = (residual @ system) * scale[:, None]
    assert torch.allclose(output["forward_response"], forward)
    assert torch.allclose(output["raw_residual"], residual)
    assert torch.allclose(output["raw_backprojection"], expected_adjoint)


def test_profiled_amplitude_recovers_measurement_scale() -> None:
    torch.manual_seed(23)
    system = torch.randn(6, 8)
    state = torch.randn(2, 8)
    response = state @ system.t()
    expected_amplitude = torch.tensor([0.4, 2.5])
    measurement = expected_amplitude[:, None] * response
    operator = ScaleCalibratedFEMMeasurementResidual(system)
    output = operator(state, measurement)
    assert torch.allclose(output["amplitude"], expected_amplitude, atol=1e-5)
    assert torch.max(output["forward_residual_rms"]) < 1e-5


def test_profiled_residual_is_invariant_to_positive_state_scale() -> None:
    torch.manual_seed(29)
    system = torch.randn(6, 8)
    state = torch.randn(2, 8)
    measurement = torch.randn(2, 6)
    operator = ScaleCalibratedFEMMeasurementResidual(system)
    original = operator(state, measurement)
    scaled = operator(3.0 * state, measurement)
    assert torch.allclose(original["forward_residual"], scaled["forward_residual"], atol=1e-5)
    assert torch.allclose(original["amplitude"], 3.0 * scaled["amplitude"], atol=1e-5)


def test_v4_raw_and_si_residuals_are_exact() -> None:
    torch.manual_seed(37)
    system = torch.randn(6, 8)
    state = torch.randn(2, 8)
    measurement = torch.randn(2, 6)
    operator = DualScaleFEMMeasurementEvidence(system)
    output = operator(state, measurement)
    forward = state @ system.t()
    raw = measurement - forward
    numerator = (forward * measurement).sum(dim=-1, keepdim=True).clamp_min(0.0)
    alpha = numerator / (forward.square().sum(dim=-1, keepdim=True) + operator.eps)
    si = measurement - alpha * forward
    assert torch.allclose(output["raw_residual"], raw)
    assert torch.allclose(output["raw_backprojection"], raw @ system)
    assert torch.allclose(output["si_residual"], si)
    assert torch.allclose(output["si_backprojection"], alpha * (si @ system))
    assert torch.all(output["amplitude"] >= 0.0)
    assert sum(parameter.numel() for parameter in operator.parameters()) == 0


def test_v4_zero_profiled_adjoint_has_finite_backward() -> None:
    torch.manual_seed(41)
    system = torch.randn(6, 8)
    state = torch.randn(2, 8, requires_grad=True)
    response = state.detach() @ system.t()
    # Negative correlation gives the analytic non-negative amplitude alpha=0,
    # hence an exactly zero SI backprojection and RMS.
    measurement = -response
    operator = DualScaleFEMMeasurementEvidence(system)
    output = operator(state, measurement)
    assert torch.equal(output["si_backprojection"], torch.zeros_like(output["si_backprojection"]))
    output["si_normalized_backprojection"].square().sum().backward()
    assert state.grad is not None
    assert torch.isfinite(state.grad).all()


def test_v4_recomputes_both_evidence_channels_every_iteration() -> None:
    model = _v4_model(n_iterations=3)
    inputs = _inputs()
    calls = 0

    def count_call(_module: torch.nn.Module, _args: object, _output: object) -> None:
        nonlocal calls
        calls += 1

    handle = model.measurement_residual.register_forward_hook(count_call)
    output = model(**inputs)
    handle.remove()
    assert calls == 4  # three current states plus one post-update diagnostic
    assert output["raw_residual_rms"].shape == (2, 4)
    assert output["si_residual_rms"].shape == (2, 4)


def test_v4_update_is_bounded_shared_and_analytic_p1_only() -> None:
    model = _v4_model(n_iterations=3)
    inputs = _inputs()
    with torch.no_grad():
        model.cell.delta_head.bias.fill_(100.0)
        model.dc_step_logits.fill_(100.0)
    output = model(**inputs)
    # Neural <= .25 and DC <= .2 because the established parameterization is
    # max_dc_update * 2*sigmoid(logit).
    assert torch.all(output["step_corrections"].abs() <= 0.45 + 1e-6)
    assert len({id(model.cell) for _ in range(model.n_iterations)}) == 1
    corrected = output["corrected_nodes"]
    gathered = corrected.gather(1, inputs["query_node_indices"].reshape(2, -1)).reshape(2, 2, 4)
    expected = (gathered * inputs["query_barycentric"]).sum(dim=-1)
    assert torch.equal(output["final_prediction"], expected)
    names = {name for name, _ in model.named_modules()}
    assert not any("voxel" in name or "lift" in name for name in names)


def test_v4_terminal_hidden_export_is_opt_in_and_state_identical() -> None:
    model = _v4_model(n_iterations=3).eval()
    inputs = _inputs()
    with torch.no_grad():
        default = model.correct_nodes(
            inputs["x_h"], inputs["measurement_b"], inputs["node_coords_norm"]
        )
        exported = model.correct_nodes(
            inputs["x_h"],
            inputs["measurement_b"],
            inputs["node_coords_norm"],
            return_terminal_hidden=True,
        )
    assert "terminal_hidden" not in default
    assert exported["terminal_hidden"].shape == (2, 8, 16)
    assert set(exported) == set(default) | {"terminal_hidden"}
    assert torch.equal(default["corrected_nodes"], exported["corrected_nodes"])
    assert torch.equal(default["step_states"], exported["step_states"])


def test_zero_initialization_preserves_stage1_and_analytic_p1() -> None:
    model = _model()
    inputs = _inputs()
    output = model(**inputs)
    assert output["step_states"].shape == (2, 3, 8)
    assert output["measurement_residual_rms"].shape == (2, 4)
    for step in range(3):
        assert torch.equal(output["step_states"][:, step], inputs["x_h"])
    gathered = inputs["x_h"].gather(1, inputs["query_node_indices"].reshape(2, -1)).reshape(2, 2, 4)
    expected = (gathered * inputs["query_barycentric"]).sum(dim=-1)
    assert torch.equal(output["final_prediction"], expected)


def test_shared_cell_applies_a_true_iterative_state_update() -> None:
    model = _model(n_iterations=3)
    inputs = _inputs()
    with torch.no_grad():
        model.cell.delta_head.bias.fill_(0.1)
    output = model(**inputs)
    assert torch.allclose(output["step_states"][:, 0], inputs["x_h"] + 0.1)
    assert torch.allclose(output["step_states"][:, 1], inputs["x_h"] + 0.2)
    assert torch.allclose(output["step_states"][:, 2], inputs["x_h"] + 0.3)
    system = model.measurement_residual.system_matrix
    expected_norms = []
    for offset in (0.0, 0.1, 0.2, 0.3):
        residual = inputs["measurement_b"] - (inputs["x_h"] + offset) @ system.t()
        expected_norms.append(residual.square().mean(dim=-1).sqrt())
    assert torch.allclose(output["measurement_residual_rms"], torch.stack(expected_norms, dim=1))


def test_residual_consistent_steps_are_monotone() -> None:
    model = _residual_consistent_model()
    inputs = _inputs()
    with torch.no_grad():
        model.cell.delta_head.weight.normal_(std=0.1)
        model.cell.delta_head.bias.fill_(0.2)
    output = model(**inputs)
    residuals = output["measurement_residual_rms"]
    assert torch.all(residuals[:, 1:] <= residuals[:, :-1] + 1e-6)
    assert torch.all(output["accepted_step_scale"] >= 0.0)
    assert torch.all(output["accepted_step_scale"] <= 1.0)


def test_rejected_trust_region_proposal_keeps_task_gradient() -> None:
    model = _residual_consistent_model(n_iterations=1)
    state = torch.randn(1, 8)
    measurement = torch.randn(1, 5)
    evidence = model.measurement_residual(state, measurement)
    # A sufficiently large ascent direction is rejected in favor of scale zero.
    direction = (-evidence["jacobi"].detach() * 1e4).requires_grad_()
    next_state, selected, _ = model._monotone_update(
        state,
        direction,
        evidence["forward_response"],
        measurement,
        evidence["forward_residual_rms"],
    )
    assert torch.equal(selected, torch.zeros_like(selected))
    assert torch.equal(next_state, state)
    next_state.sum().backward()
    assert torch.equal(direction.grad, torch.ones_like(direction))


def test_residual_consistent_forward_uses_analytic_p1_identity() -> None:
    model = _residual_consistent_model()
    inputs = _inputs()
    output = model(**inputs)
    corrected = output["corrected_nodes"]
    gathered = corrected.gather(1, inputs["query_node_indices"].reshape(2, -1)).reshape(2, 2, 4)
    expected = (gathered * inputs["query_barycentric"]).sum(dim=-1)
    assert torch.equal(output["final_prediction"], expected)
    assert "system_matrix" not in model.state_dict()


def test_model_has_no_voxel_completion_or_learned_transfer() -> None:
    model = _model()
    names = {name for name, _ in model.named_modules()}
    assert not any("representation" in name for name in names)
    assert not any("voxel" in name for name in names)
    assert not any("lift" in name for name in names)
    assert sum(parameter.numel() for parameter in model.measurement_residual.parameters()) == 0


def test_inverse_state_supervision_backpropagates_to_contextual_cell() -> None:
    model = _model()
    inputs = _inputs()
    target = torch.randn(2, 8)
    output = model(**inputs)
    loss = torch.nn.functional.mse_loss(output["corrected_nodes"], target)
    loss.backward()
    gradient = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.cell.parameters()
        if parameter.grad is not None
    )
    assert gradient > 0


def test_default_contextual_cell_capacity_is_lightweight() -> None:
    n_nodes = 8
    model = IterativeResidualAwareFEMCorrector(
        system_matrix=torch.randn(5, n_nodes),
        knn_indices=torch.zeros(n_nodes, 3, dtype=torch.long),
        hidden_dim=144,
        n_context_blocks=2,
        n_iterations=3,
        view_feat_dim=32,
    )
    parameters = sum(parameter.numel() for parameter in model.cell.parameters())
    assert 200_000 <= parameters <= 500_000


def test_frozen_ensemble_combines_fem_states_before_analytic_p1() -> None:
    first = _model(n_iterations=1)
    second = _model(n_iterations=1)
    with torch.no_grad():
        first.cell.delta_head.bias.fill_(0.2)
        second.cell.delta_head.bias.fill_(0.6)
    ensemble = FrozenConvexFEMCorrectorEnsemble(first, second, second_weight=0.55)
    inputs = _inputs()
    output = ensemble(**inputs)
    expected_nodes = inputs["x_h"] + 0.45 * 0.2 + 0.55 * 0.6
    assert torch.allclose(output["corrected_nodes"], expected_nodes)
    gathered = expected_nodes.gather(1, inputs["query_node_indices"].reshape(2, -1)).reshape(
        2, 2, 4
    )
    expected_prediction = (gathered * inputs["query_barycentric"]).sum(dim=-1)
    assert torch.allclose(output["final_prediction"], expected_prediction)
    assert not any(parameter.requires_grad for parameter in ensemble.parameters())
