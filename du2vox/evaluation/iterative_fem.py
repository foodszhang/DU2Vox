"""Dense common-domain evaluation for the iterative FEM inverse corrector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from du2vox.bridge.canonical_cross_discretization import CanonicalCrossDiscretization
from du2vox.data.error_structured_dataset import ErrorStructuredFixedDomainDataset
from du2vox.evaluation.error_structured import (
    _reconstruction_metrics,
    cosine,
    relative_errors,
)
from du2vox.evaluation.continuous_field import continuous_metrics
from du2vox.models.stage2.iterative_fem_corrector import (
    IterativeResidualAwareFEMCorrector,
)
from du2vox.models.stage2.stage2_dataset import load_projection_stack
from experiments.cross_discretization_decomposition.run_analysis import focus_metadata


def _measurement_for_sample(dataset: ErrorStructuredFixedDomainDataset, sid: str) -> np.ndarray:
    measurement, _ = dataset.load_measurement_with_operator_scale(sid)
    return measurement


@torch.inference_mode()
def evaluate_iterative_fem_dense(
    model: IterativeResidualAwareFEMCorrector,
    dataset: ErrorStructuredFixedDomainDataset,
    canonical: CanonicalCrossDiscretization,
    device: torch.device,
    *,
    max_samples: int | None = None,
    save_predictions_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate Stage 1 and every corrected FEM iteration on all valid centers."""

    model.eval()
    save_dir = Path(save_predictions_dir) if save_predictions_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    n_samples = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    rows: list[dict[str, Any]] = []
    p = canonical.p
    for index in range(n_samples):
        sid = dataset.sample_ids[index]
        x_h_np = np.load(dataset.bridge_dir / sid / "coarse_d.npy").astype(np.float32)
        pi_gt_np = np.load(dataset.projection_targets_dir / f"{sid}.npy").astype(np.float32)
        gt, _ = dataset.load_gt(sid)
        measurement_np, operator_scale = dataset.load_measurement_with_operator_scale(sid)
        x_h = torch.from_numpy(x_h_np).unsqueeze(0).to(device)
        measurement = torch.from_numpy(measurement_np).unsqueeze(0).to(device)
        measurement_operator_scale = torch.tensor(
            [operator_scale], device=device, dtype=torch.float32
        )
        coords_norm = torch.from_numpy(dataset.node_coords_norm).unsqueeze(0).to(device)
        coords_world = torch.from_numpy(dataset.node_coords_world).unsqueeze(0).to(device)
        encoded = None
        if model.view_encoder is not None:
            projections, _ = load_projection_stack(
                dataset.samples_dir / sid,
                projection_file=dataset.projection_file,
                projection_norm=dataset.projection_norm,
                projection_transform=dataset.projection_transform,
            )
            proj_imgs = torch.from_numpy(projections[:, None]).unsqueeze(0).to(device)
            encoded = model.view_encoder.encode_images(proj_imgs)
        output = model.correct_nodes(
            x_h,
            measurement,
            coords_norm,
            node_coords_world=coords_world,
            encoded_views=encoded,
            measurement_operator_scale=measurement_operator_scale,
        )
        step_states = output["step_states"].squeeze(0).float().cpu().numpy()
        residual_rms = output["measurement_residual_rms"].squeeze(0).float().cpu().numpy()
        stage1_voxel = np.asarray(p @ x_h_np, dtype=np.float64).ravel()
        step_voxels = [np.asarray(p @ state, dtype=np.float64).ravel() for state in step_states]
        tumor = focus_metadata(dataset.samples_dir / sid)["tumor_params"]
        row: dict[str, Any] = {"sample_id": sid}
        stage1_metrics = _reconstruction_metrics(stage1_voxel, gt, dataset, tumor)
        row.update({f"stage1_{key}": value for key, value in stage1_metrics.items()})
        row.update(
            {
                f"stage1_field_{key}": value
                for key, value in continuous_metrics(
                    stage1_voxel, gt, data_range=dataset.metric_data_range
                ).items()
            }
        )
        previous_dice = stage1_metrics["dice"]
        for step, prediction in enumerate(step_voxels, start=1):
            metrics = _reconstruction_metrics(prediction, gt, dataset, tumor)
            row.update({f"step{step}_{key}": value for key, value in metrics.items()})
            row.update(
                {
                    f"step{step}_field_{key}": value
                    for key, value in continuous_metrics(
                        prediction, gt, data_range=dataset.metric_data_range
                    ).items()
                }
            )
            row[f"step{step}_delta_dice"] = metrics["dice"] - previous_dice
            previous_dice = metrics["dice"]
        total_correction = step_states[-1] - x_h_np
        inverse_target = pi_gt_np - x_h_np
        rel_l1, rel_l2 = relative_errors(total_correction, inverse_target)
        row.update(
            {
                "inverse_target_cosine": cosine(total_correction, inverse_target),
                "inverse_relative_l1": rel_l1,
                "inverse_relative_l2": rel_l2,
                "initial_measurement_residual_rms": float(residual_rms[0]),
                "final_measurement_residual_rms": float(residual_rms[-1]),
                "measurement_residual_ratio": float(residual_rms[-1] / max(residual_rms[0], 1e-30)),
            }
        )
        if "accepted_step_scale" in output:
            accepted = output["accepted_step_scale"].squeeze(0).float().cpu().numpy()
            for step, scale in enumerate(accepted, start=1):
                row[f"step{step}_accepted_scale"] = float(scale)
        if "profiled_amplitude" in output:
            amplitudes = output["profiled_amplitude"].squeeze(0).float().cpu().numpy()
            for step, amplitude in enumerate(amplitudes, start=1):
                row[f"step{step}_profiled_amplitude"] = float(amplitude)
        diagnostic_outputs = {
            "raw_residual_rms": "raw_residual_rms",
            "si_residual_rms": "si_residual_rms",
            "raw_adjoint_rms": "raw_adjoint_rms",
            "si_adjoint_rms": "si_adjoint_rms",
            "relative_forward_rms": "relative_forward_rms",
            "raw_si_adjoint_cosine": "raw_si_adjoint_cosine",
            "update_rms": "update_rms",
            "state_rms": "state_rms",
            "dc_rms": "dc_rms",
            "si_fusion_weight": "si_fusion_weight",
        }
        for output_key, row_suffix in diagnostic_outputs.items():
            if output_key not in output:
                continue
            values = output[output_key].squeeze(0).float().cpu().numpy()
            for step, value in enumerate(values[: len(step_states)], start=1):
                row[f"step{step}_{row_suffix}"] = float(value)
        if save_dir is not None:
            arrays: dict[str, np.ndarray] = {
                "stage1_fem_nodes": x_h_np,
                "stage1_fem": stage1_voxel.astype(np.float32),
                "final_prediction": step_voxels[-1].astype(np.float32),
                "valid_flat_indices": dataset.valid_flat_indices,
                "grid_shape": np.asarray(dataset.grid_shape),
            }
            for step, (state, prediction) in enumerate(
                zip(step_states, step_voxels, strict=True), start=1
            ):
                arrays[f"step{step}_fem_nodes"] = state.astype(np.float32)
                arrays[f"step{step}_fem"] = prediction.astype(np.float32)
            np.savez_compressed(save_dir / f"{sid}.npz", **arrays)
        rows.append(row)
        print(
            f"[dense {index + 1}/{n_samples}] {sid} "
            f"stage1={row['stage1_dice']:.4f} "
            f"final={row[f'step{len(step_voxels)}_dice']:.4f}",
            flush=True,
        )
    numeric_keys = sorted(
        key
        for key, value in rows[0].items()
        if key != "sample_id" and isinstance(value, (int, float, np.floating))
    )
    summary = {key: float(np.nanmean([float(row[key]) for row in rows])) for key in numeric_keys}
    return {
        "n_samples": n_samples,
        "measurement_operator_scaling": dataset.measurement_operator_scaling,
        "summary": summary,
        "per_sample": rows,
    }


def save_iterative_fem_result(result: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
