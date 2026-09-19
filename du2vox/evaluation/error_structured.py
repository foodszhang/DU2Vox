"""Dense fixed-domain evaluation and semantic diagnostics for ESCB."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from du2vox.bridge.canonical_cross_discretization import (
    CanonicalCrossDiscretization,
    canonical_p1_torch,
)
from du2vox.data.error_structured_dataset import (
    ErrorStructuredFixedDomainDataset,
)
from du2vox.models.stage2.error_structured_bridge import (
    ErrorStructuredCrossDiscretizationBridge,
    PlainTargetFirstVoxelResidual,
)
from experiments.cross_discretization_decomposition.decomposition import (
    binary_metrics,
    component_metrics,
    surface_metrics,
)
from experiments.cross_discretization_decomposition.run_analysis import (
    focus_metadata,
    weak_recall,
)


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-30) -> float:
    a64 = np.asarray(a, dtype=np.float64).ravel()
    b64 = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a64, b64) / max(np.linalg.norm(a64) * np.linalg.norm(b64), eps))


def relative_errors(
    predicted: np.ndarray, target: np.ndarray, eps: float = 1e-30
) -> tuple[float, float]:
    error = np.asarray(predicted, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    rel_l1 = float(np.abs(error).sum() / max(np.abs(target64).sum(), eps))
    rel_l2 = float(np.linalg.norm(error) / max(np.linalg.norm(target64), eps))
    return rel_l1, rel_l2


def coarse_space_leakage(
    canonical: CanonicalCrossDiscretization,
    values: np.ndarray,
    eps: float = 1e-30,
) -> float:
    coefficients = canonical.project_coefficients(values)
    coarse = canonical.prolong(coefficients)
    values64 = np.asarray(values, dtype=np.float64)
    return float(np.dot(coarse, coarse) / (np.dot(values64, values64) + eps))


def _to_device(item: dict[str, Any], key: str, device: torch.device) -> torch.Tensor:
    value = item[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{key} is not a tensor")
    return value.unsqueeze(0).to(device, non_blocking=True)


def _reconstruction_metrics(
    prediction: np.ndarray,
    gt: np.ndarray,
    dataset: ErrorStructuredFixedDomainDataset,
    tumor: dict,
) -> dict[str, float]:
    metrics = binary_metrics(prediction, gt)
    metrics.update(
        surface_metrics(
            prediction,
            gt,
            dataset.valid_flat_indices,
            dataset.grid_shape,
            0.2,
        )
    )
    metrics.update(
        component_metrics(
            prediction,
            gt,
            dataset.valid_flat_indices,
            dataset.grid_shape,
            0.2,
        )
    )
    metrics["weak_recall"] = weak_recall(prediction, gt, dataset.coords_world, tumor)
    return metrics


@torch.inference_mode()
def evaluate_dense(
    model: torch.nn.Module,
    dataset: ErrorStructuredFixedDomainDataset,
    canonical: CanonicalCrossDiscretization,
    device: torch.device,
    *,
    batch_points: int = 32768,
    phase_a_only: bool = False,
    save_predictions_dir: str | Path | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Evaluate every canonical valid voxel center; never sampled-query validation."""

    model.eval()
    save_dir = Path(save_predictions_dir) if save_predictions_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    n_samples = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    p = canonical.p
    for sample_index in range(n_samples):
        sid = dataset.sample_ids[sample_index]
        x_h_np = np.load(dataset.bridge_dir / sid / "coarse_d.npy").astype(np.float32)
        pi_gt_np = np.load(dataset.projection_targets_dir / f"{sid}.npy").astype(np.float32)
        gt, _ = dataset.load_gt(sid)
        x_h = torch.from_numpy(x_h_np).unsqueeze(0).to(device)
        node_coords_norm = torch.from_numpy(dataset.node_coords_norm).unsqueeze(0).to(device)
        node_coords_world = torch.from_numpy(dataset.node_coords_world).unsqueeze(0).to(device)
        encoded = None
        if getattr(model, "view_encoder", None) is not None:
            projections, _ = load_projection_stack_for_eval(dataset, sid)
            proj_imgs = torch.from_numpy(projections[:, None]).unsqueeze(0).to(device)
            encoded = model.view_encoder.encode_images(proj_imgs)

        if isinstance(model, ErrorStructuredCrossDiscretizationBridge):
            node_view = None
            if encoded is not None:
                node_view, _ = model.view_encoder.sample_encoded(
                    encoded, node_coords_world, coords_vox_norm=None
                )
            delta_x = model.inverse_net(x_h, node_coords_norm, node_view)
            corrected_x = x_h + delta_x
            delta_x_np = delta_x.squeeze(0).float().cpu().numpy()
            corrected_x_np = corrected_x.squeeze(0).float().cpu().numpy()
        elif isinstance(model, PlainTargetFirstVoxelResidual):
            delta_x_np = np.zeros_like(x_h_np)
            corrected_x_np = x_h_np
        else:
            raise TypeError(f"Unsupported dense model: {type(model).__name__}")

        original_voxel = np.asarray(p @ x_h_np, dtype=np.float64).ravel()
        corrected_voxel = np.asarray(p @ corrected_x_np, dtype=np.float64).ravel()
        pi_gt_voxel = np.asarray(p @ pi_gt_np, dtype=np.float64).ravel()
        inverse_target_nodes = pi_gt_np.astype(np.float64) - x_h_np
        inverse_target_voxel = pi_gt_voxel - original_voxel
        representation_target = gt.astype(np.float64) - pi_gt_voxel

        representation_prediction = np.zeros(len(gt), dtype=np.float32)
        if not phase_a_only:
            for start in range(0, len(gt), batch_points):
                end = min(start + batch_points, len(gt))
                coords_norm = (
                    torch.from_numpy(dataset.coords_norm[start:end]).unsqueeze(0).to(device)
                )
                coords_world = (
                    torch.from_numpy(dataset.coords_world[start:end]).unsqueeze(0).to(device)
                )
                node_indices = (
                    torch.from_numpy(dataset.query_node_indices[start:end]).unsqueeze(0).to(device)
                )
                barycentric = (
                    torch.from_numpy(dataset.query_barycentric[start:end]).unsqueeze(0).to(device)
                )
                query_view = None
                if encoded is not None:
                    query_view, _ = model.view_encoder.sample_encoded(
                        encoded, coords_world, coords_vox_norm=None
                    )
                if isinstance(model, ErrorStructuredCrossDiscretizationBridge):
                    original_p1 = canonical_p1_torch(x_h, node_indices, barycentric)
                    corrected_p1 = canonical_p1_torch(corrected_x, node_indices, barycentric)
                    original_nodes = model._gather_nodes(x_h, node_indices)
                    corrected_nodes = model._gather_nodes(corrected_x, node_indices)
                    predicted = model.representation_net(
                        coords_norm,
                        original_nodes,
                        corrected_nodes,
                        barycentric,
                        original_p1,
                        corrected_p1,
                        query_view,
                    )
                else:
                    original_p1 = canonical_p1_torch(x_h, node_indices, barycentric)
                    original_nodes = ErrorStructuredCrossDiscretizationBridge._gather_nodes(
                        x_h, node_indices
                    )
                    predicted = model.residual_net(
                        coords_norm,
                        original_nodes,
                        original_nodes,
                        barycentric,
                        original_p1,
                        original_p1,
                        query_view,
                    )
                representation_prediction[start:end] = predicted.squeeze(0).float().cpu().numpy()

        final_prediction = corrected_voxel + representation_prediction
        tumor = focus_metadata(dataset.samples_dir / sid)["tumor_params"]
        stage1_metrics = _reconstruction_metrics(original_voxel, gt, dataset, tumor)
        corrected_metrics = _reconstruction_metrics(corrected_voxel, gt, dataset, tumor)
        final_metrics = _reconstruction_metrics(final_prediction, gt, dataset, tumor)
        row: dict[str, Any] = {"sample_id": sid}
        for prefix, metrics in (
            ("fem", stage1_metrics),
            ("corrected", corrected_metrics),
            ("final", final_metrics),
        ):
            row.update({f"{prefix}_{key}": value for key, value in metrics.items()})
        row["delta_fem_correction_dice"] = corrected_metrics["dice"] - stage1_metrics["dice"]
        row["delta_repr_completion_dice"] = final_metrics["dice"] - corrected_metrics["dice"]

        if isinstance(model, ErrorStructuredCrossDiscretizationBridge):
            inv_l1, inv_l2 = relative_errors(delta_x_np, inverse_target_nodes)
            repr_l1, repr_l2 = relative_errors(representation_prediction, representation_target)
            inv_voxel_prediction = corrected_voxel - original_voxel
            row.update(
                {
                    "inverse_target_cosine": cosine(delta_x_np, inverse_target_nodes),
                    "inverse_relative_l1": inv_l1,
                    "inverse_relative_l2": inv_l2,
                    "representation_target_cosine": cosine(
                        representation_prediction, representation_target
                    ),
                    "representation_relative_l1": repr_l1,
                    "representation_relative_l2": repr_l2,
                    "representation_leakage": coarse_space_leakage(
                        canonical, representation_prediction
                    ),
                    "representation_target_leakage": coarse_space_leakage(
                        canonical, representation_target
                    ),
                    "cross_inv_pred_inv_target": cosine(inv_voxel_prediction, inverse_target_voxel),
                    "cross_inv_pred_repr_target": cosine(
                        inv_voxel_prediction, representation_target
                    ),
                    "cross_repr_pred_inv_target": cosine(
                        representation_prediction, inverse_target_voxel
                    ),
                    "cross_repr_pred_repr_target": cosine(
                        representation_prediction, representation_target
                    ),
                }
            )
        if save_dir:
            np.savez_compressed(
                save_dir / f"{sid}.npz",
                stage1_fem_nodes=x_h_np,
                stage1_fem=original_voxel.astype(np.float32),
                corrected_fem_before_repr=corrected_voxel.astype(np.float32),
                representation_prediction=representation_prediction,
                final_prediction=final_prediction.astype(np.float32),
                valid_flat_indices=dataset.valid_flat_indices,
                grid_shape=np.asarray(dataset.grid_shape),
            )
        rows.append(row)
        print(
            f"[dense {sample_index + 1}/{n_samples}] {sid} "
            f"fem={row['fem_dice']:.4f} corrected={row['corrected_dice']:.4f} "
            f"final={row['final_dice']:.4f}",
            flush=True,
        )

    numeric_keys = sorted(
        key
        for key in rows[0]
        if key != "sample_id" and isinstance(rows[0][key], (int, float, np.floating))
    )
    summary = {key: float(np.nanmean([float(row[key]) for row in rows])) for key in numeric_keys}
    return {"n_samples": n_samples, "summary": summary, "per_sample": rows}


def load_projection_stack_for_eval(
    dataset: ErrorStructuredFixedDomainDataset, sid: str
) -> tuple[np.ndarray, str]:
    from du2vox.models.stage2.stage2_dataset import load_projection_stack

    return load_projection_stack(
        dataset.samples_dir / sid,
        projection_file=dataset.projection_file,
        projection_norm=dataset.projection_norm,
        projection_transform=dataset.projection_transform,
    )


def save_dense_result(result: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
