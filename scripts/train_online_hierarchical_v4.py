#!/usr/bin/env python3
"""Train matched frozen-online and joint-cell hard-Q hierarchical models."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.evaluation.iterative_fem import _measurement_for_sample
from du2vox.models.stage2.complement_voxel_detail import ComplementConstrainedVoxelDetail
from scripts.train_complement_voxel_detail import FullDomainEngine
from scripts.train_iterative_fem_corrector import build_dataset, build_model, load_ids, seed_all


ARMS = ("frozen_online", "joint_v4_cell")


def configure_trainable_parameters(
    v4: torch.nn.Module,
    detail_model: ComplementConstrainedVoxelDetail,
    arm: str,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Freeze the hierarchy, then expose only the preregistered parameter groups."""

    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}")
    v4.requires_grad_(False)
    detail_model.requires_grad_(True)
    cell_parameters: list[torch.nn.Parameter] = []
    if arm == "joint_v4_cell":
        v4.cell.requires_grad_(True)
        cell_parameters = list(v4.cell.parameters())
    return list(detail_model.parameters()), cell_parameters


class OnlineHierarchicalEngine(FullDomainEngine):
    """Full canonical voxel engine driven by a live V4 FEM computation graph."""

    def __init__(self, cfg: dict[str, Any], device: torch.device, arm: str) -> None:
        super().__init__(cfg, device)
        self.arm = arm
        self.backbone_cfg = yaml.safe_load(Path(cfg["frozen_backbone"]["config"]).read_text())
        self.v4 = build_model(self.backbone_cfg).to(device)
        checkpoint = torch.load(
            cfg["frozen_backbone"]["checkpoint"], map_location=device, weights_only=False
        )
        if checkpoint.get("model_type") != "unified_dual_evidence_fem_v4":
            raise RuntimeError("Online backbone checkpoint is not unified V4")
        if checkpoint.get("epoch") != int(cfg["frozen_backbone"]["expected_epoch"]):
            raise RuntimeError("Online backbone checkpoint is not the declared epoch")
        self.v4.load_state_dict(checkpoint["model"])
        self.v4.eval().requires_grad_(False)
        # Use the exact checkpoint encoder once, shared by node and voxel sampling.
        self.encoder = self.v4.view_encoder
        if self.encoder is None:
            raise RuntimeError("V4 online hierarchy requires its frozen multiview encoder")
        self.encoder.eval().requires_grad_(False)
        self.backbone_data = self.backbone_cfg["data"]
        self.datasets = {
            split: build_dataset(
                self.backbone_cfg,
                split,
                load_ids(self.backbone_data[f"{split}_split"]),
                int(self.backbone_cfg["training"].get("n_query_points", 8192)),
            )
            for split in ("train", "val", "test")
        }
        reference = self.datasets["train"]
        self.node_coords_norm_online = torch.from_numpy(reference.node_coords_norm).unsqueeze(0).to(device)
        self.node_coords_world_online = torch.from_numpy(reference.node_coords_world).unsqueeze(0).to(device)

    def _stage1_and_measurement(self, sid: str, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        dataset = self.datasets[split]
        stage1 = torch.from_numpy(
            np.load(dataset.bridge_dir / sid / "coarse_d.npy").astype(np.float32)
        ).unsqueeze(0).to(self.device)
        measurement = torch.from_numpy(_measurement_for_sample(dataset, sid)).unsqueeze(0).to(self.device)
        return stage1, measurement

    def online_v4(
        self, sid: str, split: str
    ) -> tuple[torch.Tensor, torch.Tensor | dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        stage1, measurement = self._stage1_and_measurement(sid, split)
        encoded = self._encoded_views(sid)
        output = self.v4.correct_nodes(
            stage1,
            measurement,
            self.node_coords_norm_online,
            node_coords_world=self.node_coords_world_online,
            encoded_views=encoded,
        )
        return output["corrected_nodes"].squeeze(0), encoded, output

    def predict_online(
        self, model: ComplementConstrainedVoxelDetail, sid: str, split: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        state, encoded, v4_output = self.online_v4(sid, split)
        proposal, coarse, detail, final = self.predict_from_state(model, state, encoded)
        return proposal, coarse, detail, final, state, v4_output

    @torch.no_grad()
    def audit_cached_equivalence(self, sid: str, split: str) -> float:
        self.v4.eval()
        online, _, _ = self.online_v4(sid, split)
        cached = torch.from_numpy(
            np.load(Path(self.data["v4_states_root"]) / split / f"{sid}.npy").astype(np.float32)
        ).to(self.device)
        return float(torch.linalg.vector_norm(online - cached) / torch.linalg.vector_norm(cached).clamp_min(1e-30))


@torch.inference_mode()
def validate_online(
    engine: OnlineHierarchicalEngine,
    model: ComplementConstrainedVoxelDetail,
    ids: list[str],
) -> dict[str, float]:
    engine.v4.eval()
    model.eval()
    dice_values: list[float] = []
    leakage_values: list[float] = []
    threshold = float(engine.cfg["validation"]["threshold"])
    for index, sid in enumerate(ids):
        _, _, detail, final, _, _ = engine.predict_online(model, sid, "val")
        gt, _ = engine._targets(sid)
        predicted = final.squeeze(0) >= threshold
        truth = gt >= 0.5
        dice_values.append(float(2 * torch.sum(predicted & truth) / (torch.sum(predicted) + torch.sum(truth)).clamp_min(1)))
        detail_np = detail.squeeze(0).float().cpu().numpy()
        projected = engine.complement.project_numpy(detail_np)
        leakage_values.append(float(np.dot(projected, projected) / max(np.dot(detail_np, detail_np), 1e-30)))
        if (index + 1) % 25 == 0:
            print(f"[online val {index + 1}/{len(ids)}]", flush=True)
    return {"dice": float(np.mean(dice_values)), "coarse_leakage": float(np.mean(leakage_values))}


def gradient_contract(
    engine: OnlineHierarchicalEngine,
    detail_model: ComplementConstrainedVoxelDetail,
    arm: str,
) -> dict[str, float]:
    def grad_sum(module: torch.nn.Module) -> float:
        return float(sum(p.grad.detach().abs().sum() for p in module.parameters() if p.grad is not None))

    values = {
        "voxel_branch": grad_sum(detail_model),
        "v4_cell": grad_sum(engine.v4.cell),
        "v4_encoder": grad_sum(engine.v4.view_encoder),
        "measurement_operator": grad_sum(engine.v4.measurement_residual),
    }
    if values["voxel_branch"] <= 0:
        raise RuntimeError("Voxel branch received no gradient")
    if arm == "joint_v4_cell" and values["v4_cell"] <= 0:
        raise RuntimeError("Joint V4 cell received no gradient")
    if arm == "frozen_online" and values["v4_cell"] != 0:
        raise RuntimeError("Frozen V4 cell unexpectedly received a gradient")
    if values["v4_encoder"] != 0 or values["measurement_operator"] != 0:
        raise RuntimeError("A frozen encoder/operator unexpectedly received a gradient")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--experiment-name")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--audit-online-cached", action="store_true")
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    backbone_cfg = yaml.safe_load(Path(cfg["frozen_backbone"]["config"]).read_text())
    backbone_data = backbone_cfg["data"]
    os.environ["DU2VOX_SHARED_DIR"] = str(backbone_data["shared_dir"])
    if backbone_data.get("allow_stale_frame_manifest", False):
        os.environ["DU2VOX_ALLOW_STALE_FRAME_MANIFEST"] = "1"
    os.environ["DU2VOX_FRAME_MANIFEST_SHA256"] = str(backbone_data["frame_manifest_sha256"])
    seed_all(int(cfg["training"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = OnlineHierarchicalEngine(cfg, device, args.arm)
    train_ids = load_ids(cfg["data"]["train_split"])
    val_ids = load_ids(cfg["data"]["val_split"])
    if args.max_samples is not None:
        train_ids = train_ids[: args.max_samples]
    if args.max_val_samples is not None:
        val_ids = val_ids[: args.max_val_samples]
    if args.smoke:
        train_ids, val_ids = train_ids[:1], val_ids[:1]
    if args.audit_online_cached:
        rows = []
        for index, audit_id in enumerate(val_ids):
            error = engine.audit_cached_equivalence(audit_id, "val")
            rows.append({"sample_id": audit_id, "relative_l2": error})
            if error >= 1e-6:
                raise RuntimeError(f"Online V4 differs from cached state for {audit_id}")
            if (index + 1) % 25 == 0:
                print(f"[online-cache-contract {index + 1}/{len(val_ids)}]", flush=True)
        result = {
            "n_samples": len(rows),
            "maximum_relative_l2": max(row["relative_l2"] for row in rows),
            "tolerance": 1e-6,
            "status": "passed",
            "per_sample": rows,
            "confirmation_data_used": False,
        }
        if args.audit_output is not None:
            args.audit_output.parent.mkdir(parents=True, exist_ok=True)
            args.audit_output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key != "per_sample"}))
        return
    audit_id = val_ids[0]
    online_error = engine.audit_cached_equivalence(audit_id, "val")
    print(f"[online-cache-contract] sample={audit_id} relative_l2={online_error:.8e}")
    if online_error >= 1e-6:
        raise RuntimeError("Online V4 differs from the frozen cached state")

    # Re-seed after building/loading V4 so both arms get identical voxel initialization.
    seed_all(int(cfg["training"]["seed"]))
    detail_model = engine.build_model("hard")
    voxel_parameters, cell_parameters = configure_trainable_parameters(
        engine.v4, detail_model, args.arm
    )
    groups: list[dict[str, Any]] = [
        {"params": voxel_parameters, "lr": float(cfg["training"]["lr"]), "name": "voxel"}
    ]
    if cell_parameters:
        groups.append(
            {"params": cell_parameters, "lr": float(cfg["training"].get("v4_cell_lr", 1e-5)), "name": "v4_cell"}
        )
    optimizer = torch.optim.AdamW(groups, weight_decay=float(cfg["training"]["weight_decay"]))
    experiment = args.experiment_name or f"p0_{args.arm}_hard_q"
    run_dir = Path(cfg["runs_root"]) / experiment
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, run_dir / "config.yaml")
    epochs = args.max_epochs or int(cfg["training"]["epochs"])
    if args.smoke:
        epochs = 1
    best_dice = -np.inf
    stale = 0
    history: list[dict[str, Any]] = []
    gradient_checked = False
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, epochs + 1):
        detail_model.train()
        engine.v4.eval()
        if args.arm == "joint_v4_cell":
            engine.v4.cell.train()
        rng = np.random.default_rng(int(cfg["training"]["seed"]) + epoch)
        totals: list[float] = []
        started = time.perf_counter()
        for position, sample_index in enumerate(rng.permutation(len(train_ids))):
            sid = train_ids[int(sample_index)]
            gt, target = engine._targets(sid)
            optimizer.zero_grad(set_to_none=True)
            proposal, _, detail, final, _, _ = engine.predict_online(detail_model, sid, "train")
            loss, loss_details = engine.loss(
                detail_model, proposal, detail, final, gt, target
            )
            if not torch.isfinite(loss):
                finite = {
                    "proposal": bool(torch.isfinite(proposal).all()),
                    "detail": bool(torch.isfinite(detail).all()),
                    "final": bool(torch.isfinite(final).all()),
                }
                raise FloatingPointError(
                    f"Non-finite online hierarchical loss at epoch={epoch} "
                    f"position={position} sample={sid}; tensors={finite}; "
                    f"components={loss_details}"
                )
            loss.backward()
            if not gradient_checked:
                contract = gradient_contract(engine, detail_model, args.arm)
                print(f"[gradient-contract] {contract}", flush=True)
                gradient_checked = True
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                voxel_parameters + cell_parameters,
                float(cfg["training"]["grad_clip_norm"]),
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"Non-finite online hierarchical gradient at epoch={epoch} "
                    f"position={position} sample={sid}; loss={float(loss.detach())}; "
                    f"components={loss_details}"
                )
            optimizer.step()
            totals.append(float(loss.detach()))
            if (position + 1) % 25 == 0:
                print(f"[epoch {epoch}/{epochs} train {position + 1}/{len(train_ids)}] loss={np.mean(totals[-25:]):.6f}", flush=True)
        validation = validate_online(engine, detail_model, val_ids)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(totals)),
            "val_dice": validation["dice"],
            "val_coarse_leakage": validation["coarse_leakage"],
            "epoch_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        }
        history.append(row)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        payload = {
            "detail_model": detail_model.state_dict(),
            "v4_model": engine.v4.state_dict(),
            "arm": args.arm,
            "mode": "hard",
            "epoch": epoch,
            "dense_val_dice": validation["dice"],
            "online_cached_relative_l2": online_error,
            "config": cfg,
        }
        torch.save(payload, checkpoint_dir / "last.pth")
        if validation["dice"] > best_dice:
            best_dice = validation["dice"]
            stale = 0
            shutil.copy2(checkpoint_dir / "last.pth", checkpoint_dir / "best_dense_val_dice.pth")
        else:
            stale += 1
        print(json.dumps(row), flush=True)
        if stale >= int(cfg["training"]["early_stopping_patience"]):
            break


if __name__ == "__main__":
    main()
