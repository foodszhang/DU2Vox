#!/usr/bin/env python3
"""Train one matched hard-Q A0--A3 information-source audit arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.models.stage2.complement_voxel_detail import (  # noqa: E402
    ComplementConstrainedVoxelDetail,
)
from du2vox.models.stage2.information_audit import (  # noqa: E402
    InformationFeatureLayout,
    assemble_information_features,
    get_information_arm,
    select_coarse_state,
)
from scripts.train_complement_voxel_detail import (  # noqa: E402
    FullDomainEngine,
    seed_all,
)
from scripts.train_iterative_fem_corrector import load_ids  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_initialization_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_audit_config(path: Path) -> dict[str, Any]:
    """Load an arm config and its optional shared, immutable audit base."""

    arm_cfg = yaml.safe_load(path.read_text())
    base_path = arm_cfg.pop("base_config", None)
    if base_path is None:
        return arm_cfg
    candidate = Path(base_path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    base = yaml.safe_load(candidate.read_text())

    def merge(target: dict[str, Any], source: dict[str, Any]) -> None:
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = value

    merge(base, arm_cfg)
    return base


class InformationAuditEngine(FullDomainEngine):
    """Full-domain engine with fixed G/S/L/V slots for all eight arms."""

    def __init__(self, cfg: dict[str, Any], device: torch.device) -> None:
        self.arm = get_information_arm(cfg["audit"]["arm"])
        super().__init__(cfg, device)
        self.layout = InformationFeatureLayout(
            coordinate_dim=self.pe.out_dim,
            barycentric_dim=4,
            state_dim=12,
            latent_dim=int(cfg["model"]["latent_dim"]),
            view_dim=int(cfg["model"]["view_feat_dim"]),
        )
        self.input_dim = self.layout.input_dim
        self.latent_root = Path(self.data["v4_latents_root"])
        self._latent_manifests: dict[str, dict[str, Any]] = {}
        expected = int(cfg["model"].get("matched_input_dim", self.input_dim))
        if self.input_dim != expected:
            raise RuntimeError(f"Matched input dimension {self.input_dim} != {expected}")

    def _latent_record(self, split: str, sid: str) -> dict[str, Any]:
        if split not in self._latent_manifests:
            manifest_path = self.latent_root / split / "cache_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            backbone = self.cfg["frozen_backbone"]
            shared_dir = Path(self.data["shared_dir"])
            expected = {
                "v4_config_sha256": file_sha256(Path(backbone["config"])),
                "v4_checkpoint_sha256": file_sha256(Path(backbone["checkpoint"])),
                "mesh_sha256": file_sha256(shared_dir / "mesh.npz"),
                "frame_manifest_sha256": file_sha256(
                    shared_dir / "frame_manifest.json"
                ),
                "generation_seed": int(self.cfg["training"]["seed"]),
                "v4_checkpoint_epoch": int(backbone["expected_epoch"]),
                "terminal_hidden_shape": [
                    self.complement.n_fem_nodes,
                    self.layout.latent_dim,
                ],
                "terminal_hidden_dtype": "float16",
            }
            for key, value in expected.items():
                if manifest.get(key) != value:
                    raise RuntimeError(
                        f"Latent cache manifest {key}={manifest.get(key)!r}, expected {value!r}"
                    )
            audit = manifest.get("fp16_audit", {})
            if not audit.get("passed") or not manifest.get(
                "latent_export_state_identity"
            ):
                raise RuntimeError("Latent FP16/state-identity cache gate did not pass")
            if audit.get("completed_samples") != audit.get("requested_samples"):
                raise RuntimeError("Canonical latent cache did not complete its split audit quota")
            manifest["sample_records"] = {
                row["sample_id"]: row for row in manifest["samples"]
            }
            self._latent_manifests[split] = manifest
        records = self._latent_manifests[split]["sample_records"]
        if sid not in records:
            raise RuntimeError(f"Sample {sid} is absent from the {split} latent manifest")
        return records[sid]

    def build_model(self, mode: str | bool = "hard") -> ComplementConstrainedVoxelDetail:
        if str(mode).lower() not in {"hard", "constrained", "true"}:
            raise ValueError("Information audit permits exact hard-Q only")
        return ComplementConstrainedVoxelDetail(
            input_dim=self.input_dim,
            complement=self.complement,
            hidden_dim=int(self.cfg["model"]["hidden_dim"]),
            n_hidden_layers=int(self.cfg["model"]["hidden_layers"]),
            mode="hard",
        ).to(self.device)

    def _state_features(
        self, state: torch.Tensor, start: int, end: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self.node_indices[start:end]
        bary = self.barycentric[start:end]
        local = state[ids]
        coarse = torch.sum(local * bary, dim=-1)
        gradient = torch.sum(
            local[:, :, None] * self.gradient_weights[start:end], dim=1
        )
        stats = torch.stack(
            [
                local.mean(-1),
                local.std(-1, unbiased=False),
                local.amax(-1) - local.amin(-1),
            ],
            dim=-1,
        )
        boundary = torch.exp(
            -torch.abs(coarse - 0.5) / self.boundary_temperature
        ).unsqueeze(-1)
        return coarse, torch.cat([coarse[:, None], gradient, local, stats, boundary], -1)

    def predict(
        self, model: ComplementConstrainedVoxelDetail, sid: str, split: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        frozen_state = torch.from_numpy(
            np.load(Path(self.data["v4_states_root"]) / split / f"{sid}.npy").astype(
                np.float32
            )
        ).to(self.device)
        oracle_state = None
        if self.arm.oracle_state:
            oracle_state = torch.from_numpy(
                np.load(Path(self.data["projection_targets_dir"]) / f"{sid}.npy").astype(
                    np.float32
                )
            ).to(self.device)
        state = select_coarse_state(
            self.arm, frozen_state=frozen_state, oracle_state=oracle_state
        )
        terminal = None
        if self.arm.use_latent:
            record = self._latent_record(split, sid)
            latent_path = self.latent_root / split / f"{sid}.npy"
            latent_array = np.load(latent_path)
            if list(latent_array.shape) != record["terminal_hidden_shape"]:
                raise RuntimeError(f"Latent shape does not match manifest for {sid}")
            if str(latent_array.dtype) != record["terminal_hidden_dtype"]:
                raise RuntimeError(f"Latent dtype does not match manifest for {sid}")
            if not np.isfinite(latent_array).all():
                raise RuntimeError(f"Latent contains non-finite values for {sid}")
            terminal = torch.from_numpy(latent_array.astype(np.float32)).to(self.device)
        encoded = self._encoded_views(sid) if self.arm.use_views else None
        return self.predict_from_sources(model, state, terminal, encoded)

    def predict_from_sources(
        self,
        model: ComplementConstrainedVoxelDetail,
        coarse_state: torch.Tensor,
        terminal_hidden: torch.Tensor | None,
        encoded_views: torch.Tensor | dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        proposals: list[torch.Tensor] = []
        coarse_chunks: list[torch.Tensor] = []
        amp = self.device.type == "cuda"
        for start in range(0, self.complement.n_voxels, self.chunk_size):
            end = min(start + self.chunk_size, self.complement.n_voxels)
            ids = self.node_indices[start:end]
            bary = self.barycentric[start:end]
            coarse, state_features = self._state_features(coarse_state, start, end)
            latent = None
            if terminal_hidden is not None:
                latent = torch.sum(
                    terminal_hidden[ids] * bary[:, :, None], dim=1
                )
            views = None
            if encoded_views is not None:
                with torch.no_grad():
                    sampled, _ = self.encoder.sample_encoded(
                        encoded_views,
                        self.coords_world[start:end][None],
                        coords_vox_norm=None,
                    )
                views = sampled.squeeze(0)
            with torch.autocast(
                device_type=self.device.type, dtype=torch.bfloat16, enabled=amp
            ):
                features = assemble_information_features(
                    self.arm,
                    coordinate_pe=self.pe(self.coords_norm[start:end]),
                    barycentric=bary,
                    state=state_features,
                    latent=latent,
                    views=views,
                    layout=self.layout,
                )
                proposal = (
                    model.proposal_checkpointed(features)
                    if model.training
                    else model.proposal(features)
                )
            proposals.append(proposal)
            coarse_chunks.append(coarse)
        proposal_full = torch.cat(proposals).unsqueeze(0)
        coarse_full = torch.cat(coarse_chunks).unsqueeze(0)
        detail = model.constrain(proposal_full)
        return proposal_full, coarse_full, detail, coarse_full + detail


@torch.inference_mode()
def validate(
    engine: InformationAuditEngine,
    model: ComplementConstrainedVoxelDetail,
    ids: list[str],
) -> dict[str, float]:
    model.eval()
    values = []
    threshold = float(engine.cfg["validation"]["threshold"])
    for index, sid in enumerate(ids):
        _, _, _, final = engine.predict(model, sid, "val")
        gt, _ = engine._targets(sid)
        prediction = final.squeeze(0) >= threshold
        truth = gt >= 0.5
        values.append(
            float(
                2
                * torch.sum(prediction & truth)
                / (torch.sum(prediction) + torch.sum(truth)).clamp_min(1)
            )
        )
        if (index + 1) % 50 == 0:
            print(f"[{engine.arm.name} val {index + 1}/{len(ids)}]", flush=True)
    return {"dice": float(np.mean(values))}


def memory_smoke(
    engine: InformationAuditEngine,
    model: ComplementConstrainedVoxelDetail,
    sid: str,
    output: Path,
) -> None:
    model.train()
    if engine.device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(engine.device)
    gt, target = engine._targets(sid)
    proposal, _, detail, final = engine.predict(model, sid, "train")
    loss, _ = engine.loss(model, proposal, detail, final, gt, target)
    loss.backward()
    if engine.device.type == "cuda":
        torch.cuda.synchronize(engine.device)
        peak_bytes = torch.cuda.max_memory_reserved(engine.device)
        source = "cuda_max_memory_reserved"
    else:
        peak_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        source = "process_max_rss"
    limit_bytes = int(float(engine.cfg["audit"]["maximum_peak_gib"]) * 1024**3)
    result = {
        "arm": engine.arm.name,
        "sample_id": sid,
        "full_domain_voxels": engine.complement.n_voxels,
        "activation_checkpointing": True,
        "measurement": source,
        "peak_bytes": int(peak_bytes),
        "peak_gib": peak_bytes / 1024**3,
        "limit_gib": limit_bytes / 1024**3,
        "passed": peak_bytes <= limit_bytes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    if peak_bytes > limit_bytes:
        raise RuntimeError(
            f"Full-resolution smoke peak {result['peak_gib']:.3f} GiB exceeds "
            f"{result['limit_gib']:.3f} GiB; canonical domain will not be reduced"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-name")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    cfg = load_audit_config(args.config)
    backbone = yaml.safe_load(Path(cfg["frozen_backbone"]["config"]).read_text())
    os.environ["DU2VOX_SHARED_DIR"] = str(backbone["data"]["shared_dir"])
    if backbone["data"].get("allow_stale_frame_manifest", False):
        os.environ["DU2VOX_ALLOW_STALE_FRAME_MANIFEST"] = "1"
    os.environ["DU2VOX_FRAME_MANIFEST_SHA256"] = str(
        backbone["data"]["frame_manifest_sha256"]
    )
    seed = int(cfg["training"]["seed"])
    if seed != 20260901 or int(cfg["training"]["epochs"]) != 5:
        raise RuntimeError("Canonical audit requires seed 20260901 and exactly 5 epochs")
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = InformationAuditEngine(cfg, device)
    train_ids = load_ids(cfg["data"]["train_split"])
    val_ids = load_ids(cfg["data"]["val_split"])
    if args.max_samples is None and (len(train_ids), len(val_ids)) != (2400, 300):
        raise RuntimeError(
            f"D0 split sizes are train={len(train_ids)}, val={len(val_ids)}; "
            "expected 2400/300"
        )
    if args.max_samples is not None:
        train_ids = train_ids[: args.max_samples]
        val_ids = val_ids[: min(args.max_samples, len(val_ids))]
    experiment = args.experiment_name or cfg["experiment"]["name"]
    run_dir = Path(cfg["runs_root"]) / experiment
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, run_dir / "config.yaml")
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False)
    )
    model = engine.build_model("hard")
    memory_smoke(engine, model, train_ids[0], run_dir / "memory_smoke.json")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    seed_all(seed)
    model = engine.build_model("hard")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    initialization_sha256 = model_initialization_sha256(model)
    order_contract_sha256 = hashlib.sha256(
        ("\n".join(train_ids) + "\n").encode("utf-8")
    ).hexdigest()
    (run_dir / "matched_contract.json").write_text(
        json.dumps(
            {
                "arm": engine.arm.name,
                "input_dim": engine.input_dim,
                "parameter_count": parameter_count,
                "initialization_sha256": initialization_sha256,
                "seed": seed,
                "train_split_order_sha256": order_contract_sha256,
                "epoch_order_rule": "numpy.default_rng(seed + epoch).permutation",
                "hard_q": True,
                "activation_checkpointing": True,
            },
            indent=2,
        )
        + "\n"
    )
    epochs = args.max_epochs or int(cfg["training"]["epochs"])
    best_dice = -np.inf
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(len(train_ids))
        totals = []
        started = time.perf_counter()
        for position, sample_index in enumerate(order):
            sid = train_ids[int(sample_index)]
            gt, target = engine._targets(sid)
            optimizer.zero_grad(set_to_none=True)
            proposal, _, detail, final = engine.predict(model, sid, "train")
            loss, _ = engine.loss(model, proposal, detail, final, gt, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(cfg["training"]["grad_clip_norm"])
            )
            optimizer.step()
            totals.append(float(loss.detach()))
            if (position + 1) % 50 == 0:
                print(
                    f"[{engine.arm.name} epoch {epoch}/{epochs} "
                    f"{position + 1}/{len(train_ids)}] loss={np.mean(totals[-50:]):.6f}",
                    flush=True,
                )
        validation = validate(engine, model, val_ids)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(totals)),
            "val_final_dice": validation["dice"],
            "epoch_seconds": time.perf_counter() - started,
        }
        history.append(row)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        payload = {
            "model": model.state_dict(),
            "mode": "hard",
            "audit_arm": engine.arm.name,
            "epoch": epoch,
            "dense_val_dice": validation["dice"],
            "input_dim": engine.input_dim,
            "parameter_count": parameter_count,
            "initialization_sha256": initialization_sha256,
            "train_split_order_sha256": order_contract_sha256,
            "config": cfg,
        }
        torch.save(payload, checkpoint_dir / "last.pth")
        if validation["dice"] > best_dice:
            best_dice = validation["dice"]
            shutil.copy2(
                checkpoint_dir / "last.pth",
                checkpoint_dir / "best_dense_val_dice.pth",
            )
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
