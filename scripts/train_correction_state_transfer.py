#!/usr/bin/env python3
"""Train full-domain Correction-State Cross-Space Transfer models."""

from __future__ import annotations

import argparse
import hashlib
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

from du2vox.models.stage2.correction_state_transfer import CorrectionStateTransfer  # noqa: E402
from scripts.train_complement_voxel_detail import FullDomainEngine, seed_all  # noqa: E402
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
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    base_path = config.pop("base_config", None)
    if base_path is None:
        return config
    base = yaml.safe_load((Path.cwd() / base_path).read_text())

    def merge(target: dict[str, Any], source: dict[str, Any]) -> None:
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = value

    merge(base, config)
    return base


class CSTEngine(FullDomainEngine):
    """Dense fixed-operator engine using cached V4 states and hidden transition."""

    state_dim = 28

    def __init__(self, cfg: dict[str, Any], device: torch.device) -> None:
        super().__init__(cfg, device)
        self.initial_hidden_root = Path(cfg["data"]["initial_hidden_root"])
        self.terminal_hidden_root = Path(cfg["data"]["terminal_hidden_root"])
        self.latent_dim = int(cfg["model"]["latent_dim"])
        self.use_one_ring = int(cfg["model"].get("one_ring_context_dim", 0)) > 0
        if self.use_one_ring:
            self._load_element_topology()
        self._audit_latent_manifests()

    def _load_element_topology(self) -> None:
        cache_root = Path("precomputed/final_dual_space/topology")
        cache_root.mkdir(parents=True, exist_ok=True)
        row_path = cache_root / "canonical_voxel_element_indices.npy"
        neighbor_path = cache_root / "tetra_face_neighbors.npy"
        elements_np = np.load(Path(self.data["shared_dir"]) / "mesh.npz")["elements"].astype(
            np.int64
        )
        if row_path.exists() and neighbor_path.exists():
            row_elements = np.load(row_path)
            face_neighbors = np.load(neighbor_path)
        else:
            element_lookup = {
                tuple(sorted(map(int, nodes))): index
                for index, nodes in enumerate(elements_np)
            }
            row_nodes = self.canonical.p.indices.reshape(-1, 4)
            row_elements = np.fromiter(
                (element_lookup[tuple(sorted(map(int, nodes)))] for nodes in row_nodes),
                dtype=np.int32,
                count=len(row_nodes),
            )
            face_neighbors = np.repeat(
                np.arange(len(elements_np), dtype=np.int32)[:, None], 4, axis=1
            )
            face_owner: dict[tuple[int, int, int], tuple[int, int]] = {}
            for element_index, nodes in enumerate(elements_np):
                for opposite in range(4):
                    face = tuple(sorted(int(nodes[i]) for i in range(4) if i != opposite))
                    prior = face_owner.pop(face, None)
                    if prior is None:
                        face_owner[face] = (element_index, opposite)
                    else:
                        other_element, other_opposite = prior
                        face_neighbors[element_index, opposite] = other_element
                        face_neighbors[other_element, other_opposite] = element_index
            np.save(row_path, row_elements)
            np.save(neighbor_path, face_neighbors)
        if row_elements.shape != (self.complement.n_voxels,):
            raise RuntimeError("Canonical voxel-to-element topology cache shape mismatch")
        if face_neighbors.shape != (len(elements_np), 4):
            raise RuntimeError("Tetrahedron face-neighbor cache shape mismatch")
        self.elements = torch.from_numpy(elements_np).to(self.device)
        self.row_elements = torch.from_numpy(row_elements.astype(np.int64)).to(self.device)
        self.face_neighbors = torch.from_numpy(face_neighbors.astype(np.int64)).to(self.device)

    def _audit_latent_manifests(self) -> None:
        expected = {
            "v4_config_sha256": file_sha256(Path(self.cfg["frozen_backbone"]["config"])),
            "v4_checkpoint_sha256": file_sha256(
                Path(self.cfg["frozen_backbone"]["checkpoint"])
            ),
            "mesh_sha256": file_sha256(Path(self.data["shared_dir"]) / "mesh.npz"),
            "frame_manifest_sha256": file_sha256(
                Path(self.data["shared_dir"]) / "frame_manifest.json"
            ),
            "terminal_hidden_shape": [self.complement.n_fem_nodes, self.latent_dim],
            "terminal_hidden_dtype": "float16",
        }
        for split in ("train", "val", "test"):
            for root, role in (
                (self.terminal_hidden_root, "terminal"),
                (self.initial_hidden_root, "initial"),
            ):
                path = root / split / "cache_manifest.json"
                if not path.exists():
                    raise FileNotFoundError(f"Missing {role} latent manifest: {path}")
                manifest = json.loads(path.read_text())
                for key, value in expected.items():
                    if manifest.get(key) != value:
                        raise RuntimeError(
                            f"{role} {split} latent manifest {key} mismatch"
                        )

    def build_model(self, mode: str | bool = "hard") -> CorrectionStateTransfer:
        if str(mode).lower() not in {"hard", "true", "constrained"}:
            raise ValueError("CST permits exact hard-Q only")
        model_cfg = self.cfg["model"]
        return CorrectionStateTransfer(
            coordinate_dim=self.pe.out_dim,
            state_dim=self.state_dim,
            latent_dim=self.latent_dim,
            transfer_dim=int(model_cfg["transfer_dim"]),
            modulation_groups=int(model_cfg["modulation_groups"]),
            decoder_width=int(model_cfg["decoder_width"]),
            decoder_depth=int(model_cfg["decoder_depth"]),
            complement=self.complement,
            innovation=str(model_cfg["innovation"]),
            one_ring_context_dim=int(model_cfg.get("one_ring_context_dim", 0)),
            direct_view_dim=(
                int(model_cfg["view_feat_dim"])
                if bool(model_cfg.get("use_views", False))
                else 0
            ),
        ).to(self.device)

    def _load_latent(self, root: Path, split: str, sid: str) -> torch.Tensor:
        array = np.load(root / split / f"{sid}.npy", mmap_mode="r")
        expected = (self.complement.n_fem_nodes, self.latent_dim)
        if array.shape != expected or array.dtype != np.float16:
            raise RuntimeError(f"Invalid latent cache {root / split / f'{sid}.npy'}")
        return torch.from_numpy(np.asarray(array, dtype=np.float32)).to(self.device)

    def predict(
        self, model: CorrectionStateTransfer, sid: str, split: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x0 = torch.from_numpy(
            np.load(
                Path(self.data[f"{split}_initial_state_root"]) / sid / "coarse_d.npy"
            ).astype(np.float32)
        ).to(self.device)
        xc = torch.from_numpy(
            np.load(Path(self.data["v4_states_root"]) / split / f"{sid}.npy").astype(np.float32)
        ).to(self.device)
        h0 = self._load_latent(self.initial_hidden_root, split, sid)
        hc = self._load_latent(self.terminal_hidden_root, split, sid)
        encoded = self._encoded_views(sid) if self.use_views else None
        return self.predict_from_sources(model, x0, xc, h0, hc, encoded)

    def predict_from_sources(
        self,
        model: CorrectionStateTransfer,
        x0: torch.Tensor,
        xc: torch.Tensor,
        h0_nodes: torch.Tensor,
        hc_nodes: torch.Tensor,
        encoded_views: torch.Tensor | dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        proposals: list[torch.Tensor] = []
        coarse_chunks: list[torch.Tensor] = []
        amp = self.device.type == "cuda"
        dx = xc - x0
        element_context = (
            model.build_one_ring_context(
                x0, xc, h0_nodes, hc_nodes, self.elements, self.face_neighbors
            )
            if self.use_one_ring
            else None
        )
        for start in range(0, self.complement.n_voxels, self.chunk_size):
            end = min(start + self.chunk_size, self.complement.n_voxels)
            ids = self.node_indices[start:end]
            bary = self.barycentric[start:end]
            gradients = self.gradient_weights[start:end]
            local0, localc, locald = x0[ids], xc[ids], dx[ids]
            x0q = torch.sum(local0 * bary, dim=-1)
            xcq = torch.sum(localc * bary, dim=-1)
            dxq = torch.sum(locald * bary, dim=-1)
            grad0 = torch.sum(local0[:, :, None] * gradients, dim=1)
            gradc = torch.sum(localc[:, :, None] * gradients, dim=1)
            gradd = torch.sum(locald[:, :, None] * gradients, dim=1)
            explicit = torch.cat(
                [
                    x0q[:, None],
                    xcq[:, None],
                    dxq[:, None],
                    grad0,
                    gradc,
                    gradd,
                    bary,
                    gradients.reshape(end - start, -1),
                ],
                dim=-1,
            )
            h0 = torch.sum(h0_nodes[ids] * bary[:, :, None], dim=1)
            hc = torch.sum(hc_nodes[ids] * bary[:, :, None], dim=1)
            views = (
                self.sample_view_features(
                    encoded_views, start=start, end=end, dtype=xcq.dtype
                ).squeeze(0)
                if self.use_views
                else None
            )
            with torch.autocast(
                device_type=self.device.type, dtype=torch.bfloat16, enabled=amp
            ):
                proposals.append(
                    model.proposal_checkpointed(
                        self.pe(self.coords_norm[start:end]),
                        explicit,
                        h0,
                        hc,
                        *(
                            (element_context[self.row_elements[start:end]],)
                            if element_context is not None
                            else ()
                        ),
                        *((None, views) if element_context is None and views is not None else ()),
                        *((views,) if element_context is not None and views is not None else ()),
                    )
                )
            coarse_chunks.append(xcq)
        proposal = torch.cat(proposals).unsqueeze(0)
        coarse = torch.cat(coarse_chunks).unsqueeze(0)
        detail = model.constrain(proposal)
        return proposal, coarse, detail, coarse + detail


@torch.inference_mode()
def validate(
    engine: CSTEngine, model: CorrectionStateTransfer, ids: list[str]
) -> dict[str, float]:
    model.eval()
    values = []
    threshold = float(engine.cfg["validation"]["threshold"])
    for index, sid in enumerate(ids):
        _, _, _, final = engine.predict(model, sid, "val")
        gt, _ = engine._targets(sid)
        prediction, truth = final.squeeze(0) >= threshold, gt >= 0.5
        values.append(
            float(2 * (prediction & truth).sum() / (prediction.sum() + truth.sum()).clamp_min(1))
        )
        if (index + 1) % 50 == 0:
            print(f"[CST val {index + 1}/{len(ids)}]", flush=True)
    return {"dice": float(np.mean(values))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-name")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume an interrupted run from an epoch-complete checkpoint.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    backbone = yaml.safe_load(Path(cfg["frozen_backbone"]["config"]).read_text())
    os.environ["DU2VOX_SHARED_DIR"] = str(backbone["data"]["shared_dir"])
    os.environ["DU2VOX_ALLOW_STALE_FRAME_MANIFEST"] = "1"
    os.environ["DU2VOX_FRAME_MANIFEST_SHA256"] = str(
        backbone["data"]["frame_manifest_sha256"]
    )
    seed = int(cfg["training"]["seed"])
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = CSTEngine(cfg, device)
    model = engine.build_model()
    train_ids = load_ids(cfg["data"]["train_split"])
    val_ids = load_ids(cfg["data"]["val_split"])
    if args.max_samples is None and (len(train_ids), len(val_ids)) != (2400, 300):
        raise RuntimeError("Formal CST requires the fixed 2400/300 train/validation split")
    if args.max_samples is not None:
        train_ids = train_ids[: args.max_samples]
        val_ids = val_ids[: min(args.max_samples, len(val_ids))]
    epochs = args.max_epochs or int(cfg["training"]["epochs"])
    experiment = args.experiment_name or cfg["experiment"]["name"]
    run_dir = Path(cfg["runs_root"]) / experiment
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.resume is not None and not args.resume.exists():
        raise FileNotFoundError(args.resume)
    shutil.copy2(args.config, run_dir / "config.yaml")
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    parameter_count = sum(p.numel() for p in model.parameters())
    init_hash = model_initialization_sha256(model)
    (run_dir / "model_contract.json").write_text(
        json.dumps(
            {
                "parameter_count": parameter_count,
                "strong_baseline_parameter_count": 193121,
                "relative_parameter_difference": parameter_count / 193121 - 1,
                "initialization_sha256": init_hash,
                "hard_q": True,
                "coarse_states_frozen": True,
                "detail_gradient_into_coarse": False,
                "seed": seed,
            },
            indent=2,
        )
        + "\n"
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    best_dice = -np.inf
    history: list[dict[str, float | int]] = []
    start_epoch = 1
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if "optimizer" not in checkpoint:
            raise RuntimeError(
                "Resume checkpoint predates the optimizer-state contract; restart the "
                "formal run rather than silently changing AdamW state."
            )
        if checkpoint.get("initialization_sha256") != init_hash:
            raise RuntimeError("Resume checkpoint initialization contract mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        history_path = run_dir / "history.json"
        if not history_path.exists():
            raise FileNotFoundError(
                f"Resume requires the matching epoch history: {history_path}"
            )
        history = json.loads(history_path.read_text())
        if not history or int(history[-1]["epoch"]) != start_epoch - 1:
            raise RuntimeError("Resume checkpoint and history epoch mismatch")
        best_dice = max(float(row["val_final_dice"]) for row in history)
        print(
            f"[CST resume] checkpoint={args.resume} start_epoch={start_epoch} "
            f"best_val_dice={best_dice:.9f}",
            flush=True,
        )
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        totals = []
        started = time.perf_counter()
        order = np.random.default_rng(seed + epoch).permutation(len(train_ids))
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
                    f"[CST epoch {epoch}/{epochs} {position + 1}/{len(train_ids)}] "
                    f"loss={np.mean(totals[-50:]):.6f}",
                    flush=True,
                )
        validation = validate(engine, model, val_ids)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(totals)),
            "val_final_dice": validation["dice"],
            "alpha": float(model.alpha.detach()),
            "epoch_seconds": time.perf_counter() - started,
        }
        history.append(row)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "dense_val_dice": validation["dice"],
            "parameter_count": parameter_count,
            "initialization_sha256": init_hash,
            "config": cfg,
            "hard_q": True,
        }
        torch.save(payload, checkpoint_dir / "last.pth")
        if validation["dice"] > best_dice:
            best_dice = validation["dice"]
            shutil.copy2(checkpoint_dir / "last.pth", checkpoint_dir / "best_dense_val_dice.pth")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
