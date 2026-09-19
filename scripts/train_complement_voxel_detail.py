#!/usr/bin/env python3
"""Train matched frozen-V4 voxel detail models on the full canonical domain."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.bridge.canonical_cross_discretization import CanonicalCrossDiscretization
from du2vox.models.stage2.complement_voxel_detail import (
    ComplementConstrainedVoxelDetail,
    ExactVoxelComplement,
    normalize_detail_mode,
)
from du2vox.models.stage2.residual_inr import PositionalEncoding
from du2vox.models.stage2.stage2_dataset import load_projection_stack
from du2vox.evaluation.continuous_field import continuous_metrics
from du2vox.utils.frame import FrameManifest
from du2vox.utils.gt_io import load_canonical_gt, load_normalization_scale
from scripts.train_error_structured_bridge import build_view_encoder


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_ids(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_dict_sha256(module: torch.nn.Module) -> str:
    """Hash tensor names, dtypes, shapes, and values deterministically."""

    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def parameter_counts(*modules: torch.nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for module in modules for parameter in module.parameters()),
        "trainable": sum(
            parameter.numel()
            for module in modules
            for parameter in module.parameters()
            if parameter.requires_grad
        ),
    }


def build_gradient_weights(
    canonical: CanonicalCrossDiscretization, shared_dir: Path, cache_path: Path
) -> np.ndarray:
    expected_shape = (canonical.p.shape[0], 4, 3)
    if cache_path.exists():
        cached = np.load(cache_path, mmap_mode="r")
        if cached.shape != expected_shape:
            raise RuntimeError(f"Gradient cache shape {cached.shape} != {expected_shape}")
        return cached
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    nodes, _ = FrameManifest.load_mesh_nodes(shared_dir)
    node_ids = canonical.p.indices.reshape(-1, 4)
    output = np.lib.format.open_memmap(
        cache_path, mode="w+", dtype=np.float32, shape=expected_shape
    )
    chunk = 131072
    for start in range(0, len(node_ids), chunk):
        end = min(start + chunk, len(node_ids))
        vertices = nodes[node_ids[start:end]].astype(np.float64)
        edge = np.stack(
            [
                vertices[:, 1] - vertices[:, 0],
                vertices[:, 2] - vertices[:, 0],
                vertices[:, 3] - vertices[:, 0],
            ],
            axis=2,
        )
        inverse = np.linalg.inv(edge)
        output[start:end, 1:] = inverse.astype(np.float32)
        output[start:end, 0] = -inverse.sum(axis=1).astype(np.float32)
    output.flush()
    return np.load(cache_path, mmap_mode="r")


class FullDomainEngine:
    def __init__(self, cfg: dict[str, Any], device: torch.device) -> None:
        self.cfg = cfg
        self.data = cfg["data"]
        self.device = device
        self.canonical = CanonicalCrossDiscretization(
            self.data["operator_cache"], shared_dir=self.data["shared_dir"], factorize=False
        )
        self.complement = ExactVoxelComplement(
            self.canonical.p, quadrature_weight=self.canonical.quadrature_weight
        )
        p = self.canonical.p.tocsr()
        self.node_indices = torch.from_numpy(p.indices.reshape(-1, 4).astype(np.int64)).to(device)
        self.barycentric = torch.from_numpy(p.data.reshape(-1, 4).astype(np.float32)).to(device)
        self.coords_world = torch.from_numpy(self.canonical.operator.coords_world).to(device)
        frame = FrameManifest.load(self.data["shared_dir"])
        bbox_min = torch.as_tensor(frame.mcx_bbox_min, device=device, dtype=torch.float32)
        bbox_max = torch.as_tensor(frame.mcx_bbox_max, device=device, dtype=torch.float32)
        self.coords_norm = 2.0 * (self.coords_world - bbox_min) / (bbox_max - bbox_min) - 1.0
        gradient_path = Path("precomputed/complement_voxel_detail/canonical_gradient_weights.npy")
        gradient = build_gradient_weights(
            self.canonical, Path(self.data["shared_dir"]), gradient_path
        )
        self.gradient_weights = torch.from_numpy(np.array(gradient, copy=True)).to(device)
        self.pe = PositionalEncoding(
            n_freqs=int(cfg["model"]["coordinate_frequencies"]), include_input=True
        ).to(device)
        self.use_views = bool(cfg["model"].get("use_views", True))
        self.input_contract = str(cfg["model"].get("input_contract", "legacy_local_descriptor"))
        self.train_view_encoder = bool(cfg["model"].get("train_view_encoder", False))
        self.use_terminal_latent = bool(cfg["model"].get("use_terminal_latent", False))
        self.terminal_hidden_root = (
            Path(self.data["terminal_hidden_root"]) if self.use_terminal_latent else None
        )
        self.latent_dim = int(cfg["model"].get("latent_dim", 0))
        if self.use_terminal_latent and self.latent_dim <= 0:
            raise ValueError("use_terminal_latent requires a positive model.latent_dim")
        if self.input_contract == "approximation_space_separated":
            if self.use_views:
                raise ValueError("approximation_space_separated forbids direct voxel-side views")
            if not self.use_terminal_latent:
                raise ValueError("approximation_space_separated requires terminal Hc concatenation")
        elif self.input_contract != "legacy_local_descriptor":
            raise ValueError(f"Unknown model.input_contract: {self.input_contract}")
        self.coarse_source = cfg.get("coarse_source")
        if self.coarse_source is None:
            # Compatibility for the frozen-V4 experiments. New Stage1 hard-Q
            # configurations must use coarse_source and never enter this branch.
            from scripts.train_iterative_fem_corrector import build_model

            backbone_config_path = Path(cfg["frozen_backbone"]["config"])
            expected_config_sha = cfg["frozen_backbone"].get("config_sha256")
            if (
                expected_config_sha is not None
                and sha256_file(backbone_config_path) != expected_config_sha
            ):
                raise RuntimeError("Frozen V4 config SHA256 mismatch")
            backbone_cfg = yaml.safe_load(backbone_config_path.read_text())
            v4 = build_model(backbone_cfg).to(device)
            checkpoint_path = Path(cfg["frozen_backbone"]["checkpoint"])
            expected_checkpoint_sha = cfg["frozen_backbone"].get("checkpoint_sha256")
            if (
                expected_checkpoint_sha is not None
                and sha256_file(checkpoint_path) != expected_checkpoint_sha
            ):
                raise RuntimeError("Frozen V4 checkpoint SHA256 mismatch")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if checkpoint.get("epoch") != int(cfg["frozen_backbone"]["expected_epoch"]):
                raise RuntimeError("Frozen V4 checkpoint epoch does not match the config")
            v4.load_state_dict(checkpoint["model"])
            self.encoder = build_view_encoder(backbone_cfg["model"]).to(device)
            self.encoder.load_state_dict(v4.view_encoder.state_dict())
            self.encoder.eval().requires_grad_(False)
            del v4
        else:
            if self.coarse_source.get("type") != "stage1_bridge":
                raise ValueError("coarse_source.type must be 'stage1_bridge'")
            self._verify_stage1_checkpoint()
            self.encoder = build_view_encoder(cfg["model"]).to(device)
            if self.encoder is None:
                raise ValueError("Stage1 hard-Q arms must instantiate an isomorphic view encoder")
            # The no-view arm retains the same trainable parameter inventory. Its
            # encoder is deliberately not called, so the view slot is exactly zero.
            self.encoder.requires_grad_(self.train_view_encoder)
        self.chunk_size = int(cfg["model"]["chunk_size"])
        self.boundary_temperature = float(cfg["model"]["boundary_temperature"])
        if self.input_contract == "approximation_space_separated":
            # [PE(q), I_h x_h^c(q), lambda(q), G_e, I_h H_h^c(q)].
            # G_e is the flattened 4x3 analytic P1 basis-gradient matrix.
            self.input_dim = self.pe.out_dim + 1 + 4 + 12 + self.latent_dim
        else:
            view_dim = int(cfg["model"]["view_feat_dim"])
            self.input_dim = (
                self.pe.out_dim
                + 1
                + 3
                + 4
                + 4
                + 3
                + 1
                + view_dim
                + (self.latent_dim if self.use_terminal_latent else 0)
            )

    def _verify_stage1_checkpoint(self) -> None:
        checkpoint_path = Path(self.coarse_source["checkpoint"])
        actual_sha = sha256_file(checkpoint_path)
        expected_sha = str(self.coarse_source["checkpoint_sha256"]).lower()
        if actual_sha != expected_sha:
            raise RuntimeError(f"Stage1 checkpoint SHA256 mismatch: {actual_sha} != {expected_sha}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        expected_epoch = int(self.coarse_source["expected_epoch"])
        if int(checkpoint.get("epoch", -1)) != expected_epoch:
            raise RuntimeError(
                f"Stage1 checkpoint epoch {checkpoint.get('epoch')} != {expected_epoch}"
            )
        self.stage1_checkpoint_sha256 = actual_sha

    def coarse_path(self, sid: str, split: str) -> Path:
        if self.coarse_source is not None:
            key = f"{split}_bridge_dir"
            return Path(self.coarse_source[key]) / sid / "coarse_d.npy"
        return Path(self.data["v4_states_root"]) / split / f"{sid}.npy"

    def audit_coarse_cache(self, split: str, ids: list[str]) -> dict[str, Any]:
        """Validate the complete declared Stage1 cache before using any sample."""

        if len(ids) != len(set(ids)):
            raise RuntimeError(f"Duplicate sample IDs in {split} split")
        dtypes: set[str] = set()
        for sid in ids:
            path = self.coarse_path(sid, split)
            if not path.exists():
                raise FileNotFoundError(f"Missing coarse state for {sid}: {path}")
            values = np.load(path, mmap_mode="r")
            if values.size != self.complement.n_fem_nodes:
                raise RuntimeError(
                    f"{path}: {values.size} values != {self.complement.n_fem_nodes} FEM nodes"
                )
            if not np.issubdtype(values.dtype, np.floating):
                raise TypeError(f"{path}: expected floating dtype, got {values.dtype}")
            if not np.isfinite(values).all():
                raise ValueError(f"{path}: coarse state contains non-finite values")
            dtypes.add(str(values.dtype))
            if self.use_terminal_latent:
                latent_path = self.terminal_hidden_root / split / f"{sid}.npy"
                if not latent_path.exists():
                    raise FileNotFoundError(f"Missing terminal V4 latent for {sid}: {latent_path}")
                latent = np.load(latent_path, mmap_mode="r")
                expected = (self.complement.n_fem_nodes, self.latent_dim)
                if latent.shape != expected or not np.issubdtype(latent.dtype, np.floating):
                    raise RuntimeError(
                        f"Invalid terminal latent {latent_path}: {latent.shape}/{latent.dtype}, "
                        f"expected {expected}/floating"
                    )
        expected_dtype = (
            self.coarse_source.get("expected_dtype") if self.coarse_source is not None else None
        )
        if expected_dtype is not None and dtypes != {str(expected_dtype)}:
            raise TypeError(
                f"{split} coarse cache dtypes {sorted(dtypes)} != {[str(expected_dtype)]}"
            )
        return {
            "split": split,
            "n_samples": len(ids),
            "n_fem_nodes": self.complement.n_fem_nodes,
            "dtypes": sorted(dtypes),
            "checkpoint_sha256": getattr(self, "stage1_checkpoint_sha256", None),
        }

    def build_model(self, mode: str | bool) -> ComplementConstrainedVoxelDetail:
        return ComplementConstrainedVoxelDetail(
            input_dim=self.input_dim,
            complement=self.complement,
            hidden_dim=int(self.cfg["model"]["hidden_dim"]),
            n_hidden_layers=int(self.cfg["model"]["hidden_layers"]),
            mode=normalize_detail_mode(mode),
        ).to(self.device)

    def _targets(self, sid: str) -> tuple[torch.Tensor, torch.Tensor]:
        scale_filename = self.data.get("normalization_scale_filename")
        normalization_scale = (
            load_normalization_scale(Path(self.data["samples_dir"]) / sid, scale_filename)
            if scale_filename
            else None
        )
        gt_np, _ = load_canonical_gt(
            Path(self.data["samples_dir"]) / sid,
            self.canonical.operator.valid_flat_indices,
            gt_mode=self.data.get("gt_mode", "binary"),
            normalize=self.data.get("normalize_gt", "none"),
            binary_threshold=float(self.data.get("binary_threshold", 0.05)),
            normalization_scale=normalization_scale,
        )
        pi_gt = np.load(Path(self.data["projection_targets_dir"]) / f"{sid}.npy").astype(np.float32)
        detail_np = gt_np - np.asarray(self.canonical.p @ pi_gt).ravel().astype(np.float32)
        return torch.from_numpy(gt_np).to(self.device), torch.from_numpy(detail_np).to(self.device)

    def _encoded_views(self, sid: str) -> torch.Tensor | dict[str, torch.Tensor]:
        projections, _ = load_projection_stack(
            Path(self.data["samples_dir"]) / sid,
            projection_file=self.data["projection_file"],
            projection_norm=self.data["projection_norm"],
            projection_transform=self.data["projection_transform"],
        )
        images = torch.from_numpy(projections[:, None]).unsqueeze(0).to(self.device)
        return self.encoder.encode_images(images)

    def predict(
        self, model: ComplementConstrainedVoxelDetail, sid: str, split: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        coarse_state = torch.from_numpy(
            np.load(self.coarse_path(sid, split)).reshape(-1).astype(np.float32)
        ).to(self.device)
        terminal_hidden = None
        if self.use_terminal_latent:
            terminal_hidden = torch.from_numpy(
                np.load(self.terminal_hidden_root / split / f"{sid}.npy").astype(np.float32)
            ).to(self.device)
        encoded = self._encoded_views(sid) if self.use_views else None
        return self.predict_from_state(
            model, coarse_state, encoded, terminal_hidden=terminal_hidden
        )

    def predict_from_state(
        self,
        model: ComplementConstrainedVoxelDetail,
        coarse_state: torch.Tensor,
        encoded: torch.Tensor | dict[str, torch.Tensor] | None,
        terminal_hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the dense detail result from fixed FEM coefficients."""

        proposals = []
        coarse_chunks = []
        amp = self.device.type == "cuda"
        for start in range(0, self.complement.n_voxels, self.chunk_size):
            end = min(start + self.chunk_size, self.complement.n_voxels)
            ids = self.node_indices[start:end]
            bary = self.barycentric[start:end]
            local = coarse_state[ids]
            coarse = torch.sum(local * bary, dim=-1)
            latent = None
            if self.use_terminal_latent:
                if terminal_hidden is None:
                    raise RuntimeError("Configured terminal latent is missing")
                latent = torch.sum(terminal_hidden[ids] * bary[:, :, None], dim=1)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=amp):
                if self.input_contract == "approximation_space_separated":
                    feature_parts = [
                        self.pe(self.coords_norm[start:end]),
                        coarse[:, None],
                        bary,
                        self.gradient_weights[start:end].reshape(end - start, 12),
                    ]
                else:
                    gradient = torch.sum(
                        local[:, :, None] * self.gradient_weights[start:end], dim=1
                    )
                    local_stats = torch.stack(
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
                    view = self.sample_view_features(
                        encoded, start=start, end=end, dtype=coarse.dtype
                    )
                    feature_parts = [
                        self.pe(self.coords_norm[start:end]),
                        coarse[:, None],
                        gradient,
                        local,
                        bary,
                        local_stats,
                        boundary,
                        view.squeeze(0),
                    ]
                if latent is not None:
                    # Terminal concat is intentionally last: no Delta-H route and
                    # no direct voxel-side view bypass.
                    feature_parts.append(latent)
                features = torch.cat(feature_parts, dim=-1)
                proposals.append(model.proposal_checkpointed(features))
            coarse_chunks.append(coarse)
        proposal = torch.cat(proposals).unsqueeze(0)
        coarse = torch.cat(coarse_chunks).unsqueeze(0)
        detail = model.constrain(proposal)
        return proposal, coarse, detail, coarse + detail

    def sample_view_features(
        self,
        encoded: torch.Tensor | dict[str, torch.Tensor] | None,
        *,
        start: int,
        end: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the fixed 32-D slot; the no-view arm is identically zero."""

        if not self.use_views:
            return torch.zeros(
                1,
                end - start,
                int(self.cfg["model"]["view_feat_dim"]),
                device=self.device,
                dtype=dtype,
            )
        if encoded is None:
            raise RuntimeError("The direct-view arm requires encoded projections")
        view, _ = self.encoder.sample_encoded(
            encoded, self.coords_world[start:end][None], coords_vox_norm=None
        )
        return view

    def loss(
        self,
        model: ComplementConstrainedVoxelDetail,
        proposal: torch.Tensor,
        detail: torch.Tensor,
        final: torch.Tensor,
        gt: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss_cfg = self.cfg["loss"]
        smooth = F.smooth_l1_loss(
            detail.float(), target[None], beta=float(loss_cfg["smooth_l1_beta"])
        )
        mse = F.mse_loss(detail.float(), target[None])
        final_error_sq = (final.float() - gt[None]).square()
        source_mask = gt[None] > float(loss_cfg.get("source_threshold", 0.0))
        source_mse = final_error_sq[source_mask].mean()
        global_mse = final_error_sq.mean()
        # Support is defined RELATIVE TO THE PER-CASE PEAK, matching the Dice@50%
        # metric. ``gt`` here is the full canonical valid domain, so its max is the
        # true per-case peak. Dividing both sides by it also leaves
        # ``support_temperature`` on its original dimensionless intensity scale.
        support_fraction = float(loss_cfg.get("support_fraction", 0.5))
        gt_peak = gt.max().clamp_min(1e-8)
        probability = torch.sigmoid(
            (final.float() / gt_peak - support_fraction)
            / float(loss_cfg["support_temperature"])
        )
        # Tversky scores set overlap, so it needs a support mask. Under the binary
        # contract the peak is exactly 1, so this reduces to ``gt >= 0.5``; under
        # the continuous contract it yields the peak-relative support being scored.
        truth = (gt[None] / gt_peak >= support_fraction).to(probability.dtype)
        tp = torch.sum(probability * truth)
        fp = torch.sum(probability * (1.0 - truth))
        fn = torch.sum((1.0 - probability) * truth)
        tversky = 1.0 - (tp + 1e-6) / (
            tp + float(loss_cfg["tversky_alpha"]) * fp + float(loss_cfg["tversky_beta"]) * fn + 1e-6
        )
        # Non-negativity prior on the final voxel field this decoder emits. It
        # constrains the physical output rather than the approximation-space
        # proposal, so the coarse/representation separation is preserved; the
        # gradient still reaches z through the exact Q. A fluorophore
        # concentration cannot be negative. See
        # diagnosis/lpr_negativity_audit_val300.json for the measured size of the
        # negative excursion this counterweights.
        voxel_negativity = torch.clamp(final.float(), max=0.0).square().mean()
        total = (
            float(loss_cfg["lambda_detail_l1"]) * smooth
            + float(loss_cfg["lambda_detail_mse"]) * mse
            + float(loss_cfg["lambda_support"]) * tversky
            + float(loss_cfg.get("lambda_final_source_mse", 0.0)) * source_mse
            + float(loss_cfg.get("lambda_final_global_mse", 0.0)) * global_mse
            + float(loss_cfg.get("lambda_negativity", 0.0)) * voxel_negativity
        )
        soft_penalty = proposal.new_zeros(())
        if model.mode == "soft":
            soft_penalty = model.soft_coarse_penalty(
                proposal, eps=float(loss_cfg.get("soft_epsilon", 1e-12))
            )
            total = total + float(loss_cfg["lambda_coarse_soft"]) * soft_penalty
        return total, {
            "smooth_l1": float(smooth.detach()),
            "detail_mse": float(mse.detach()),
            "support": float(tversky.detach()),
            "final_source_mse": float(source_mse.detach()),
            "final_global_mse": float(global_mse.detach()),
            "voxel_negativity": float(voxel_negativity.detach()),
            "soft_coarse_penalty": float(soft_penalty.detach()),
        }


@torch.inference_mode()
def validate(
    engine: FullDomainEngine,
    model: ComplementConstrainedVoxelDetail,
    ids: list[str],
) -> dict[str, float]:
    model.eval()
    engine.encoder.eval()
    dice_values = []
    leakage_values = []
    field_values: dict[str, list[float]] = {}
    threshold = float(engine.cfg["validation"]["threshold"])
    for index, sid in enumerate(ids):
        _, _, detail, final = engine.predict(model, sid, "val")
        gt, _ = engine._targets(sid)
        predicted = final.squeeze(0) >= threshold
        truth = gt >= 0.5
        dice_values.append(
            float(
                2
                * torch.sum(predicted & truth)
                / (torch.sum(predicted) + torch.sum(truth)).clamp_min(1)
            )
        )
        detail_np = detail.squeeze(0).float().cpu().numpy()
        projected = engine.complement.project_numpy(detail_np)
        leakage_values.append(
            float(np.dot(projected, projected) / max(np.dot(detail_np, detail_np), 1e-30))
        )
        for key, value in continuous_metrics(
            final.squeeze(0).float().cpu().numpy(),
            gt.float().cpu().numpy(),
            data_range=float(engine.data.get("data_range", 2.0)),
        ).items():
            field_values.setdefault(key, []).append(float(value))
        if (index + 1) % 50 == 0:
            print(f"[dense val {index + 1}/{len(ids)}]", flush=True)
    return {
        "dice": float(np.mean(dice_values)),
        "coarse_leakage": float(np.mean(leakage_values)),
        **{key: float(np.nanmean(values)) for key, values in field_values.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["unconstrained", "soft", "hard", "constrained"])
    parser.add_argument("--lambda-coarse-soft", type=float)
    parser.add_argument("--experiment-name")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--disable-early-stopping", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.max_epochs is not None:
        cfg["training"]["epochs"] = args.max_epochs
    environment_data = cfg["data"]
    if "frozen_backbone" in cfg:
        backbone_cfg = yaml.safe_load(Path(cfg["frozen_backbone"]["config"]).read_text())
        environment_data = backbone_cfg["data"]
    os.environ["DU2VOX_SHARED_DIR"] = str(environment_data["shared_dir"])
    if environment_data.get("allow_stale_frame_manifest", False):
        os.environ["DU2VOX_ALLOW_STALE_FRAME_MANIFEST"] = "1"
    if environment_data.get("frame_manifest_sha256"):
        os.environ["DU2VOX_FRAME_MANIFEST_SHA256"] = str(environment_data["frame_manifest_sha256"])
    seed_all(int(cfg["training"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = FullDomainEngine(cfg, device)
    configured_mode = cfg["model"].get("mode")
    if args.mode is None and configured_mode is None:
        raise ValueError("Set model.mode in the config or pass --mode")
    mode = normalize_detail_mode(args.mode or configured_mode)
    if configured_mode is not None and mode != normalize_detail_mode(configured_mode):
        raise ValueError(
            f"CLI mode {args.mode!r} disagrees with config model.mode {configured_mode!r}"
        )
    if cfg.get("coarse_source") is not None and mode != "hard":
        raise ValueError("The direct Stage1 experiment permits only exact hard-Q mode")
    if args.lambda_coarse_soft is not None:
        cfg["loss"]["lambda_coarse_soft"] = args.lambda_coarse_soft
    if mode == "soft" and "lambda_coarse_soft" not in cfg["loss"]:
        raise ValueError("soft mode requires loss.lambda_coarse_soft or --lambda-coarse-soft")
    model = engine.build_model(mode)
    initial_decoder_sha256 = state_dict_sha256(model)
    initial_encoder_sha256 = state_dict_sha256(engine.encoder)
    optimized_parameters = list(model.parameters()) + list(engine.encoder.parameters())
    optimizer = torch.optim.AdamW(
        optimized_parameters,
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    train_ids = load_ids(cfg["data"]["train_split"])
    val_ids = load_ids(cfg["data"]["val_split"])
    if args.max_samples is not None:
        train_ids = train_ids[: args.max_samples]
        val_ids = val_ids[: min(args.max_samples, len(val_ids))]
    coarse_audit = {
        "train": engine.audit_coarse_cache("train", train_ids),
        "val": engine.audit_coarse_cache("val", val_ids),
    }
    experiment = args.experiment_name or f"{cfg['experiment']['name']}_{mode}"
    run_dir = Path(cfg["runs_root"]) / experiment
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Save the effective configuration, including bounded LR/epoch overrides, so
    # a selected tuning run can be evaluated from one self-contained config.
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (run_dir / "coarse_cache_audit.json").write_text(json.dumps(coarse_audit, indent=2) + "\n")
    epochs = int(cfg["training"]["epochs"])
    selection_metric = str(cfg["validation"].get("selection_metric", "dice"))
    selection_mode = str(cfg["validation"].get("selection_mode", "max"))
    if selection_mode not in {"max", "min"}:
        raise ValueError("validation.selection_mode must be max or min")
    best_score = -np.inf if selection_mode == "max" else np.inf
    best_name = str(cfg["validation"].get("checkpoint_name", "best_dense_val_dice.pth"))
    stale = 0
    history = []
    order_hashes = [
        hashlib.sha256(
            np.random.default_rng(int(cfg["training"]["seed"]) + epoch)
            .permutation(len(train_ids))
            .tobytes()
        ).hexdigest()
        for epoch in range(1, epochs + 1)
    ]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, epochs + 1):
        model.train()
        engine.encoder.train(engine.train_view_encoder and engine.use_views)
        rng = np.random.default_rng(int(cfg["training"]["seed"]) + epoch)
        order = rng.permutation(len(train_ids))
        if hashlib.sha256(order.tobytes()).hexdigest() != order_hashes[epoch - 1]:
            raise RuntimeError("Deterministic training order contract failed")
        totals = []
        components: dict[str, list[float]] = {
            "smooth_l1": [],
            "detail_mse": [],
            "support": [],
            "final_source_mse": [],
            "final_global_mse": [],
            "voxel_negativity": [],
            "soft_coarse_penalty": [],
        }
        start_time = time.perf_counter()
        for position, sample_index in enumerate(order):
            sid = train_ids[int(sample_index)]
            gt, target = engine._targets(sid)
            optimizer.zero_grad(set_to_none=True)
            proposal, _, detail, final = engine.predict(model, sid, "train")
            loss, values = engine.loss(model, proposal, detail, final, gt, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                optimized_parameters, float(cfg["training"]["grad_clip_norm"])
            )
            optimizer.step()
            totals.append(float(loss.detach()))
            for key, value in values.items():
                components[key].append(value)
            if (position + 1) % 50 == 0:
                print(
                    f"[epoch {epoch}/{epochs} train {position + 1}/{len(train_ids)}] "
                    f"loss={np.mean(totals[-50:]):.6f}",
                    flush=True,
                )
        validation = validate(engine, model, val_ids)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(totals)),
            **{f"train_{key}": float(np.mean(value)) for key, value in components.items()},
            "val_dice": validation["dice"],
            "val_coarse_leakage": validation["coarse_leakage"],
            **{
                f"val_{key}": value
                for key, value in validation.items()
                if key not in {"dice", "coarse_leakage"}
            },
            "epoch_seconds": time.perf_counter() - start_time,
            "peak_gpu_memory_gib": (
                torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
            ),
        }
        memory_limit = float(cfg["training"].get("max_peak_gpu_gib", 16.0))
        if row["peak_gpu_memory_gib"] > memory_limit:
            raise RuntimeError(
                f"Peak GPU memory {row['peak_gpu_memory_gib']:.3f} GiB exceeds "
                f"{memory_limit:.3f} GiB"
            )
        history.append(row)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        torch.save(
            {
                "model": model.state_dict(),
                "view_encoder": engine.encoder.state_dict(),
                "mode": mode,
                "epoch": epoch,
                "dense_val_dice": validation["dice"],
                "dense_val_metric": validation[selection_metric],
                "selection_metric": selection_metric,
                "config": cfg,
                "coarse_cache_audit": coarse_audit,
                "initial_decoder_sha256": initial_decoder_sha256,
                "initial_encoder_sha256": initial_encoder_sha256,
                "training_order_sha256": list(order_hashes),
                "parameter_counts": parameter_counts(model, engine.encoder),
            },
            checkpoint_dir / "last.pth",
        )
        score = float(validation[selection_metric])
        improved = score > best_score if selection_mode == "max" else score < best_score
        if improved:
            best_score = score
            stale = 0
            shutil.copy2(checkpoint_dir / "last.pth", checkpoint_dir / best_name)
        else:
            stale += 1
        print(json.dumps(row), flush=True)
        if not args.disable_early_stopping and stale >= int(
            cfg["training"]["early_stopping_patience"]
        ):
            break


if __name__ == "__main__":
    main()
