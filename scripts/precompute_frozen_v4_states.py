#!/usr/bin/env python3
"""Cache frozen V4 FEM states without materializing dense voxel predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.data.error_structured_dataset import ErrorStructuredFixedDomainDataset
from du2vox.models.stage2.stage2_dataset import load_projection_stack
from scripts.train_iterative_fem_corrector import build_dataset, build_model, load_ids


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stratified_audit_ids(ids: list[str], count: int, seed: int) -> set[str]:
    """Take one deterministic member from evenly spaced split strata."""

    if not ids or count <= 0:
        return set()
    rng = random.Random(seed)
    strata = np.array_split(np.asarray(ids, dtype=object), min(count, len(ids)))
    return {str(rng.choice(list(stratum))) for stratum in strata if len(stratum)}


def metadata_stratified_audit_ids(
    ids: list[str], samples_dir: Path, count: int, seed: int
) -> set[str]:
    """Proportionally sample joint source-count/depth/type strata."""

    if not ids or count <= 0:
        return set()
    groups: dict[tuple[int, str, str], list[str]] = {}
    for sid in ids:
        params = json.loads((samples_dir / sid / "tumor_params.json").read_text())
        key = (
            len(params.get("foci", [])),
            str(params.get("depth_tier", "unknown")),
            str(params.get("source_type", "unknown")),
        )
        groups.setdefault(key, []).append(sid)
    target = min(count, len(ids))
    exact = {key: target * len(members) / len(ids) for key, members in groups.items()}
    allocation = {key: min(len(groups[key]), int(value)) for key, value in exact.items()}
    remaining = target - sum(allocation.values())
    priority = sorted(
        groups,
        key=lambda key: (exact[key] - allocation[key], len(groups[key]), key),
        reverse=True,
    )
    while remaining:
        progressed = False
        for key in priority:
            if allocation[key] < len(groups[key]):
                allocation[key] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break
    rng = random.Random(seed)
    return {sid for key, members in groups.items() for sid in rng.sample(members, allocation[key])}


def fp16_relative_l2(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize a finite terminal hidden array and report its relative L2 error."""

    array = np.asarray(values, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Terminal hidden contains non-finite FP32 values")
    quantized = array.astype(np.float16)
    if not np.isfinite(quantized).all():
        raise ValueError("Terminal hidden is not finite after FP16 quantization")
    error = float(
        np.linalg.norm(array - quantized.astype(np.float32)) / max(np.linalg.norm(array), 1e-30)
    )
    return quantized, error


def validate_cache_array(path: Path, *, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    array = np.load(path, mmap_mode="r")
    if array.shape != shape:
        raise RuntimeError(f"Cache shape {array.shape} != {shape}: {path}")
    if array.dtype != dtype:
        raise RuntimeError(f"Cache dtype {array.dtype} != {dtype}: {path}")
    if not np.isfinite(array).all():
        raise RuntimeError(f"Cache contains non-finite values: {path}")
    return array


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-terminal-hidden", action="store_true")
    parser.add_argument("--latent-output-dir", type=Path)
    parser.add_argument("--initial-latent-output-dir", type=Path)
    parser.add_argument("--generation-seed", type=int, default=20260901)
    parser.add_argument("--expected-epoch", type=int, default=15)
    parser.add_argument(
        "--fp16-audit-samples",
        type=int,
        help="Override the canonical split allocation (train/val/test = 24/3/3)",
    )
    parser.add_argument("--fp16-relative-l2-limit", type=float, default=5e-4)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    random.seed(args.generation_seed)
    np.random.seed(args.generation_seed)
    torch.manual_seed(args.generation_seed)
    torch.cuda.manual_seed_all(args.generation_seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    cfg = yaml.safe_load(args.config.read_text())
    data = cfg["data"]
    os.environ["DU2VOX_SHARED_DIR"] = str(data["shared_dir"])
    if data.get("allow_stale_frame_manifest", False):
        os.environ["DU2VOX_ALLOW_STALE_FRAME_MANIFEST"] = "1"
    if data.get("frame_manifest_sha256"):
        os.environ["DU2VOX_FRAME_MANIFEST_SHA256"] = str(data["frame_manifest_sha256"])
    ids = load_ids(data[f"{args.split}_split"])
    expected_count = {"train": 2400, "val": 300, "test": 300}[args.split]
    if args.max_samples is None and len(ids) != expected_count:
        raise RuntimeError(f"{args.split} split has {len(ids)} samples, expected {expected_count}")
    if args.max_samples is not None:
        ids = ids[: args.max_samples]
    dataset: ErrorStructuredFixedDomainDataset = build_dataset(
        cfg, args.split, ids, int(cfg["training"].get("n_query_points", 8192))
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("model_type") != "unified_dual_evidence_fem_v4":
        raise RuntimeError("Frozen backbone checkpoint is not V4")
    if checkpoint.get("epoch") != args.expected_epoch:
        raise RuntimeError(
            f"Frozen V4 checkpoint epoch {checkpoint.get('epoch')} != {args.expected_epoch}"
        )
    model.load_state_dict(checkpoint["model"])
    model.eval().requires_grad_(False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    latent_dir = args.latent_output_dir
    if args.cache_terminal_hidden:
        if latent_dir is None:
            raise ValueError("--latent-output-dir is required with --cache-terminal-hidden")
        latent_dir.mkdir(parents=True, exist_ok=True)
    initial_latent_dir = args.initial_latent_output_dir
    if initial_latent_dir is not None:
        if not args.cache_terminal_hidden:
            raise ValueError("--initial-latent-output-dir requires --cache-terminal-hidden")
        initial_latent_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = Path(data["shared_dir"])
    identity = {
        "generation_seed": args.generation_seed,
        "v4_config_sha256": sha256(args.config),
        "v4_checkpoint_sha256": sha256(args.checkpoint),
        "mesh_sha256": sha256(shared_dir / "mesh.npz"),
        "frame_manifest_sha256": sha256(shared_dir / "frame_manifest.json"),
    }
    trusted_existing_latents = False
    if latent_dir is not None and (latent_dir / "cache_manifest.json").exists():
        prior = json.loads((latent_dir / "cache_manifest.json").read_text())
        trusted_existing_latents = all(prior.get(key) == value for key, value in identity.items())
    audit_count = (
        args.fp16_audit_samples
        if args.fp16_audit_samples is not None
        else {"train": 24, "val": 3, "test": 3}[args.split]
    )
    audit_ids = (
        metadata_stratified_audit_ids(
            ids, Path(data["samples_dir"]), audit_count, args.generation_seed
        )
        if args.cache_terminal_hidden
        else set()
    )
    expected_audit_count = len(audit_ids)
    rows: list[dict[str, object]] = []
    fp16_errors: list[float] = []
    node_coords_norm = torch.from_numpy(dataset.node_coords_norm).unsqueeze(0).to(device)
    node_coords_world = torch.from_numpy(dataset.node_coords_world).unsqueeze(0).to(device)
    for index, sid in enumerate(ids):
        output_path = args.output_dir / f"{sid}.npy"
        latent_path = latent_dir / f"{sid}.npy" if latent_dir is not None else None
        initial_latent_path = (
            initial_latent_dir / f"{sid}.npy" if initial_latent_dir is not None else None
        )
        cached_state = None
        if output_path.exists():
            cached_state = validate_cache_array(
                output_path,
                shape=(len(dataset.node_coords_world),),
                dtype=np.dtype(np.float32),
            )
        if latent_path is not None and latent_path.exists() and trusted_existing_latents:
            validate_cache_array(
                latent_path,
                shape=(len(dataset.node_coords_world), int(cfg["model"]["hidden_dim"])),
                dtype=np.dtype(np.float16),
            )
        if initial_latent_path is not None and initial_latent_path.exists():
            validate_cache_array(
                initial_latent_path,
                shape=(len(dataset.node_coords_world), int(cfg["model"]["hidden_dim"])),
                dtype=np.dtype(np.float16),
            )
        if (
            output_path.exists()
            and (
                not args.cache_terminal_hidden
                or (
                    trusted_existing_latents
                    and latent_path is not None
                    and latent_path.exists()
                    and (initial_latent_path is None or initial_latent_path.exists())
                )
            )
            and sid not in audit_ids
        ):
            rows.append(
                {
                    "sample_id": sid,
                    "state_path": str(output_path.resolve()),
                    "state_shape": [len(dataset.node_coords_world)],
                    "state_dtype": "float32",
                    "state_sha256": sha256(output_path),
                    **(
                        {
                            "terminal_hidden_path": str(latent_path.resolve()),
                            "terminal_hidden_shape": [
                                len(dataset.node_coords_world),
                                int(cfg["model"]["hidden_dim"]),
                            ],
                            "terminal_hidden_dtype": "float16",
                            "terminal_hidden_sha256": sha256(latent_path),
                        }
                        if latent_path is not None
                        else {}
                    ),
                    **(
                        {
                            "initial_hidden_path": str(initial_latent_path.resolve()),
                            "initial_hidden_shape": [
                                len(dataset.node_coords_world),
                                int(cfg["model"]["hidden_dim"]),
                            ],
                            "initial_hidden_dtype": "float16",
                            "initial_hidden_sha256": sha256(initial_latent_path),
                        }
                        if initial_latent_path is not None
                        else {}
                    ),
                }
            )
            continue
        x_h = (
            torch.from_numpy(np.load(dataset.bridge_dir / sid / "coarse_d.npy").astype(np.float32))
            .unsqueeze(0)
            .to(device)
        )
        measurement_np, operator_scale = dataset.load_measurement_with_operator_scale(sid)
        measurement = torch.from_numpy(measurement_np).unsqueeze(0).to(device)
        measurement_operator_scale = torch.tensor(
            [operator_scale], device=device, dtype=torch.float32
        )
        encoded = None
        if model.view_encoder is not None:
            projections, _ = load_projection_stack(
                dataset.samples_dir / sid,
                projection_file=dataset.projection_file,
                projection_norm=dataset.projection_norm,
                projection_transform=dataset.projection_transform,
            )
            images = torch.from_numpy(projections[:, None]).unsqueeze(0).to(device)
            encoded = model.view_encoder.encode_images(images)
        output = model.correct_nodes(
            x_h,
            measurement,
            node_coords_norm,
            node_coords_world=node_coords_world,
            encoded_views=encoded,
            return_terminal_hidden=args.cache_terminal_hidden,
            return_hidden_trajectory=initial_latent_path is not None,
            measurement_operator_scale=measurement_operator_scale,
        )
        corrected = output["corrected_nodes"]
        corrected_np = corrected.squeeze(0).float().cpu().numpy()
        if cached_state is None:
            np.save(output_path, corrected_np)
        elif not np.array_equal(np.asarray(cached_state), corrected_np):
            raise RuntimeError(
                f"Recomputed frozen V4 state differs from the existing FP32 cache for {sid}"
            )
        row: dict[str, object] = {
            "sample_id": sid,
            "state_path": str(output_path.resolve()),
            "state_shape": list(corrected.shape[1:]),
            "state_dtype": "float32",
            "state_sha256": sha256(output_path),
        }
        if args.cache_terminal_hidden:
            assert latent_path is not None
            terminal = output["terminal_hidden"].squeeze(0).float().cpu().numpy()
            terminal_fp16, relative_l2 = fp16_relative_l2(terminal)
            np.save(latent_path, terminal_fp16)
            row.update(
                {
                    "terminal_hidden_path": str(latent_path.resolve()),
                    "terminal_hidden_shape": list(terminal_fp16.shape),
                    "terminal_hidden_dtype": "float16",
                    "terminal_hidden_sha256": sha256(latent_path),
                }
            )
            if sid in audit_ids:
                without_latent = model.correct_nodes(
                    x_h,
                    measurement,
                    node_coords_norm,
                    node_coords_world=node_coords_world,
                    encoded_views=encoded,
                    return_terminal_hidden=False,
                )["corrected_nodes"]
                if not torch.equal(corrected, without_latent):
                    raise RuntimeError(f"Terminal-hidden export changed V4 state for {sid}")
                fp16_errors.append(relative_l2)
                row["fp16_relative_l2"] = relative_l2
        if initial_latent_path is not None:
            initial = output["hidden_trajectory"][:, 0].squeeze(0).float().cpu().numpy()
            initial_fp16, initial_relative_l2 = fp16_relative_l2(initial)
            np.save(initial_latent_path, initial_fp16)
            row.update(
                {
                    "initial_hidden_path": str(initial_latent_path.resolve()),
                    "initial_hidden_shape": list(initial_fp16.shape),
                    "initial_hidden_dtype": "float16",
                    "initial_hidden_sha256": sha256(initial_latent_path),
                }
            )
            if sid in audit_ids:
                row["initial_fp16_relative_l2"] = initial_relative_l2
        rows.append(row)
        if (index + 1) % 25 == 0:
            print(f"[{args.split} V4 states {index + 1}/{len(ids)}]", flush=True)

    if fp16_errors and max(fp16_errors) > args.fp16_relative_l2_limit:
        raise RuntimeError(
            f"FP16 terminal-hidden error {max(fp16_errors):.6g} exceeds "
            f"{args.fp16_relative_l2_limit:.6g}"
        )
    metadata = {
        "split": args.split,
        "sample_count": len(rows),
        **identity,
        "v4_config": str(args.config.resolve()),
        "v4_checkpoint": str(args.checkpoint.resolve()),
        "v4_checkpoint_epoch": checkpoint.get("epoch"),
        "state_shape": [len(dataset.node_coords_world)],
        "state_dtype": "float32",
        "terminal_hidden_shape": (
            [len(dataset.node_coords_world), int(cfg["model"]["hidden_dim"])]
            if args.cache_terminal_hidden
            else None
        ),
        "terminal_hidden_dtype": "float16" if args.cache_terminal_hidden else None,
        "fp16_audit": {
            "strata": ["num_foci", "depth_tier", "source_type"],
            "requested_samples": audit_count,
            "expected_available_samples": expected_audit_count,
            "completed_samples": len(fp16_errors),
            "maximum_relative_l2": max(fp16_errors, default=None),
            "limit": args.fp16_relative_l2_limit,
            "passed": (
                len(fp16_errors) == expected_audit_count
                and expected_audit_count > 0
                and max(fp16_errors) <= args.fp16_relative_l2_limit
            ),
        },
        "latent_export_state_identity": (
            len(fp16_errors) == expected_audit_count and expected_audit_count > 0
        ),
        "initial_hidden_semantics": (
            "shared V4 correction-cell hidden after iteration 0, before its scalar update"
            if initial_latent_dir is not None
            else None
        ),
        "samples": rows,
    }
    metadata_path = (latent_dir or args.output_dir) / "cache_manifest.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    if initial_latent_dir is not None:
        initial_metadata = dict(metadata)
        initial_metadata["latent_role"] = "initial_correction_process_hidden"
        (initial_latent_dir / "cache_manifest.json").write_text(
            json.dumps(initial_metadata, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
