#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from du2vox.bridge.roi_derivation import derive_roi, save_roi_results
from du2vox.utils.frame import FrameManifest


def load_split(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_op_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return {
        "name": cfg.get("name", Path(path).stem),
        "tau_active": float(cfg.get("tau_active", 0.5)),
        "tau_core": float(cfg.get("tau_core", 0.5)),
        "roi_dilation_layers": int(cfg.get("roi_dilation_layers", cfg.get("dilate_layers", 1))),
        "min_component_size": int(cfg.get("min_component_size", 0)),
    }


def apply_overrides(op: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = dict(op)
    for key in ["name", "tau_active", "tau_core", "roi_dilation_layers", "min_component_size"]:
        value = getattr(args, key)
        if value is not None:
            out[key] = value
    out["tau_active"] = float(out["tau_active"])
    out["tau_core"] = float(out["tau_core"])
    out["roi_dilation_layers"] = int(out["roi_dilation_layers"])
    out["min_component_size"] = int(out["min_component_size"])
    return out


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(os.path.relpath(src, start=dst.parent), dst)
    except OSError:
        shutil.copy2(src, dst)


def write_op_into_roi_info(path: Path, op: dict[str, Any]) -> None:
    with open(path) as f:
        info = json.load(f)
    info["stage1_operating_point"] = {
        "name": str(op["name"]),
        "tau_active": float(op["tau_active"]),
        "tau_core": float(op["tau_core"]),
        "roi_dilation_layers": int(op["roi_dilation_layers"]),
        "min_component_size": int(op["min_component_size"]),
    }
    with open(path, "w") as f:
        json.dump(info, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild Stage2 bridge ROI from existing Stage1 coarse_d")
    parser.add_argument("--input_bridge_dir", required=True)
    parser.add_argument("--output_bridge_dir", required=True)
    parser.add_argument("--shared_dir", required=True)
    parser.add_argument("--split_file", required=True)
    parser.add_argument("--op_config", required=True)
    parser.add_argument("--copy_coarse", action="store_true", help="Copy coarse_d.npy instead of symlinking")
    parser.add_argument("--name", default=None)
    parser.add_argument("--tau_active", type=float, default=None)
    parser.add_argument("--tau_core", type=float, default=None)
    parser.add_argument("--roi_dilation_layers", type=int, default=None)
    parser.add_argument("--min_component_size", type=int, default=None)
    args = parser.parse_args()

    op = apply_overrides(load_op_config(args.op_config), args)
    input_bridge_dir = Path(args.input_bridge_dir)
    output_bridge_dir = Path(args.output_bridge_dir)
    output_bridge_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = load_split(args.split_file)
    nodes, elements = FrameManifest.load_mesh_nodes(args.shared_dir)

    rows = []
    for i, sid in enumerate(sample_ids, 1):
        src = input_bridge_dir / sid / "coarse_d.npy"
        if not src.exists():
            print(f"[{i}/{len(sample_ids)}] {sid}: missing {src}, skip")
            continue
        out_dir = output_bridge_dir / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / "coarse_d.npy"
        link_or_copy(src, dst, args.copy_coarse)

        coarse_d = np.load(src).astype(np.float64)
        result = derive_roi(
            coarse_d,
            nodes,
            elements,
            tau=op["tau_active"],
            dilate_layers=op["roi_dilation_layers"],
            min_component_size=op["min_component_size"],
        )
        save_roi_results(result, out_dir)
        write_op_into_roi_info(out_dir / "roi_info.json", op)
        rows.append(result)
        if i == 1 or i % 25 == 0:
            print(
                f"[{i}/{len(sample_ids)}] {sid}: active={result['n_active_nodes']} "
                f"roi_tets={result['n_roi_tets']} ({result['roi_tet_ratio']:.3f})"
            )

    stats = {
        "stage1_operating_point": op,
        "n_samples": len(rows),
        "roi_tet_ratio_mean": float(np.mean([r["roi_tet_ratio"] for r in rows])) if rows else 0.0,
        "roi_tet_ratio_std": float(np.std([r["roi_tet_ratio"] for r in rows])) if rows else 0.0,
        "activation_ratio_mean": float(np.mean([r["activation_ratio"] for r in rows])) if rows else 0.0,
    }
    with open(output_bridge_dir / "bridge_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[Rebuild] wrote {output_bridge_dir}, samples={len(rows)}")


if __name__ == "__main__":
    main()
