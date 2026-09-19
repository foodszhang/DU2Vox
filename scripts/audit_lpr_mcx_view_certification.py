#!/usr/bin/env python3
"""Gate 1: certify the reconstructed LPR MCX view set.

Checks every ``proj.npz`` in the cohort for the frozen seven-view contract:
expected angle keys, 256x256 shape, all-finite values, non-negativity, and a
non-degenerate signal. Also cross-checks the ``.jnii`` count and the MCX/Projection
failure log, distinguishing genuine failures from the known pause artefacts.

This is validation only: it reads artefacts and writes a JSON receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EXPECTED_ANGLES = ["-90", "-60", "-30", "0", "30", "60", "90"]
EXPECTED_SHAPE = (256, 256)
# MCX view noise realisations that were truncated by a WSL-side SIGSTOP while the
# Windows MCX child kept running; the .jnii and proj.npz are complete and valid.
KNOWN_PAUSE_ARTEFACTS = {"sample_0444", "sample_1870"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default=None, help="Restrict to one split's ids")
    parser.add_argument("--log", type=Path, help="MCX pipeline log to scan for failures")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    samples_dir = args.dataset_root / "samples"
    if args.split:
        ids = [
            line
            for line in (args.dataset_root / "splits" / f"{args.split}.txt").read_text().splitlines()
            if line
        ]
    else:
        ids = sorted(p.name for p in samples_dir.glob("sample_*") if p.is_dir())

    print(f"auditing {len(ids)} cases under {samples_dir}")

    missing_jnii: list[str] = []
    missing_proj: list[str] = []
    bad_keys: list[dict] = []
    bad_shape: list[dict] = []
    non_finite: list[str] = []
    negative: list[str] = []
    degenerate: list[str] = []
    per_case: list[dict] = []

    for index, sid in enumerate(ids, start=1):
        sample_dir = samples_dir / sid
        jnii = sample_dir / f"{sid}.jnii"
        proj = sample_dir / "proj.npz"
        if not jnii.exists():
            missing_jnii.append(sid)
        if not proj.exists():
            missing_proj.append(sid)
            continue

        with np.load(proj) as archive:
            keys = list(archive.files)
            arrays = {key: archive[key] for key in keys}

        if sorted(keys) != sorted(EXPECTED_ANGLES):
            bad_keys.append({"id": sid, "keys": keys})
            continue

        shapes = {key: tuple(value.shape) for key, value in arrays.items()}
        if any(shape != EXPECTED_SHAPE for shape in shapes.values()):
            bad_shape.append({"id": sid, "shapes": shapes})
            continue

        finite = all(bool(np.isfinite(value).all()) for value in arrays.values())
        if not finite:
            non_finite.append(sid)
        if any(bool((value < 0).any()) for value in arrays.values()):
            negative.append(sid)

        # Signal must be non-degenerate in the most-visible views; a zero view
        # means the projection silently produced nothing.
        maxes = [float(value.max()) for value in arrays.values()]
        non_zero = [
            float(np.count_nonzero(value)) / value.size for value in arrays.values()
        ]
        if min(maxes) <= 0.0 or min(non_zero) < 0.01:
            degenerate.append(sid)
        per_case.append(
            {
                "id": sid,
                "view_max": maxes,
                "view_nonzero_fraction": non_zero,
            }
        )

        if index % 250 == 0:
            print(f"  [{index}/{len(ids)}]", flush=True)

    view_max_array = np.asarray([row["view_max"] for row in per_case]) if per_case else np.zeros((0, 7))
    nonzero_array = (
        np.asarray([row["view_nonzero_fraction"] for row in per_case]) if per_case else np.zeros((0, 7))
    )

    failure_scan: dict[str, object] = {}
    if args.log and args.log.exists():
        mcx_failures: list[str] = []
        projection_failures: list[str] = []
        for line in args.log.read_text().splitlines():
            if "MCX " in line and "FAIL" in line:
                for token in line.split():
                    if token.startswith("sample_"):
                        mcx_failures.append(token.rstrip(":"))
            if "Projection" in line and "FAIL" in line:
                projection_failures.append(line)
        failure_scan = {
            "log": str(args.log),
            "mcx_failures": sorted(set(mcx_failures)),
            "projection_failures": projection_failures,
            "mcx_failures_explained_as_pause_artefacts": sorted(
                set(mcx_failures) & KNOWN_PAUSE_ARTEFACTS
            ),
            "mcx_failures_unexplained": sorted(set(mcx_failures) - KNOWN_PAUSE_ARTEFACTS),
        }

    payload = {
        "gate": "1",
        "name": "lpr_mcx_view_certification",
        "dataset_root": str(args.dataset_root.resolve()),
        "n_requested": len(ids),
        "n_audited": len(per_case),
        "view_contract": {
            "angles": EXPECTED_ANGLES,
            "shape": list(EXPECTED_SHAPE),
            "dtype": "float32",
        },
        "counts": {
            "jnii_present": len(ids) - len(missing_jnii),
            "proj_present": len(ids) - len(missing_proj),
            "keys_ok": len(per_case),
        },
        "violations": {
            "missing_jnii": missing_jnii,
            "missing_proj": missing_proj,
            "bad_angle_keys": bad_keys,
            "bad_shape": bad_shape,
            "non_finite": non_finite,
            "negative_values": negative,
            "degenerate_views": degenerate,
        },
        "signal_stats": {
            "view_max": {
                "min": float(view_max_array.min()) if view_max_array.size else None,
                "median": float(np.median(view_max_array)) if view_max_array.size else None,
                "max": float(view_max_array.max()) if view_max_array.size else None,
            },
            "view_nonzero_fraction": {
                "min": float(nonzero_array.min()) if nonzero_array.size else None,
                "median": float(np.median(nonzero_array)) if nonzero_array.size else None,
                "max": float(nonzero_array.max()) if nonzero_array.size else None,
            },
        },
        "failure_scan": failure_scan,
        "artefact_hashes": {
            "frame_manifest.json": sha256(Path("/home/foods/pro/FMT-SimGen/output/shared_mesh_20k/frame_manifest.json")),
            "mesh.npz": sha256(Path("/home/foods/pro/FMT-SimGen/output/shared_mesh_20k/mesh.npz")),
        },
    }

    passed = (
        not missing_jnii
        and not missing_proj
        and not bad_keys
        and not bad_shape
        and not non_finite
        and not negative
        and not degenerate
        and not failure_scan.get("mcx_failures_unexplained")
        and not failure_scan.get("projection_failures", [])
    )
    payload["status"] = "passed" if passed else "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    print()
    print(f"jnii present      : {payload['counts']['jnii_present']}/{len(ids)}")
    print(f"proj present      : {payload['counts']['proj_present']}/{len(ids)}")
    print(f"angle keys ok     : {payload['counts']['keys_ok']}/{len(ids)}")
    print(f"missing jnii      : {len(missing_jnii)}")
    print(f"missing proj      : {len(missing_proj)}")
    print(f"bad angle keys    : {len(bad_keys)}")
    print(f"bad shape         : {len(bad_shape)}")
    print(f"non-finite        : {len(non_finite)}")
    print(f"negative values   : {len(negative)}")
    print(f"degenerate views  : {len(degenerate)}")
    if failure_scan:
        print(f"MCX failures      : {failure_scan['mcx_failures']}")
        print(f"  explained       : {failure_scan['mcx_failures_explained_as_pause_artefacts']}")
        print(f"  unexplained     : {failure_scan['mcx_failures_unexplained']}")
        print(f"Projection fails  : {len(failure_scan['projection_failures'])}")
    print()
    print(f"STATUS: {payload['status'].upper()}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
