#!/usr/bin/env python3
"""Layer-wise negativity audit of the continuous-field reconstruction chain.

The FEM nodal reference state used by Stage 2 is an L2 projection,
``x_h* = Pi_h rho* = argmin_c ||P c - rho*||^2``. Because ``(P^T P)^-1`` has
negative off-diagonals (it is the discrete Green's function of the P1 mass
matrix), that reference carries ~50% negative nodal coefficients even though the
GT volume, the MCX measurement and the prolongation are all non-negative. The
P1 basis cannot express a sharp source, so the least-squares optimum overshoots
at the source and undershoots on its neighbours.

This script reports how that negativity propagates through

    L1  x_h*                      the reference state (node layer)
    L2  x_h^0 = Stage-1 coarse_d  the non-negative starting state (node layer)
    L2' x_h^c                     the corrected state (node layer)
    L3  rho_hat = P1(x_state)     the final voxel field

and measures what ``clip(rho_hat, 0)`` does to Dice / CCC / relative L2.

``L2'`` needs a trained corrector. Without one, pass ``--alphas`` to sweep the
proxy ``x_h^0 + a (x_h* - x_h^0)``, or ``--corrected-nodes-dir`` with one
``{sample_id}.npy`` per case to audit real corrected states.

Per layer it reports, exactly as requested: negative fraction, minimum value,
negative-mass ratio (|negative mass| / positive mass) and the negative excursion
relative to the positive peak.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.bridge.canonical_cross_discretization import (  # noqa: E402
    CanonicalCrossDiscretization,
)
from du2vox.evaluation.continuous_field import concordance_correlation  # noqa: E402


def node_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    negative = values < 0
    peak = float(values.max())
    positive_mass = float(values[values > 0].sum())
    negative_mass = float(-values[negative].sum())
    return {
        "neg_frac": float(negative.mean()),
        "min": float(values.min()),
        "neg_mass_ratio": negative_mass / positive_mass if positive_mass > 0 else float("nan"),
        "excursion_over_peak": (-float(values.min()) / peak) if peak > 0 else float("nan"),
        "mean_negative_square": float((np.minimum(values, 0.0) ** 2).mean()),
    }


def dice(first: np.ndarray, second: np.ndarray) -> float:
    first_count = int(np.count_nonzero(first))
    second_count = int(np.count_nonzero(second))
    if first_count + second_count == 0:
        return 1.0
    if first_count == 0 or second_count == 0:
        return 0.0
    return 2.0 * int(np.count_nonzero(first & second)) / (first_count + second_count)


def median(rows: list[dict], key: str) -> float:
    return float(np.median([row[key] for row in rows]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--operator-cache", type=Path, required=True)
    parser.add_argument("--projection-targets-dir", type=Path, required=True)
    parser.add_argument("--bridge-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="*",
        default=[0.25, 0.5, 0.75],
        help="Proxy interpolation fractions along x_h^0 -> x_h*",
    )
    parser.add_argument(
        "--corrected-nodes-dir",
        type=Path,
        help="Optional real corrected states, one {sample_id}.npy per case",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output", type=Path, help="Optional JSON receipt")
    args = parser.parse_args()

    ids = [
        line for line in (args.dataset_root / "splits" / f"{args.split}.txt").read_text().splitlines() if line
    ]
    if args.max_samples is not None:
        ids = ids[: args.max_samples]

    canonical = CanonicalCrossDiscretization(
        args.operator_cache, shared_dir=args.shared_dir, factorize=True
    )
    prolongation = canonical.p.tocsr()
    operator = canonical.operator
    valid = operator.valid_flat_indices

    system_matrix_path = Path(args.shared_dir) / "system_matrix.A.npz"
    system_matrix = np.asarray(
        np.load(system_matrix_path, allow_pickle=True)["forward_matrix"], dtype=np.float64
    )

    node_layers: dict[str, list[dict]] = {}
    voxel_layers: dict[str, list[dict]] = {}
    clip_rows: list[dict] = []
    consistency_rows: list[dict] = []

    def record(layer: str, store: dict, values: np.ndarray) -> None:
        store.setdefault(layer, []).append(node_stats(values))

    for index, sid in enumerate(ids, start=1):
        reference = np.load(args.projection_targets_dir / f"{sid}.npy").astype(np.float64)
        initial = np.load(args.bridge_dir / sid / "coarse_d.npy").astype(np.float64)
        measurement = np.load(args.dataset_root / "samples" / sid / "measurement_b.npy").astype(
            np.float64
        )
        with np.load(args.dataset_root / "samples" / sid / "gt_voxels.npz") as archive:
            indices = archive["indices"]
            values = archive["values"].astype(np.float64)
            shape = tuple(int(s) for s in archive["shape"])
        volume = np.zeros(int(np.prod(shape)))
        volume[indices] = values
        gt = volume[valid]

        record("L1_x_h_reference(node)", node_layers, reference)
        record("L2_x_h0_initial(node)", node_layers, initial)
        for alpha in args.alphas:
            record(f"L2p_x_hc_proxy(a={alpha:g})", node_layers, initial + alpha * (reference - initial))
        if args.corrected_nodes_dir is not None:
            corrected = np.load(args.corrected_nodes_dir / f"{sid}.npy").astype(np.float64)
            record("L2p_x_hc_trained(node)", node_layers, corrected)

        voxel_initial = np.asarray(prolongation @ initial).ravel()
        voxel_reference = np.asarray(prolongation @ reference).ravel()
        record("L3_rho_hat=P1(x_h0)", voxel_layers, voxel_initial)
        record("L3_rho_hat=P1(x_h*)", voxel_layers, voxel_reference)
        if args.corrected_nodes_dir is not None:
            corrected = np.load(args.corrected_nodes_dir / f"{sid}.npy").astype(np.float64)
            record("L3_rho_hat=P1(x_hc)", voxel_layers, np.asarray(prolongation @ corrected).ravel())

        threshold = 0.5 * float(gt.max())
        norm = float(np.linalg.norm(gt)) or 1e-30
        for tag, field in (("m0", voxel_initial), ("reference", voxel_reference)):
            clipped = np.clip(field, 0.0, None)
            clip_rows.append(
                {
                    "tag": tag,
                    "d_dice": dice(clipped >= threshold, gt >= threshold)
                    - dice(field >= threshold, gt >= threshold),
                    "d_ccc": concordance_correlation(clipped, gt)
                    - concordance_correlation(field, gt),
                    "d_relative_l2": float(np.linalg.norm(clipped - gt) / norm)
                    - float(np.linalg.norm(field - gt) / norm),
                    "mean_negative_square": float((np.minimum(field, 0.0) ** 2).mean()),
                }
            )

        residual = measurement - system_matrix @ initial
        direction = system_matrix.T @ residual
        consistency_rows.append(
            {
                "frac_direction_negative": float((direction < 0).mean()),
                # Raw masses rather than a per-case ratio: the per-case positive
                # mass can approach zero, so the ratio is extremely heavy-tailed
                # across cases. The pooled ratio is reported instead.
                "direction_negative_mass": float(-direction[direction < 0].sum()),
                "direction_positive_mass": float(direction[direction > 0].sum()),
                "frac_measurement_overshoot": float((residual < 0).mean()),
            }
        )
        if index % 50 == 0:
            print(f"[{index}/{len(ids)}] {sid}", flush=True)

    print(f"\nval samples: {len(ids)}")
    print("\n" + "=" * 104)
    print("LAYER NEGATIVITY")
    print("=" * 104)
    header = (
        f"{'layer':<28}{'neg_frac':>11}{'min':>12}{'neg_mass_ratio':>17}"
        f"{'excursion/peak':>16}{'mean(min(.,0)^2)':>19}"
    )
    print(header)
    print("-" * 104)
    for layer, rows in {**node_layers, **voxel_layers}.items():
        print(
            f"{layer:<28}{median(rows,'neg_frac'):>11.4f}{median(rows,'min'):>12.5f}"
            f"{median(rows,'neg_mass_ratio'):>17.5f}{median(rows,'excursion_over_peak'):>16.5f}"
            f"{median(rows,'mean_negative_square'):>19.3e}"
        )

    print("\n" + "=" * 104)
    print("CLIP IMPACT ON THE FINAL VOXEL FIELD")
    print("=" * 104)
    for tag, label in (("m0", "M0 P1(x_h0)"), ("reference", "P1(x_h*)")):
        subset = [row for row in clip_rows if row["tag"] == tag]
        print(
            f"{label:<16} dDice={median(subset,'d_dice'):+.8f}  dCCC={median(subset,'d_ccc'):+.6f}"
            f"  dRelL2={median(subset,'d_relative_l2'):+.6f}"
            f"  mean(min(rho,0)^2)={median(subset,'mean_negative_square'):.3e}"
        )

    print("\n" + "=" * 104)
    print("DATA-CONSISTENCY DIRECTION ADDED ON TOP OF x_h0")
    print("=" * 104)
    pooled_negative = float(np.sum([row["direction_negative_mass"] for row in consistency_rows]))
    pooled_positive = float(np.sum([row["direction_positive_mass"] for row in consistency_rows]))
    print(
        f"  fraction of nodes with A^T(y - A x_h) < 0 (pushes state down): "
        f"{median(consistency_rows,'frac_direction_negative'):.4f}"
    )
    print(
        f"  pooled negative/positive mass of that direction              : "
        f"{pooled_negative / max(pooled_positive, 1e-30):.4f}"
    )
    print(
        f"  fraction of measurements with A x_h > y (overshoot)          : "
        f"{median(consistency_rows,'frac_measurement_overshoot'):.4f}"
    )

    if args.output is not None:
        def summarize(rows: list[dict]) -> dict[str, float]:
            return {key: median(rows, key) for key in rows[0] if key != "tag"}

        payload = {
            "split": args.split,
            "n_samples": len(ids),
            "alphas": args.alphas,
            "corrected_nodes_dir": str(args.corrected_nodes_dir) if args.corrected_nodes_dir else None,
            "node_layers": {k: summarize(v) for k, v in node_layers.items()},
            "voxel_layers": {k: summarize(v) for k, v in voxel_layers.items()},
            "clip_impact": {
                tag: summarize([r for r in clip_rows if r["tag"] == tag])
                for tag in ("m0", "reference")
            },
            "data_consistency_direction": {
                **{
                    key: median(consistency_rows, key)
                    for key in (
                        "frac_direction_negative",
                        "frac_measurement_overshoot",
                        "direction_negative_mass",
                        "direction_positive_mass",
                    )
                },
                "pooled_negative_over_positive": pooled_negative / max(pooled_positive, 1e-30),
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
