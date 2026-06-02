#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


ROLE_NAMES = {
    0: "bg",
    1: "core",
    2: "halo",
    3: "sentinel",
    4: "proposal",
}

TAG_NAMES = {
    0: "core",
    1: "halo",
    2: "bg",
    3: "proposal",
}


def dice_at(pred, gt, thr=0.5):
    p = pred >= thr
    g = gt >= thr
    return 2 * (p & g).sum() / (p.sum() + g.sum() + 1e-8)


def gt_pos(gt, mask, thr=0.5):
    if mask.sum() == 0:
        return 0.0
    return float((gt[mask] >= thr).mean())


def top_bottom_residual_stats(residual, gt, valid):
    if residual is None or valid.sum() == 0:
        return 0.0, 0.0
    rv = residual[valid]
    gv = gt[valid]
    top_cut = np.quantile(rv, 0.90)
    bottom_cut = np.quantile(rv, 0.50)
    top = rv >= top_cut
    bottom = rv <= bottom_cut
    top_gt = float((gv[top] >= 0.5).mean()) if top.any() else 0.0
    bottom_gt = float((gv[bottom] >= 0.5).mean()) if bottom.any() else 0.0
    return top_gt, bottom_gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--max_files", type=int, default=20)
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    paths = sorted(Path(args.dir).glob("*.npz"))[: args.max_files]
    if not paths:
        raise SystemExit(f"No npz files found in {args.dir}")

    all_rows = []
    for p in paths:
        d = np.load(p, allow_pickle=False)
        gt = d["gt_values"]
        prior = d["prior_ext"] if "prior_ext" in d.files else d["prior_8d"]
        role = d["role"] if "role" in d.files else np.zeros(len(gt), dtype=np.int64)
        query_src_tag = d["query_src_tag"] if "query_src_tag" in d.files else None
        correction_band = d["correction_band"] if "correction_band" in d.files else role
        prolongation_value = d["prolongation_value"] if "prolongation_value" in d.files else None
        correction_demand_score = d["correction_demand_score"] if "correction_demand_score" in d.files else None
        band_distance_score = d["band_distance_score"] if "band_distance_score" in d.files else None
        lift_fields = {
            name: d[name] if name in d.files else None
            for name in [
                "tet_grad_norm",
                "grad_jump_score",
                "recovery_error_score",
                "transition_score",
                "residual_indicator",
            ]
        }
        valid = d["valid_mask"].astype(bool)

        fem = (prior[:, :4] * prior[:, 4:8]).sum(axis=1)

        print(f"\n[{p.name}] valid={valid.sum()}/{len(valid)}")
        print(f"  gt_pos@0.5={((gt[valid] >= 0.5).mean() if valid.any() else 0):.4f}")
        print(f"  fem_pos@0.5={((fem[valid] >= 0.5).mean() if valid.any() else 0):.4f}")
        print(f"  fem_dice@0.5={dice_at(fem[valid], gt[valid], 0.5):.4f}")
        print(f"  gt_mean={gt[valid].mean():.4f}, fem_mean={fem[valid].mean():.4f}")
        if "prior_lift" in d.files:
            print(f"  prior_lift_dim={d['prior_lift'].shape[-1]}")
        print(f"  correction_band counts={np.bincount(correction_band[valid], minlength=5).tolist()}")
        residual = lift_fields["residual_indicator"]
        if residual is not None and valid.any():
            top_gt, bottom_gt = top_bottom_residual_stats(residual, gt, valid)
            halo = valid & (correction_band == 2)
            print(f"  residual_indicator top10 gt_pos@0.5={top_gt:.4f}")
            print(f"  residual_indicator bottom50 gt_pos@0.5={bottom_gt:.4f}")
            print(f"  halo_gt_pos={gt_pos(gt, halo):.4f}")
            proposal = valid & (correction_band == 4)
            print(f"  proposal_gt_pos={gt_pos(gt, proposal):.4f}")
            print(f"  halo_residual_mean={(residual[halo].mean() if halo.any() else 0):.4f}")
        if query_src_tag is not None:
            print("  query_src_tag stats:")
            for tag_id, tag_name in TAG_NAMES.items():
                m = query_src_tag == tag_id
                valid_m = m & valid
                residual = lift_fields["residual_indicator"]
                residual_mean = float(residual[valid_m].mean()) if residual is not None and valid_m.any() else 0.0
                print(
                    f"    tag={tag_name:8s} "
                    f"n={int(m.sum()):6d} "
                    f"valid_rate={(float(valid_m.sum()) / max(int(m.sum()), 1)):.4f} "
                    f"gt_pos@0.5={gt_pos(gt, valid_m):.4f} "
                    f"fem_pos@0.5={((fem[valid_m] >= 0.5).mean() if valid_m.any() else 0):.4f} "
                    f"residual_indicator_mean={residual_mean:.4f}"
                )
        for name, values in [
            ("prolongation_value", prolongation_value),
            ("correction_demand_score", correction_demand_score),
            ("band_distance_score", band_distance_score),
        ]:
            if values is not None and valid.any():
                print(f"  {name} mean={values[valid].mean():.4f}, std={values[valid].std():.4f}")
        for name, values in lift_fields.items():
            if values is not None and valid.any():
                print(f"  {name} mean={values[valid].mean():.4f}, std={values[valid].std():.4f}")

        for rid in [0, 1, 2, 3, 4]:
            m = valid & (role == rid)
            if m.sum() == 0:
                continue
            print(
                f"  role={ROLE_NAMES[rid]:8s} "
                f"n={m.sum():6d} "
                f"gt_pos={((gt[m] >= 0.5).mean()):.4f} "
                f"fem_pos={((fem[m] >= 0.5).mean()):.4f} "
                f"gt_mean={gt[m].mean():.4f} "
                f"fem_mean={fem[m].mean():.4f}"
            )
            bm = valid & (correction_band == rid)
            if bm.sum() > 0 and correction_demand_score is not None and prolongation_value is not None:
                print(
                    f"    band={ROLE_NAMES[rid]:8s} "
                    f"demand_mean={correction_demand_score[bm].mean():.4f} "
                    f"prolong_mean={prolongation_value[bm].mean():.4f}"
                )
            if bm.sum() > 0 and lift_fields["residual_indicator"] is not None:
                print(
                    f"    lift_band={ROLE_NAMES[rid]:8s} "
                    f"residual_mean={lift_fields['residual_indicator'][bm].mean():.4f} "
                    f"jump_mean={lift_fields['grad_jump_score'][bm].mean():.4f} "
                    f"transition_mean={lift_fields['transition_score'][bm].mean():.4f}"
                )

        all_rows.append(
            (
                gt[valid],
                fem[valid],
                role[valid],
                correction_band[valid],
                correction_demand_score[valid] if correction_demand_score is not None else None,
                prolongation_value[valid] if prolongation_value is not None else None,
                band_distance_score[valid] if band_distance_score is not None else None,
                {k: v[valid] if v is not None else None for k, v in lift_fields.items()},
                query_src_tag[valid] if query_src_tag is not None else None,
            )
        )

    gt_all = np.concatenate([r[0] for r in all_rows])
    fem_all = np.concatenate([r[1] for r in all_rows])
    role_all = np.concatenate([r[2] for r in all_rows])
    band_all = np.concatenate([r[3] for r in all_rows])
    demand_all = None if all_rows[0][4] is None else np.concatenate([r[4] for r in all_rows])
    prolong_all = None if all_rows[0][5] is None else np.concatenate([r[5] for r in all_rows])
    band_distance_all = None if all_rows[0][6] is None else np.concatenate([r[6] for r in all_rows])
    lift_all = {
        key: None if all_rows[0][7][key] is None else np.concatenate([row[7][key] for row in all_rows])
        for key in all_rows[0][7]
    }
    tag_all = None if all_rows[0][8] is None else np.concatenate([row[8] for row in all_rows])

    print("\n[Aggregate]")
    print(f"  n={len(gt_all)}")
    print(f"  gt_pos@0.5={(gt_all >= 0.5).mean():.4f}")
    print(f"  fem_pos@0.5={(fem_all >= 0.5).mean():.4f}")
    print(f"  fem_dice@0.5={dice_at(fem_all, gt_all, 0.5):.4f}")
    print(f"  correction_band counts={np.bincount(band_all, minlength=5).tolist()}")
    if lift_all["residual_indicator"] is not None:
        residual_all = lift_all["residual_indicator"]
        top_gt, bottom_gt = top_bottom_residual_stats(
            residual_all,
            gt_all,
            np.ones(len(gt_all), dtype=bool),
        )
        halo_all = band_all == 2
        proposal_all = band_all == 4
        print(f"  residual_indicator top10 gt_pos@0.5={top_gt:.4f}")
        print(f"  residual_indicator bottom50 gt_pos@0.5={bottom_gt:.4f}")
        print(f"  halo_gt_pos={gt_pos(gt_all, halo_all):.4f}")
        print(f"  proposal_gt_pos={gt_pos(gt_all, proposal_all):.4f}")
        print(f"  halo_residual_mean={(residual_all[halo_all].mean() if halo_all.any() else 0):.4f}")
    for name, values in [
        ("correction_demand_score", demand_all),
        ("prolongation_value", prolong_all),
        ("band_distance_score", band_distance_all),
    ]:
        if values is not None:
            print(f"  {name} mean={values.mean():.4f}, std={values.std():.4f}")
    for name, values in lift_all.items():
        if values is not None:
            print(f"  {name} mean={values.mean():.4f}, std={values.std():.4f}")
    print("  threshold sweep for FEM:")
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        print(f"    thr={thr:.1f} dice={dice_at(fem_all, gt_all, thr):.4f}")

    if tag_all is not None:
        print("\n  query_src_tag aggregate:")
        for tag_id, tag_name in TAG_NAMES.items():
            m = tag_all == tag_id
            if not m.any():
                continue
            residual_all = lift_all["residual_indicator"]
            residual_mean = float(residual_all[m].mean()) if residual_all is not None else 0.0
            print(
                f"    tag={tag_name:8s} "
                f"n={int(m.sum()):7d} "
                f"valid_rate=1.0000 "
                f"gt_pos@0.5={float((gt_all[m] >= 0.5).mean()):.4f} "
                f"fem_pos@0.5={float((fem_all[m] >= 0.5).mean()):.4f} "
                f"residual_indicator_mean={residual_mean:.4f}"
            )

    print("\n  role aggregate:")
    rows = [
        {
            "role": "all",
            "n": len(gt_all),
            "gt_pos_05": float((gt_all >= 0.5).mean()),
            "fem_pos_05": float((fem_all >= 0.5).mean()),
            "gt_mean": float(gt_all.mean()),
            "fem_mean": float(fem_all.mean()),
            "fem_dice_05": float(dice_at(fem_all, gt_all, 0.5)),
        }
    ]
    for rid in [0, 1, 2, 3, 4]:
        m = role_all == rid
        if m.sum() == 0:
            continue
        rows.append(
            {
                "role": ROLE_NAMES[rid],
                "n": int(m.sum()),
                "gt_pos_05": float((gt_all[m] >= 0.5).mean()),
                "fem_pos_05": float((fem_all[m] >= 0.5).mean()),
                "gt_mean": float(gt_all[m].mean()),
                "fem_mean": float(fem_all[m].mean()),
                "fem_dice_05": float(dice_at(fem_all[m], gt_all[m], 0.5)),
            }
        )
        print(
            f"    {ROLE_NAMES[rid]:8s} "
            f"n={m.sum():7d} "
            f"gt_pos={(gt_all[m] >= 0.5).mean():.4f} "
            f"fem_pos={(fem_all[m] >= 0.5).mean():.4f} "
            f"gt_mean={gt_all[m].mean():.4f} "
            f"fem_mean={fem_all[m].mean():.4f}"
        )
        bm = band_all == rid
        if bm.sum() > 0 and demand_all is not None and prolong_all is not None:
            print(
                f"      band demand_mean={demand_all[bm].mean():.4f} "
                f"prolong_mean={prolong_all[bm].mean():.4f}"
            )
        if bm.sum() > 0 and lift_all["residual_indicator"] is not None:
            print(
                f"      lift residual_mean={lift_all['residual_indicator'][bm].mean():.4f} "
                f"jump_mean={lift_all['grad_jump_score'][bm].mean():.4f} "
                f"transition_mean={lift_all['transition_score'][bm].mean():.4f}"
            )

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["role", "n", "gt_pos_05", "fem_pos_05", "gt_mean", "fem_mean", "fem_dice_05"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[Diagnose] wrote {out_path}")


if __name__ == "__main__":
    main()
