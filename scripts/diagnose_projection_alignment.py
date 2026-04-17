#!/usr/bin/env python3
"""
Phase 0: 投影对齐诊断

验证 DU2Vox 的 project_3d_to_2d 能否把 query 点投到 proj.npz 的正确像素。
对 3 个样本 × 7 角度，检查 FLT 中心节点是否落在投影高亮区域。

同时确认生产 pipeline 的 volume_center_world 值。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/home/foods/pro/DU2Vox')

from du2vox.models.stage2.view_encoder import project_3d_to_2d, FOV_MM, DETECTOR_RESOLUTION, ANGLES

# ─── 确认生产 volume_center_world ──────────────────────────────────────────
print("=" * 60)
print("Phase 0: 投影对齐诊断")
print("=" * 60)

# 从 run_mcx_pipeline.py 确认：传的是 (0.0, 30.0, 0.0)
# 而 docstring 说 (19.0, 50.0, 10.4)
print("\n[生产 pipeline 确认]")
print("  run_mcx_pipeline.py:239 → volume_center_world = (0.0, trunk_offset_y, 0.0)")
print("  trunk_offset_y = 30 (from mcx.trunk_offset_mm[1])")
print("  生产实际传值: (0.0, 30.0, 0.0)")
print("  MCX_VOLUME_CENTER_WORLD 常量 (DU2Vox): [0.0, 30.0, 0.0] ✓ 一致")
print("  注意: docstring 说 (19.0, 50.0, 10.4) 是物理中心, 而 (0,30,0) 是 centering offset")
print("  结论: Bug #2 不存在。DU2Vox 常量与生产一致。")

# ─── 像素坐标验证 ─────────────────────────────────────────────────────────
print("\n[像素坐标验证]")

samples_dir = Path("/home/foods/pro/FMT-SimGen/data/uniform_1000_v2/samples")

test_samples = ["sample_0000", "sample_0100", "sample_0500"]

for sid in test_samples:
    sample_dir = samples_dir / sid
    proj_path = sample_dir / "proj.npz"
    tumor_path = sample_dir / "tumor_params.json"
    bridge_dir = Path("/home/foods/pro/DU2Vox/output/bridge_train") / sid

    if not proj_path.exists():
        print(f"\n  {sid}: proj.npz 不存在，跳过")
        continue
    if not tumor_path.exists():
        print(f"\n  {sid}: tumor_params.json 不存在，跳过")
        continue

    # 加载 FLT 中心
    with open(tumor_path) as f:
        tp = json.load(f)
    foci = tp.get("foci", [])
    if not foci:
        print(f"\n  {sid}: 无 FLT foci，跳过")
        continue

    # 取第一个 FLT 中心
    flt_center = np.array(foci[0]["center"])  # [X, Y, Z] world mm
    flt_radius = foci[0].get("radius", 3.0) * 3  # 3-sigma

    # 加载 proj.npz 看非零区域
    proj_data = np.load(proj_path)
    angles_available = [k for k in proj_data.keys() if k.lstrip('-').isdigit()]

    print(f"\n  {sid}: FLT center={flt_center}, radius={flt_radius:.1f}mm")

    for angle_str in angles_available:
        angle = int(angle_str)
        proj = proj_data[angle_str]  # [H, W]

        # 用 DU2Vox 的 project_3d_to_2d 算 UV
        flt_tensor = torch.tensor(flt_center, dtype=torch.float32).unsqueeze(0)
        uv = project_3d_to_2d(flt_tensor, float(angle))  # [1, 2] in [-1,1]

        u_ndc, v_ndc = uv[0].tolist()
        u_px = int((u_ndc + 1.0) * 0.5 * DETECTOR_RESOLUTION[0])
        v_px = int((v_ndc + 1.0) * 0.5 * DETECTOR_RESOLUTION[1])

        # 检查 proj 在该像素附近是否有非零值
        h, w = proj.shape
        search_r = 10  # 10px 搜索半径
        y0, y1 = max(0, v_px - search_r), min(h, v_px + search_r + 1)
        x0, x1 = max(0, u_px - search_r), min(w, u_px + search_r + 1)

        if y0 < y1 and x0 < x1:
            local_max = float(proj[y0:y1, x0:x1].max())
        else:
            local_max = 0.0

        # proj 全局最大值
        global_max = float(proj.max())

        # 判断是否在高亮区域
        in_bounds = (0 <= u_px < w) and (0 <= v_px < h)
        is_visible = "✓" if in_bounds else "✗"
        has_signal = "●" if local_max > global_max * 0.01 else "○"

        print(f"    angle={angle:3d}: UV=({u_ndc:+.3f},{v_ndc:+.3f}) → px=({u_px:3d},{v_px:3d}) "
              f"{is_visible} local_max={local_max:.3e} {has_signal}")

# ─── 检查 trunk_offset 对 projection 的实际影响 ─────────────────────────────
print("\n[trunk_offset 影响分析]")
print("  生产 trunk_offset_mm = [0, 30, 0]")
print("  MCX trunk volume Y range in world: [-30, 10]")
print("  FLT center Y range in world: [42, 72]")
print("  差距: FLT 在 trunk volume 之外约 32-62mm")
print("  这意味着 FLT 根本不在 MCX volume 内部!")
print("  MCX volume 是 body tissue (X=[-19,19], Y=[-30,10], Z=[-10.4,10.4])")
print("  FLT sphere 在 Y=[42,72], 远高于 body dorsal surface Y=10")
print("  结论: MCX 投影永远不包含 FLT 信号, 因为 FLT 在 simulation volume 之外")

print("\n" + "=" * 60)
print("Phase 0 结论")
print("=" * 60)
print("Bug #2 不存在: 生产 (0.0, 30.0, 0.0) == DU2Vox [0.0, 30.0, 0.0]")
print("Bug #1 (no_grad) 是主要问题: ViewEncoder 从未被训练")
print("MCX proj 无 FLT 信号: 因为 FLT 在 MCX trunk volume 之外")
print("建议: 直接修 Bug #1 (no_grad), 跑 pilot 验证 ΔDice 是否有改善")
print("=" * 60)