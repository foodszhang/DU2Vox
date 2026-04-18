# FMT-SimGen × DU2Vox 坐标系规范 v1

## 0. 唯一的世界坐标系：`mcx_trunk_local_mm`

- **原点**：MCX volume 的 voxel(0,0,0) 角点
- **单位**：毫米 (mm)
- **轴向**：+X = 右, +Y = 后（tail 方向）, +Z = 背 (dorsal)
- **范围**：X∈[0, 38], Y∈[0, 40], Z∈[0, 20.8]
- **与 Digimouse atlas 的关系**：
  `world_mm = atlas_voxel * 0.1 − [0, 30, 0]`

所有以下数据**必须**在这个 frame 下：
mesh.nodes / gt_nodes 的隐式坐标 / gt_voxels offset / tumor_params.foci[i].center / MCX source position (只是换算成 voxel)。

## 1. MCX 体素 ↔ 世界坐标
- `voxel_xyz = world_mm / 0.2`（voxel_size = 0.2 mm, corner-aligned）
- MCX `.bin` 的 shape 是 ZYX = `[104, 200, 190]`（X-fastest）
- MCX source pos 传给 mcx 的值单位**始终是 voxel**，不管 LengthUnit

## 2. `frame_manifest.json` (output/shared/)
权威元数据，两边代码**只读不写**以外的地方都必须以它为准：

```json
{
  "version": 1,
  "world_frame": "mcx_trunk_local_mm",
  "atlas_to_world_offset_mm": [0, 30, 0],
  "mcx_volume": {
    "shape_xyz": [190, 200, 104],
    "voxel_size_mm": 0.2,
    "bbox_world_mm": {"min": [0,0,0], "max": [38.0, 40.0, 20.8]}
  },
  "fem_mesh": {
    "file": "mesh.npz",
    "frame": "mcx_trunk_local_mm",
    "n_nodes": 11535
  },
  "voxel_grid_gt": {
    "shape": [150, 150, 150],
    "spacing_mm": 0.2,
    "offset_world_mm": [4.0, 12.5, 0.4],
    "frame": "mcx_trunk_local_mm"
  }
}
```

## 3. 哪些数据是哪个 frame（修复前 vs 修复后）

| 数据 | 修复前 | 修复后 (v3) |
|---|---|---|
| mesh.npz `nodes` | atlas-corner mm (0–99 range) | **trunk-local mm** (0–40 range) |
| gt_nodes[i] | 对应 nodes[i] | 同 |
| gt_voxels + offset | atlas-corner mm | **trunk-local mm** |
| tumor_params.foci.center | atlas-corner mm | **trunk-local mm**（额外保留 `center_atlas_mm` 仅用于调试） |
| MCX source position | atlas-based 换算错 | `foci.center / 0.2`（voxel）|
| MCX `.jnii` 非零 bbox | trunk-local（本来就对） | 同 |

## 4. DU2Vox 侧使用规则

- **删除** `MCX_VOLUME_CENTER_WORLD` 常量。
- ViewEncoder 投影：`vox = world / 0.2`，world 直接用 mesh.nodes（修复后已 rebase）。
- `precompute_stage2_data.py`：GT 从 `gt_voxels.npy` 查表，offset 从 `frame_manifest.voxel_grid_gt.offset_world_mm` 读，不再用 FEM 插值。
- Stage 2 coords normalize: `(world - bbox_min) / (bbox_max - bbox_min) * 2 - 1`，bbox 从 `frame_manifest` 读（不要硬编码）。

## 5. 硬断言（CI 级别）

运行 `scripts/05_verify_frame_consistency.py`，所有 6 条必须通过：

1. `mesh.nodes.min() ≥ -1mm` & `mesh.nodes.max() ≤ [39, 41, 22]`（允许 1mm 容差）
2. `(mesh.nodes inside MCX bbox).mean() ≥ 0.65`（头尾在外是正常的，不到 65% 说明 rebase 错了）
3. 每个 sample: `all(foci.center ∈ MCX bbox)`（tumor 必须 100% 在 MCX 内）
4. `|foci.center - gt_voxels 非零重心| < 0.5mm`（以 offset 还原后）
5. MCX `.jnii` 非零 bbox 包含所有 foci.center（球半径内）
6. `proj.npz` 非零像素占比 ∈ [10%, 60%]