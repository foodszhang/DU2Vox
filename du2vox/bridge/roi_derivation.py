"""
ROI derivation from Stage 1 coarse distribution.

Given coarse_d [N_nodes], thresholds active nodes, finds the union
of tetrahedra containing at least one active node, then dilates by
one layer (adds all tets sharing any vertex with an existing ROI tet).
"""

import json
from pathlib import Path

import numpy as np


def _build_node_to_tets(elements: np.ndarray, n_nodes: int) -> list[list[int]]:
    """Build node → list of tet indices mapping."""
    node_to_tets: list[list[int]] = [[] for _ in range(n_nodes)]
    for tet_idx, tet in enumerate(elements):
        for node in tet:
            node_to_tets[node].append(tet_idx)
    return node_to_tets


def derive_roi(
    coarse_d: np.ndarray,
    nodes: np.ndarray,
    elements: np.ndarray,
    tau: float = 0.5,
    dilate_layers: int = 1,
) -> dict:
    # Defensive frame check: detect if nodes arrive in wrong frame before
    # silent corruption propagates downstream. trunk-local max < 45mm.
    assert nodes.max() < 45, (
        f"derive_roi expects trunk-local nodes (max < 45mm), got max={nodes.max():.1f}. "
        f"Load via FrameManifest.load_mesh_nodes() to ensure correct frame."
    )
    """
    Derive ROI tetrahedra from a coarse distribution.

    Args:
        coarse_d: [N_nodes] Stage 1 output, values in [0, 1].
        nodes: [N_nodes, 3] node coordinates in mm.
        elements: [N_tets, 4] tetrahedral connectivity (node indices).
        tau: Activation threshold for coarse_d.
        dilate_layers: Number of dilation layers to expand the ROI.

    Returns:
        dict with keys:
            active_node_indices: ndarray — nodes with coarse_d > tau
            roi_tet_indices: ndarray — dilated ROI tet indices
            roi_bbox_mm: {"min": [x,y,z], "max": [x,y,z]}
            n_active_nodes: int
            n_roi_tets: int
            activation_ratio: float — active_nodes / total_nodes
            roi_tet_ratio: float — roi_tets / total_tets
    """
    n_nodes = nodes.shape[0]
    n_tets = elements.shape[0]

    # Step 1: active nodes
    active_mask = coarse_d > tau
    active_node_indices = np.where(active_mask)[0]
    n_active = len(active_node_indices)

    if n_active == 0:
        # Fallback: use top-1% nodes if nothing crosses tau
        k = max(1, int(0.01 * n_nodes))
        active_node_indices = np.argsort(coarse_d)[-k:]
        active_mask = np.zeros(n_nodes, dtype=bool)
        active_mask[active_node_indices] = True
        n_active = len(active_node_indices)
        print(f"  [ROI] Warning: no nodes above tau={tau:.2f}, using top-{k} nodes instead")

    # Step 2: build node → tets map
    node_to_tets = _build_node_to_tets(elements, n_nodes)

    # Step 3: initial ROI — tets containing at least one active node
    roi_tet_set: set[int] = set()
    for node_idx in active_node_indices:
        roi_tet_set.update(node_to_tets[node_idx])

    # Step 4: dilation
    for _ in range(dilate_layers):
        # Collect all nodes in current ROI tets
        roi_nodes: set[int] = set()
        for tet_idx in roi_tet_set:
            roi_nodes.update(elements[tet_idx].tolist())
        # Expand to all tets touching those nodes
        for node_idx in roi_nodes:
            roi_tet_set.update(node_to_tets[node_idx])

    roi_tet_indices = np.array(sorted(roi_tet_set), dtype=np.int64)

    # Step 5: ROI bounding box (mm)
    roi_node_indices = np.unique(elements[roi_tet_indices].ravel())
    roi_nodes_coords = nodes[roi_node_indices]
    bbox_min = roi_nodes_coords.min(axis=0).tolist()
    bbox_max = roi_nodes_coords.max(axis=0).tolist()

    return {
        "active_node_indices": active_node_indices,
        "roi_tet_indices": roi_tet_indices,
        "roi_bbox_mm": {"min": bbox_min, "max": bbox_max},
        "n_active_nodes": n_active,
        "n_roi_tets": len(roi_tet_indices),
        "activation_ratio": n_active / n_nodes,
        "roi_tet_ratio": len(roi_tet_indices) / n_tets,
    }


def save_roi_results(result: dict, output_dir: Path) -> None:
    """Save ROI derivation results to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "roi_tet_indices.npy", result["roi_tet_indices"])

    roi_info = {
        "n_active_nodes": int(result["n_active_nodes"]),
        "n_roi_tets": int(result["n_roi_tets"]),
        "activation_ratio": float(result["activation_ratio"]),
        "roi_tet_ratio": float(result["roi_tet_ratio"]),
        "roi_bbox_mm": {
            "min": [float(v) for v in result["roi_bbox_mm"]["min"]],
            "max": [float(v) for v in result["roi_bbox_mm"]["max"]],
        },
        "bbox_size_mm": [
            float(result["roi_bbox_mm"]["max"][i] - result["roi_bbox_mm"]["min"][i])
            for i in range(3)
        ],
    }
    with open(output_dir / "roi_info.json", "w") as f:
        json.dump(roi_info, f, indent=2)
