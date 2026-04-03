#!/usr/bin/env python3
"""
Tecplot-style 3D visualization utilities.

Provides helpers for GT vs Prediction side-by-side rendering.
"""

import numpy as np
import pyvista as pv


def build_tet_mesh(nodes, elements, scalars=None, name="value"):
    """Build PyVista UnstructuredGrid from tet mesh."""
    n_tets = elements.shape[0]
    cells = np.hstack([
        np.full((n_tets, 1), 4, dtype=elements.dtype),
        elements,
    ]).ravel()
    celltypes = np.full(n_tets, pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(
        cells, celltypes, nodes.astype(np.float64)
    )
    if scalars is not None:
        grid.point_data[name] = scalars
    return grid


def get_body_wireframe(nodes, elements, tissue_labels):
    """Extract mouse body silhouette (non-background tissue), return wireframe surface."""
    elem_labels = np.zeros(elements.shape[0], dtype=int)
    for i, tet in enumerate(elements):
        vals, counts = np.unique(tissue_labels[tet], return_counts=True)
        elem_labels[i] = vals[counts.argmax()]

    mask = elem_labels > 0
    body_grid = build_tet_mesh(nodes, elements[mask])
    body_surf = body_grid.extract_surface()
    body_surf = body_surf.smooth(n_iter=30, relaxation_factor=0.1)
    return body_surf


def get_organ_surfaces(nodes, elements, tissue_labels, organ_ids):
    """Extract outer surfaces of specified organs."""
    elem_labels = np.zeros(elements.shape[0], dtype=int)
    for i, tet in enumerate(elements):
        vals, counts = np.unique(tissue_labels[tet], return_counts=True)
        elem_labels[i] = vals[counts.argmax()]

    surfaces = {}
    for oid in organ_ids:
        mask = elem_labels == oid
        if mask.sum() < 10:
            continue
        sub = build_tet_mesh(nodes, elements[mask])
        surf = sub.extract_surface().smooth(n_iter=20, relaxation_factor=0.1)
        if surf.n_points > 0:
            surfaces[oid] = surf
    return surfaces


# Organ color/opacity style map (id -> (RGB, opacity))
ORGAN_STYLE = {
    3: ((0.6, 0.8, 1.0), 0.08),   # lung - light blue
    4: ((0.9, 0.3, 0.3), 0.12),   # heart - light red
    5: ((0.55, 0.25, 0.15), 0.10), # liver - dark brown
    6: ((0.7, 0.4, 0.3), 0.10),   # kidney - brown
}

ISO_LEVELS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
CAMERA_ANGLES = [
    ("side",  "xz"),
    ("front", "yz"),
    ("top",   "xy"),
    ("iso",   [(80, 120, 60), (19, 50, 10), (0, 0, 1)]),
]
