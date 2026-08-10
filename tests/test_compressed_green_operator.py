import hashlib
import json

import numpy as np
import pytest
import scipy.sparse as sp

from du2vox.physics.compressed_green_operator import apply_surface_convention
from du2vox.physics.compressed_green_operator import build_compressed_green_operator
from du2vox.utils import frame


def test_compressed_operator_identity_and_full_surface_convention():
    rng = np.random.default_rng(3)
    n_nodes = 9
    dense = rng.normal(size=(n_nodes, n_nodes))
    mass = sp.csr_matrix(dense.T @ dense + np.eye(n_nodes))
    source = sp.csr_matrix(rng.normal(size=(n_nodes, n_nodes)))
    surface = np.array([0, 2, 4, 7, 8])
    forward = np.linalg.solve(mass.toarray(), source.toarray())[surface]
    forward_used, surface_used, applied = apply_surface_convention(
        forward, surface, np.ones(len(surface), dtype=bool), False
    )
    operator = build_compressed_green_operator(
        mass, source, forward_used, surface_used, rank=3, oversampling=2, seed=9
    )
    assert not applied
    assert operator.relative_operator_error < 1e-10
    assert np.isfinite(operator.green_node_modes).all()


def test_visible_mask_convention_crops_rows_and_indices():
    forward = np.arange(20, dtype=np.float64).reshape(5, 4)
    surface = np.array([1, 3, 5, 7, 9])
    mask = np.array([True, False, True, False, True])
    cropped, indices, applied = apply_surface_convention(forward, surface, mask, True)
    assert applied
    np.testing.assert_array_equal(cropped, forward[mask])
    np.testing.assert_array_equal(indices, surface[mask])


def test_frame_manifest_sha256_certification(tmp_path, monkeypatch):
    manifest = {
        "world_frame": "mcx_trunk_local_mm",
        "mcx_volume": {"voxel_size_mm": 0.2},
        "frame_contract": {
            "voxel_size_mm": 0.2,
            "volume_extents_mm": [38.0, 40.0, 20.8],
            "grid_shape_xyz": [190, 200, 104],
        },
    }
    payload = json.dumps(manifest).encode()
    (tmp_path / "frame_manifest.json").write_bytes(payload)
    monkeypatch.setenv("DU2VOX_FRAME_MANIFEST_SHA256", "0" * 64)
    frame._CACHED_CONSTANTS = None
    with pytest.raises(RuntimeError, match="hash mismatch"):
        frame.get_frame_constants(tmp_path)

    monkeypatch.setenv(
        "DU2VOX_FRAME_MANIFEST_SHA256", hashlib.sha256(payload).hexdigest()
    )
    frame._CACHED_CONSTANTS = None
    constants = frame.get_frame_constants(tmp_path)
    assert constants["mcx_shape_xyz"] == (190, 200, 104)
