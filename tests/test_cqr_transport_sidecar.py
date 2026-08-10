import numpy as np

from scripts.precompute_cqr_transport_sidecar import build_sidecar


def test_proposal_negative_tet_is_not_mapped_to_tet_zero(tmp_path):
    cqr = {
        "tet_ids": np.array([-1, 0, 0], dtype=np.int64),
        "prior_8d": np.array(
            [
                [0, 0, 0, 0, 0.25, 0.25, 0.25, 0.25],
                [1, 0, 0, 0, 0.25, 0.25, 0.25, 0.25],
                [1, 0, 0, 0, 0.25, 0.25, 0.25, 0.25],
            ],
            dtype=np.float32,
        ),
        "role": np.array([4, 1, 1]),
    }
    result = build_sidecar(
        cqr,
        np.array([2.0, 3.0]),
        np.array([0]),
        sample_id="sample",
        source_path=tmp_path / "sample.npz",
    )
    assert not result["candidate_valid_physics_mask"][0]
    assert result["candidate_tet_volume"][0] == 0
    assert result["candidate_tet_query_count"][0] == 0
    assert result["candidate_cell_weight"][0] == 0
    np.testing.assert_allclose(result["candidate_cell_weight"][1:], 1.0)
