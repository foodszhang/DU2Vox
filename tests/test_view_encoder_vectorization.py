from __future__ import annotations

import os

os.environ.setdefault(
    "DU2VOX_SHARED_DIR", "/home/foods/pro/FMT-SimGen/output/shared_mesh_20k"
)
os.environ.setdefault("DU2VOX_ALLOW_STALE_FRAME_MANIFEST", "1")
# No DU2VOX_FRAME_MANIFEST_SHA256 pin here. This module only imports
# view_encoder to test its vectorized implementation against the sequential one;
# it does not validate shared-asset identity. A pinned hash in frame.py takes
# precedence over ALLOW_STALE and would abort pytest *collection* the moment the
# manifest is legitimately regenerated (as it was when the LPR cohort was built),
# failing the whole suite with an error unrelated to this test's subject. Asset
# certification belongs in a run config's own frame_manifest_sha256 field.

import torch
import torch.nn.functional as F

from du2vox.models.stage2.view_encoder import (
    ANGLES,
    ProjectAndSample,
    ViewEncoderModule,
    project_3d_to_2d,
)


def sequential_project_and_sample(
    coords: torch.Tensor, feature_maps: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    per_view = []
    visibility = []
    for view_index, angle in enumerate(ANGLES):
        uv = project_3d_to_2d(coords, angle, voxel_space=False)
        valid = (uv.abs() <= 1.0).all(dim=-1)
        sampled = F.grid_sample(
            feature_maps[:, view_index],
            uv.unsqueeze(2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.squeeze(-1).transpose(1, 2)
        per_view.append(sampled * valid.unsqueeze(-1))
        visibility.append(valid)
    return torch.stack(per_view, dim=2), torch.stack(visibility, dim=2)


def test_vectorized_project_and_sample_matches_sequential() -> None:
    generator = torch.Generator().manual_seed(7)
    coords = torch.rand((2, 19, 3), generator=generator)
    coords = coords * torch.tensor([38.0, 40.0, 20.8])
    feature_maps = torch.randn((2, 7, 5, 16, 16), generator=generator)
    expected, expected_visibility = sequential_project_and_sample(
        coords, feature_maps
    )
    actual, actual_visibility = ProjectAndSample()(coords, feature_maps)
    assert torch.equal(actual_visibility, expected_visibility)
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_batched_image_encoding_matches_sequential_in_eval_mode() -> None:
    generator = torch.Generator().manual_seed(11)
    module = ViewEncoderModule(
        view_feat_dim=8,
        encoder_out_channels=8,
        encoder_base_channels=8,
        multiscale_cfg={"enabled": True, "scales": ["s1", "s2", "s3"]},
    ).eval()
    images = torch.randn((2, 7, 1, 32, 32), generator=generator)
    with torch.inference_mode():
        expected_lists = {scale: [] for scale in module.multiscale_scales}
        for view_index in range(7):
            encoded = module.encoder(images[:, view_index], return_multiscale=True)
            for scale in module.multiscale_scales:
                expected_lists[scale].append(encoded[scale])
        expected = {
            scale: torch.stack(values, dim=1)
            for scale, values in expected_lists.items()
        }
        actual = module.encode_images(images)
    for scale in module.multiscale_scales:
        assert torch.allclose(actual[scale], expected[scale], atol=1e-6, rtol=1e-6)
