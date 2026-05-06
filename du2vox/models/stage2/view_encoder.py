"""
Stage 2 View Encoder: 2D U-Net encoder for MCX multi-view projections
+ differentiable projection + multi-view fusion.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from du2vox.utils.frame import get_frame_constants

# ─── Projection constants (from FMT-SimGen view_config + mcx_config) ────────
CAMERA_DISTANCE_MM = 200.0
FOV_MM = 80.0
DETECTOR_RESOLUTION = (256, 256)  # (width, height) in pixels

# MCX trunk volume geometry — derived from frame_manifest.json (U6 audit):
#   volume_shape: [Z=104, Y=200, X=190] after 2× downsample
#   voxel_size_mm: 0.2
#
# Trunk-local frame (mcx_trunk_local_mm): MCX corner at origin (0,0,0).
# Physical volume center = half_extents in trunk-local mm.
_FRAME = get_frame_constants()
VOXEL_SIZE_MM = _FRAME["voxel_size_mm"]
MCX_HALF_EXTENTS = _FRAME["mcx_half_extents"]
MCX_VOLUME_CENTER_WORLD = _FRAME["mcx_volume_center"]

ANGLES = [-90, -60, -30, 0, 30, 60, 90]


# ─── Projection ───────────────────────────────────────────────────────────────

def project_3d_to_2d(
    points: torch.Tensor,
    angle_deg: float,
    voxel_space: bool = False,
) -> torch.Tensor:
    """
    Project 3D coordinates to normalized UV coordinates on a single detector view,
    for grid_sample.

    Two modes:

    World-space mode (voxel_space=False):
      Input is trunk-local mm (MCX corner origin). Subtracts MCX_VOLUME_CENTER_WORLD
      so that rotation happens around the physical volume center — matching the
      MCX TurntableCamera geometry.
      1. World → detector: shift by -MCX_VOLUME_CENTER_WORLD
      2. Rotate around Y axis by angle_deg
      3. Orthographic: detector (u,v) = (X_rot, Y_rot) in mm
      4. Normalize to [-1, 1]:  u_ndc = X_rot / (FOV/2),  v_ndc = Y_rot / (FOV/2)

    Voxel-space mode (voxel_space=True):
      Coords are MCX-volume NDC coordinates from FrameManifest.world_to_mcx_ndc.
      Convert back to centered mm before applying the same FOV normalization used
      by FMT-SimGen.

    Parameters
    ----------
    points : torch.Tensor [..., 3]
        3D coordinates. World mm if voxel_space=False, voxel indices if voxel_space=True.
    angle_deg : float
        Camera rotation angle in degrees.
    voxel_space : bool
        If True, input is MCX-volume NDC and is converted back to centered mm.
        If False, use world-space with FOV normalization.

    Returns
    -------
    torch.Tensor [..., 2]
        Normalized UV coordinates in [-1, 1] for F.grid_sample.
    """
    half_fov = FOV_MM / 2.0

    if voxel_space:
        # Voxel-space: convert MCX-volume NDC back to centered physical mm.
        # FMT-SimGen's proj.npz is normalized by detector FOV, not by volume bbox.
        hx, hy, hz = MCX_HALF_EXTENTS
        x_c = points[..., 0] * hx
        y_c = points[..., 1] * hy
        z_c = points[..., 2] * hz
    else:
        # World-space: trunk-local mm, MCX corner at (0,0,0).
        # Shift to volume-center frame BEFORE rotating so rotation happens around
        # the volume center — matching MCX TurntableCamera geometry.
        cx, cy, cz = MCX_VOLUME_CENTER_WORLD
        x_c = points[..., 0] - cx
        y_c = points[..., 1] - cy
        z_c = points[..., 2] - cz

    # Rotate around Y axis (column-vector convention: new = old @ R.T)
    angle_rad = torch.deg2rad(torch.tensor(angle_deg, device=points.device, dtype=points.dtype))
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)

    # R.T for (x, y, z) @ R.T:
    #   x' = x*cos + z*sin
    #   y' = y
    #   z' = -x*sin + z*cos
    x_rot = x_c * cos_a + z_c * sin_a
    y_rot = y_c  # orthographic: Y stays Y

    # Normalize to [-1, 1]
    u = x_rot / half_fov
    v = y_rot / half_fov

    return torch.stack([u, v], dim=-1)


def world_to_detector_px(
    points: torch.Tensor,
    angle_deg: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Project 3D world coordinates to detector pixel indices.

    Returns (u_px, v_px) in [0, width-1] x [0, height-1] float pixels,
    and a visibility mask (True if within FOV).
    """
    half_fov = FOV_MM / 2.0
    w, h = DETECTOR_RESOLUTION
    pixel_size = FOV_MM / w  # = 0.1953 mm/px

    # Get UV in mm
    uv_mm = project_3d_to_2d(points, angle_deg) * half_fov  # [..., 2]

    # Convert to pixel indices
    u_px = (uv_mm[..., 0] + half_fov) / pixel_size  # [0, w]
    v_px = (uv_mm[..., 1] + half_fov) / pixel_size  # [0, h]

    # FOV clipping mask
    visible = (u_px >= 0) & (u_px < w) & (v_px >= 0) & (v_px < h)

    return u_px, v_px, visible


# ─── View Encoder (2D U-Net) ─────────────────────────────────────────────────

class ViewEncoder(nn.Module):
    """
    Lightweight 2D U-Net encoder for MCX projection images.

    Input:  [B, 1, 256, 256] — single-channel projection image
    Output: [B, feat_dim, H', W'] — feature map at reduced resolution

    Design choices:
    - Single channel input (MCX projections are fluence, single-channel)
    - 4× downsampled output (64×64) to reduce memory + capture global context
    - Skip connections for multi-scale feature aggregation
    - No skip at the bottleneck (too small benefit vs complexity)
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        n_downsample: int = 2,
        out_channels: int = 32,
    ):
        super().__init__()

        # Encoder: 1 → 32 → 64
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)

        ch2 = base_channels * 2
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, ch2, 3, padding=1),
            nn.BatchNorm2d(ch2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch2, ch2, 3, padding=1),
            nn.BatchNorm2d(ch2),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        ch3 = base_channels * 4
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ch2, ch3, 3, padding=1),
            nn.BatchNorm2d(ch3),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3, ch3, 3, padding=1),
            nn.BatchNorm2d(ch3),
            nn.ReLU(inplace=True),
        )

        # Decoder: 64 → 32 → out_channels
        self.up2 = nn.ConvTranspose2d(ch3, ch2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(ch2 + ch2, ch2, 3, padding=1),
            nn.BatchNorm2d(ch2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch2, ch2, 3, padding=1),
            nn.BatchNorm2d(ch2),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(ch2, base_channels, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels + base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Final projection to output channels
        self.final = nn.Conv2d(base_channels, out_channels, 1)

        self.feat_dim = out_channels
        self.h_out = 256 // (2 ** n_downsample)  # 64
        self.w_out = 256 // (2 ** n_downsample)  # 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, 256, 256] → out: [B, feat_dim, 64, 64]
        """
        # Encoder
        e1 = self.enc1(x)          # [B, 32, 256, 256]
        p1 = self.pool1(e1)        # [B, 32, 128, 128]

        e2 = self.enc2(p1)         # [B, 64, 128, 128]
        p2 = self.pool2(e2)        # [B, 64, 64, 64]

        # Bottleneck
        b = self.bottleneck(p2)    # [B, 128, 64, 64]

        # Decoder
        d2 = self.up2(b)           # [B, 64, 128, 128]
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)         # [B, 64, 128, 128]

        d1 = self.up1(d2)          # [B, 32, 256, 256]
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)         # [B, 32, 256, 256]

        out = self.final(d1)       # [B, feat_dim, 256, 256]

        # Downsample 4× if needed (n_downsample=2 means 64×64 output)
        if out.shape[-1] != self.w_out:
            out = F.adaptive_avg_pool2d(out, (self.h_out, self.w_out))

        return out


# ─── ProjectAndSample ────────────────────────────────────────────────────────

class ProjectAndSample(nn.Module):
    """
    Project 3D query points onto 7 view feature maps and sample per-view features.

    Supports two projection modes:
    - world-space: uses FOV-based normalization (legacy)
    - voxel-space: uses physical volume half-extents (Phase 3, preserves aspect ratio)
    """

    def __init__(self, feat_map_size: int = 64):
        super().__init__()
        self.feat_map_size = feat_map_size
        self.angles = ANGLES

    def forward(
        self,
        coords_world: torch.Tensor,
        feat_maps: torch.Tensor,
        coords_vox_norm: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        coords_world : torch.Tensor [B, N, 3]
            Query point world coordinates in mm (detector-centered).
            Used for world-space projection when coords_vox_norm is None.
        feat_maps : torch.Tensor [B, 7, C, H', W']
            Encoded feature maps for each of the 7 views.
        coords_vox_norm : torch.Tensor [B, N, 3] or None
            Voxel-space normalized coordinates (centered by physical volume center,
            normalized by half-extents). If provided, uses voxel-space projection.

        Returns
        -------
        multi_view_feat : torch.Tensor [B, N, 7, C]
            Per-query, per-view features. Points outside FOV get zero features.
        visibility : torch.Tensor [B, N, 7]
            Bool mask: True if point is visible in that view (within FOV).
        """
        B, N = coords_world.shape[:2]
        n_views = len(self.angles)
        C = feat_maps.shape[2]

        use_voxel = coords_vox_norm is not None

        multi_view_feat = torch.zeros(B, N, n_views, C, device=feat_maps.device, dtype=feat_maps.dtype)
        visibility = torch.zeros(B, N, n_views, device=feat_maps.device, dtype=torch.bool)

        for view_idx, angle in enumerate(self.angles):
            if use_voxel:
                # Voxel-space projection: coords already centered + normalized
                uv = project_3d_to_2d(coords_vox_norm, angle, voxel_space=True)
            else:
                # World-space projection
                uv = project_3d_to_2d(coords_world, angle, voxel_space=False)

            # Check visibility (within FOV, clamping for voxel-space)
            valid = (uv.abs() <= 1.0).all(dim=-1)

            # Normalize UV for grid_sample: need [B, N, 1, 2] format
            uv_grid = uv.unsqueeze(2)

            # Sample from feat_maps[:, view_idx]: [B, C, H', W']
            view_feat = F.grid_sample(
                feat_maps[:, view_idx],
                uv_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )  # → [B, C, N, 1]

            view_feat = view_feat.squeeze(-1).transpose(1, 2)  # [B, N, C]

            # Zero out features for out-of-FOV points
            view_feat = view_feat * valid.unsqueeze(-1).float()

            multi_view_feat[:, :, view_idx] = view_feat
            visibility[:, :, view_idx] = valid

        return multi_view_feat, visibility


# ─── Multi-View Fusion ────────────────────────────────────────────────────────

class MultiViewFusion(nn.Module):
    """
    Fuse per-view features into a single fused representation.

    Methods:
      - "mean":     simple average of visible views
      - "attn":     learned attention (one attention head per query)
    """

    def __init__(self, feat_dim: int, method: str = "attn"):
        super().__init__()
        self.method = method

        if method == "attn":
            # Simple single-head attention
            self.attn = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim // 2, 1),
            )
        elif method == "mean":
            pass
        else:
            raise ValueError(f"Unknown fusion method: {method}")

    def forward(
        self,
        multi_view_feat: torch.Tensor,
        visibility: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        multi_view_feat : torch.Tensor [B, N, 7, C]
        visibility : torch.Tensor [B, N, 7] bool

        Returns
        -------
        fused : torch.Tensor [B, N, C]
        """
        if self.method == "mean":
            # Set invisible views to 0 and average over all 7 views
            masked = multi_view_feat.clone()
            masked[~visibility] = 0.0
            # Divide by count of visible views (add 1e-8 to avoid div by zero)
            count = visibility.sum(dim=-1, keepdim=True).float()  # [B, N, 1]
            fused = masked.sum(dim=-2) / (count + 1e-8)             # [B, N, C]
            return fused

        elif self.method == "attn":
            B, N, n_views, C = multi_view_feat.shape

            # Attention scores per view
            # multi_view_feat: [B, N, 7, C] → [B*N*7, C]
            flat = multi_view_feat.reshape(B * N * n_views, C)
            scores = self.attn(flat)  # [B*N*7, 1]
            scores = scores.reshape(B, N, n_views)  # [B, N, 7]

            # Mask invisible views with large negative
            scores = scores.masked_fill(~visibility, -1e9)
            attn_w = F.softmax(scores, dim=-1)  # [B, N, 7]

            # Weighted sum
            fused = (multi_view_feat * attn_w.unsqueeze(-1)).sum(dim=-2)  # [B, N, C]
            return fused


# ─── Full ViewEncoderModule ───────────────────────────────────────────────────

class ViewEncoderModule(nn.Module):
    """
    Complete view encoding pipeline:
      1. Encode 7 projection images → 7 feature maps (shared encoder)
      2. Project query points to each view → per-view features
      3. Fuse → fused view features

    Parameters
    ----------
    view_feat_dim : int
        Dimension of output fused features per query point (default 32).
    fusion_method : str
        "mean" or "attn" (default "attn").
    encoder_out_channels : int
        Channels in encoder output feature maps (default 32).
    encoder_base_channels : int
        Base channels in encoder (default 32).
    """

    def __init__(
        self,
        view_feat_dim: int = 32,
        fusion_method: str = "attn",
        encoder_out_channels: int = 32,
        encoder_base_channels: int = 32,
    ):
        super().__init__()

        # Shared 2D U-Net encoder
        self.encoder = ViewEncoder(
            in_channels=1,
            base_channels=encoder_base_channels,
            out_channels=encoder_out_channels,
        )

        # Project and sample
        self.project_and_sample = ProjectAndSample(
            feat_map_size=self.encoder.h_out,
        )

        # Multi-view fusion
        self.fusion = MultiViewFusion(
            feat_dim=encoder_out_channels,
            method=fusion_method,
        )

        self.view_feat_dim = view_feat_dim

        # Project fused features to desired output dim
        if view_feat_dim != encoder_out_channels:
            self.fuse_proj = nn.Linear(encoder_out_channels, view_feat_dim)
        else:
            self.fuse_proj = nn.Identity()

    def forward(
        self,
        proj_imgs: torch.Tensor,
        coords_world: torch.Tensor,
        coords_vox_norm: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        proj_imgs : torch.Tensor [B, 7, 1, 256, 256]
            7 MCX projection images (fluences). Values should be non-negative.
            Zero-mean normalization applied internally.
        coords_world : torch.Tensor [B, N, 3]
            Query points in world mm coords (detector-centered).
        coords_vox_norm : torch.Tensor [B, N, 3] or None
            Voxel-space normalized coords (centered by physical volume center,
            normalized by half-extents). If provided, uses voxel-space projection.

        Returns
        -------
        view_feat : torch.Tensor [B, N, view_feat_dim]
            Fused per-query view features.
        visibility : torch.Tensor [B, N, 7]
            Visibility mask per view.
        """
        # Sanity check: coords should be trunk-local (max valid is ~40mm in Y)
        assert coords_world.max() < 45, (
            f"ViewEncoder received coords in wrong frame (max={coords_world.max():.1f}), "
            f"should be trunk-local (< 45mm)"
        )

        B, n_views = proj_imgs.shape[:2]

        # Normalize projections: log-scale for better dynamic range
        proj_norm = torch.log1p(proj_imgs.clamp(min=0))

        # Encode each view (shared encoder, batch over views)
        feat_maps_list = []
        for v in range(n_views):
            feat = self.encoder(proj_norm[:, v])
            feat_maps_list.append(feat)

        # Stack: [B, 7, C, H', W']
        feat_maps = torch.stack(feat_maps_list, dim=1)

        # Project and sample (voxel-space if coords_vox_norm provided)
        multi_view_feat, visibility = self.project_and_sample(
            coords_world,
            feat_maps,
            coords_vox_norm=coords_vox_norm,
        )

        # Fuse
        fused = self.fusion(multi_view_feat, visibility)
        view_feat = self.fuse_proj(fused)

        return view_feat, visibility
