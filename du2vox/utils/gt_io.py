"""GT voxel-volume loading with support for both storage formats.

The GT voxel volume of a sample is a full MCX-grid array
(``190 x 200 x 104``) that is only ~0.2-0.4% non-zero. Datasets may store it
in either of two formats:

``dense``
    ``gt_voxels.npy`` — the historical float32 array.

``sparse``
    ``gt_voxels.npz`` — a lossless compressed archive of the non-zero
    voxels only. Written by FMT-SimGen (``fmt_simgen/utils/volume_io.py``).

Both decode to the same dense float32 array, so downstream metrics are
unaffected by the choice.

Sparse archive layout
---------------------
``format``  : 0-d array holding the ASCII marker ``b"sparse_flat_c"``
``shape``   : int32 array, the dense C-order shape
``indices`` : int32 array ``[nnz]``, C-order flattened non-zero indices
``values``  : float32 array ``[nnz]``, the corresponding values

This is a deliberate second implementation of the format frozen in
FMT-SimGen; keep ``GT_VOXELS_SPARSE_FORMAT`` and the field names in sync if
the layout ever changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

GT_VOXELS_SPARSE_FORMAT = b"sparse_flat_c"
"""ASCII marker expected in the ``format`` field of a sparse volume archive."""

GT_VOXELS_SPARSE_NAME = "gt_voxels.npz"
GT_VOXELS_DENSE_NAME = "gt_voxels.npy"

GT_MODES = ("binary", "continuous")
"""How the GT volume is turned into a supervision target.

``binary``     threshold at ``binary_threshold`` — the historical D0 contract,
               which only teaches source *support*.
``continuous`` keep the raw intensity, so a spatially varying source keeps its
               internal profile and its per-focus amplitude.
"""

GT_NORMALIZATIONS = ("none", "per_sample_peak")
"""How the target is scaled.

``per_sample_peak`` divides by that sample's own peak, so the target peak is
exactly 1. Absolute fluorophore yield is not recoverable from a single FMT
measurement, so rescaling to a per-sample peak loses no recoverable
information, while it removes the 0.63-1.75 peak spread across the cohort and
makes every sample contribute comparably to the loss. Source-to-source contrast
ratios are unaffected because the whole sample shares one divisor.
"""


def _as_sample_dir(path: Union[str, Path]) -> Path:
    """Accept either a sample directory or a path to a volume file."""
    path = Path(path)
    if path.is_dir():
        return path
    if path.name in (GT_VOXELS_SPARSE_NAME, GT_VOXELS_DENSE_NAME):
        return path.parent
    return path


def resolve_gt_volume_path(path: Union[str, Path]) -> Optional[Path]:
    """Return the GT volume file for a sample, preferring the sparse form.

    Parameters
    ----------
    path : str or Path
        Sample directory, or a path to either volume file.

    Returns
    -------
    Path or None
        The sparse archive if present, else the dense array, else None.
    """
    sample_dir = _as_sample_dir(path)
    sparse_path = sample_dir / GT_VOXELS_SPARSE_NAME
    if sparse_path.exists():
        return sparse_path
    dense_path = sample_dir / GT_VOXELS_DENSE_NAME
    if dense_path.exists():
        return dense_path
    return None


def has_gt_volume(path: Union[str, Path]) -> bool:
    """Return True if a GT voxel volume exists in either format."""
    return resolve_gt_volume_path(path) is not None


def _decode_sparse_archive(archive) -> np.ndarray:
    """Decode a sparse archive into a dense float32 volume."""
    marker = np.asarray(archive["format"]).tobytes()
    if marker != GT_VOXELS_SPARSE_FORMAT:
        raise ValueError(
            f"Unsupported sparse GT volume format marker: {marker!r}; "
            f"expected {GT_VOXELS_SPARSE_FORMAT!r}"
        )
    shape = tuple(int(s) for s in np.asarray(archive["shape"]))
    out = np.zeros(shape, dtype=np.float32)
    indices = np.asarray(archive["indices"])
    if indices.size:
        out.ravel()[indices] = np.asarray(archive["values"], dtype=np.float32)
    return out


def load_gt_volume(
    path: Union[str, Path],
    mmap_mode: Optional[str] = None,
) -> np.ndarray:
    """Load a sample's GT voxel volume, accepting either storage format.

    Parameters
    ----------
    path : str or Path
        Sample directory, or a path to either volume file.
    mmap_mode : str, optional
        Passed through to ``np.load`` for the dense format. The sparse format
        is materialized in memory and ignores this argument; callers that only
        need array semantics are unaffected.

    Returns
    -------
    np.ndarray
        Dense float32 GT volume of shape ``(190, 200, 104)``.

    Raises
    ------
    FileNotFoundError
        If the sample has no GT voxel volume in either format.
    """
    resolved = resolve_gt_volume_path(path)
    if resolved is None:
        sample_dir = _as_sample_dir(path)
        raise FileNotFoundError(
            f"No GT voxel volume in {sample_dir}; expected "
            f"{GT_VOXELS_SPARSE_NAME} or {GT_VOXELS_DENSE_NAME}"
        )
    if resolved.name == GT_VOXELS_SPARSE_NAME:
        with np.load(resolved) as archive:
            return _decode_sparse_archive(archive)
    return np.load(resolved, mmap_mode=mmap_mode).astype(np.float32)


def canonical_gt_values(
    volume: np.ndarray,
    valid_flat_indices: np.ndarray,
    *,
    gt_mode: str = "binary",
    normalize: str = "none",
    binary_threshold: float = 0.05,
    normalization_scale: float | None = None,
) -> tuple[np.ndarray, float]:
    """Extract the supervision target on the canonical valid-voxel domain.

    This is the single implementation shared by target precomputation, the
    training dataset, and evaluation, so the three can never disagree about
    what the target is.

    When ``normalization_scale`` is supplied, every discretization uses that
    explicit divisor. Otherwise the historical fallback computes the peak over
    the whole valid domain before query subsampling.

    Parameters
    ----------
    volume : np.ndarray
        Dense GT voxel volume.
    valid_flat_indices : np.ndarray
        C-order flat indices of the canonical valid voxel centers.
    gt_mode : {"binary", "continuous"}
        Binarize the target, or keep the raw intensity.
    normalize : {"none", "per_sample_peak"}
        Leave values as-is, or divide by this sample's own peak.
    binary_threshold : float
        Threshold used by ``gt_mode="binary"``.
    normalization_scale : float, optional
        Explicit positive divisor for ``per_sample_peak``. Continuous datasets
        should use a scale saved beside the authoritative full voxel field so
        FEM-node, valid-voxel, and full-volume targets cannot silently choose
        different peaks.

    Returns
    -------
    tuple[np.ndarray, float]
        ``(values, scale)`` where ``values`` is float32 on the valid domain and
        ``scale`` is the divisor applied (1.0 when ``normalize="none"``, or when
        the sample is all-zero and no rescaling was possible).
    """
    if gt_mode not in GT_MODES:
        raise ValueError(f"gt_mode must be one of {GT_MODES}, got {gt_mode!r}")
    if normalize not in GT_NORMALIZATIONS:
        raise ValueError(f"normalize must be one of {GT_NORMALIZATIONS}, got {normalize!r}")
    if normalization_scale is not None:
        normalization_scale = float(normalization_scale)
        if normalize != "per_sample_peak":
            raise ValueError("normalization_scale is only valid with normalize='per_sample_peak'")
        if not np.isfinite(normalization_scale) or normalization_scale <= 0.0:
            raise ValueError("normalization_scale must be finite and positive")

    flat = np.asarray(volume).ravel()
    values = flat[np.asarray(valid_flat_indices)].astype(np.float32)

    if gt_mode == "binary":
        values = (values > binary_threshold).astype(np.float32)

    scale = 1.0
    if normalize == "per_sample_peak":
        peak = (
            normalization_scale
            if normalization_scale is not None
            else (float(values.max()) if values.size else 0.0)
        )
        if peak > 0.0:
            values = (values / peak).astype(np.float32)
            scale = peak
    return values, scale


def load_canonical_gt(
    path: Union[str, Path],
    valid_flat_indices: np.ndarray,
    *,
    gt_mode: str = "binary",
    normalize: str = "none",
    binary_threshold: float = 0.05,
    normalization_scale: float | None = None,
) -> tuple[np.ndarray, float]:
    """Load a sample's GT volume and return its canonical-domain target.

    Convenience wrapper around :func:`load_gt_volume` and
    :func:`canonical_gt_values`; see those for the parameter semantics.
    """
    volume = load_gt_volume(path)
    return canonical_gt_values(
        volume,
        valid_flat_indices,
        gt_mode=gt_mode,
        normalize=normalize,
        binary_threshold=binary_threshold,
        normalization_scale=normalization_scale,
    )


def load_normalization_scale(path: Union[str, Path], filename: str = "gt_scale.npy") -> float:
    """Load and validate a sample-level scalar normalization contract."""

    scale_path = _as_sample_dir(path) / filename
    if not scale_path.exists():
        raise FileNotFoundError(f"Missing normalization scale: {scale_path}")
    raw = np.asarray(np.load(scale_path))
    if raw.size != 1:
        raise ValueError(f"Normalization scale must be scalar: {scale_path}")
    scale = float(raw.reshape(()))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Normalization scale must be finite and positive: {scale_path}={scale}")
    return scale
