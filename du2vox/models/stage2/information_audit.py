"""Matched A0--A3 information-source audit contracts."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class InformationArm:
    name: str
    use_state: bool
    use_latent: bool
    use_views: bool
    oracle_state: bool


_STRICT = {
    "A0": InformationArm("A0", False, False, False, False),
    "A1": InformationArm("A1", True, False, False, False),
    "A2": InformationArm("A2", True, True, False, False),
    "A3": InformationArm("A3", True, False, False, True),
}
INFORMATION_ARMS = {
    **_STRICT,
    **{
        f"V{name}": InformationArm(
            f"V{name}", arm.use_state, arm.use_latent, True, arm.oracle_state
        )
        for name, arm in _STRICT.items()
    },
}


def get_information_arm(name: str) -> InformationArm:
    normalized = name.upper()
    try:
        return INFORMATION_ARMS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unknown information-audit arm {name!r}; expected {tuple(INFORMATION_ARMS)}"
        ) from exc


def select_coarse_state(
    arm: InformationArm,
    *,
    frozen_state: torch.Tensor,
    oracle_state: torch.Tensor | None,
) -> torch.Tensor:
    """Keep oracle FEM state structurally confined to A3/VA3."""

    if not arm.oracle_state:
        return frozen_state
    if oracle_state is None:
        raise ValueError(f"{arm.name} requires the oracle projected FEM state")
    return oracle_state


@dataclass(frozen=True)
class InformationFeatureLayout:
    coordinate_dim: int = 39
    barycentric_dim: int = 4
    state_dim: int = 12
    latent_dim: int = 144
    view_dim: int = 32

    @property
    def geometry_dim(self) -> int:
        return self.coordinate_dim + self.barycentric_dim

    @property
    def input_dim(self) -> int:
        return self.geometry_dim + self.state_dim + self.latent_dim + self.view_dim

    @property
    def slices(self) -> dict[str, slice]:
        g = self.geometry_dim
        s = g + self.state_dim
        latent = s + self.latent_dim
        return {
            "geometry": slice(0, g),
            "state": slice(g, s),
            "latent": slice(s, latent),
            "views": slice(latent, latent + self.view_dim),
        }


def assemble_information_features(
    arm: InformationArm,
    *,
    coordinate_pe: torch.Tensor,
    barycentric: torch.Tensor,
    state: torch.Tensor | None,
    latent: torch.Tensor | None,
    views: torch.Tensor | None,
    layout: InformationFeatureLayout,
) -> torch.Tensor:
    """Place authorized groups in fixed slots and zero-fill every absent group."""

    if coordinate_pe.shape[:-1] != barycentric.shape[:-1]:
        raise ValueError("Coordinate PE and barycentric leading shapes differ")
    if coordinate_pe.shape[-1] != layout.coordinate_dim:
        raise ValueError("Coordinate PE dimension does not match the audit layout")
    if barycentric.shape[-1] != layout.barycentric_dim:
        raise ValueError("Barycentric dimension does not match the audit layout")
    leading = coordinate_pe.shape[:-1]

    def authorized(
        value: torch.Tensor | None, dimension: int, enabled: bool, label: str
    ) -> torch.Tensor:
        if not enabled:
            return coordinate_pe.new_zeros(*leading, dimension)
        if value is None or value.shape != (*leading, dimension):
            actual = None if value is None else tuple(value.shape)
            raise ValueError(f"{label} must have shape {(*leading, dimension)}, got {actual}")
        return value.to(device=coordinate_pe.device, dtype=coordinate_pe.dtype)

    return torch.cat(
        [
            coordinate_pe,
            barycentric.to(coordinate_pe.dtype),
            authorized(state, layout.state_dim, arm.use_state, "state"),
            authorized(latent, layout.latent_dim, arm.use_latent, "latent"),
            authorized(views, layout.view_dim, arm.use_views, "views"),
        ],
        dim=-1,
    )
