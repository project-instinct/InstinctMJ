from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.sensor import RayCastSensorCfg

from .grouped_ray_caster import GroupedRayCaster


@dataclass(kw_only=True)
class GroupedRayCasterCfg(RayCastSensorCfg):
    """Configuration for the GroupedRayCaster sensor."""

    class_type: type = GroupedRayCaster

    min_distance: float = 0.0
    """The minimum distance from the sensor to ray cast to. aka ignore the hits closer than this distance."""

    mesh_prim_paths: list[str] = field(default_factory=list)
    """Optional mesh path regex list from InstinctLab configs.

    When empty, all raycastable geoms are considered (default mjlab behavior).
    When non-empty, GroupedRayCaster filters hits to geoms matched by these expressions.
    """

    aux_mesh_and_link_names: dict[str, str | None] = field(default_factory=dict)
    """Optional alias mapping from mesh names to link/body names.

    This mirrors InstinctLab config semantics and is used as additional name hints when
    resolving ``mesh_prim_paths`` against MuJoCo geom/body names.
    """

    mesh_filter_max_hops: int = 6
    """Maximum number of re-raycast hops when skipping disallowed hits."""

    mesh_filter_epsilon: float = 1e-4
    """Small forward offset (meters) used when continuing rays past disallowed hits."""

    def build(self) -> GroupedRayCaster:
        return GroupedRayCaster(self)
