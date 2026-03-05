from __future__ import annotations
from dataclasses import MISSING, dataclass, field

import math
from typing import TYPE_CHECKING, Literal

from instinct_mjlab.visualization.marker_cfg import VisualizationMarkersCfg
from mjlab.sensor import GridPatternCfg

from .edge_cylinder import GreedyconcatEdgeCylinder, PluckerEdgeCylinder, RansacEdgeCylinder, RayEdgeCylinder
from .virtual_obstacle_base import VirtualObstacleCfg


@dataclass
class _PreviewSurfaceCfg:
    diffuse_color: tuple[float, float, float]
    opacity: float = 1.0


@dataclass
class _CylinderMarkerCfg:
    radius: float
    height: float
    visual_material: _PreviewSurfaceCfg


@dataclass
class _SphereMarkerCfg:
    radius: float
    visual_material: _PreviewSurfaceCfg


@dataclass(kw_only=True)
class EdgeCylinderCfg(VirtualObstacleCfg):
    """The class to use for the edge cylinder detector."""

    class_type: type = MISSING
    """The class to use for the edge detector."""
    angle_threshold: float = 70.0
    """The angle threshold to consider an edge as sharp."""

    cylinder_radius: float = 0.2
    """The radius of the edge cylinder, which is used to treat the edge cylinders as a virtual obstacle."""
    num_grid_cells: int = 64**3
    """The number of grid cells to use for spatial partitioning of the edge cylinders.
    Usually the power of 2, e.g., 64^3 = 262144.
    """

    visualizer: VisualizationMarkersCfg = field(
        default_factory=lambda: VisualizationMarkersCfg(
            prim_path="/Visuals/edgeMarkers",
            markers={
                "cylinder": _CylinderMarkerCfg(
                    radius=1.0,
                    height=1.0,
                    visual_material=_PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.9), opacity=0.2),
                )
            },
        )
    )


@dataclass(kw_only=True)
class PluckerEdgeCylinderCfg(EdgeCylinderCfg):
    """Configuration for the plucker edge cylinder generator."""

    class_type: type = PluckerEdgeCylinder
    """The class to use for the sharp edge detector."""


@dataclass(kw_only=True)
class RansacEdgeCylinderCfg(EdgeCylinderCfg):
    """The class to use for the ransac edge cylinder generator."""

    class_type: type = RansacEdgeCylinder

    max_iter: int = 500
    """The maximum number of iterations."""

    point_distance_threshold: float = 0.04
    """The distance threshold to consider a point as an inlier."""

    min_points: int = 5
    """The minimum number of points required to fit."""

    cluster_eps: float = 0.08
    """The maximum distance between points in a cluster."""


@dataclass(kw_only=True)
class GreedyconcatEdgeCylinderCfg(EdgeCylinderCfg):
    """The class to use for the greedy-concat edge cylinder generator."""

    class_type: type = GreedyconcatEdgeCylinder

    adjacent_angle_threshold: float = 30.0
    """The angle threshold to consider two edges as adjacent."""

    point_distance_threshold: float = 0.06
    """The distance threshold to consider a point as an inlier."""

    min_points: int = 5
    """The minimum number of points in one line."""

    merge_collinear_gap: float = 0.0
    """Max endpoint gap (meters) allowed when post-merging collinear segments.

    Set to 0 to disable the post-merge stage and keep legacy behavior.
    """

    merge_collinear_angle_threshold: float = 25.0
    """Max angular deviation (degrees) between segments for post-merge."""

    merge_collinear_line_distance: float | None = None
    """Line-distance threshold (meters) for post-merge collinearity test.

    If None, fall back to ``point_distance_threshold``.
    """

    merge_collinear_max_passes: int = 3
    """Maximum iterative passes for collinear post-merge."""

    merge_collinear_max_segments: int = 4000
    """Skip collinear post-merge when segment count exceeds this threshold.

    This keeps startup cost bounded on very dense terrains.
    """


@dataclass(kw_only=True)
class RayEdgeCylinderCfg(VirtualObstacleCfg):
    """The class to use for the ray-based edge cylinder generator."""

    class_type: type = RayEdgeCylinder

    cylinder_radius: float = 0.2
    """The radius of the edge cylinder, which is used to treat the edge cylinders as a virtual obstacle."""
    num_grid_cells: int = 64**3
    """The number of grid cells to use for spatial partitioning of the edge cylinders.
    Usually the power of 2, e.g., 64^3 = 262144.
    """
    max_iter: int = 500
    """The maximum number of iterations."""

    point_distance_threshold: float = 0.005
    """The distance threshold to consider a point as an inlier."""

    min_points: int = 15
    """The minimum number of points required to fit."""

    cluster_eps: float = 0.08
    """The maximum distance between points in a cluster."""

    ray_pattern: GridPatternCfg = field(
        default_factory=lambda: GridPatternCfg(
            resolution=0.01,
            size=[6, 6],
            direction=(0.0, 0.0, -1.0),
        )
    )
    """The pattern to use for ray sampling."""

    ray_offset_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    """The offset position of the rays."""

    ray_rotate_axes: list[list[float]] = field(
        default_factory=lambda: [
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, -1.0, 0.0],
        ]
    )

    ray_rotate_angle: list[float] = field(
        default_factory=lambda: [math.pi * 0.25, math.pi * 0.25, math.pi * 0.25, math.pi * 0.25]
    )
    """The axes and angles to rotate the rays."""

    max_ray_depth: float = 8.0
    """The maximum depth of the rays to sample."""

    depth_canny_thresholds: list[float] = field(default_factory=lambda: [250, 300])
    """The thresholds for the Canny edge detector to detect edges in the depth image."""

    normal_canny_thresholds: list[float] = field(default_factory=lambda: [80, 250])
    """The thresholds for the Canny edge detector to detect edges in the normal image."""

    cutoff_z_height: float = 0.1
    """The height threshold to filter out rays that are too close to the ground."""

    visualizer: VisualizationMarkersCfg = field(
        default_factory=lambda: VisualizationMarkersCfg(
            prim_path="/Visuals/edgeMarkers",
            markers={
                "cylinder": _CylinderMarkerCfg(
                    radius=1.0,
                    height=1.0,
                    visual_material=_PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.9), opacity=0.2),
                )
            },
        )
    )

    points_visualizer: VisualizationMarkersCfg = field(
        default_factory=lambda: VisualizationMarkersCfg(
            prim_path="/Visuals/edgePoints",
            markers={
                "sphere": _SphereMarkerCfg(
                    radius=0.01,
                    visual_material=_PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 0.5), opacity=1.0),
                ),
            },
        )
    )
