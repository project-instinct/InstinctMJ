from __future__ import annotations
from dataclasses import MISSING, dataclass

import torch
import trimesh
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from instinct_mjlab.visualization.marker_cfg import VisualizationMarkersCfg


@dataclass(kw_only=True)
class VirtualObstacleCfg:
    """Configuration for a virtual obstacle."""

    class_type: type = MISSING
    """The class to use for the virtual obstacle."""

    visualizer: VisualizationMarkersCfg = MISSING
    """The visualizer configuration for the virtual obstacle."""


class VirtualObstacleBase(ABC):
    def __init__(self, cfg: VirtualObstacleCfg):
        self.cfg = cfg
        self.supports_edge_segment_generation = False

    @abstractmethod
    def generate(self, mesh: trimesh.Trimesh, device: torch.device | str = "cpu") -> None:
        """Generate the virtual obstacle mesh based on the provided terrain mesh.
        NOTE: This interface might be updated in the future to support more complex generation logic.

        Args:
            mesh (trimesh.Trimesh): The terrain mesh to generate the virtual obstacle from.

        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def disable_visualizer(self) -> None:
        """Disable the visualizer for the virtual obstacle if there is one."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    """
    Operations only after being generated.
    If called before generation, it should skip and print a warning.
    """

    @abstractmethod
    def visualize(self):
        """Visualize the virtual obstacle."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def generate_from_edge_segments(self, edge_segments, device: torch.device | str = "cpu") -> None:
        del edge_segments, device
        raise NotImplementedError("This virtual obstacle does not support edge-segment generation.")

    def debug_vis(self, visualizer) -> None:
        del visualizer
        return

    @abstractmethod
    def get_points_penetration_offset(self, points: torch.Tensor) -> torch.Tensor:
        """Get the penetration offset for the given points.

        Args:
            points (torch.Tensor): Shape (N, 3) The points to check for penetration.

        Returns:
            torch.Tensor: Shape (N, 3) The penetration offsets for the points. pointing from the surface to the point.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
