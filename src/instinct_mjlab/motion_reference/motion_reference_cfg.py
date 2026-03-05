from __future__ import annotations
from dataclasses import dataclass, field

from collections.abc import Sequence
from typing import Literal

from instinct_mjlab.visualization.marker_cfg import VisualizationMarkersCfg
from mjlab.sensor import SensorCfg

from .motion_buffer import MotionBuffer, MotionReferenceData
from .motion_reference_manager import MotionReferenceManager


@dataclass(kw_only=True)
class MotionBufferCfg:
    """Configuration for the motion buffer."""

    class_type: type = MotionBuffer

    clip_joint_ref_to_robot_limits: bool = False
    """ clip the joint reference to the robot joint limits. """


@dataclass(kw_only=True)
class MotionReferenceManagerCfg(SensorCfg):
    """Configuration for the motion reference manager."""

    class_type: type = MotionReferenceManager

    def build(self):
        """Build sensor instance from this config (mjlab SensorCfg interface)."""
        return self.class_type(self)

    name: str = "motion_reference"

    entity_name: str = "robot"
    """Entity name of the robot to track (key in scene.entities)."""

    data_class_type: type = MotionReferenceData
    """ The class type of the motion reference data. Use this config to override the default motion reference data class. """

    robot_model_path: str | None = None
    """ Robot model (MJCF xml or URDF) path to build the robot kinematics chain.
        If None, assuming the motion_data file is already in the robot model's form.
    """

    link_of_interests: Sequence[str] | None = None
    """ if set, the link poses in base_link will be computed. """

    num_frames: int = 1
    """ The number of frames for which the command is generated. """

    data_start_from: Literal["one_frame_interval", "current_time"] = "one_frame_interval"
    """ The data start from the frame interval or the current time.
    - "one_frame_interval": the 0-th frame is at `t+frame_interval_s.
    - "current_time": the 0-th frame is at `t`.
    Due to the previous experiments, the default is set to "one_frame_interval".
    """

    frame_interval_s: float | Sequence[float] = 0.1
    """ the frame interval in seconds.
        If a list, the frame interval is randomly selected from/between the
        list-range when reset.
    """

    update_period: float | Sequence[float] = 0.0
    """ The update period of the motion buffer time indices (in seconds)

    If a list/tuple of two floats, the initialization samples for each env at reset.
    """

    update_period_sample_strategy: Literal["uniform", "uniform_frame_limits"] | None = "uniform"
    """ The update period sample strategy. It will use the ranges of the update_period

    - None: no update period sampling. (use the smallest update_period if a list)
    - "uniform": sample uniformly from the update_period range.
    - "uniform_frame_limits": ignore `update_period`, sample uniformly across the longest time
        range of the motion frame sequence.
    """

    motion_buffers: dict[str, MotionBufferCfg] = field(default_factory=dict)
    """ The dictionary for all types of motion buffers to construct.
    """

    mp_split_method: Literal["None", "Even", "Segment"] = "None"
    """ The method to split the entire motion data for multiple processes.
    Only rank/world_size and weight are communicated between processes.
    ## Options:
        - "None": not splitting the motion data. Recommending each motion buffer in each process
            loads the motion data into GPU memory passively.
        - "Even": split the motion data evenly among the processes. If putting all trajecties
            sequentially, the trajectory assignment to each process is like following:
            (assuming 4 processes)
                1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, ...
        - "Segment": split the motion data into segments and assign each segment to each process.
            (assuming 4 processes)
                1, 1, 1, 1, 1, ...., 2, 2, 2, 2, 2, ...., 3, 3, 3, 3, 3, ...., 4, 4, 4, 4, 4, ...
    """

    ### Data Augmentation ###
    symmetric_augmentation_joint_mapping: Sequence[int] | None = None
    """ the joint indices to augment the motion data symmetrically. """
    symmetric_augmentation_joint_reverse_buf: Sequence | None = None
    """ the joint indices to augment the motion data symmetrically,
        If None, no symmetric augmentation is performed.
    """
    symmetric_augmentation_link_mapping: Sequence[int] | None = None
    """ link mapping is in the order of `link_of_interests` """

    ### visualizations ###
    debug_vis: bool = False
    """Enable reference-robot visualization semantics from InstinctLab."""

    reference_entity_name: str | None = None
    """ The entity name of the reference robot in scene.entities.
        To activate the robot model visualization, please spawn another articulation with no
        collisions with any other objects in the scene (but can be visualized in the viewer).
        Then provide the entity name of that reference articulation.
        If None, the reference robot is not visualized.
    """

    visualizer_cfg: VisualizationMarkersCfg = field(default_factory=lambda: VisualizationMarkersCfg(
        prim_path="/Visuals/MotionReference",
        markers={
            "root_frame_ref": {
                "scale": (0.15, 0.15, 0.15),
                "color": (1.0, 1.0, 1.0, 1.0),
            },
            "link_ref": {
                "radius": 0.04,
                "color": (0.0, 1.0, 0.0, 1.0),
            },
            "relative_link_ref": {
                "scale": (0.05, 0.05, 0.05),
                "color": (1.0, 1.0, 1.0, 1.0),
            },
        },
    ))
    """ Visualization config for link reference and base_pose reference. """

    visualizing_marker_types: list[str] = field(default_factory=list)
    """List of marker types to visualize.
    ## Available options:
        - 'root' for root transform
        - 'links' for link transforms
        - 'relative_links' for relative link transforms
    """

    visualizing_robot_offset: Sequence[float] = (0.0, 0.0, 0.0)
    """ To reduce overlapping of the real robot and the reference robot,
    we can offset the reference robot.
    """

    visualizing_robot_from: Literal["aiming_frame", "reference_frame"] = "aiming_frame"
    """ The data source to get and to set the reference robot state.
    - 'aiming_frame': get aiming frame idx and select from the motion_reference data
    - 'reference_frame': get the reference state directly from motion_reference.reference_frame
    """

    def __post_init__(self):
        # Keep InstinctLab-style debug_vis while using mjlab's reference_entity_name.
        if self.reference_entity_name is not None:
            self.debug_vis = True
        elif self.debug_vis:
            self.reference_entity_name = "robot_reference"


@dataclass(kw_only=True)
class NoCollisionPropertiesCfg:
    collision_enabled: bool = False
