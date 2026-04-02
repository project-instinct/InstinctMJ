from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import mujoco_warp as mjwarp
import torch
import warp as wp
from mjlab.utils.lab_api import math as math_utils

from .grouped_ray_caster import GroupedRayCaster

if TYPE_CHECKING:
    from mjlab.viewer.debug_visualizer import DebugVisualizer

    from .grouped_ray_caster_camera_cfg import GroupedRayCasterCameraCfg


@dataclass
class CameraData:
    """Data container for the camera sensor."""

    ##
    # Frame state.
    ##

    pos_w: torch.Tensor | None = None
    """Position of the sensor origin in world frame, following ROS convention.

    Shape is (N, 3) where N is the number of sensors.
    """

    quat_w_world: torch.Tensor | None = None
    """Quaternion orientation `(w, x, y, z)` of the sensor origin in world frame, following the world coordinate frame

    .. note::
        World frame convention follows the camera aligned with forward axis +X and up axis +Z.

    Shape is (N, 4) where N is the number of sensors.
    """

    ##
    # Camera data
    ##

    image_shape: tuple[int, int] | None = None
    """A tuple containing (height, width) of the camera sensor."""

    intrinsic_matrices: torch.Tensor | None = None
    """The intrinsic matrices for the camera.

    Shape is (N, 3, 3) where N is the number of sensors.
    """

    output: dict[str, torch.Tensor] | None = None
    """The retrieved sensor data with sensor types as key."""

    info: list[dict[str, Any]] | None = None
    """The retrieved sensor info with sensor types as key."""

    ##
    # Additional Frame orientation conventions
    ##

    @property
    def quat_w_ros(self) -> torch.Tensor:
        """Quaternion orientation `(w, x, y, z)` of the sensor origin in the world frame, following ROS convention."""
        return math_utils.convert_camera_frame_orientation_convention(self.quat_w_world, origin="world", target="ros")

    @property
    def quat_w_opengl(self) -> torch.Tensor:
        """Quaternion orientation `(w, x, y, z)` of the sensor origin in the world frame, following OpenGL convention."""
        return math_utils.convert_camera_frame_orientation_convention(
            self.quat_w_world, origin="world", target="opengl"
        )


class GroupedRayCasterCamera(GroupedRayCaster):
    """Grouped ray-cast camera sensor."""

    cfg: GroupedRayCasterCameraCfg
    """The configuration parameters."""

    UNSUPPORTED_TYPES: ClassVar[set[str]] = {
        "rgb",
        "instance_id_segmentation",
        "instance_id_segmentation_fast",
        "instance_segmentation",
        "instance_segmentation_fast",
        "semantic_segmentation",
        "skeleton_data",
        "motion_vectors",
        "bounding_box_2d_tight",
        "bounding_box_2d_tight_fast",
        "bounding_box_2d_loose",
        "bounding_box_2d_loose_fast",
        "bounding_box_3d",
        "bounding_box_3d_fast",
    }
    """A set of sensor types that are not supported by the ray-caster camera."""

    def __init__(self, cfg: GroupedRayCasterCameraCfg):
        """Initializes the camera object.

        Args:
            cfg: The configuration parameters.

        Raises:
            ValueError: If the provided data types are not supported by the grouped-ray-caster camera.
        """
        # perform check on supported data types
        self._check_supported_data_types(cfg)
        # initialize base class
        super().__init__(cfg)
        # create empty variables for storing output data
        self._camera_data = CameraData()
        self._frame = torch.empty(0, dtype=torch.long)

        self._offset_quat: torch.Tensor | None = None
        self._offset_pos: torch.Tensor | None = None
        self._focal_length: float = 1.0
        self._update_period_s: float = max(float(cfg.update_period), 0.0)
        self._elapsed_since_refresh: torch.Tensor = torch.empty(0)
        self._refresh_mask: torch.Tensor = torch.empty(0, dtype=torch.bool)
        self._has_fresh_sense = False

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        update_period_str = f"{self._update_period_s:.6f}" if self._update_period_s > 0.0 else "every sense()"
        return (
            f"Grouped-Ray-Caster-Camera @ '{self.cfg.name}': \n"
            f"\tupdate period (s)    : {update_period_str}\n"
            f"\tnumber of sensors    : {self._num_envs}\n"
            f"\tnumber of rays/sensor: {self.num_rays}\n"
            f"\ttotal number of rays : {self.num_rays * self._num_envs}\n"
            f"\timage shape          : {self.image_shape}"
        )

    """
    Properties
    """

    @property
    def image_shape(self) -> tuple[int, int]:
        """A tuple containing (height, width) of the camera sensor."""
        return (self.cfg.pattern.height, self.cfg.pattern.width)

    @property
    def frame(self) -> torch.Tensor:
        """Frame number when the measurement took place."""
        return self._frame

    @property
    def refresh_mask(self) -> torch.Tensor:
        """Mask of environments that refreshed in the latest sense call."""
        return self._refresh_mask

    """
    Operations.
    NOTE: Since RayCasterCamera is a direct subclass of RayCaster, GroupedRayCasterCamera has to copy some of the code
    from RayCasterCamera. (Code duplication is not ideal, shall be optimized in the future.)
    """

    def initialize(self, mj_model, model, data, device: str) -> None:
        super().initialize(mj_model, model, data, device)
        # Create all indices buffer
        self._ALL_INDICES = torch.arange(self._num_envs, device=device, dtype=torch.long)
        # Create frame count buffer
        self._frame = torch.zeros(self._num_envs, device=device, dtype=torch.long)
        # Keep per-env refresh timers (seconds).
        self._elapsed_since_refresh = torch.zeros(self._num_envs, device=device, dtype=torch.float32)
        self._refresh_mask = torch.ones(self._num_envs, device=device, dtype=torch.bool)
        if self._update_period_s > 0.0:
            # Force first sense to produce a fresh frame.
            self._elapsed_since_refresh.fill_(self._update_period_s)
        # create buffers
        self._create_buffers()
        # compute intrinsic matrices
        self._compute_intrinsic_matrices()
        # Compute camera rays from intrinsic matrices.
        self.ray_starts, self.ray_directions = self._compute_pinhole_rays_from_intrinsics(
            self._camera_data.intrinsic_matrices
        )
        if self.ray_directions.shape[1] != self._num_rays:
            raise ValueError(
                "Pattern ray count mismatch between base sensor initialization and camera intrinsic reconstruction."
            )
        # set offsets
        quat_w = math_utils.convert_camera_frame_orientation_convention(
            torch.tensor([self.cfg.offset.rot], device=device), origin=self.cfg.offset.convention, target="world"
        )
        self._offset_quat = quat_w.repeat(self._num_envs, 1)
        self._offset_pos = torch.tensor(list(self.cfg.offset.pos), device=device, dtype=torch.float32).repeat(
            self._num_envs, 1
        )
        self._has_fresh_sense = False

    def set_intrinsic_matrices(
        self, matrices: torch.Tensor, focal_length: float = 1.0, env_ids: Sequence[int] | None = None
    ):
        """Set the intrinsic matrix of the camera.

        Args:
            matrices: The intrinsic matrices for the camera. Shape is (N, 3, 3).
            focal_length: Focal length to use when computing aperture values (in cm). Defaults to 1.0.
            env_ids: A sensor ids to manipulate. Defaults to None, which means all sensor indices.
        """
        # resolve env_ids
        if env_ids is None:
            env_ids = self._ALL_INDICES
        device = self._runtime_device
        assert device is not None
        env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)
        # save new intrinsic matrices and focal length
        self._camera_data.intrinsic_matrices[env_ids] = matrices.to(device)
        self._focal_length = focal_length
        # recompute ray directions
        ray_starts, ray_directions = self._compute_pinhole_rays_from_intrinsics(
            self._camera_data.intrinsic_matrices[env_ids]
        )
        self.ray_starts[env_ids] = ray_starts
        self.ray_directions[env_ids] = ray_directions
        self._has_fresh_sense = False

    def reset(self, env_ids: Sequence[int] | None = None):
        # reset the timestamps
        super().reset(env_ids)
        # resolve None
        if env_ids is None:
            env_ids = self._ALL_INDICES
        device = self._runtime_device
        assert device is not None
        env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)
        # reset the data
        # note: this recomputation is useful if one performs events such as randomizations on the camera poses.
        pos_w, quat_w = self._compute_camera_world_poses(env_ids)
        self._camera_data.pos_w[env_ids] = pos_w
        self._camera_data.quat_w_world[env_ids] = quat_w
        # Reset the frame count
        self._frame[env_ids] = 0
        if self._update_period_s > 0.0:
            self._elapsed_since_refresh[env_ids] = self._update_period_s
        self._refresh_mask[env_ids] = True
        self._has_fresh_sense = False

    def update(self, dt: float) -> None:
        super().update(dt)
        if self._update_period_s <= 0.0:
            self._has_fresh_sense = False
            return
        self._elapsed_since_refresh += float(dt)
        self._has_fresh_sense = False

    def set_world_poses(
        self,
        positions: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        env_ids: Sequence[int] | None = None,
        convention: Literal["opengl", "ros", "world"] = "ros",
    ):
        """Set the pose of the camera w.r.t. the world frame using specified convention.

        Since different fields use different conventions for camera orientations, the method allows users to
        set the camera poses in the specified convention. Possible conventions are:

        - :obj:`"opengl"` - forward axis: -Z - up axis +Y - Offset is applied in the OpenGL (Usd.Camera) convention
        - :obj:`"ros"`    - forward axis: +Z - up axis -Y - Offset is applied in the ROS convention
        - :obj:`"world"`  - forward axis: +X - up axis +Z - Offset is applied in the World Frame convention

        See :func:`convert_orientation_convention` for more details on these
        conventions.

        Args:
            positions: The cartesian coordinates (in meters). Shape is (N, 3).
                Defaults to None, in which case the camera position in not changed.
            orientations: The quaternion orientation in (w, x, y, z). Shape is (N, 4).
                Defaults to None, in which case the camera orientation in not changed.
            env_ids: A sensor ids to manipulate. Defaults to None, which means all sensor indices.
            convention: The convention in which the poses are fed. Defaults to "ros".

        Raises:
            RuntimeError: If the camera prim is not set. Need to call :meth:`initialize` method first.
        """
        # resolve env_ids
        if env_ids is None:
            env_ids = self._ALL_INDICES
        device = self._runtime_device
        assert device is not None
        env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)

        # get current positions
        pos_w, quat_w = self._compute_view_world_poses(env_ids)
        if positions is not None:
            # transform to camera frame
            pos_offset_world_frame = positions.to(device) - pos_w
            self._offset_pos[env_ids] = math_utils.quat_apply(math_utils.quat_inv(quat_w), pos_offset_world_frame)
        if orientations is not None:
            # convert rotation matrix from input convention to world
            quat_w_set = math_utils.convert_camera_frame_orientation_convention(
                orientations.to(device), origin=convention, target="world"
            )
            self._offset_quat[env_ids] = math_utils.quat_mul(math_utils.quat_inv(quat_w), quat_w_set)

        # update the data
        pos_w, quat_w = self._compute_camera_world_poses(env_ids)
        self._camera_data.pos_w[env_ids] = pos_w
        self._camera_data.quat_w_world[env_ids] = quat_w
        self._has_fresh_sense = False

    def set_world_poses_from_view(
        self, eyes: torch.Tensor, targets: torch.Tensor, env_ids: Sequence[int] | None = None
    ):
        """Set the poses of the camera from the eye position and look-at target position.

        Args:
            eyes: The positions of the camera's eye. Shape is N, 3).
            targets: The target locations to look at. Shape is (N, 3).
            env_ids: A sensor ids to manipulate. Defaults to None, which means all sensor indices.

        Raises:
            RuntimeError: If the camera prim is not set. Need to call :meth:`initialize` method first.
            NotImplementedError: If the stage up-axis is not "Y" or "Z".

        Note:
            In MuJoCo (mjlab), the up-axis is always "Z", so the NotImplementedError will not be raised.
        """
        device = self._runtime_device
        assert device is not None
        # camera position and rotation in opengl convention
        orientations = math_utils.quat_from_matrix(
            math_utils.create_rotation_matrix_from_view(eyes, targets, up_axis="Z", device=device)
        )
        self.set_world_poses(eyes, orientations, env_ids, convention="opengl")

    """
    Implementation.
    """

    def prepare_rays(self) -> None:
        """PRE-GRAPH: Prepare grouped camera rays from camera world poses."""

        env_ids = self._ALL_INDICES
        pos_w, quat_w = self._compute_camera_world_poses(env_ids)
        if self._update_period_s > 0.0:
            refresh_mask = self._elapsed_since_refresh >= (self._update_period_s - 1e-8)
            self._refresh_mask = refresh_mask
            if torch.any(refresh_mask):
                refresh_ids = refresh_mask.nonzero(as_tuple=False).squeeze(-1)
                self._elapsed_since_refresh[refresh_ids] = torch.remainder(
                    self._elapsed_since_refresh[refresh_ids], self._update_period_s
                )
                self._camera_data.pos_w[refresh_ids] = pos_w[refresh_ids]
                self._camera_data.quat_w_world[refresh_ids] = quat_w[refresh_ids]
                self._frame[refresh_ids] += 1
        else:
            self._refresh_mask.fill_(True)
            self._camera_data.pos_w[env_ids] = pos_w
            self._camera_data.quat_w_world[env_ids] = quat_w
            self._frame += 1

        quat_w_expanded = quat_w.unsqueeze(1).expand(-1, self.num_rays, -1).reshape(-1, 4)
        ray_starts_flat = self.ray_starts.reshape(-1, 3)
        ray_dirs_flat = self.ray_directions.reshape(-1, 3)
        ray_starts_w = math_utils.quat_apply(quat_w_expanded, ray_starts_flat).view(self._num_envs, self.num_rays, 3)
        ray_starts_w += pos_w.unsqueeze(1)
        ray_directions_w = math_utils.quat_apply(quat_w_expanded, ray_dirs_flat).view(self._num_envs, self.num_rays, 3)

        self._write_world_rays_to_backend(ray_starts_w, ray_directions_w)

        self._cached_world_origins = ray_starts_w
        self._cached_world_rays = ray_directions_w
        self._cached_frame_pos = pos_w.unsqueeze(1)
        self._cached_frame_mat = math_utils.matrix_from_quat(quat_w).unsqueeze(1)

    def postprocess_rays(self) -> None:
        super().postprocess_rays()
        stale_env_ids: torch.Tensor | None = None
        stale_cached_outputs: dict[str, torch.Tensor] = {}
        if self._update_period_s > 0.0:
            stale_mask = ~self._refresh_mask
            if torch.any(stale_mask):
                stale_env_ids = stale_mask.nonzero(as_tuple=False).squeeze(-1)
                for data_type in self.cfg.data_types:
                    stale_cached_outputs[data_type] = self._camera_data.output[data_type][stale_env_ids].clone()

        raycast_data = self.raycast_data
        ray_directions_w = self._cached_world_rays

        quat_w = self._camera_data.quat_w_world
        ray_depth_camera = raycast_data.distances.clone()
        ray_depth_camera[ray_depth_camera < 0] = float("inf")
        ray_depth_image = raycast_data.distances.clone()
        ray_depth_image[ray_depth_image < 0] = float("nan")

        # update output buffers
        if "distance_to_image_plane" in self.cfg.data_types:
            # Use the camera frame with +X as the forward axis.
            quat_inv = math_utils.quat_inv(quat_w).unsqueeze(1).expand(-1, self.num_rays, -1).reshape(-1, 4)
            image_ray_vec = (ray_depth_image[:, :, None] * ray_directions_w).reshape(-1, 3)
            distance_to_image_plane = math_utils.quat_apply(quat_inv, image_ray_vec).view(
                self._num_envs, self.num_rays, 3
            )[:, :, 0]
            # apply the maximum distance after the transformation
            if self.cfg.depth_clipping_behavior == "max":
                distance_to_image_plane = torch.clip(distance_to_image_plane, max=self.cfg.max_distance)
                distance_to_image_plane[torch.isnan(distance_to_image_plane)] = self.cfg.max_distance
            elif self.cfg.depth_clipping_behavior == "zero":
                distance_to_image_plane[distance_to_image_plane > self.cfg.max_distance] = 0.0
                distance_to_image_plane[torch.isnan(distance_to_image_plane)] = 0.0
            self._camera_data.output["distance_to_image_plane"] = distance_to_image_plane.view(-1, *self.image_shape, 1)

        if "distance_to_camera" in self.cfg.data_types:
            if self.cfg.depth_clipping_behavior == "max":
                ray_depth_camera = torch.clip(ray_depth_camera, max=self.cfg.max_distance)
            elif self.cfg.depth_clipping_behavior == "zero":
                ray_depth_camera[ray_depth_camera > self.cfg.max_distance] = 0.0
            self._camera_data.output["distance_to_camera"] = ray_depth_camera.view(-1, *self.image_shape, 1)

        if "normals" in self.cfg.data_types:
            self._camera_data.output["normals"] = raycast_data.normals_w.view(-1, *self.image_shape, 3)

        if stale_env_ids is not None:
            for data_type, cached_output in stale_cached_outputs.items():
                self._camera_data.output[data_type][stale_env_ids] = cached_output
        self._has_fresh_sense = True

    def debug_vis(self, visualizer: DebugVisualizer) -> None:
        """Debug visualization for the grouped ray-caster camera.

        Uses mjlab DebugVisualizer native API.
        """
        if not self.cfg.debug_vis:
            return

        env_ids = list(visualizer.get_env_indices(self._num_envs))
        if not env_ids:
            return

        meansize = max(float(visualizer.meansize), 1e-6)
        point_radius = max(0.003 * meansize, 0.01)
        frame_scale = max(0.15 * meansize, 0.1)
        frame_axis_radius = max(0.01 * meansize, 0.005)
        raycast_data = self.raycast_data

        # Only visualize rays that actually hit a surface (distance >= 0).
        # Miss rays keep the ray origin as their hit position, which would cluster
        # all miss points inside the robot head and make them invisible.
        hit_mask = (raycast_data.distances[env_ids] >= 0).reshape(-1)  # (len(env_ids)*num_rays,)
        viz_points = raycast_data.hit_pos_w[env_ids].reshape(-1, 3)
        viz_points = viz_points[hit_mask]

        max_points = 512
        if viz_points.shape[0] > max_points:
            stride = max(1, viz_points.shape[0] // max_points)
            viz_points = viz_points[::stride]

        for point in viz_points:
            visualizer.add_sphere(
                center=point,
                radius=point_radius,
                color=(1.0, 0.1, 0.1, 0.6),
            )

        camera_pos = self._camera_data.pos_w[env_ids]
        camera_quat = self._camera_data.quat_w_world[env_ids]
        camera_rot = math_utils.matrix_from_quat(camera_quat)
        for idx in range(len(env_ids)):
            visualizer.add_frame(
                position=camera_pos[idx],
                rotation_matrix=camera_rot[idx],
                scale=frame_scale,
                axis_radius=frame_axis_radius,
                alpha=0.9,
            )

    """
    Private Helpers
    """

    def _check_supported_data_types(self, cfg: GroupedRayCasterCameraCfg):
        """Checks if the data types are supported by the grouped-ray-caster camera."""
        # check if there is any intersection in unsupported types
        # reason: we cannot obtain this data from simplified warp-based ray caster
        common_elements = set(cfg.data_types) & GroupedRayCasterCamera.UNSUPPORTED_TYPES
        if common_elements:
            raise ValueError(
                f"GroupedRayCasterCamera class does not support the following sensor types: {common_elements}."
                "\n\tThis is because these sensor types cannot be obtained in a fast way using ''warp''."
                "\n\tHint: If you need these sensor types, use a full camera sensor"
                " backend instead of the grouped ray-caster path."
            )

    def _create_buffers(self):
        """Create buffers for storing data."""
        device = self._runtime_device
        assert device is not None
        # prepare drift
        self.drift = torch.zeros(self._num_envs, 3, device=device)
        # create the data object
        # -- pose of the cameras
        self._camera_data.pos_w = torch.zeros((self._num_envs, 3), device=device)
        self._camera_data.quat_w_world = torch.zeros((self._num_envs, 4), device=device)
        # -- intrinsic matrix
        self._camera_data.intrinsic_matrices = torch.zeros((self._num_envs, 3, 3), device=device)
        self._camera_data.intrinsic_matrices[:, 2, 2] = 1.0
        self._camera_data.image_shape = self.image_shape
        # -- output data
        # create the buffers to store the annotator data.
        self._camera_data.output = {}
        self._camera_data.info = [{name: None for name in self.cfg.data_types}] * self._num_envs
        for name in self.cfg.data_types:
            if name in ["distance_to_image_plane", "distance_to_camera"]:
                shape = (self.cfg.pattern.height, self.cfg.pattern.width, 1)
            elif name in ["normals"]:
                shape = (self.cfg.pattern.height, self.cfg.pattern.width, 3)
            else:
                raise ValueError(f"Received unknown data type: {name}. Please check the configuration.")
            # allocate tensor to store the data
            self._camera_data.output[name] = torch.zeros((self._num_envs, *shape), device=device)

    def _compute_intrinsic_matrices(self):
        """Computes the intrinsic matrices for the camera based on the config provided."""
        # get the sensor properties
        pattern_cfg = self.cfg.pattern
        if self.cfg.focal_length is not None and self.cfg.horizontal_aperture is not None:
            vertical_aperture = self.cfg.vertical_aperture
            # Compute vertical aperture from the image aspect ratio if omitted.
            if vertical_aperture is None:
                vertical_aperture = self.cfg.horizontal_aperture * pattern_cfg.height / pattern_cfg.width
            f_x = pattern_cfg.width * self.cfg.focal_length / self.cfg.horizontal_aperture
            f_y = pattern_cfg.height * self.cfg.focal_length / vertical_aperture
            c_x = self.cfg.horizontal_aperture_offset * f_x + pattern_cfg.width / 2
            c_y = self.cfg.vertical_aperture_offset * f_y + pattern_cfg.height / 2
            self._focal_length = self.cfg.focal_length
        else:
            half_fovy = 0.5 * math.radians(pattern_cfg.fovy)
            f_y = 0.5 * pattern_cfg.height / max(math.tan(half_fovy), 1e-8)
            f_x = f_y
            c_x = pattern_cfg.width / 2
            c_y = pattern_cfg.height / 2
            self._focal_length = 1.0

        # allocate the intrinsic matrices
        self._camera_data.intrinsic_matrices[:, 0, 0] = f_x
        self._camera_data.intrinsic_matrices[:, 0, 2] = c_x
        self._camera_data.intrinsic_matrices[:, 1, 1] = f_y
        self._camera_data.intrinsic_matrices[:, 1, 2] = c_y

    def _compute_pinhole_rays_from_intrinsics(
        self, intrinsic_matrices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute pinhole rays from intrinsic matrices."""
        device = self._runtime_device
        assert device is not None
        # get image plane mesh grid
        grid = torch.meshgrid(
            torch.arange(start=0, end=self.cfg.pattern.width, dtype=torch.int32, device=device),
            torch.arange(start=0, end=self.cfg.pattern.height, dtype=torch.int32, device=device),
            indexing="xy",
        )
        pixels = torch.vstack(list(map(torch.ravel, grid))).T
        # convert to homogeneous coordinate system
        pixels = torch.hstack([pixels, torch.ones((len(pixels), 1), device=device)])
        # move each pixel coordinate to the center of the pixel
        pixels += torch.tensor([[0.5, 0.5, 0]], device=device)
        pixels = pixels.to(dtype=intrinsic_matrices.dtype)
        # get pixel coordinates in camera frame
        pix_in_cam_frame = torch.matmul(torch.inverse(intrinsic_matrices), pixels.T)
        # Convert from image camera frame (x-right, y-down, z-forward) to
        # world camera frame (x-forward, y-left, z-up).
        transform_vec = (
            torch.tensor([1, -1, -1], device=device, dtype=intrinsic_matrices.dtype).unsqueeze(0).unsqueeze(2)
        )
        pix_in_cam_frame = pix_in_cam_frame[:, [2, 0, 1], :] * transform_vec
        # normalize ray directions
        ray_directions = (pix_in_cam_frame / torch.norm(pix_in_cam_frame, dim=1, keepdim=True)).permute(0, 2, 1)
        # for camera, we always ray-cast from the sensor's origin
        ray_starts = torch.zeros_like(ray_directions, device=device)
        return ray_starts, ray_directions

    def _compute_view_world_poses(self, env_ids: Sequence[int] | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Obtains the pose of the view the camera is attached to in the world frame.

        Returns:
            A tuple of the position (in meters) and quaternion (w, x, y, z).
        """
        device = self._runtime_device
        runtime_data = self._runtime_data
        metadata = self._require_attachment_frame_metadata()
        assert device is not None and runtime_data is not None
        env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)
        pos_w, frame_mat = self._compute_attached_frame_world_pose(env_ids)
        if metadata.frame_type == "body":
            quat_w = runtime_data.xquat[env_ids, metadata.body_id]
        else:
            quat_w = math_utils.quat_from_matrix(frame_mat)
        # return the pose
        return pos_w.clone(), quat_w.clone()

    def _compute_camera_world_poses(self, env_ids: Sequence[int] | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the pose of the camera in the world frame.

        This function applies the offset pose to the pose of the view the camera is attached to.

        Returns:
            A tuple of the position (in meters) and quaternion (w, x, y, z) in "world" convention.
        """
        device = self._runtime_device
        assert device is not None
        env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)
        # get the pose of the view the camera is attached to
        pos_w, quat_w = self._compute_view_world_poses(env_ids)
        # apply offsets
        # need to apply quat because offset relative to parent frame
        pos_w += math_utils.quat_apply(quat_w, self._offset_pos[env_ids])
        quat_w = math_utils.quat_mul(quat_w, self._offset_quat[env_ids])

        return pos_w, quat_w

    def _sense_on_demand(self) -> None:
        """Run sensing on demand when the camera data is requested."""
        if self._runtime_ctx is None:
            return
        if self._runtime_model is None or self._runtime_data is None or self._runtime_wp_device is None:
            return
        self.prepare_rays()
        render_context = self._get_backend_render_context()
        with wp.ScopedDevice(self._runtime_wp_device):
            mjwarp.refit_bvh(self._runtime_model, self._runtime_data, render_context)
            self.raycast_kernel(rc=render_context)
        self.postprocess_rays()

    def _compute_data(self) -> CameraData:
        if not self._has_fresh_sense:
            self._sense_on_demand()
        return self._camera_data
