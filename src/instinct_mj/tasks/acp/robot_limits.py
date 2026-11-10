"""Joint velocity limits [rad/s] of the ACP G1 (authors' playground model, 23 controlled joints; the wrists are welded).

The authors' MJCF (``assets/resources/acp_g1/g1_29dof.xml``) gives joint ranges and ``actuatorfrcrange`` but NO velocity limit, so the values are defined here explicitly: the
Unitree G1 motor datasheet maxima per joint group (32 rad/s for the 7520-14 hip pitch / yaw and waist yaw, 20 for the 7520-22 hip roll / knee, 37 for the 5020 / 4010 ankle,
waist pitch / roll, shoulders and elbows). They are numerically identical to the ones the task used before (it read the Unitree G1 datasheet table of the shared G1 asset, minus the wrist entries
that match no joint of the welded model), so ``dof_vel_limits`` and its imagined counterpart are unchanged."""

ACP_G1_JOINT_VELOCITY_LIMITS: dict[str, float] = {
    ".*_hip_pitch_joint": 32.0,
    ".*_hip_roll_joint": 20.0,
    ".*_hip_yaw_joint": 32.0,
    ".*_knee_joint": 20.0,
    ".*_ankle_.*_joint": 37.0,
    "waist_pitch_joint": 37.0,
    "waist_roll_joint": 37.0,
    "waist_yaw_joint": 32.0,
    ".*_shoulder_.*_joint": 37.0,
    ".*_elbow_joint": 37.0,
}
