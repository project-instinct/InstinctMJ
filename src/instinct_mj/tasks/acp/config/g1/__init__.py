"""Register Instinct Mj ACP G1 tasks."""

from instinct_mj.tasks.registry import register_instinct_task

# Joint order of the deploy policy (authors' g1_rsl_cpp, REPORT.md section 3.4) of the 23 controlled joints. The env keeps
# the MJCF joint order (``waist_*`` first, then legs, then arms); the ONNX export remaps with this list.
DEPLOY_POLICY_JOINT_ORDER = tuple(
    f"{name}_joint"
    for name in (
        "left_hip_pitch",
        "right_hip_pitch",
        "waist_yaw",
        "left_hip_roll",
        "right_hip_roll",
        "waist_roll",
        "left_hip_yaw",
        "right_hip_yaw",
        "waist_pitch",
        "left_knee",
        "right_knee",
        "left_shoulder_pitch",
        "right_shoulder_pitch",
        "left_ankle_pitch",
        "right_ankle_pitch",
        "left_shoulder_roll",
        "right_shoulder_roll",
        "left_ankle_roll",
        "right_ankle_roll",
        "left_shoulder_yaw",
        "right_shoulder_yaw",
        "left_elbow",
        "right_elbow",
    )
)

# The ACP tracker is trained with SuperMimic (no PPO).
ACP_RUNNER_ENTRY_POINT = "instinct_mj.rl.acp.runner:ACPRunner"


def g1_acp_track_instinct_rl_cfg():
    from .rl_cfg import g1_acp_track_instinct_rl_cfg as _cfg

    return _cfg()


def _acp_track_env_cfg():
    from .acp_tracking_cfg import G1AcpTrackEnvCfg

    return G1AcpTrackEnvCfg()


def _acp_track_play_env_cfg():
    from .acp_tracking_cfg import G1AcpTrackEnvCfg_PLAY

    return G1AcpTrackEnvCfg_PLAY()


def g1_acp_stage2_instinct_rl_cfg():
    from .rl_cfg import g1_acp_stage2_instinct_rl_cfg as _cfg

    return _cfg()


def _acp_stage2_env_cfg():
    from .acp_tracking_cfg import G1AcpStage2EnvCfg

    return G1AcpStage2EnvCfg()


register_instinct_task(
    task_id="Instinct-ACP-Stage2-G1-v0",
    env_cfg_factory=_acp_stage2_env_cfg,
    play_env_cfg_factory=_acp_track_play_env_cfg,
    instinct_rl_cfg_factory=g1_acp_stage2_instinct_rl_cfg,
    runner_entry_point=ACP_RUNNER_ENTRY_POINT,
)

register_instinct_task(
    task_id="Instinct-ACP-Track-G1-v0",
    env_cfg_factory=_acp_track_env_cfg,
    play_env_cfg_factory=_acp_track_play_env_cfg,
    instinct_rl_cfg_factory=g1_acp_track_instinct_rl_cfg,
    runner_entry_point=ACP_RUNNER_ENTRY_POINT,
)

def _acp_track_nolinkterm_env_cfg():
    from .acp_tracking_cfg import G1AcpTrackNoLinkTermEnvCfg

    return G1AcpTrackNoLinkTermEnvCfg()


register_instinct_task(
    task_id="Instinct-ACP-Track-NoLinkTerm-G1-v0",
    env_cfg_factory=_acp_track_nolinkterm_env_cfg,
    play_env_cfg_factory=_acp_track_play_env_cfg,
    instinct_rl_cfg_factory=g1_acp_track_instinct_rl_cfg,
    runner_entry_point=ACP_RUNNER_ENTRY_POINT,
)

def _acp_track_sm_env_cfg():
    from .acp_tracking_cfg import G1AcpTrackSMEnvCfg

    return G1AcpTrackSMEnvCfg()


register_instinct_task(
    task_id="Instinct-ACP-Track-SM-G1-v0",
    env_cfg_factory=_acp_track_sm_env_cfg,
    play_env_cfg_factory=_acp_track_play_env_cfg,
    instinct_rl_cfg_factory=g1_acp_track_instinct_rl_cfg,
    runner_entry_point=ACP_RUNNER_ENTRY_POINT,
)

def _acp_stage2_sm_env_cfg():
    from .acp_tracking_cfg import G1AcpStage2SMEnvCfg

    return G1AcpStage2SMEnvCfg()


register_instinct_task(
    task_id="Instinct-ACP-Stage2-SM-G1-v0",
    env_cfg_factory=_acp_stage2_sm_env_cfg,
    play_env_cfg_factory=_acp_track_play_env_cfg,
    instinct_rl_cfg_factory=g1_acp_stage2_instinct_rl_cfg,
    runner_entry_point=ACP_RUNNER_ENTRY_POINT,
)

def _acp_track_sm_xy_env_cfg():
    from .acp_tracking_cfg import G1AcpTrackSMXYEnvCfg

    return G1AcpTrackSMXYEnvCfg()


register_instinct_task(
    task_id="Instinct-ACP-Track-SM-XY-G1-v0",
    env_cfg_factory=_acp_track_sm_xy_env_cfg,
    play_env_cfg_factory=_acp_track_play_env_cfg,
    instinct_rl_cfg_factory=g1_acp_track_instinct_rl_cfg,
    runner_entry_point=ACP_RUNNER_ENTRY_POINT,
)

def _acp_track_sm_bm_env_cfg():
    from .acp_tracking_cfg import G1AcpTrackSMBMEnvCfg

    return G1AcpTrackSMBMEnvCfg()


register_instinct_task(
    task_id="Instinct-ACP-Track-SM-BM-G1-v0",
    env_cfg_factory=_acp_track_sm_bm_env_cfg,
    play_env_cfg_factory=_acp_track_play_env_cfg,
    instinct_rl_cfg_factory=g1_acp_track_instinct_rl_cfg,
    runner_entry_point=ACP_RUNNER_ENTRY_POINT,
)

def _acp_track_sm_c_env_cfg():
    from .acp_tracking_cfg import G1AcpTrackSMCEnvCfg

    return G1AcpTrackSMCEnvCfg()


register_instinct_task(
    task_id="Instinct-ACP-Track-SM-C-G1-v0",
    env_cfg_factory=_acp_track_sm_c_env_cfg,
    play_env_cfg_factory=_acp_track_play_env_cfg,
    instinct_rl_cfg_factory=g1_acp_track_instinct_rl_cfg,
    runner_entry_point=ACP_RUNNER_ENTRY_POINT,
)

def _acp_track_sm_ch5_env_cfg():
    from .acp_tracking_cfg import G1AcpTrackSMCH5EnvCfg

    return G1AcpTrackSMCH5EnvCfg()


register_instinct_task(
    task_id="Instinct-ACP-Track-SM-CH5-G1-v0",
    env_cfg_factory=_acp_track_sm_ch5_env_cfg,
    play_env_cfg_factory=_acp_track_play_env_cfg,
    instinct_rl_cfg_factory=g1_acp_track_instinct_rl_cfg,
    runner_entry_point=ACP_RUNNER_ENTRY_POINT,
)

def _acp_track_sm_ch5x5_env_cfg():
    from .acp_tracking_cfg import G1AcpTrackSMCH5X5EnvCfg

    return G1AcpTrackSMCH5X5EnvCfg()


def _acp_track_sm_ch5_play_env_cfg():
    from .acp_tracking_cfg import G1AcpTrackSMCH5EnvCfg_PLAY

    return G1AcpTrackSMCH5EnvCfg_PLAY()


register_instinct_task(
    task_id="Instinct-ACP-Track-SM-CH5-G1-Play-v0",
    env_cfg_factory=_acp_track_sm_ch5_play_env_cfg,
    play_env_cfg_factory=_acp_track_sm_ch5_play_env_cfg,
    instinct_rl_cfg_factory=g1_acp_track_instinct_rl_cfg,
    runner_entry_point=ACP_RUNNER_ENTRY_POINT,
)

register_instinct_task(
    task_id="Instinct-ACP-Track-SM-CH5X5-G1-v0",
    env_cfg_factory=_acp_track_sm_ch5x5_env_cfg,
    play_env_cfg_factory=_acp_track_play_env_cfg,
    instinct_rl_cfg_factory=g1_acp_track_instinct_rl_cfg,
    runner_entry_point=ACP_RUNNER_ENTRY_POINT,
)

def _acp_track_sm_ch5x5p_env_cfg():
    from .acp_tracking_cfg import G1AcpTrackSMCH5X5PenEnvCfg

    return G1AcpTrackSMCH5X5PenEnvCfg()


register_instinct_task(
    task_id="Instinct-ACP-Track-SM-CH5X5P-G1-v0",
    env_cfg_factory=_acp_track_sm_ch5x5p_env_cfg,
    play_env_cfg_factory=_acp_track_play_env_cfg,
    instinct_rl_cfg_factory=g1_acp_track_instinct_rl_cfg,
    runner_entry_point=ACP_RUNNER_ENTRY_POINT,
)

register_instinct_task(
    task_id="Instinct-ACP-Track-G1-Play-v0",
    env_cfg_factory=_acp_track_play_env_cfg,
    play_env_cfg_factory=_acp_track_play_env_cfg,
    instinct_rl_cfg_factory=g1_acp_track_instinct_rl_cfg,
    runner_entry_point=ACP_RUNNER_ENTRY_POINT,
)

# Stage 3: reference tracking + goal reaching on the procedural terrains, dual-critic SuperMimic runner.
ACP_STAGE3_RUNNER_ENTRY_POINT = "instinct_mj.rl.acp.stage3_runner:ACPStage3Runner"


def g1_acp_stage3_instinct_rl_cfg():
    from .rl_stage3_cfg import g1_acp_stage3_instinct_rl_cfg as _cfg

    return _cfg()


def _acp_stage3_env_cfg():
    from .acp_stage3_cfg import G1AcpStage3EnvCfg

    return G1AcpStage3EnvCfg()


def _acp_stage3_play_env_cfg():
    from .acp_stage3_cfg import G1AcpStage3EnvCfg_PLAY

    return G1AcpStage3EnvCfg_PLAY()


register_instinct_task(
    task_id="Instinct-ACP-Stage3-G1-v0",
    env_cfg_factory=_acp_stage3_env_cfg,
    play_env_cfg_factory=_acp_stage3_play_env_cfg,
    instinct_rl_cfg_factory=g1_acp_stage3_instinct_rl_cfg,
    runner_entry_point=ACP_STAGE3_RUNNER_ENTRY_POINT,
)


def _acp_stage3_sm_env_cfg():
    from .acp_stage3_cfg import G1AcpStage3SMEnvCfg

    return G1AcpStage3SMEnvCfg()


register_instinct_task(
    task_id="Instinct-ACP-Stage3-SM-G1-v0",
    env_cfg_factory=_acp_stage3_sm_env_cfg,
    play_env_cfg_factory=_acp_stage3_play_env_cfg,
    instinct_rl_cfg_factory=g1_acp_stage3_instinct_rl_cfg,
    runner_entry_point=ACP_STAGE3_RUNNER_ENTRY_POINT,
)
