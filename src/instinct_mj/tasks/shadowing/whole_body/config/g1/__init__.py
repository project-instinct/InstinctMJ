"""Register Instinct Mj whole-body shadowing G1 tasks."""

from instinct_mj.tasks.registry import register_instinct_task

from .rl_cfgs import g1_shadowing_instinct_rl_cfg


def _plane_shadowing_env_cfg(play: bool):
    from .plane_shadowing_cfg import g1_plane_shadowing_env_cfg

    return g1_plane_shadowing_env_cfg(play=play)


register_instinct_task(
    task_id="Instinct-Shadowing-WholeBody-Plane-G1-v0",
    env_cfg_factory=lambda: _plane_shadowing_env_cfg(play=False),
    play_env_cfg_factory=lambda: _plane_shadowing_env_cfg(play=True),
    instinct_rl_cfg_factory=g1_shadowing_instinct_rl_cfg,
)

register_instinct_task(
    task_id="Instinct-Shadowing-WholeBody-Plane-G1-Play-v0",
    env_cfg_factory=lambda: _plane_shadowing_env_cfg(play=True),
    play_env_cfg_factory=lambda: _plane_shadowing_env_cfg(play=True),
    instinct_rl_cfg_factory=g1_shadowing_instinct_rl_cfg,
)
