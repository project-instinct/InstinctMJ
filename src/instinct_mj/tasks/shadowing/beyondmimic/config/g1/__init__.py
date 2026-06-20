"""Register Instinct Mj BeyondMimic G1 tasks."""

from instinct_mj.tasks.registry import register_instinct_task

from .rl_cfgs import g1_beyondmimic_instinct_rl_cfg


def _beyondmimic_plane_env_cfg(play: bool):
    from .beyondmimic_plane_cfg import g1_beyondmimic_plane_env_cfg

    return g1_beyondmimic_plane_env_cfg(play=play)


register_instinct_task(
    task_id="Instinct-BeyondMimic-Plane-G1-v0",
    env_cfg_factory=lambda: _beyondmimic_plane_env_cfg(play=False),
    play_env_cfg_factory=lambda: _beyondmimic_plane_env_cfg(play=True),
    instinct_rl_cfg_factory=g1_beyondmimic_instinct_rl_cfg,
)

register_instinct_task(
    task_id="Instinct-BeyondMimic-Plane-G1-Play-v0",
    env_cfg_factory=lambda: _beyondmimic_plane_env_cfg(play=True),
    play_env_cfg_factory=lambda: _beyondmimic_plane_env_cfg(play=True),
    instinct_rl_cfg_factory=g1_beyondmimic_instinct_rl_cfg,
)
