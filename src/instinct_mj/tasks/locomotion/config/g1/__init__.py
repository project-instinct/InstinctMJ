"""Register Instinct Mj locomotion G1 tasks."""

from instinct_mj.tasks.registry import register_instinct_task

from .rl_cfgs import g1_locomotion_instinct_rl_cfg


def _locomotion_flat_env_cfg(play: bool):
    from .flat_env_cfg import instinct_g1_locomotion_flat_env_cfg

    return instinct_g1_locomotion_flat_env_cfg(play=play)


register_instinct_task(
    task_id="Instinct-Locomotion-Flat-G1-v0",
    env_cfg_factory=lambda: _locomotion_flat_env_cfg(play=False),
    play_env_cfg_factory=lambda: _locomotion_flat_env_cfg(play=True),
    instinct_rl_cfg_factory=g1_locomotion_instinct_rl_cfg,
)

register_instinct_task(
    task_id="Instinct-Locomotion-Flat-G1-Play-v0",
    env_cfg_factory=lambda: _locomotion_flat_env_cfg(play=True),
    play_env_cfg_factory=lambda: _locomotion_flat_env_cfg(play=True),
    instinct_rl_cfg_factory=g1_locomotion_instinct_rl_cfg,
)
