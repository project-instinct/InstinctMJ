"""Register Instinct Mj perceptive G1 tasks."""

from instinct_mj.tasks.registry import register_instinct_task

# The perceptive shadowing tasks are trained with SuperMimic (no PPO); the Vae task keeps the stock runner.
SUPERMIMIC_RUNNER_ENTRY_POINT = "instinct_mj.rl.supermimic.runner:SuperMimicRunner"


def g1_perceptive_shadowing_instinct_rl_cfg():
    from .rl_cfg import g1_perceptive_shadowing_instinct_rl_cfg as _cfg

    return _cfg()


def g1_perceptive_shadowing_latent_instinct_rl_cfg():
    from .rl_cfg import g1_perceptive_shadowing_latent_instinct_rl_cfg as _cfg

    return _cfg()


def g1_perceptive_shadowing_render_instinct_rl_cfg():
    from .rl_cfg import g1_perceptive_shadowing_render_instinct_rl_cfg as _cfg

    return _cfg()


def g1_perceptive_shadowing_one_motion_instinct_rl_cfg():
    from .rl_cfg import g1_perceptive_shadowing_one_motion_instinct_rl_cfg as _cfg

    return _cfg()


def g1_perceptive_vae_instinct_rl_cfg():
    from .rl_cfg import g1_perceptive_vae_instinct_rl_cfg as _cfg

    return _cfg()


def _perceptive_shadowing_env_cfg():
    from .perceptive_shadowing_cfg import G1PerceptiveShadowingEnvCfg

    return G1PerceptiveShadowingEnvCfg()


def _perceptive_shadowing_play_env_cfg():
    from .perceptive_shadowing_cfg import G1PerceptiveShadowingEnvCfg_PLAY

    return G1PerceptiveShadowingEnvCfg_PLAY()


def _perceptive_shadowing_one_motion_env_cfg():
    from .perceptive_shadowing_cfg import G1PerceptiveShadowingOneMotionEnvCfg

    return G1PerceptiveShadowingOneMotionEnvCfg()


def _perceptive_shadowing_one_motion_play_env_cfg():
    from .perceptive_shadowing_cfg import G1PerceptiveShadowingOneMotionEnvCfg_PLAY

    return G1PerceptiveShadowingOneMotionEnvCfg_PLAY()


def _perceptive_vae_env_cfg():
    from .perceptive_vae_cfg import G1PerceptiveVaeEnvCfg

    return G1PerceptiveVaeEnvCfg()


def _perceptive_vae_play_env_cfg():
    from .perceptive_vae_cfg import G1PerceptiveVaeEnvCfg_PLAY

    return G1PerceptiveVaeEnvCfg_PLAY()


register_instinct_task(
    task_id="Instinct-Perceptive-Shadowing-G1-v0",
    env_cfg_factory=_perceptive_shadowing_env_cfg,
    play_env_cfg_factory=_perceptive_shadowing_play_env_cfg,
    instinct_rl_cfg_factory=g1_perceptive_shadowing_instinct_rl_cfg,
    runner_entry_point=SUPERMIMIC_RUNNER_ENTRY_POINT,
)

register_instinct_task(
    task_id="Instinct-Perceptive-Shadowing-G1-Play-v0",
    env_cfg_factory=_perceptive_shadowing_play_env_cfg,
    play_env_cfg_factory=_perceptive_shadowing_play_env_cfg,
    instinct_rl_cfg_factory=g1_perceptive_shadowing_instinct_rl_cfg,
    runner_entry_point=SUPERMIMIC_RUNNER_ENTRY_POINT,
)

# The two SuperMimic vision methods with their settings (same env, same policy network; see rl/supermimic/README.md).
for _method, _rl_cfg in (("Latent", g1_perceptive_shadowing_latent_instinct_rl_cfg), ("Render", g1_perceptive_shadowing_render_instinct_rl_cfg)):
    register_instinct_task(
        task_id=f"Instinct-Perceptive-Shadowing-{_method}-G1-v0",
        env_cfg_factory=_perceptive_shadowing_env_cfg,
        play_env_cfg_factory=_perceptive_shadowing_play_env_cfg,
        instinct_rl_cfg_factory=_rl_cfg,
        runner_entry_point=SUPERMIMIC_RUNNER_ENTRY_POINT,
    )
    register_instinct_task(
        task_id=f"Instinct-Perceptive-Shadowing-{_method}-G1-Play-v0",
        env_cfg_factory=_perceptive_shadowing_play_env_cfg,
        play_env_cfg_factory=_perceptive_shadowing_play_env_cfg,
        instinct_rl_cfg_factory=_rl_cfg,
        runner_entry_point=SUPERMIMIC_RUNNER_ENTRY_POINT,
    )

register_instinct_task(
    task_id="Instinct-Perceptive-Shadowing-G1-OneMotion-v0",
    env_cfg_factory=_perceptive_shadowing_one_motion_env_cfg,
    play_env_cfg_factory=_perceptive_shadowing_one_motion_play_env_cfg,
    instinct_rl_cfg_factory=g1_perceptive_shadowing_one_motion_instinct_rl_cfg,
    runner_entry_point=SUPERMIMIC_RUNNER_ENTRY_POINT,
)

register_instinct_task(
    task_id="Instinct-Perceptive-Shadowing-G1-OneMotion-Play-v0",
    env_cfg_factory=_perceptive_shadowing_one_motion_play_env_cfg,
    play_env_cfg_factory=_perceptive_shadowing_one_motion_play_env_cfg,
    instinct_rl_cfg_factory=g1_perceptive_shadowing_one_motion_instinct_rl_cfg,
    runner_entry_point=SUPERMIMIC_RUNNER_ENTRY_POINT,
)

register_instinct_task(
    task_id="Instinct-Perceptive-Vae-G1-v0",
    env_cfg_factory=_perceptive_vae_env_cfg,
    play_env_cfg_factory=_perceptive_vae_play_env_cfg,
    instinct_rl_cfg_factory=g1_perceptive_vae_instinct_rl_cfg,
)

register_instinct_task(
    task_id="Instinct-Perceptive-Vae-G1-Play-v0",
    env_cfg_factory=_perceptive_vae_play_env_cfg,
    play_env_cfg_factory=_perceptive_vae_play_env_cfg,
    instinct_rl_cfg_factory=g1_perceptive_vae_instinct_rl_cfg,
)
