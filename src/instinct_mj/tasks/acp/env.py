"""Environment class of the ACP tasks: ``InstinctRlEnv`` with fresh kinematics for the terminations and rewards."""

from __future__ import annotations

from instinct_mj.envs import InstinctRlEnv

ACP_ENV_ENTRY_POINT = "instinct_mj.tasks.acp.env:AcpRlEnv"


class AcpRlEnv(InstinctRlEnv):
    """``ManagerBasedRlEnv.step`` calls ``sim.forward()`` once, after the reset block and before the observations; the terminations and rewards before it
    read ``xpos`` / ``xquat`` / ``cvel`` that lag ``qpos`` / ``qvel`` by the last physics substep (the last ``mj_step`` computed them from the pre-integration state).
    The ACP rewards and terminations compare body positions and velocities with the reference at the 1 cm / 0.1 m/s level (and the analytic MDP of
    the imagination evaluates them on the exact end state), so the kinematics are refreshed once before the termination manager. No other
    step logic changes (nothing between the physics loop and the terminations writes state; the actuator forces ``qfrc_actuator`` come from the
    control the last substep used, which the refresh keeps)."""

    def load_managers(self) -> None:
        super().load_managers()
        compute = self.termination_manager.compute

        def compute_with_fresh_kinematics(*args, **kwargs):
            self.sim.forward()
            return compute(*args, **kwargs)

        self.termination_manager.compute = compute_with_fresh_kinematics
