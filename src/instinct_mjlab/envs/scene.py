from __future__ import annotations

from mjlab.scene import Scene
from mjlab.terrains.terrain_entity import TerrainEntity


class InstinctScene(Scene):
  """Scene variant that honors terrain cfg.class_type in Instinct tasks."""

  def _add_terrain(self) -> None:
    if self._cfg.terrain is None:
      return
    self._cfg.terrain.num_envs = self._cfg.num_envs
    self._cfg.terrain.env_spacing = self._cfg.env_spacing
    terrain_cls = getattr(self._cfg.terrain, "class_type", TerrainEntity)
    terrain = terrain_cls(self._cfg.terrain, device=self._device)
    self._terrain = terrain
    self._entities["terrain"] = terrain
    frame = self._spec.worldbody.add_frame()
    self._spec.attach(terrain.spec, prefix="", frame=frame)
