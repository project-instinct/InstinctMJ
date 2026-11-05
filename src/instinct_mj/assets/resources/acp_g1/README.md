# ACP playground G1

Unmodified copy of the simplified Unitree G1 of the ACP web playground (Humanoid Mobility via Attention-based Contacts,
<https://leggedrobotics.github.io/ACP/playground/>): `g1_29dof.xml` and the 37 mesh files it references (`meshes/`, the
`g1_mocap_markers.xml` and terrain files of the playground are not needed).

* pelvis-rooted free joint (`floating_base_joint`), 23 hinge joints (legs 12, waist 3, arms 8); the wrist links exist but are welded
* collision: the link meshes (convex hulls in MuJoCo), 7 capsules per foot (`*_foot[1-7]_collision`), `half_sphere.stl` on the wrist yaw
  links as hand; the rubber hand mesh is commented out in the file
* joint `armature`, `frictionloss`, `damping`, ranges and `actuatorfrcrange` are defined per joint class in the file
* `<option>`: iterations 50, Newton solver, impratio 1, pyramidal cone, no timestep (MuJoCo default 0.002); the mjlab simulation config
  of the task overrides the whole option block (timestep 0.005, iterations 10, ...)

`instinct_mj.assets.acp_g1.get_acp_g1_spec()` loads the file and strips, at load time only: the mocap body `target_sphere`, all 77 `<sensor>`
elements, the 23 `<motor>` actuators and the group-0 default of the collision meshes (moved to geom group 3: the task's height scanner casts rays against group 0, the terrain, only) (mjlab adds its position / velocity actuators with the deploy gains); the pelvis start pose (the file
spawns it prone) is set to identity orientation at 0.78 m.
