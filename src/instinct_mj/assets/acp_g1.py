"""Simplified Unitree G1 of the ACP playground (authors' ``g1_29dof.xml``): pelvis-rooted free joint, 23 actuated joints (the wrists are
welded, their links remain), foot = 7 capsules per ankle, hand collision = ``half_sphere.stl`` on the wrist yaw link.

The file is an unmodified copy (``resources/acp_g1``, see its README). What the task does not use is stripped when the spec is loaded:
the mocap goal sphere, the sensors and the motor actuators (mjlab adds its own position / velocity actuators); the geoms that default to
geom group 0 (the collision meshes) move to group 3, because the task's height scanner casts against group 0 (the terrain) only.

``simplified_collision=True`` (default off) replaces the collision *meshes* of the upper body / arms with analytic primitives copied from
InstinctMJ's G1 popsicle model (``assets/resources/unitree_g1/xml/g1_29dof_torsobase_popsicle.xml``); the legs (hip pitch/roll/yaw, knee,
ankle pitch/roll) and the 7 foot capsules keep the authors' meshes exactly. Visual meshes are never touched, the body masses / inertias are
identical (the primitives have ``density=0`` and every body keeps its ``<inertial>``). See ``COLLISION_SIMPLIFY.md``.
"""

from __future__ import annotations

import hashlib
import os

import mujoco

__file_dir__ = os.path.dirname(os.path.realpath(__file__))

ACP_G1_XML: str = os.path.join(__file_dir__, "resources/acp_g1/g1_29dof.xml")
ACP_G1_MESHES_DIR: str = os.path.join(__file_dir__, "resources/acp_g1/meshes")

# Bodies whose collision mesh copies are removed by ``simplified_collision``. The ankle pitch / roll links are deliberately absent: the
# authors' 7 foot capsules define the ``foot`` contact body, they stay exactly as they are.
SIMPLIFIED_COLLISION_BODIES: tuple[str, ...] = (
    "pelvis",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
)

_MESH = mujoco.mjtGeom.mjGEOM_MESH
_CAPSULE = mujoco.mjtGeom.mjGEOM_CAPSULE
_SPHERE = mujoco.mjtGeom.mjGEOM_SPHERE

# Primitives of the replaced bodies, in the body frame of the ACP model (same Unitree URDF frames). Tuple: (type, size, pos, quat).
# The torso capsules are from ``g1_29dof_torsobase_popsicle.xml``. The ACP model has its own analytic cylinders on the shoulder pitch /
# roll links (the authors' collision for the shoulder ball); they are kept.
#
# Everything else is fitted to the ACP meshes: the primitive stays inside the outer shell, not around it. That is deliberate
# (``# ACP-unspecified``): a capsule that contains the whole mesh looks and behaves 1-2 cm fatter than the robot, so each primitive is
# sized to the shell's outer surface where the mesh is convex and sits a few mm inside the plastic. Costs: the robot sinks a few mm into
# itself before self-contact (only relaxes the self-collision term) and into the terrain before a contact is generated (measured below
# per contact body, all <= 2 mm).
#
# Notes on single bodies (all measured on the compiled models):
# * shoulder_yaw / elbow: one capsule along each bone, cropped at the joints, radius 3.1 / 3.0 cm (the shells are 6.4-7.0 cm across;
#   the radii are sized to their outer surface, so the collision limb is as thick as the visual one). The two shells are separate rings
#   around the elbow joint (a 3.6 mm hollow between them), and the upper-arm capsule is 1.6 cm off the elbow joint axis, so the joint
#   itself read as a hole; a sphere r=3.1 on the elbow link, centred under the shoulder-yaw axis, fills it and matches the hinge width
#   (# ACP-unspecified: the authors have no elbow collision primitive to copy). The 5.7 cm shoulder cup and the elbow fork still stay
#   outside it.
# * wrist: the popsicle ``wrist_yaw`` capsule is its rubber *hand*, not the wrist. The three welded wrist links (roll / pitch / yaw, the
#   ACP model keeps them as bodies) each carry their own capsule along x, fitted to their own shell, plus the hand sphere on ``yaw``, so
#   the ``hand`` contact body is the sphere + the 4.2 cm yaw shell, and the middle of the lower arm is collision-only. ``label_contacts``
#   samples the meshes (never this file), so the labels are unaffected.
# * pelvis: a y-axis capsule 12 cm across / 13.4 cm wide (the mesh is 12.0 x 13.6 x 12.4 cm) instead of the popsicle's 14 cm sphere.
#   With the hip motors it forms the ``pelvis`` contact body; sitting on it lands 2 mm high, lying on the back 1.4 mm high.
# * hip_pitch (the motor the G1 sits on): sphere r=0.048, lowest point 0.3 mm above the mesh's.
# * hip_roll / hip_yaw (thigh): slim capsules, 8.4 cm and 7.2 cm across. The mesh's 15.3 cm x-extent is the knee-guard plate.
#   The thigh capsule is tilted onto the hip -> knee line (knee joint at (-0.0783, 0.0021, -0.1773): 0.416 rad, 23.8 deg about y) and
#   pushed 1.0 cm forward of that line (along the shell's wide axis), into the middle of the mesh: the shell is lopsided around the
#   joint line (5.3 cm in front of it, 1.9 cm behind at mid-thigh), so a capsule centred on the line stuck out of the back by ~1.7 cm.
#   The shape is not representable by one capsule (the knee-guard plate), so the surface still deviates by p95 1.9 cm, max 3.2 cm.
# * knee (shank): a capsule along the shank (7.2 cm across, as wide as the mesh's thin axis, 32 cm long) plus a short x-axis capsule that
#   reproduces the knee cap (7.2 cm wide, reaching 6.4 cm in front of the joint like the mesh). Rounded, so the knee slides over step
#   edges instead of catching on them; kneeling lands 0.4 mm high. ``*_knee_link`` and its parent thigh are a parent / child pair, so the
#   filterparent does not let them collide at any flexion (checked to hip -2.4 rad / knee 2.8 rad, and against the pelvis it clears).
_SIMPLIFIED_COLLISION_GEOMS: dict[str, tuple[tuple[int, tuple[float, float, float], tuple[float, float, float], tuple[float, float, float, float]], ...]] = {
    "pelvis": ((_CAPSULE, (0.06, 0.008, 0.0), (0.0, 0.0, -0.09), (0.707105, 0.707108, 0.0, 0.0)),),
    "torso_link": (
        (_CAPSULE, (0.073, 0.032, 0.0), (0.005, 0.0, 0.22), (0.707105, 0.707108, 0.0, 0.0)),
        (_CAPSULE, (0.07, 0.028, 0.0), (0.005, 0.0, 0.13), (0.707105, 0.707108, 0.0, 0.0)),
        (_CAPSULE, (0.065, 0.02, 0.0), (0.005, 0.0, 0.06), (0.707105, 0.707108, 0.0, 0.0)),
        (_CAPSULE, (0.068, 0.005, 0.0), (0.01, 0.0, 0.415), (-3.67321e-06, 1.0, 0.0, 0.0)),
    ),
    # Arms: capsule axis along each link, cropped at the joints so neither capsule pokes through the neighbouring segment
    # (upper arm spans shoulder->elbow, 8.2 cm of bone / 11.4 cm of capsule; forearm spans elbow->wrist_roll, 10.1 cm of bone /
    # 10.0 cm of capsule). Without the crop each capsule ran ~3 cm past the elbow, which is what made the arm read as "too long".
    # Radius sized to the outer shell (6.2 / 6.0 cm across vs the shell's 6.4-7.0 cm): the arm reads as thick as the visual model,
    # and the joint sphere below is the same width. The elbow joint gets its own sphere: the two shells are separate rings (a 3.6 mm
    # hollow in the ACP meshes) and the upper-arm capsule sits 1.6 cm off the joint axis, so without it the joint reads as a hole
    # between the two capsules (and the joint would not be a contact surface at all). It is centred under the shoulder-yaw axis
    # (x = -1.6 cm in the elbow frame), the way the two shells close the joint there.
    "left_shoulder_yaw_link": ((_CAPSULE, (0.031, 0.026, 0.0), (0.0, 0.0, -0.025), (-3.67321e-06, 1.0, 0.0, 0.0)),),
    "right_shoulder_yaw_link": ((_CAPSULE, (0.031, 0.026, 0.0), (0.0, 0.0, -0.025), (-3.67321e-06, 1.0, 0.0, 0.0)),),
    "left_elbow_link": (
        (_CAPSULE, (0.030, 0.020, 0.0), (0.050, 0.0, -0.01), (0.707105, 0.0, -0.707108, 0.0)),
        (_SPHERE, (0.031, 0.0, 0.0), (-0.016, 0.0, 0.002), (1.0, 0.0, 0.0, 0.0)),
    ),
    "right_elbow_link": (
        (_CAPSULE, (0.030, 0.020, 0.0), (0.050, 0.0, -0.01), (0.707105, 0.0, -0.707108, 0.0)),
        (_SPHERE, (0.031, 0.0, 0.0), (-0.016, 0.0, 0.002), (1.0, 0.0, 0.0, 0.0)),
    ),
    # The three wrist links are welded in the ACP model but stay separate bodies, so each keeps its own capsule, fitted to its own
    # shell (x extent in its own frame, radius 2.6 cm kept inside the 5.6-6.0 cm shell): roll [0.0, 5.6] cm, pitch [-2.9, 7.5],
    # yaw [-1.8, 4.2]. Only the yaw link is a contact body (``left_hand``), the two in between are collision-only, so the middle of
    # the lower arm carries no contact label. Replacing them by a single capsule on the yaw link (the first version) merged three
    # links' worth of collision into the hand contact body, which read as "the whole forearm is a contact".
    # The hand ``half_sphere.stl`` -> analytic sphere. Measured from the STL (46258 triangles): the dome is a sphere of radius
    # 0.039948 m centred 0.02207 m above its flat face in the mesh frame; the geom maps the mesh +z to the link +x at pos
    # (0.03, 0, 0), so the sphere is centred at (0.03 + 0.02207, 0, 0) = (0.05207, 0, 0); radius rounded to 0.04 contains the
    # discretised mesh (max 0.039992 m from that centre).
    "left_wrist_roll_link": ((_CAPSULE, (0.026, 0.002, 0.0), (0.028, 0.0, 0.0), (0.707105, 0.0, -0.707108, 0.0)),),
    "right_wrist_roll_link": ((_CAPSULE, (0.026, 0.002, 0.0), (0.028, 0.0, 0.0), (0.707105, 0.0, -0.707108, 0.0)),),
    "left_wrist_pitch_link": ((_CAPSULE, (0.026, 0.0255, 0.0), (0.023, 0.0, 0.0), (0.707105, 0.0, -0.707108, 0.0)),),
    "right_wrist_pitch_link": ((_CAPSULE, (0.026, 0.0255, 0.0), (0.023, 0.0, 0.0), (0.707105, 0.0, -0.707108, 0.0)),),
    "left_wrist_yaw_link": (
        (_CAPSULE, (0.026, 0.00375, 0.0), (0.01175, 0.0, 0.0), (0.707105, 0.0, -0.707108, 0.0)),
        (_SPHERE, (0.04, 0.0, 0.0), (0.05207, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
    ),
    "right_wrist_yaw_link": (
        (_CAPSULE, (0.026, 0.00375, 0.0), (0.01175, 0.0, 0.0), (0.707105, 0.0, -0.707108, 0.0)),
        (_SPHERE, (0.04, 0.0, 0.0), (0.05207, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
    ),
    # Legs (see the note above): hip_pitch = the motor ball, hip_roll / hip_yaw = thigh, knee = shank capsule + knee cap.
    "left_hip_pitch_link": ((_SPHERE, (0.048, 0.0, 0.0), (0.0, 0.0494, -0.032), (1.0, 0.0, 0.0, 0.0)),),
    "right_hip_pitch_link": ((_SPHERE, (0.048, 0.0, 0.0), (0.0, -0.0494, -0.032), (1.0, 0.0, 0.0, 0.0)),),
    "left_hip_roll_link": ((_CAPSULE, (0.042, 0.030, 0.0), (0.027, 0.0, -0.038), (1.0, 0.0, 0.0, 0.0)),),
    "right_hip_roll_link": ((_CAPSULE, (0.042, 0.030, 0.0), (0.027, 0.0, -0.038), (1.0, 0.0, 0.0, 0.0)),),
    "left_hip_yaw_link": ((_CAPSULE, (0.036, 0.060, 0.0), (-0.03845, 0.0, -0.11204), (0.978481, 0.0, 0.206335, 0.0)),),
    "right_hip_yaw_link": ((_CAPSULE, (0.036, 0.060, 0.0), (-0.03845, 0.0, -0.11204), (0.978481, 0.0, 0.206335, 0.0)),),
    "left_knee_link": (
        (_CAPSULE, (0.036, 0.124, 0.0), (0.012, 0.0, -0.155), (1.0, 0.0, 0.0, 0.0)),
        (_CAPSULE, (0.036, 0.015, 0.0), (0.013, 0.0, -0.020), (0.707105, 0.0, 0.707108, 0.0)),
    ),
    "right_knee_link": (
        (_CAPSULE, (0.036, 0.124, 0.0), (0.012, 0.0, -0.155), (1.0, 0.0, 0.0, 0.0)),
        (_CAPSULE, (0.036, 0.015, 0.0), (0.013, 0.0, -0.020), (0.707105, 0.0, 0.707108, 0.0)),
    ),
}


def _simplify_collision(spec: mujoco.MjSpec) -> None:
    """Remove the collision mesh copies of ``SIMPLIFIED_COLLISION_BODIES`` and attach the analytic primitives (group 3, contype /
    conaffinity 1, density 0, contact parameters copied from the mesh they replace)."""
    for body_name in SIMPLIFIED_COLLISION_BODIES:
        body = spec.body(body_name)
        # Defaults are the ACP collision class; they are overwritten by the values of the mesh being replaced below.
        friction = [1.0, 0.005, 1e-4]
        condim = 3
        solref = [0.02, 1.0]
        solimp = [0.9, 0.95, 0.001, 0.5, 2.0]
        priority = 0
        margin = 0.0
        gap = 0.0
        solmix = 1.0
        rgba = [0.7, 0.7, 0.7, 1.0]
        removed = []
        for geom in list(body.geoms):
            if geom.type == _MESH and geom.group == 3:  # group 1 = the visual mesh, which is kept
                removed.append(geom)
        if removed:
            # Contact parameters of the meshes being replaced (all default in the ACP file: friction 1 0.005 1e-4, condim 3,
            # solref 0.02 1, solimp 0.9 0.95 0.001). Copy them so the primitives behave exactly like the meshes at the contact solver.
            ref = removed[0]
            friction = list(ref.friction)
            condim = ref.condim
            solref = list(ref.solref)
            solimp = list(ref.solimp)
            priority = ref.priority
            margin = ref.margin
            gap = ref.gap
            solmix = ref.solmix
            rgba = list(ref.rgba)
            for geom in removed:
                spec.delete(geom)
        for index, (geom_type, size, pos, quat) in enumerate(_SIMPLIFIED_COLLISION_GEOMS.get(body_name, ())):
            geom = body.add_geom()
            geom.name = f"simplified_{body_name}_{index}"
            geom.type = geom_type
            geom.size[:] = size
            geom.pos[:] = pos
            geom.quat[:] = quat
            geom.group = 3
            geom.contype = 1
            geom.conaffinity = 1
            geom.density = 0.0  # every body has an explicit <inertial>: keep masses / inertias exactly those of the authors' model
            geom.friction[:] = friction
            geom.condim = condim
            geom.solref[:] = solref
            geom.solimp[:] = solimp
            geom.priority = priority
            geom.margin = margin
            geom.gap = gap
            geom.solmix = solmix
            geom.rgba[:] = rgba


def get_acp_g1_spec(simplified_collision: bool = False) -> mujoco.MjSpec:
    """The playground G1 without mocap target, sensors and motors; the pelvis (root) at a standing pose with identity orientation (the
    file spawns it prone).

    ``simplified_collision``: replace the upper-body / arm collision meshes by capsules / spheres, keeping the whole leg's meshes and the
    foot capsules (default off: the authors' model unchanged)."""
    spec = mujoco.MjSpec.from_file(ACP_G1_XML)
    spec.delete(spec.body("target_sphere"))
    for sensor in list(spec.sensors):
        spec.delete(sensor)
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    for geom in spec.geoms:
        if geom.group == 0:
            geom.group = 3  # group 0 is the terrain's: the height scanner casts against it only (the collision meshes default to 0)
    pelvis = spec.body("pelvis")
    pelvis.pos = [0.0, 0.0, 0.78]
    pelvis.quat = [1.0, 0.0, 0.0, 0.0]
    if simplified_collision:
        _simplify_collision(spec)
    return spec


def write_acp_g1_mjcf(simplified_collision: bool = False) -> str:
    """The stripped model as a flat MJCF file (``~/.cache/instinct_mj``, content-addressed) for the kinematics of the motion reference
    (``robot_model_path``: no default classes, absolute mesh directory). Collision geoms do not affect the forward kinematics, so the
    default (meshes) file is also valid for the simplified model."""
    spec = get_acp_g1_spec(simplified_collision=simplified_collision)
    spec.meshdir = ACP_G1_MESHES_DIR
    xml = spec.to_xml()
    cache = os.path.expanduser("~/.cache/instinct_mj")
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, f"acp_g1_{hashlib.sha1(xml.encode()).hexdigest()[:12]}.xml")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(xml)
    return path
