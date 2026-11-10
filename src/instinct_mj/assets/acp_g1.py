"""Simplified Unitree G1 of the ACP playground (authors' ``g1_29dof.xml``): pelvis-rooted free joint, 23 actuated joints (the wrists are
welded, their links remain), foot = 7 capsules per ankle, hand collision = ``half_sphere.stl`` on the wrist yaw link.

The file is an unmodified copy (``resources/acp_g1``, see its README). What the task does not use is stripped when the spec is loaded:
the mocap goal sphere, the sensors and the motor actuators (mjlab adds its own position / velocity actuators); the geoms that default to
geom group 0 (the collision meshes) move to group 3, because the task's height scanner casts against group 0 (the terrain) only.

``simplified_collision=True`` (default off; geometry v3) replaces ALL the collision *meshes* of the body and the legs / arms (and the
authors' shoulder pitch / roll cylinders) with analytic primitives. The torso capsules are copied from InstinctMJ's G1 popsicle model
(``assets/resources/unitree_g1/xml/g1_29dof_torsobase_popsicle.xml``); each limb segment is one continuous capsule: the upper arm is a single
capsule on ``shoulder_yaw`` (popsicle's length / position, radius fitted to the authors' shell), the thigh is a single capsule on ``hip_yaw``
that overlaps the ``hip_roll`` primitives (no gap at the hip; the hip_roll itself is two capsules, an L for the block + front plate), the
shank is one capsule on ``knee``. The robot's collision shell is then mesh-free: only the 7 foot capsules per ankle (``ankle_roll``, they
define the ``foot`` contact body) stay as the authors made them; the small ankle_pitch collision plate is deleted, not replaced. Visual
meshes are never touched, the body masses / inertias are identical (the primitives have ``density=0`` and every body keeps its
``<inertial>``). See ``COLLISION_SIMPLIFY.md``.
"""

from __future__ import annotations

import hashlib
import os

import mujoco

__file_dir__ = os.path.dirname(os.path.realpath(__file__))

ACP_G1_XML: str = os.path.join(__file_dir__, "resources/acp_g1/g1_29dof.xml")
ACP_G1_MESHES_DIR: str = os.path.join(__file_dir__, "resources/acp_g1/meshes")

# Bodies whose collision meshes / cylinders are removed by ``simplified_collision``. The ankle *roll* links are deliberately absent: the
# authors' 7 foot capsules define the ``foot`` contact body, they stay exactly as they are. The ankle *pitch* bodies get their small
# collision plate deleted and are not replaced (v3): it is 3.6 x 1.3 x 3.8 cm, thin enough that no capsule / sphere stays within 5 mm of
# it, and the shank capsule already covers its upper half. What it leaves is a collision-free band between the shank capsule and the
# foot capsules (17.6 mm at the default pose, 17.6-18.7 mm over the ankle pitch / roll limits, see the note below): a hole in the
# self-collision shell, not in the ground contact (the foot capsules still touch the terrain first).
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
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
)

_MESH = mujoco.mjtGeom.mjGEOM_MESH
_CAPSULE = mujoco.mjtGeom.mjGEOM_CAPSULE
_SPHERE = mujoco.mjtGeom.mjGEOM_SPHERE
_CYLINDER = mujoco.mjtGeom.mjGEOM_CYLINDER

# Primitives of the replaced bodies, in the body frame of the ACP model (same Unitree URDF frames). Tuple: (type, size, pos, quat).
# The torso capsules are from ``g1_29dof_torsobase_popsicle.xml``. The authors' analytic cylinders on the shoulder pitch / roll links are
# removed (v2): like the popsicle, the shoulder is covered by the one upper-arm capsule.
#
# Everything else is fitted to the ACP meshes: the primitive stays inside the outer shell, not around it. That is deliberate
# (``# ACP-unspecified``): a capsule that contains the whole mesh looks and behaves 1-2 cm fatter than the robot, so each primitive is
# sized to the shell's outer surface where the mesh is convex and sits a few mm inside the plastic. Costs: the robot sinks a few mm into
# itself before self-contact (only relaxes the self-collision term) and into the terrain before a contact is generated (measured below
# per contact body, all <= 2 mm).
#
# Notes on single bodies (all measured on the compiled models):
# * shoulder_yaw / elbow (v2): the upper arm is one capsule on shoulder_yaw (popsicle's pos / half-length, r = 3.1 cm: the shells are
#   6.4-7.0 cm across, sized to their outer surface), spanning the shoulder to the elbow; the forearm is one capsule on the elbow link
#   (r = 3.0 cm). The two shells are separate rings around the elbow joint (a 3.6 mm hollow between them), and the upper-arm capsule is
#   1.6 cm off the elbow joint axis, so the joint itself read as a hole; a sphere r=3.1 on the elbow link, centred under the shoulder-yaw
#   axis, fills it and matches the hinge width (# ACP-unspecified: the authors have no elbow collision primitive to copy). The capsule
#   reaches ~1.4 cm from the torso capsules with the arm hanging; upper arm / torso are not parent and child, so they do collide when the
#   arm is pressed in (shoulder roll beyond ~0.15 rad of adduction), the same onset as the v1 geometry, never at rest or in the reference
#   motions.
# * wrist: the popsicle ``wrist_yaw`` capsule is its rubber *hand*, not the wrist. The three welded wrist links (roll / pitch / yaw, the
#   ACP model keeps them as bodies) each carry their own capsule along x, fitted to their own shell, plus the hand sphere on ``yaw``, so
#   the ``hand`` contact body is the sphere + the 4.2 cm yaw shell, and the middle of the lower arm is collision-only. ``label_contacts``
#   samples the meshes (never this file), so the labels are unaffected.
# * pelvis: a y-axis capsule 12 cm across / 13.4 cm wide (the mesh is 12.0 x 13.6 x 12.4 cm) instead of the popsicle's 14 cm sphere.
#   With the hip motors it forms the ``pelvis`` contact body; sitting on it lands 2 mm high, lying on the back 1.4 mm high.
# * hip_pitch (the motor the G1 sits on): sphere r=0.048, lowest point 0.3 mm above the mesh's.
# * hip_roll (v3, two primitives): the mesh is an L in the x-z side view: a bottom block x [-1.8, +7.2] cm, z [-11.1, -5.8] cm (8.4 cm
#   wide in y) and a front plate x [+3.3, +7.2] cm, z [-5.8, +3.4] cm. One z capsule cannot be both: v2 (r = 4.2 cm, 8.4 cm across, centre
#   (2.7, 0, -3.8) cm) matched the block, but its rear hemisphere ran up to z = +3.4 cm at x = -1.5 cm, i.e. 4.8 cm of extra volume behind
#   the plate's back wall (x = +3.25 cm) -- the fat hip / butt in the renders. v3 splits it at the L's step (z = -5.8 cm):
#   - block: x-axis capsule, r = 4.0 cm, half length 0.5 cm, centre (2.75, 0, -9.8) cm -> x [-1.75, +7.25], z [-13.8, -5.8] cm.
#   - plate: z-axis capsule, r = 2.0 cm, half length 2.6 cm, centre (5.0, 0, -1.2) cm -> x [+3.0, +7.0], z [-5.8, +3.4] cm. Thinner than
#     the 3.9 cm plate on purpose: anything thicker in x or reaching above z = -5.8 cm re-creates the rear bulge.
#   Fit to the visual mesh (surface samples / mesh vertices): primitive-outside-mesh p95 2.3 cm, max 2.7 cm; mesh-not-covered p95 2.2 cm,
#   max 2.6 cm (v2: 3.6 / 4.8 and 2.1 / 2.3). Inside the mesh's own z range the protrusion is p95 0.6 cm, max 0.9 cm; the 2.7 cm maximum is
#   the block's bottom cap 2.4 cm below the mesh floor, 1.4 cm inside the thigh mesh at the default pose (hip_roll is the thigh's parent,
#   so it is filtered). The 2.2 cm uncovered is the plate's sides (|y| 2.0 vs 4.2 cm), where the hip_pitch sphere (r = 4.8 cm) sits next
#   to it. Overlaps hip_pitch by 1.8 cm (v2 6.2 cm) and the thigh / hip_yaw by 3.3 cm (v2 0.5 cm) at the default pose, -1.8 / -3.3 cm at
#   the worst hip joint limit: still attached at every pose, no new seam.
# * hip_yaw (thigh, v2): a slim capsule, 7.2 cm across. The mesh's 15.3 cm x-extent is the knee-guard plate.
#   The thigh capsule is tilted onto the hip -> knee line (knee joint at (-0.0783, 0.0021, -0.1773): 0.416 rad, 23.8 deg about y) and
#   pushed 1.0 cm forward of that line (along the shell's wide axis), into the middle of the mesh: the shell is lopsided around the
#   joint line (5.3 cm in front of it, 1.9 cm behind at mid-thigh), so a capsule centred on the line stuck out of the back by ~1.7 cm.
#   The shape is not representable by one capsule (the knee-guard plate), so the surface still deviates by p95 1.9 cm, max 3.2 cm.
#   v1 stopped the thigh capsule 0.096 m up the axis from its centre and left a 3.7 cm hole (z 0.506 .. 0.543 at the default pose) to the
#   hip_roll capsule although the two shells are joined (along this axis the thigh mesh spans [-0.106, +0.118] from the capsule centre and
#   hip_roll's [0.113, 0.266]). v2 extends the thigh capsule up to +0.140 on the axis (half length 0.060 -> 0.082, centre moved 2.2 cm along
#   the axis), overlapping the hip_roll capsule by 5 mm at the default pose (4.8 mm at the hip_yaw limit); the two are parent / child, so
#   the overlap never collides. The bottom end (-0.096) is unchanged: it sits inside the knee capsule (3.9 cm overlap).
# * knee (shank): a capsule along the shank (7.2 cm across, as wide as the mesh's thin axis, 32 cm long) plus a short x-axis capsule that
#   reproduces the knee cap (7.2 cm wide, reaching 6.4 cm in front of the joint like the mesh). Rounded, so the knee slides over step
#   edges instead of catching on them; kneeling lands 0.4 mm high. ``*_knee_link`` and its parent thigh are a parent / child pair, so the
#   filterparent does not let them collide at any flexion (checked to hip -2.4 rad / knee 2.8 rad, and against the pelvis it clears).
# * ankle_pitch (v3): the authors' collision mesh (3.6 x 1.3 x 3.8 cm plate around the pitch axis) is removed, not replaced: it is thin
#   enough that no capsule / sphere fits it inside 5 mm, and the shank capsule already covers its upper half. The ankle is therefore
#   collision-free between the shank capsule and the authors' 7 foot capsules: 1.76 cm at the default pose, 2.08 cm at the pitch upper
#   limit, 0.87 cm at the pitch lower limit (toe down), 1.62-1.63 cm at the roll limits. Foot contact is unchanged: the deleted plate sat
#   1.76 cm above the capsules' tops, and nothing else was added, so the foot capsules are still the lowest geoms of the leg and the
#   terrain meets them first at every pose. With ``simplified_collision`` the robot has no collision mesh left at all (39 capsules, 6
#   spheres); mujoco_warp's MULTICCD warning then lists only ``CAPSULE-CYLINDER`` (the terrain's cylinders).
_SIMPLIFIED_COLLISION_GEOMS: dict[str, tuple[tuple[int, tuple[float, float, float], tuple[float, float, float], tuple[float, float, float, float]], ...]] = {
    "pelvis": ((_CAPSULE, (0.06, 0.008, 0.0), (0.0, 0.0, -0.09), (0.707105, 0.707108, 0.0, 0.0)),),
    "torso_link": (
        (_CAPSULE, (0.073, 0.032, 0.0), (0.005, 0.0, 0.22), (0.707105, 0.707108, 0.0, 0.0)),
        (_CAPSULE, (0.07, 0.028, 0.0), (0.005, 0.0, 0.13), (0.707105, 0.707108, 0.0, 0.0)),
        (_CAPSULE, (0.065, 0.02, 0.0), (0.005, 0.0, 0.06), (0.707105, 0.707108, 0.0, 0.0)),
        (_CAPSULE, (0.068, 0.005, 0.0), (0.01, 0.0, 0.415), (-3.67321e-06, 1.0, 0.0, 0.0)),
    ),
    # Arms (geometry v2): the upper arm is ONE capsule on the shoulder_yaw link, like InstinctMJ's popsicle (same pos / length / flip), from
    # the shoulder (the authors' shoulder pitch / roll cylinders are gone) down to the elbow. Spans z in [-0.080, +0.050] of the yaw frame
    # plus the 3.1 cm hemispherical caps, i.e. up to the lower edge of the shoulder cup and down to the elbow joint sphere below.
    # The forearm is the elbow link's capsule. Radius 3.1 cm as before (sized to the shell, whose width is 6.4-7.0 cm).
    # The elbow joint gets its own sphere: the two shells are separate rings (a 3.6 mm hollow in the ACP meshes) and the upper-arm
    # capsule sits 1.6 cm off the joint axis, so without it the joint reads as a hole (# ACP-unspecified).
    "left_shoulder_yaw_link": ((_CAPSULE, (0.031, 0.065, 0.0), (0.0, 0.0, -0.015), (-3.67321e-06, 1.0, 0.0, 0.0)),),
    "right_shoulder_yaw_link": ((_CAPSULE, (0.031, 0.065, 0.0), (0.0, 0.0, -0.015), (-3.67321e-06, 1.0, 0.0, 0.0)),),
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
    "left_hip_roll_link": (
        (_CAPSULE, (0.040, 0.005, 0.0), (0.0275, 0.0, -0.098), (0.707105, 0.0, 0.707108, 0.0)),
        (_CAPSULE, (0.020, 0.026, 0.0), (0.050, 0.0, -0.012), (1.0, 0.0, 0.0, 0.0)),
    ),
    "right_hip_roll_link": (
        (_CAPSULE, (0.040, 0.005, 0.0), (0.0275, 0.0, -0.098), (0.707105, 0.0, 0.707108, 0.0)),
        (_CAPSULE, (0.020, 0.026, 0.0), (0.050, 0.0, -0.012), (1.0, 0.0, 0.0, 0.0)),
    ),
    "left_hip_yaw_link": ((_CAPSULE, (0.036, 0.082, 0.0), (-0.02956, 0.0, -0.09192), (0.978481, 0.0, 0.206335, 0.0)),),
    "right_hip_yaw_link": ((_CAPSULE, (0.036, 0.082, 0.0), (-0.02956, 0.0, -0.09192), (0.978481, 0.0, 0.206335, 0.0)),),
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
            # group 1 = the visual mesh, which is kept; the authors' shoulder pitch / roll cylinders go too (v2: one upper-arm capsule)
            if geom.group == 3 and geom.type in (_MESH, _CYLINDER):
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
    # No self-collision (as zita-ch/supermimic_vibed, ``enabled_self_collisions=False``): contype 2 / conaffinity 1 never matches between two robot
    # geoms, while the terrain (contype 1) still collides with every robot collision geom (its contype & the robot's conaffinity).
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:
            geom.contype, geom.conaffinity = 2, 1
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
