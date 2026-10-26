# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for the OpenArm cabinet door-opening task (IsaacLab 3.0.0).

The scene is a floor-standing full-height black storage cabinet (RPL
"603599BK", articulated USD authored under ``src/rpl_cabinet/assets/
rpl_cabinet.usda``) with the bimanual OpenArm facing it from across its front.
This mirrors the real ``openarm_open_lab_door`` dataset: the operator pulls the
LEFT door with the LEFT arm, the right arm stays near home. The cabinet
articulation carries four revolute DOFs:

  hinge_left   -110..0  deg      (door revolute Z at the left stile)
  hinge_right    0..110 deg      (door revolute Z at the right stile)
  latch_left     0..90  deg      (handle lever revolute Y at the pivot)
  latch_right  -90..0   deg      (handle lever revolute Y, mirrored)

Scene geometry notes:

* The world frame is the stack-task convention: ground at z=-1.05, props at
  z=0. The cabinet is placed on the ground (base at z=-1.05 so its toe-kick
  rests on the floor), the OpenArm stands beside it at a comfortable torso
  mount height. The cabinet is yawed -90 deg (xyzw: 0,0,-sin45,cos45) so its
  authored -Y front face points toward -X -- the direction the robot (which
  faces +X) is facing.
* The OpenArm USD ships without a stand (~0.773 m tall native); we drop in the
  shared Isaac Nucleus ``Stand`` prop under it, z-scaled to the ground, same
  as the centrifuge pack. Base lift is tuned so the hand envelope covers the
  handle (authored at 1.020 m above the cabinet floor).
* The cabinet uses its own authored USD articulation (fixed base, four
  revolute joints). No ``prepare_assets`` step is needed -- the USD already
  carries ``PhysicsArticulationRootAPI`` + per-link ``PhysicsRigidBodyAPI`` /
  ``PhysicsCollisionAPI``, which Isaac Lab reads directly.

What's STARTING POINT and needs in-sim tuning (same spirit as the centrifuge
pack):

* Robot base pose + XR anchor -- the HANDOFF.md known-gap "A rigid base shift
  computed from the recorded pose drives the torso 230 mm into the cabinet"
  is not solved here. The constants below put the robot in a plausible pose
  (hand at handle height, torso not inside the cabinet) and leave the final
  measurement to a lab-time pass.
* The ``door_opened`` angle threshold -- half the full 110 deg swing is a
  conservative starting point, re-measure once teleop demos exist.

Concrete variants plug in the robot actions:
  * ``ik_abs_env_cfg`` / ``ik_rel_env_cfg`` -- VR teleop (dual-source retargeters).
  * ``visuomotor_env_cfg`` -- adds cameras + image obs for headless dataset gen.

Quaternion conventions (they differ, keep them straight):
  * ``InitialStateCfg.rot`` (assets) is **(x, y, z, w)**; identity = (0,0,0,1).
  * ``CameraCfg.OffsetCfg.rot`` is **(w, x, y, z)**.
"""

from dataclasses import MISSING
from importlib.resources import files as _pkg_files

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg

from .. import mdp

# Cabinet USD bundled inside the package wheel (see
# ``src/rpl_cabinet/assets/``). ``importlib.resources.files`` resolves the path
# for both editable and built-wheel installs.
CABINET_ASSETS_DIR = str(_pkg_files("rpl_cabinet") / "assets")
CABINET_USD_PATH = f"{CABINET_ASSETS_DIR}/rpl_cabinet.usda"

# ---------------------------------------------------------------------------
# Scene geometry. Ground at z=-1.05 (matches the stack-task convention). The
# cabinet's authored origin is at the centre of the footprint on the floor, so
# dropping it at z=-1.05 lets the toe-kick + base rest on the ground.
# ---------------------------------------------------------------------------
_GROUND_Z = -1.05

# Cabinet pose (world frame). Yaw -90 deg about Z sends the authored +X face to
# -Y and the authored -Y "front" face to -X -- i.e. the doors face toward the
# robot, which stands at negative x and faces +X. STARTING POINT -- the exact
# standoff needs lab measurement (see HANDOFF.md: "a rigid base shift computed
# from the recorded pose drives the torso 230 mm into the cabinet").
_CABINET_POS_W = (0.35, 0.0, _GROUND_Z)
_CABINET_ROT_XYZW = (0.0, 0.0, -0.7071068, 0.7071068)  # yaw -90 deg

# Bimanual OpenArm torso mount. x < cabinet x by about one arm-reach (~0.7 m)
# so the hand lands at the handle, which is 75 mm inset from the door's free
# edge (door-left outer edge is at cabinet-local y = +0.4575 - 0.036 - 0.075 =
# +0.347, cabinet is yawed -90, so world x_handle = cabinet_x + 0.347 = 0.697
# for the left door; the right handle mirrors on the other side). Base z puts
# the robot's 0.78 m shoulder lift at ~0.26 m world -- shoulders ~0.29 m above
# the handle (1.02 m above the cabinet floor = -0.03 m world), i.e. the natural
# reach-down posture the real demos use. Lowered 30 cm from the first guess of
# -0.22 m: that put the shoulders too high over the handle. STARTING POINT.
_ROBOT_BASE_POS_W = (-0.35, 0.0, -0.52)

# Support stand under the robot base (nucleus ``Props/Mounts/Stand``). Same as
# the centrifuge pack: origin is at the TOP of the stand, scale z so the bottom
# lands on the ground. The effective in-env height per unit z-scale is
# ~0.515 m. A 1 cm overshoot avoids any visible float gap.
_STAND_NATIVE_HEIGHT = 0.515
_STAND_SCALE_Z = (_ROBOT_BASE_POS_W[2] - _GROUND_Z + 0.01) / _STAND_NATIVE_HEIGHT


##
# Scene
##


@configclass
class OpenDoorSceneCfg(InteractiveSceneCfg):
    """Ground + dome light + robot stand + bimanual OpenArm + rigged cabinet.

    No camera here -- see the module docstring and ``visuomotor_env_cfg`` for
    the image-rendering variant used by headless dataset generation.
    """

    # robot + EE frames: filled by the concrete variant (ik_abs / ik_rel).
    robot: ArticulationCfg = MISSING
    left_ee_frame: FrameTransformerCfg = MISSING
    right_ee_frame: FrameTransformerCfg = MISSING

    # ground + light (stack-task defaults)
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, _GROUND_Z)),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # support stand under the robot base (top at the base, z-scaled to the
    # ground). Keeps the robot from floating.
    stand = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RobotStand",
        init_state=AssetBaseCfg.InitialStateCfg(pos=_ROBOT_BASE_POS_W, rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd",
            scale=(1.0, 1.0, _STAND_SCALE_Z),
        ),
    )

    # Cabinet -- articulated asset bundled in the package. The USD carries its
    # own PhysicsArticulationRootAPI, per-link rigid-body / collision APIs, and
    # joint limits + drives. The base link is fixed to the world through the
    # authored articulation root, so writing to the root pose on reset
    # repositions the whole cabinet without physics drift.
    #
    # Initial joint positions are all zero (doors shut, levers neutral); the
    # authored limits are (USD stores limits in DEGREES; Isaac Lab exposes
    # the live joint_pos in radians):
    #   hinge_left   [-110,   0] deg
    #   hinge_right  [   0, 110] deg
    #   latch_left   [   0,  90] deg
    #   latch_right  [ -90,   0] deg
    # so joint_pos=0.0 is the "closed, undisturbed" starting state.
    cabinet = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=_CABINET_POS_W,
            rot=_CABINET_ROT_XYZW,
            joint_pos={
                "hinge_left": 0.0,
                "hinge_right": 0.0,
                "latch_left": 0.0,
                "latch_right": 0.0,
            },
        ),
        spawn=UsdFileCfg(
            usd_path=CABINET_USD_PATH,
            # Rigid-body props set on the root propagate as PhysX overrides for
            # the whole articulation; keep gravity on so the toe-kick rests on
            # the ground (even though the root is articulation-root-fixed, the
            # door bodies are not).
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        ),
        actuators={},  # the USD-authored joint drives are used as-is
    )


##
# MDP
##


@configclass
class ActionsCfg:
    """Bimanual action terms -- left/right arm IK + left/right gripper. Filled by variants."""

    left_arm_action: ActionTerm = MISSING
    left_gripper_action: ActionTerm = MISSING
    right_arm_action: ActionTerm = MISSING
    right_gripper_action: ActionTerm = MISSING


@configclass
class ObservationsCfg:
    """Canonical 3.0.0 observation groups: low-dim policy, (empty) RGB, subtask signals."""

    @configclass
    class PolicyCfg(ObsGroup):
        """State-only observations (matches the stack tasks' low-dim PolicyCfg)."""

        # robot proprioception
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        # end-effector poses (env frame) -- keys read by the Mimic env wrapper
        left_eef_pos = ObsTerm(
            func=mdp.ee_frame_position_in_env_frame, params={"ee_frame_cfg": SceneEntityCfg("left_ee_frame")}
        )
        left_eef_quat = ObsTerm(
            func=mdp.ee_frame_orientation, params={"ee_frame_cfg": SceneEntityCfg("left_ee_frame")}
        )
        right_eef_pos = ObsTerm(
            func=mdp.ee_frame_position_in_env_frame, params={"ee_frame_cfg": SceneEntityCfg("right_ee_frame")}
        )
        right_eef_quat = ObsTerm(
            func=mdp.ee_frame_orientation, params={"ee_frame_cfg": SceneEntityCfg("right_ee_frame")}
        )

        # cabinet state: four joint angles, exposed so the policy can see the
        # door swing (and the handle for a future lever-turn variant).
        cabinet_door_angles = ObsTerm(func=mdp.cabinet_door_angles)
        cabinet_handle_angles = ObsTerm(func=mdp.cabinet_handle_angles)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Empty placeholder -- the Visuomotor variant fills this with image terms."""

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Boolean subtask signals for Mimic annotation (read via obs_buf['subtask_terms'])."""

        left_grasp_handle = ObsTerm(
            func=mdp.left_grasp_handle,
            params={
                "left_ee_frame_cfg": SceneEntityCfg("left_ee_frame"),
                "cabinet_cfg": SceneEntityCfg("cabinet"),
            },
        )

        # NOT a Mimic subtask signal -- the Mimic wrapper's
        # get_subtask_term_signals() whitelists ``left_grasp_handle`` only, so
        # this term is invisible to annotation and generation. It is here
        # because the ObservationManager is the one manager that runs every
        # step in record_demos, annotate_demos AND generate_dataset -- that's
        # what keeps the success dwell counter live once those scripts set
        # ``terminations = None``. See
        # :func:`~rpl_cabinet.mdp.terminations.door_opened`. Same either-door
        # ~80 deg defaults as the TerminationsCfg.success term below, so this
        # signal and that termination evaluate the same condition.
        door_opened_signal = ObsTerm(func=mdp.door_opened, params={"settle_time": 1.0})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:
    """Termination terms."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Success once EITHER door has been held open past ~80 deg for ``settle_time``
    # seconds (defaults inside ``door_opened``: both hinges, 1.40 rad threshold).
    # The dwell gives the operator a moment to let go and withdraw the arm
    # before the episode ends. No separate failure termination -- rely on
    # ``time_out`` (40 s) to discard runs where the operator cannot open the
    # door; mirrors the simplest centrifuge-style data-collection setup.
    success = DoneTerm(func=mdp.door_opened, params={"settle_time": 1.0})


@configclass
class EventsCfg:
    """Reset to spawn state (safe for kinematic bodies), then randomize cabinet pose."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default_safe, mode="reset")

    # Clear the success dwell counter so a new episode cannot inherit progress
    # from the previous one.
    reset_settle_counter = EventTerm(func=mdp.reset_door_settle_counter, mode="reset")

    # Cabinet pose jitter. Tight so the handle stays inside the OpenArm reach
    # envelope; widen once the lab-measured base pose is correct.
    randomize_cabinet = EventTerm(
        func=mdp.reset_cabinet_pose,
        mode="reset",
        params={
            "pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "yaw": (-0.05, 0.05)},
            "cabinet_cfg": SceneEntityCfg("cabinet"),
        },
    )


##
# Env
##


@configclass
class OpenDoorEnvCfg(ManagerBasedRLEnvCfg):
    """Base env config for the OpenArm cabinet door-opening task (camera-free)."""

    scene: OpenDoorSceneCfg = OpenDoorSceneCfg(num_envs=1, env_spacing=3.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    # Unused managers (match the stack tasks).
    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 5
        # 40 s matches the real-episode durations in openarm_open_lab_door
        # (median ~27 s, 95th pct ~37 s per the HANDOFF kinematic analysis).
        self.episode_length_s = 40.0
        self.sim.dt = 0.01  # 100 Hz
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg(
            bounce_threshold_velocity=0.01,
            friction_correlation_distance=0.00625,
        )
        self.viewer.eye = (-1.8, -1.3, 0.4)
        self.viewer.lookat = (0.3, 0.0, 0.0)
