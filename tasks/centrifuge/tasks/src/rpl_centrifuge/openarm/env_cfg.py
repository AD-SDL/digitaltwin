# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for the bimanual OpenArm centrifuge task (IsaacLab 3.0.0).

Rebuilt to mirror the canonical 3.0.0 manipulation-task structure (see
``isaaclab_tasks/contrib/stack/stack_env_cfg.py`` and the Franka/SO101 stack
configs):

* Scene uses the shared nucleus **SeattleLabTable** work surface + ground +
  dome light (the same assets the stack tasks use), with our custom centrifuge
  ``rack`` / ``tube`` / ``bucket`` USDs standing in for the reference cubes.
* Observations are split into the canonical three groups: ``PolicyCfg``
  (low-dimensional state, no images), an empty ``RGBCameraPolicyCfg`` placeholder
  (filled only by the Visuomotor variant), and ``SubtaskCfg`` (grasp signals for
  Mimic annotation).
* **No camera lives in this base or in the teleop tasks.** In 3.0.0 the teleop /
  record tasks are camera-free; camera images are produced by the separate
  ``visuomotor_env_cfg`` variant rendered *headless* by ``generate_dataset``.
  (Rendering a camera observation live under the Kit XR pipeline crashes Kit —
  the examples avoid it exactly this way.)

Concrete variants plug in the robot actions:
  * ``ik_abs_env_cfg`` / ``ik_rel_env_cfg`` — VR teleop (dual-source retargeters).
  * ``visuomotor_env_cfg`` — adds cameras + image obs for headless dataset gen.

Geometry note: the world frame follows the stack examples — ground at z=-1.05,
SeattleLabTable top at z≈0, props resting at z≈0. The robot mount and prop
positions below are STARTING POINTS and need in-sim tuning (the OpenArm reach
envelope must cover the rack and bucket); tune them while watching the sim.

Quaternion conventions (they differ, keep them straight):
  * ``InitialStateCfg.rot`` (assets) is **(x, y, z, w)**; identity = (0,0,0,1).
  * ``CameraCfg.OffsetCfg.rot`` is **(w, x, y, z)**.
"""

from dataclasses import MISSING
from importlib.resources import files as _pkg_files

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sim.spawners.meshes.meshes_cfg import MeshCuboidCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg

from .. import mdp

# Rigid-body-prepared centrifuge USDs bundled inside the package wheel
# (see ``scripts/prepare_assets.py``). ``importlib.resources.files`` resolves the
# path for both editable and built-wheel installs. Imported by the GR1T2 task too.
CENTRIFUGE_DATASET_DIR = str(_pkg_files("rpl_centrifuge") / "assets")

# ---------------------------------------------------------------------------
# Scene geometry (SeattleLabTable frame). MEASURED in-sim (raycast): the table's
# collision top surface is at z≈0.0 (its bbox extends higher, but the flat work
# surface where props rest is z≈0). ground @ z=-1.05.
#
# The OpenArm is a tall torso (base=pelvis; shoulders ~0.78 m above the base;
# hands hang to ~0.16 m above the base in the default pose) with a SHORT forward
# reach (~0.16 m). So it must stand *beside* the table's near (-x) edge — NOT over
# the table footprint (x < -0.36), or the torso intersects the tabletop and the
# arms come up underneath it (the original bug). We mount the base ~0.22 m below
# the work surface (matching the previously-tuned reach) and place the props right
# at the near edge, ~0.13–0.19 m in front of the base. STILL tune in sim.
# ---------------------------------------------------------------------------
_TABLE_POS = (0.5, 0.0, 0.0)
_TABLE_ROT_XYZW = (0.0, 0.0, 0.70710678, 0.70710678)  # 90° yaw, as the stack tasks use
_GROUND_Z = -1.05
_SURFACE_Z = 0.0  # SeattleLabTable collision top surface (measured)
_PROP_CLEARANCE = 0.005

# Bimanual OpenArm torso mount: beside the table's -x edge (x=-0.45, table starts
# at x≈-0.36), 0.22 m below the work surface so the shoulders clear the tabletop
# and the arms reach forward+down onto the near edge. Faces +x toward the props.
_ROBOT_BASE_POS_W = (-0.45, 0.0, -0.22)

# Support stand under the robot base (nucleus ``Props/Mounts/Stand``). Origin is at
# the TOP of the stand; we place the top at the robot base and z-scale it so the
# bottom reaches the ground, so the robot no longer floats. Height auto-derives
# from base + ground, so moving the robot keeps the stand grounded. The effective
# in-env height per unit z-scale is ~0.515 m (measured live; the standalone USD
# bbox reads a bit larger). A 1 cm overshoot avoids any visible float gap.
_STAND_NATIVE_HEIGHT = 0.515  # measured effective height in-env at scale 1.0
_STAND_SCALE_Z = (_ROBOT_BASE_POS_W[2] - _GROUND_Z + 0.01) / _STAND_NATIVE_HEIGHT

# Prop positions: at the table's near edge, in front of the robot (+x) and spread
# in y (rack on the right/-y → bucket toward centre). Offsets from the base
# replicate the previously-tuned reach (~0.13–0.19 m fwd, ≤0.21 m side). TUNE.
_RACK_POS = (-0.32, -0.21, _SURFACE_Z + _PROP_CLEARANCE + 0.006)
_RACK_FLOOR_THICKNESS = 0.01
_RACK_FLOOR_BOTTOM_Z = _SURFACE_Z + _PROP_CLEARANCE
_TUBE_POS = (-0.29, -0.18, _SURFACE_Z + 0.15)  # above the rack well; drops in on reset
_BUCKET_POS = (-0.26, 0.0, _SURFACE_Z + _PROP_CLEARANCE)


##
# Scene
##


@configclass
class CentrifugeSceneCfg(InteractiveSceneCfg):
    """Nucleus table + light/ground + bimanual OpenArm + centrifuge rack/tube/bucket.

    No camera here — see the module docstring and ``visuomotor_env_cfg`` for the
    image-rendering variant.
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

    # work surface — shared nucleus SeattleLabTable (same asset the stack tasks use)
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=_TABLE_POS, rot=_TABLE_ROT_XYZW),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # support stand under the robot base (top at the base, z-scaled to the ground).
    stand = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RobotStand",
        init_state=AssetBaseCfg.InitialStateCfg(pos=_ROBOT_BASE_POS_W, rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd",
            scale=(1.0, 1.0, _STAND_SCALE_Z),
        ),
    )

    # vial rack — tube holder (triangle-mesh collision authored in the USD).
    # KINEMATIC rigid body (not a static AssetBase) so it can be repositioned at
    # reset for domain randomization; kinematic = immovable during the episode.
    rack = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Rack",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_RACK_POS, rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=UsdFileCfg(
            usd_path=f"{CENTRIFUGE_DATASET_DIR}/vial_rack_simple.usda",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        ),
    )

    # rack floor — invisible kinematic collision backstop under the rack wells.
    # Kinematic so it moves with the rack under randomization (same group).
    rack_floor = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RackFloor",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(_RACK_POS[0] + 0.06, _RACK_POS[1] + 0.06, _RACK_FLOOR_BOTTOM_Z + _RACK_FLOOR_THICKNESS / 2),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        spawn=MeshCuboidCfg(
            size=(0.12, 0.12, _RACK_FLOOR_THICKNESS),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05)),
        ),
    )

    # plastic centrifuge tube (rigid) — starts above the rack well and drops in.
    tube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Tube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_TUBE_POS, rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=UsdFileCfg(
            usd_path=f"{CENTRIFUGE_DATASET_DIR}/centrifuge_tube_big.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=0.5,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.82, 0.82, 0.82)),
        ),
    )

    # centrifuge bucket (kinematic receptacle — immovable so the tube depenetrates
    # against a fixed surface on insertion; SDF well collision authored in the USD).
    bucket = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bucket",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_BUCKET_POS, rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=UsdFileCfg(
            usd_path=f"{CENTRIFUGE_DATASET_DIR}/centrifuge_bucket_big.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=0.5,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.7, 0.25)),
        ),
    )


##
# MDP
##


@configclass
class ActionsCfg:
    """Bimanual action terms — left/right arm IK + left/right gripper. Filled by variants."""

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

        # end-effector poses (env frame) — keys read by the Mimic env wrapper
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

        # object poses
        tube_pos = ObsTerm(func=mdp.object_position_in_env_frame, params={"asset_cfg": SceneEntityCfg("tube")})
        tube_quat = ObsTerm(func=mdp.object_orientation, params={"asset_cfg": SceneEntityCfg("tube")})
        bucket_pos = ObsTerm(func=mdp.object_position_in_env_frame, params={"asset_cfg": SceneEntityCfg("bucket")})
        bucket_quat = ObsTerm(func=mdp.object_orientation, params={"asset_cfg": SceneEntityCfg("bucket")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Empty placeholder — the Visuomotor variant fills this with image terms."""

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Boolean subtask signals for Mimic annotation (read via obs_buf['subtask_terms'])."""

        right_grasp_tube = ObsTerm(
            func=mdp.right_grasp_tube,
            params={
                "tube_cfg": SceneEntityCfg("tube"),
                "right_ee_frame_cfg": SceneEntityCfg("right_ee_frame"),
                "rack_cfg": SceneEntityCfg("rack"),
            },
        )

        # NOT a Mimic subtask signal — the Mimic wrapper's get_subtask_term_signals()
        # whitelists ``right_grasp_tube`` only, so this term is invisible to
        # annotation and generation. It is here because the ObservationManager is
        # the one manager that runs every step in record_demos, annotate_demos AND
        # generate_dataset, which is what keeps the success dwell counter live once
        # those scripts set ``terminations = None``. See
        # :func:`~rpl_centrifuge.mdp.terminations.tube_inside_bucket`.
        tube_settled_in_bucket = ObsTerm(func=mdp.tube_inside_bucket, params={"settle_time": 3.0})

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
    tube_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": _SURFACE_Z - 0.30, "asset_cfg": SceneEntityCfg("tube")},
    )
    # Success only after the vial has sat in a bucket well for `settle_time`
    # seconds — the dwell window lets the teleoperator withdraw the arm to home
    # before the episode ends (raise/lower it to taste).
    success = DoneTerm(func=mdp.tube_inside_bucket, params={"settle_time": 3.0})


@configclass
class EventsCfg:
    """Reset to spawn state, then randomize the pickup station and the bucket.

    For data collection: the pickup station (rack + floor + tube, kept together so
    the tube stays in the well) and the bucket are each shifted by a small random
    in-plane offset every reset. Ranges are kept tight so both stay inside the
    OpenArm's reach envelope — widen them once you've confirmed reach in sim.
    """

    reset_all = EventTerm(func=mdp.reset_scene_to_default_safe, mode="reset")

    # Clear the success dwell counter so a new episode cannot inherit progress
    # from the previous one.
    reset_settle_counter = EventTerm(func=mdp.reset_tube_settle_counter, mode="reset")

    # Pickup station: shift the rack (+ floor) by a small random delta, and drop the
    # tube into one of the 4 rack wells at random (well centers are rack-local xy,
    # measured from the rack corner origin; the tube drops from `tube_drop_height`
    # above the rack). Keeps positions within the OpenArm reach; widen after tuning.
    randomize_pickup = EventTerm(
        func=mdp.reset_tube_in_random_well,
        mode="reset",
        params={
            "station_pose_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "yaw": (-0.15, 0.15)},
            "well_offsets": [(0.03, 0.03), (0.03, 0.09), (0.09, 0.03), (0.09, 0.09)],
            "tube_drop_height": 0.15,
            "rack_cfg": SceneEntityCfg("rack"),
            "tube_cfg": SceneEntityCfg("tube"),
            "rack_floor_cfg": SceneEntityCfg("rack_floor"),
        },
    )

    # Bucket (placement target) randomized independently.
    randomize_bucket = EventTerm(
        func=mdp.reset_object_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
            "asset_cfg": SceneEntityCfg("bucket"),
        },
    )


##
# Env
##


@configclass
class CentrifugeEnvCfg(ManagerBasedRLEnvCfg):
    """Base env config for the bimanual OpenArm centrifuge task (camera-free)."""

    scene: CentrifugeSceneCfg = CentrifugeSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
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
        self.episode_length_s = 30.0
        self.sim.dt = 0.01  # 100 Hz
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg(
            bounce_threshold_velocity=0.01,
            friction_correlation_distance=0.00625,
        )
        self.viewer.eye = (1.2, 1.2, 0.9)
        self.viewer.lookat = (0.35, 0.0, 0.0)
