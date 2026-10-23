# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for the single-arm SO101 centrifuge task (IsaacLab 3.0.0).

Rebuilt to mirror the upstream SO101 cube-stack config
(``isaaclab_tasks/contrib/stack/config/so101``): a small SO101 arm seated on the
shared nucleus **SeattleLabTable**, picking a centrifuge tube from a vial rack and
dropping it into the centrifuge bucket. Our custom ``rack`` / ``tube`` / ``bucket``
USDs stand in for the reference cubes.

Observations follow the canonical three-group layout (``PolicyCfg`` low-dim,
empty ``RGBCameraPolicyCfg`` placeholder, ``SubtaskCfg`` grasp signal). **No
camera in this base or in the teleop tasks** — images come from the headless
``visuomotor_env_cfg`` variant (see that file / the OpenArm docstring).

Concrete variants:
  * ``ik_abs_env_cfg`` — VR teleop (dual-source retargeter + upstream SO101 pose-IK).
  * ``lerobot_leader_env_cfg`` — physical SO101 leader-arm teleop via LeRobot,
    read in-process; adds live cameras and a ``visual`` obs group.
  * ``visuomotor_env_cfg`` — cameras + image obs for headless dataset gen.

Geometry follows the stack frame (ground z=-1.05, table top z≈0). Positions are
STARTING POINTS for in-sim tuning.
"""

from dataclasses import MISSING

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
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.meshes.meshes_cfg import MeshCuboidCfg, MeshRectangleCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg

from .. import mdp
from ..openarm.env_cfg import CENTRIFUGE_DATASET_DIR

# ---------------------------------------------------------------------------
# Scene geometry (SeattleLabTable frame). STARTING POINTS — tune in sim so the
# 5-DOF SO101 reach envelope covers the rack + bucket.
# ---------------------------------------------------------------------------
_TABLE_POS = (0.5, 0.0, 0.0)
_TABLE_ROT_XYZW = (0.0, 0.0, 0.70710678, 0.70710678)
_GROUND_Z = -1.05
_SURFACE_Z = 0.0
_PROP_CLEARANCE = 0.005

# SO101 seated mount: base planted on the table top (mirrors the upstream SO101
# stack ``_SO101_MOUNT_Z``), yawed 90° about +z so the arm faces the props.
_ROBOT_BASE = (0.0, 0.0, _SURFACE_Z - 0.03008)
_ROBOT_YAW_XYZW = (0.0, 0.0, 0.70710678, 0.70710678)

# Props in the SO101 workspace. Pushed +6 cm in world +X vs. the initial
# starting values so the arm has to stretch to reach the pickup zone (previous
# positions at X≈0.22-0.25 kept the elbow bent most of the time).
_RACK_POS = (0.28, 0.06, _SURFACE_Z + _PROP_CLEARANCE + 0.006)
_RACK_FLOOR_THICKNESS = 0.01
_RACK_FLOOR_BOTTOM_Z = _SURFACE_Z + _PROP_CLEARANCE
_TUBE_POS = (0.31, 0.09, _SURFACE_Z + 0.12)
_BUCKET_POS = (0.30, -0.06, _SURFACE_Z + _PROP_CLEARANCE)


##
# Scene
##


@configclass
class CentrifugeSO101SceneCfg(InteractiveSceneCfg):
    """Nucleus table + light/ground + SO101 (variant-filled) + centrifuge rack/tube/bucket."""

    # robot + EE frame: filled by the concrete variant.
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    # Bare visual + collision ground. The default ``GroundPlaneCfg()`` loads
    # Isaac's ``default_ground_plane.usda``, which bundles a grid-room backdrop
    # (walls + sky) that occludes the Persp camera at this scene's scale. A
    # flat mesh rectangle gives a clean studio-like background — the dome light
    # above provides illumination; the collision plane catches dropped tubes.
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, _GROUND_Z)),
        spawn=MeshRectangleCfg(
            size=(10.0, 10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
        ),
    )
    # ``visible_in_primary_ray=False`` keeps the dome light's illumination but
    # hides the dome sphere itself from the camera — otherwise, with the grid-
    # room backdrop removed above, the camera renders a glaring white sky.
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=3000.0,
            visible_in_primary_ray=False,
        ),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=_TABLE_POS, rot=_TABLE_ROT_XYZW),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # Kinematic rigid bodies (not static) so the pickup station can be
    # repositioned at reset for randomization — same as the OpenArm scene.
    rack = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Rack",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_RACK_POS, rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=UsdFileCfg(
            usd_path=f"{CENTRIFUGE_DATASET_DIR}/vial_rack_simple.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        ),
    )
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
    """Single arm action term — filled by the concrete variant."""

    arm_action: ActionTerm = MISSING


@configclass
class ObservationsCfg:
    """Canonical 3.0.0 observation groups (low-dim policy, empty RGB, subtask)."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        eef_pos = ObsTerm(
            func=mdp.ee_frame_position_in_env_frame, params={"ee_frame_cfg": SceneEntityCfg("ee_frame")}
        )
        eef_quat = ObsTerm(func=mdp.ee_frame_orientation, params={"ee_frame_cfg": SceneEntityCfg("ee_frame")})

        tube_pos = ObsTerm(func=mdp.object_position_in_env_frame, params={"asset_cfg": SceneEntityCfg("tube")})
        tube_quat = ObsTerm(func=mdp.object_orientation, params={"asset_cfg": SceneEntityCfg("tube")})
        bucket_pos = ObsTerm(func=mdp.object_position_in_env_frame, params={"asset_cfg": SceneEntityCfg("bucket")})
        bucket_quat = ObsTerm(func=mdp.object_orientation, params={"asset_cfg": SceneEntityCfg("bucket")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Empty placeholder — filled by the Visuomotor variant."""

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        grasp_tube = ObsTerm(
            func=mdp.right_grasp_tube,
            params={"tube_cfg": SceneEntityCfg("tube"), "right_ee_frame_cfg": SceneEntityCfg("ee_frame")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    tube_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": _SURFACE_Z - 0.30, "asset_cfg": SceneEntityCfg("tube")},
    )
    # Success only after the vial has sat in a bucket well for `settle_time`
    # seconds — the dwell window lets the teleoperator withdraw the arm to home.
    success = DoneTerm(func=mdp.tube_inside_bucket, params={"settle_time": 3.0})


@configclass
class EventsCfg:
    """Reset, then randomize the pickup station + bucket (same scheme as OpenArm).

    Ranges are tighter than OpenArm's because the SO101 has a smaller reach
    envelope; widen after confirming reach in sim.
    """

    reset_all = EventTerm(func=mdp.reset_scene_to_default_safe, mode="reset")

    # Pickup station: shift rack (+ floor), drop the tube into a random rack well.
    randomize_pickup = EventTerm(
        func=mdp.reset_tube_in_random_well,
        mode="reset",
        params={
            "station_pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "yaw": (-0.1, 0.1)},
            "well_offsets": [(0.03, 0.03), (0.03, 0.09), (0.09, 0.03), (0.09, 0.09)],
            "tube_drop_height": 0.15,
            "rack_cfg": SceneEntityCfg("rack"),
            "tube_cfg": SceneEntityCfg("tube"),
            "rack_floor_cfg": SceneEntityCfg("rack_floor"),
        },
    )

    randomize_bucket = EventTerm(
        func=mdp.reset_object_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
            "asset_cfg": SceneEntityCfg("bucket"),
        },
    )


##
# Env
##


@configclass
class CentrifugeSO101EnvCfg(ManagerBasedRLEnvCfg):
    """Base env config for the single-arm SO101 centrifuge task (camera-free)."""

    scene: CentrifugeSO101SceneCfg = CentrifugeSO101SceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 5
        self.episode_length_s = 60.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg(
            bounce_threshold_velocity=0.01,
            friction_correlation_distance=0.00625,
        )
        # Over-the-shoulder tabletop view framing the SO101 workspace. If the
        # Kit viewport still shows the default origin view on launch (perspective
        # rotation isn't always picked up from viewer.lookat), click any scene
        # prim in the stage tree and press F in the viewport to frame-selected.
        # (The LeRobot-leader agent additionally force-switches the active
        # viewport to a scene camera at startup, since the default Persp has
        # broken rendering in this Kit app.)
        self.viewer.eye = (0.55, 0.45, 0.40)
        self.viewer.lookat = (0.20, 0.00, 0.05)
