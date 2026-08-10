# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for the bimanual centrifuge pick-and-place task.

The task: a bimanual OpenArm robot picks a plastic centrifuge tube from the
table and places it into a centrifuge bucket. Both arms are teleoperated; the
task does not prescribe which hand does which step.

This base cfg leaves the robot articulation and action terms ``MISSING`` so
that concrete variants (e.g. IK-rel) can plug in robot-specific bindings.
"""

from dataclasses import MISSING
from importlib.resources import files as _pkg_files

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sim.spawners.meshes.meshes_cfg import MeshCuboidCfg
from isaaclab.utils.configclass import configclass

from . import mdp

# Rigid-body-prepared USDs bundled inside the package wheel. ``importlib.resources.files``
# resolves the path correctly for both editable (``pip install -e .``) and built-wheel
# installs. See ``scripts/prepare_assets.py`` for how to regenerate these from raw
# geometry-only originals.
CENTRIFUGE_DATASET_DIR = str(_pkg_files("rpl_centrifuge") / "assets")

# ---------------------------------------------------------------------------
# Shared geometry constants
# ---------------------------------------------------------------------------
# The scene has a strict vertical stack (ground -> table -> props on table),
# so every prop's z depends on the table height. Centralising these values
# means retuning the table (e.g. the upcoming swap to a longer horizon
# table) only touches one line.
TABLE_HEIGHT = 0.80  # table cuboid thickness (also its top z since bottom sits on ground)
TABLE_TOP_Z = TABLE_HEIGHT  # world z of the table's top face

# Quaternion convention reminder: ``InitialStateCfg.rot`` is (x, y, z, w),
# NOT the USD-native (w, x, y, z). Identity is (0, 0, 0, 1). A 90 deg
# rotation about world +Z is (0, 0, sin(45), cos(45)) = (0, 0, 0.7071, 0.7071).
# Mixing these up was previously dropping the rack and bucket 100+ mm into
# the table because they were being rotated about +X instead of +Z.

# Small vertical gap between a prop's mesh bottom and its support surface.
# PhysX contact-offset (default ~4 mm on rigid bodies) treats coplanar
# surfaces as inter-penetrating, causing visible jitter and — worse — the
# solver may push a rigid prop *down* into the table on frame 0. A ~5 mm
# clearance is far below anything a human can see but well above the
# contact-offset threshold.
PROP_CLEARANCE = 0.005

# Rack-floor geometry (invisible collision plate inside the rack wells).
_RACK_FLOOR_THICKNESS = 0.005
_RACK_FLOOR_BOTTOM_Z = TABLE_TOP_Z + PROP_CLEARANCE
_RACK_FLOOR_TOP_Z = _RACK_FLOOR_BOTTOM_Z + _RACK_FLOOR_THICKNESS

# Tube USD origin is 20 mm above its mesh bottom (verified via UsdGeom.BBoxCache:
# mesh min z = -0.020). To rest the tube tip on the rack_floor top, place the
# origin that far above _RACK_FLOOR_TOP_Z.
_TUBE_ORIGIN_OFFSET_Z = 0.020
_TUBE_ORIGIN_Z = _RACK_FLOOR_TOP_Z + _TUBE_ORIGIN_OFFSET_Z


##
# Scene
##


@configclass
class CentrifugeSceneCfg(InteractiveSceneCfg):
    """Scene: ground, light, table, robot pedestal, bimanual robot, tube rack, tube, bucket.

    Geometry tuned so the OpenArm bimanual can comfortably reach the rack and
    bucket from a natural shoulders-above-table pose. World/env frame:
      * Ground plane at z=0.
      * Robot pedestal: 0.413 x 0.413 x 0.579 m cube centred at (-0.62, 0, 0.289)
        so its top is at z=0.578 (preserved from ``aiet_scene.usd``).
      * Table: top at z=0.80, painted yellow for contrast against the props.
      * Robot base mounted on the pedestal top at (-0.62, 0, 0.5786).
      * Tube rack: static fixture on the table at (-0.35, -0.15, 0.80) on the
        robot's right (-y) side, rotated 90 deg CCW about +z so its long axis
        runs along world y. The row of 4 large wells faces robot-centre (+y);
        the row of 6 small wells faces -y. Painted black.
      * Rack floor: invisible-by-design collision plate inside the rack
        (the rack USD's wells are open through-bores), so the tube tip rests
        inside the rack rather than dropping onto the table top.
      * Tube: light-gray rigid body, starts inserted into the back-right large
        well at (-0.332, -0.1425, 0.825). The tube body Ø is within ~0.1 mm of
        the well bore Ø. Two cooperating fixes prevent friction lock:
          1. The rack uses triangle-mesh collision and the bucket uses SDF
             collision (instead of convexDecomposition) so the well bores are
             faithfully empty in collision space -- convex decomp hulls would
             otherwise intrude into the wells and block the tube body
             regardless of clearance.
          2. The tube's PhysX collision surface is shrunk by 1 mm
             (``rest_offset = -0.001`` in ``collision_props``), giving ~1 mm
             radial clearance through both rack and bucket wells. The wider
             cap still collides with the rim as a real centrifuge tube does.
      * Bucket: green rigid body at (-0.35, 0, 0.80) -- same depth as the rack,
        on the robot's centre line. Forms an in-line layout with the rack along
        world y so right-hand pick-from-rack + place-into-bucket is a short
        translation.
      * Chest camera: Intel RealSense D435-like sensor attached to
        ``openarm_body_link`` so it travels with the robot torso. Pitched 45
        deg down to frame the rack + bucket. Not wired into the policy obs by
        default -- see the ``chest_camera`` field's docstring to opt in.

    The asymmetric layout (rack on -y, bucket on centre) reflects the
    "right hand picks first, then transports to centre" intended operation
    order. Mirror to +y if you want left-hand-first operation.

    Rack / tube / bucket positions are starting values intended for in-sim
    tuning. The success criterion (tube inside the bucket) is invariant to
    these positions, so retuning doesn't change the task definition.
    """

    # robot: filled in by the concrete variant
    robot: ArticulationCfg = MISSING
    # end-effector frames: filled in by the concrete variant (need to know body names)
    left_ee_frame: FrameTransformerCfg = MISSING
    right_ee_frame: FrameTransformerCfg = MISSING

    # ground
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=GroundPlaneCfg(),
    )

    # dome light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # robot pedestal — static cuboid; bottom flush with ground, top at 0.579.
    # MeshCuboidCfg centres the box on its prim origin, so pos.z = size.z / 2.
    pedestal = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Pedestal",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.62, 0.0, 0.579 / 2), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=MeshCuboidCfg(
            size=(0.413, 0.413, 0.579),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
        ),
    )

    # table — static cuboid; bottom flush with ground, top at TABLE_TOP_Z.
    # Footprint chosen so the -x edge (x=-0.4) doesn't overlap the pedestal
    # (whose +x edge is at x=-0.413), leaving a small visual gap.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.1, 0.0, TABLE_HEIGHT / 2), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=MeshCuboidCfg(
            size=(1.0, 0.6, TABLE_HEIGHT),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.2)),
        ),
    )

    # tube rack — static fixture sitting on the table. Collision approximation
    # is set to ``triangleMesh`` inside the rack USD itself (see
    # scripts/prepare_assets.py): a convex-hull or convex-decomp approximation
    # cannot represent the well bores faithfully, since convex hulls can't
    # have holes and the decomp hulls intrude into the wells, blocking the
    # tube body regardless of any rest-offset clearance. Triangle mesh is
    # only allowed on static bodies, which is fine here. Placed on the
    # robot's -y (right) side so the right hand operates first. Rotated 90
    # deg CCW about +z so the long axis points along world y; the row of 4
    # large wells faces robot-centre (+y) and the row of 6 small wells faces
    # away (-y). Painted black so the well openings read clearly.
    rack = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Rack",
        init_state=AssetBaseCfg.InitialStateCfg(
            # Rack USD's origin is at its own bottom (verified via
            # UsdGeom.BBoxCache: mesh min z = 0.000), so setting pos.z =
            # table_top + clearance puts the rack visibly above the table.
            pos=(-0.35, -0.15, TABLE_TOP_Z + PROP_CLEARANCE),
            rot=(0.0, 0.0, 0.7071068, 0.7071068),
        ),
        spawn=UsdFileCfg(
            usd_path=f"{CENTRIFUGE_DATASET_DIR}/centrifuge_tube_rack.usd",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05)),
        ),
    )

    # rack floor — invisible collision plate inside the rack. The rack USD's
    # wells are open through-bores (no authored floor), so without this the
    # tube tip would fall to the table top through the well. The plate is
    # axis-aligned in world frame after the rack rotation: 82 mm (x) x 125 mm
    # (y) x 5 mm (z), spanning the rack footprint. Painted to match the rack so
    # it's hidden visually but provides the floor PhysX needs.
    # Rack floor sits inside the rack, coplanar with the rack's bottom (both
    # at TABLE_TOP_Z + PROP_CLEARANCE). MeshCuboidCfg centres the box on its
    # prim origin, so center z = bottom z + thickness/2. Thickness and z
    # bookkeeping live at module scope so @configclass doesn't try to treat
    # them as scene fields.
    rack_floor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RackFloor",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.35, -0.15, _RACK_FLOOR_BOTTOM_Z + _RACK_FLOOR_THICKNESS / 2),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        spawn=MeshCuboidCfg(
            size=(0.082, 0.125, _RACK_FLOOR_THICKNESS),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05)),
        ),
    )

    # plastic centrifuge tube (rigid) — inserted into one of the four large
    # wells of the rotated rack. Tube body Ø ~28.7 mm fits the well Ø ~29 mm
    # (~0.05 mm raw clearance per side), which without intervention would
    # friction-lock and make picking/placing impossible. We shrink the tube's
    # PhysX collision surface by 1 mm via ``rest_offset = -0.001`` so the
    # effective body Ø is ~26.7 mm — that gives ~1 mm radial clearance through
    # the well bore for both the rack and bucket wells, but the cap (Ø ~35 mm)
    # is still wider than the bore so it catches on the rim as intended.
    # Tip rests on the rack_floor plate (top face z=0.805) at init; cap sits
    # ~33 mm above the rack rim. xy = (-0.332, -0.1425) is the back-right
    # large well in the rotated layout.
    # Tube z (_TUBE_ORIGIN_Z) is computed at module scope from the rack floor
    # geometry above — kept out of the class body for the same @configclass
    # reason: only scene assets belong here.
    tube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Tube",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.332, -0.1425, _TUBE_ORIGIN_Z),
            rot=(0.0, 0.0, 0.7071068, 0.7071068),
        ),
        spawn=UsdFileCfg(
            usd_path=f"{CENTRIFUGE_DATASET_DIR}/centrifuge_tube_big.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=-0.001,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.82, 0.82, 0.82)),
        ),
    )

    # OpenArm chest camera — Intel RealSense D435-like sensor mounted on the
    # robot's chest, looking forward at the work area. The OpenArm USD ships
    # without a camera prim authored (the ``openarm_bimanual_sensor.usd``
    # layer is a stub), so we attach one here as a child of openarm_body_link.
    # If the body link ever rotates, the camera moves with it.
    #
    # Pose is approximate — front face of upper chest, looking forward (+x in
    # robot body frame). Tune in-sim after seeing the camera frustum.
    #
    # Notes:
    #  * The camera's pixel output is NOT wired into the policy observation
    #    by default (would change obs shape and break downstream IL configs).
    #    To use it: add an ObsTerm like
    #        chest_rgb = ObsTerm(func=mdp.image,
    #                            params={"sensor_cfg": SceneEntityCfg("chest_camera"),
    #                                    "data_type": "rgb"})
    #  * Cameras render only when the render pipeline is active. For headless
    #    runs without livestream, pass ``--enable_cameras``.
    #  * Intrinsics target ~69 deg horizontal FOV (RealSense D435 color stream
    #    in 4:3 mode). Tune ``focal_length`` to change FOV.
    chest_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/openarm_body_link/chest_camera",
        update_period=0.0333,  # ~30 Hz, matches RealSense default
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15.25,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.08, 0.0, 0.60),   # 8 cm forward of body axis, world z=1.179 -- tuned for XR head pose
            # Pitched 45 deg downward (rotation about +y by +45 deg) so the
            # frustum covers the centrifuge bucket (~0.34 m forward, ~0.28 m
            # below the camera) and the tube on the rack. Quaternion is
            # (cos 22.5, 0, sin 22.5, 0).
            rot=(0.9238795, 0.0, 0.3826834, 0.0),
            convention="world",       # forward = +x (robot body frame), up = +z
        ),
    )

    # centrifuge bucket (rigid — heavy enough to act as a stable receptacle).
    # Sits on the table at the same x as the rack (-0.35), centred on the
    # robot's y axis. The bucket extends the rack's rotated long axis toward
    # the robot's centre, so right-hand pick-and-place from rack to bucket is
    # a short y-translation. Bucket mesh origin is at its bottom -> z=0.80
    # places it flush on the table top.
    # Bucket USD's origin is at its own bottom (verified via UsdGeom.BBoxCache:
    # mesh min z = 0.000), so pos.z = table_top + clearance places it visibly
    # above the table without spawning in contact with it.
    bucket = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bucket",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.35, 0.0, TABLE_TOP_Z + PROP_CLEARANCE),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        spawn=UsdFileCfg(
            usd_path=f"{CENTRIFUGE_DATASET_DIR}/centrifuge_bucket_big.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            # Collision approximation is set to ``sdf`` inside the bucket USD
            # itself (see scripts/prepare_assets.py): convexHull or convex
            # decomposition cannot represent the wells, and SDF is the only
            # approximation valid for dynamic bodies that preserves concave
            # geometry. Without this the wells would be filled and the tube
            # could not be inserted.
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.7, 0.25)),
        ),
    )


##
# MDP
##


@configclass
class ActionsCfg:
    """Bimanual action terms — left/right arm IK + left/right gripper binary."""

    left_arm_action: ActionTerm = MISSING
    left_gripper_action: ActionTerm = MISSING
    right_arm_action: ActionTerm = MISSING
    right_gripper_action: ActionTerm = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """State-only observation group used for record_demos + robomimic IL."""

        # robot proprioception
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        # left end-effector pose
        left_eef_pos = ObsTerm(
            func=mdp.ee_frame_position_in_env_frame,
            params={"ee_frame_cfg": SceneEntityCfg("left_ee_frame")},
        )
        left_eef_quat = ObsTerm(
            func=mdp.ee_frame_orientation,
            params={"ee_frame_cfg": SceneEntityCfg("left_ee_frame")},
        )

        # right end-effector pose
        right_eef_pos = ObsTerm(
            func=mdp.ee_frame_position_in_env_frame,
            params={"ee_frame_cfg": SceneEntityCfg("right_ee_frame")},
        )
        right_eef_quat = ObsTerm(
            func=mdp.ee_frame_orientation,
            params={"ee_frame_cfg": SceneEntityCfg("right_ee_frame")},
        )

        # object poses
        tube_pos = ObsTerm(
            func=mdp.object_position_in_env_frame, params={"asset_cfg": SceneEntityCfg("tube")}
        )
        tube_quat = ObsTerm(func=mdp.object_orientation, params={"asset_cfg": SceneEntityCfg("tube")})
        bucket_pos = ObsTerm(
            func=mdp.object_position_in_env_frame, params={"asset_cfg": SceneEntityCfg("bucket")}
        )
        bucket_quat = ObsTerm(
            func=mdp.object_orientation, params={"asset_cfg": SceneEntityCfg("bucket")}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # tube fell off the table — table top is at z=0.80, so anything well below
    # that counts as a drop.
    tube_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.3, "asset_cfg": SceneEntityCfg("tube")},
    )

    # success — record_demos.py picks this term up by name
    success = DoneTerm(func=mdp.tube_inside_bucket)


##
# Env
##


@configclass
class CentrifugeEnvCfg(ManagerBasedRLEnvCfg):
    """Base env config for bimanual centrifuge pick-and-place."""

    # Scene
    scene: CentrifugeSceneCfg = CentrifugeSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP
    terminations: TerminationsCfg = TerminationsCfg()

    # unused managers.
    # ``commands`` and ``curriculum`` accept ``None`` on ManagerBasedRLEnvCfg,
    # but ``rewards`` and ``events`` are typed ``object`` (no ``| None``) and
    # the corresponding managers iterate ``cfg.__dict__`` unconditionally —
    # passing ``None`` there yields ``'NoneType' object has no attribute
    # '__dict__'`` at env construction. Use empty configclass containers so
    # the managers see zero terms and do nothing.
    commands = None
    curriculum = None

    @configclass
    class _EmptyRewardsCfg:
        pass

    @configclass
    class _EmptyEventsCfg:
        pass

    rewards: object = _EmptyRewardsCfg()
    events: object = _EmptyEventsCfg()

    def __post_init__(self):
        # general settings
        self.decimation = 5
        self.episode_length_s = 30.0
        # simulation settings (100 Hz physics, rendered every other step)
        self.sim.dt = 0.01
        self.sim.render_interval = 2
        # PhysX tuning. Isaac Lab now selects the physics backend via
        # ``SimulationCfg.physics``; PhysX-specific fields live on ``PhysxCfg``.
        # (Older versions exposed them under ``sim.physx.*``.)
        # NOTE: previously overrode gpu_found_lost_aggregate_pairs_capacity and
        # gpu_total_aggregate_pairs_capacity here, but the old values (esp.
        # 16*1024 for the total-aggregate cap, ~128x smaller than the default
        # 2**21) triggered SIGSEGV in libomni.physx.cooking on this Isaac Sim
        # build. Leave those at PhysxCfg defaults unless a specific profiling
        # run motivates lower values.
        self.sim.physics = PhysxCfg(
            bounce_threshold_velocity=0.01,
            friction_correlation_distance=0.00625,
        )
        # viewer — look at the work area from in front of and above the robot
        self.viewer.eye = (1.2, 1.0, 1.6)
        self.viewer.lookat = (-0.25, 0.0, 0.85)
