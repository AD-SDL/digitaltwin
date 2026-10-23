# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SO101 centrifuge task teleoperated by a physical SO101 leader arm via **LeRobot**.

This is the LeRobot "data-transfer" env (modeled on the Sim-to-Real-SO-101-Workshop
``Lerobot-So101-Teleop-Vials-To-Rack`` tasks). The physical leader arm is read
directly in-process by ``scripts/so101_lerobot_agent.py`` via lerobot's
``so_leader`` driver — no UDP bridge, no sibling venv. The agent steps this env
and records demos straight to an IsaacLab-style HDF5 that can be post-converted
to a LeRobot dataset via ``scripts/convert_to_lerobot.py``.

Contract required by ``scripts/so101_lerobot_agent.py``:
  * a 6-DOF **absolute** joint-position action, ordered
    ``[Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw]`` (``preserve_order``
    so it matches the leader's ``shoulder_pan..gripper`` stream 1:1);
  * a ``policy`` obs term ``joint_pos_obs`` giving those 6 joints (radians, same
    order) — recorded as ``observation.state``;
  * a ``visual`` obs group with ``rgb_<cam>`` terms, and scene cameras keyed
    ``camera_<cam>`` (the agent discovers cameras by that prefix).

Unlike the VR tasks this env runs cameras **live** (it is a plain gym env stepped
from USB input, not the XR pipeline), so launch the agent with
``--enable_cameras`` (set automatically by the agent via AppLauncher kwarg).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg

from isaaclab_tasks.utils.presets import set_isaac_rtx_global_settings

from .. import mdp
from .env_cfg import CentrifugeSO101EnvCfg, ObservationsCfg, _ROBOT_BASE, _ROBOT_YAW_XYZW
from .so101_cfg import (
    ALL_JOINT_NAMES,
    BASE_LINK_NAME,
    END_EFFECTOR_LINK_NAME,
    SO101_CFG,
)

# Camera scene keys carry a ``camera_`` prefix; the agent strips it to get the
# LeRobot dataset feature name (``camera_table`` -> ``table`` -> ``rgb_table``).
_CAMERA_HEIGHT = 240
_CAMERA_WIDTH = 320


def _pinhole(focal_length: float = 24.0, near_clip: float = 0.1, far_clip: float = 2.0):
    """Shared pinhole camera template. Default hFOV ≈ 47° (focal 24mm, aperture
    20.955mm). The wrist camera overrides focal_length / near_clip to get a wide
    enough view and allow close-up gripper visibility; the overview cameras
    keep the default."""
    return sim_utils.PinholeCameraCfg(
        focal_length=focal_length,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(near_clip, far_clip),
    )


@configclass
class _LeRobotLeaderObservationsCfg(ObservationsCfg):
    """Base obs + a joint_pos_obs term (recorded state) + a live RGB visual group."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        # Absolute joint positions (radians) in leader order — recorded as
        # ``observation.state``. ``preserve_order`` guarantees the ordering
        # regardless of the articulation's natural DOF order.
        joint_pos_obs = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(ALL_JOINT_NAMES), preserve_order=True)},
        )

    @configclass
    class VisualCfg(ObsGroup):
        rgb_table = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("camera_table"), "data_type": "rgb", "normalize": False},
        )
        rgb_wrist = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("camera_wrist"), "data_type": "rgb", "normalize": False},
        )
        rgb_robot = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("camera_robot"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    visual: VisualCfg = VisualCfg()


@configclass
class CentrifugeSO101LeRobotLeaderEnvCfg(CentrifugeSO101EnvCfg):
    """SO101 centrifuge task driven by a physical SO101 leader arm via LeRobot."""

    observations: _LeRobotLeaderObservationsCfg = _LeRobotLeaderObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Robot: mount SO101 on the table, yawed toward the props (shared consts).
        self.scene.robot = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = _ROBOT_BASE
        self.scene.robot.init_state.rot = _ROBOT_YAW_XYZW

        # EE frame: robot base -> gripper link.
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{BASE_LINK_NAME}",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/{END_EFFECTOR_LINK_NAME}",
                    name="end_effector",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                ),
            ],
        )

        # Absolute 6-DOF joint mirror. preserve_order pins the action vector to the
        # leader stream order [Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw].
        self.actions.arm_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=list(ALL_JOINT_NAMES),
            scale=1.0,
            use_default_offset=False,
            preserve_order=True,
        )

        # Live cameras (RGB) named with the ``camera_`` prefix the agent scans for.
        self.scene.camera_wrist = CameraCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{END_EFFECTOR_LINK_NAME}/camera_wrist",
            update_period=0.0,
            height=_CAMERA_HEIGHT,
            width=_CAMERA_WIDTH,
            data_types=["rgb"],
            # Wider FOV (~76°) + 1 cm near-clip so the gripper geometry close
            # to the lens actually renders instead of being culled.
            spawn=_pinhole(focal_length=13.5, near_clip=0.01, far_clip=10.0),
            offset=CameraCfg.OffsetCfg(
                # Position: at the SO101's authored ``lens`` sub-prim location
                # (world (0.337, 0.073, 0.233)). This is the ground-truth
                # on-arm camera location per the asset. Gripper-local
                # decomposition (gripper +X -> world +Z, +Y -> +Y, +Z -> -X):
                #   lx = world_z - 0.237 = -0.004
                #   ly = world_y - 0.021 =  0.052
                #   lz = 0.277 - world_x = -0.060
                # Rotation: tuned iteratively against live snapshots to a view
                # looking down-and-forward from the lens with image-up along
                # the arm's reach direction. Final composition: identity base
                # (camera +Z aligned with gripper +Z) + yaw CCW 90° + pitches/
                # pans to center the pickup zone and the gripper fingers.
                pos=(-0.004, 0.052, -0.060),
                rot=(0.90760, 0.09710, 0.01851, 0.40804),
                convention="ros",
            ),
        )
        self.scene.camera_table = CameraCfg(
            prim_path="{ENV_REGEX_NS}/camera_table",
            update_period=0.0,
            height=_CAMERA_HEIGHT,
            width=_CAMERA_WIDTH,
            data_types=["rgb"],
            spawn=_pinhole(),
            offset=CameraCfg.OffsetCfg(
                # Pulled +6 cm along world +X (to 0.66) so the camera continues
                # to frame the pickup zone after rack/bucket were moved to
                # X≈0.28-0.30.
                pos=(0.66, 0.0, 0.35), rot=(-0.61237, -0.61237, 0.35355, 0.35355), convention="ros"
            ),
        )
        # Robot's-perspective overview camera: mounted at (-0.10, 0.25, 0.45) —
        # slightly behind the base, 25 cm to the +Y side (clears the robot body
        # silhouette which spans Y≈±0.1), 45 cm up. Looks at pickup-zone center
        # (0.30, 0, 0.08), ~37° below horizontal. Image-up = world +Z, matching
        # camera_table's floor-at-image-bottom orientation.
        self.scene.camera_robot = CameraCfg(
            prim_path="{ENV_REGEX_NS}/camera_robot",
            update_period=0.0,
            height=_CAMERA_HEIGHT,
            width=_CAMERA_WIDTH,
            data_types=["rgb"],
            spawn=_pinhole(),
            offset=CameraCfg.OffsetCfg(
                pos=(-0.10, 0.25, 0.45),
                rot=(-0.43591, 0.78649, -0.38266, 0.21209),
                convention="ros",
            ),
        )

        self.num_rerenders_on_reset = 3
        for camera_cfg in (self.scene.camera_table, self.scene.camera_wrist, self.scene.camera_robot):
            set_isaac_rtx_global_settings(camera_cfg.renderer_cfg, antialiasing_mode="DLAA")


@configclass
class CentrifugeSO101LeRobotLeaderDREnvCfg(CentrifugeSO101LeRobotLeaderEnvCfg):
    """Domain-randomized variant: widened prop/rack/bucket reset ranges.

    NOTE: this is a *partial* port of the Workshop ``-DR`` task. The photoreal
    visual DR there (HDRI sky swaps, lightbox exposure, robot recolor, mat jitter)
    depends on assets not shipped in this repo and is intentionally omitted;
    only the pose randomization feasible with the current scene is widened here.
    """

    def __post_init__(self):
        super().__post_init__()

        # Widen the pickup-station shift and bucket placement (base uses ±0.02).
        self.events.randomize_pickup.params["station_pose_range"] = {
            "x": (-0.04, 0.04),
            "y": (-0.04, 0.04),
            "yaw": (-0.3, 0.3),
        }
        self.events.randomize_bucket.params["pose_range"] = {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
        }
