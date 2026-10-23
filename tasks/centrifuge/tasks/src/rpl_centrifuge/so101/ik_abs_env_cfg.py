# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SO101 + absolute task-space (IK) VR teleop, using the upstream clutch retargeter.

The SO101 is 5-DOF, so a 6-DOF pose target is over-determined by one DOF. Rather
than feed an absolute controller pose straight to the IK (which made the 5-DOF
soft-orientation solver flip between two solutions as the stick moved), we use the
upstream **:class:`SO101ClutchRetargeter`** — the same purpose-built, stable path
as ``IsaacContrib-Stack-Cube-SO101-IK-Abs``. It rebases the controller motion
around an origin captured on engage (so no teleport/snap), emits a smooth
absolute 7-D pose, and drives the upstream ``SO101PoseIKAction`` (full-pose IK:
position exact, orientation soft-weighted on the wrist). Controller-driven
(joystick); the analog jaw follows the trigger via ``SO101GripperRetargeter``.

``target_frame_prim_path`` rebases the controller stream into the robot base
frame, so the clutch works in the IK command frame. The clutch seeds its engage
home from ``_HOME_BASE_T_GRIPPER`` — the gripper's base-frame pose at our init
joint pose (measured in sim); the robot ``init_state`` resets there so engaging
does not jump.

This is the VR counterpart to :mod:`rpl_centrifuge.so101.lerobot_leader_env_cfg`
(physical SO101 leader read in-process). Both share the scene, randomization,
and success.

Action tensor layout (8-D): [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, jaw].

Tuning (on-headset): ``_ORIENTATION_OFFSET_XYZW`` calibrates the controller-grip →
gripper orientation. ``None`` uses the clutch default (Rz(pi)); set the clutch's
``debug_log_engage=True`` to capture your controller orientation at engage and
compute ``R_off = quat_inv(q_grip) (x) q_G0`` (q_G0 measured base gripper quat).
"""

from dataclasses import MISSING

import numpy as np

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass

from isaaclab_teleop import IsaacTeleopCfg
from isaaclab_teleop.xr_cfg import XrCfg

# Upstream SO101 task-space IK action/controller — imported (no IsaacLab edits).
from isaaclab_tasks.contrib.stack.config.so101.pose_ik_action import SO101PoseIKActionCfg
from isaaclab_tasks.contrib.stack.config.so101.pose_ik_controller import SO101PoseIKControllerCfg

from .env_cfg import CentrifugeSO101EnvCfg, _ROBOT_BASE, _ROBOT_YAW_XYZW
from .so101_cfg import (
    ARM_JOINT_NAMES,
    BASE_LINK_NAME,
    END_EFFECTOR_LINK_NAME,
    GRIPPER_JOINT_NAME,
    JAW_CLOSE,
    JAW_OPEN,
    SO101_CFG,
)

# Robot base prim path. num_envs=1, so ``{ENV_REGEX_NS}`` = ``/World/envs/env_0``.
_BASE_PRIM_PATH = f"/World/envs/env_0/Robot/{BASE_LINK_NAME}"

# XR anchor: parented on the robot base. -90° yaw pan so the headset view aligns
# with the robot (base is yawed +90°). STARTING POINT — retune per operator.
_XR_ANCHOR_LOCAL_POS = (0.0, 0.0, -1.2)
_XR_ANCHOR_LOCAL_ROT_XYZW = (0.0, 0.0, -0.70710678, 0.70710678)

# Clutch engage-home: gripper pose in the robot BASE frame at our init joint pose,
# MEASURED in sim (ee_frame.target_pos_source at reset). Only the translation
# drives the position command; orientation is the live grip + _ORIENTATION_OFFSET.
# The robot init_state must FK to this (it does — this was read from it), so the
# first engage does not snap. Re-measure if the SO101 init joint pose changes.
_HOME_BASE_T_GRIPPER = np.eye(4, dtype=np.float32)
_HOME_BASE_T_GRIPPER[:3, 3] = (-0.04073, -0.06017, 0.1315)

# Fixed controller-grip -> gripper orientation calibration (xyzw), body-frame
# right-multiplied onto the grip orientation. None = clutch default Rz(pi). Tune
# on a headset (see module docstring); wrong orientation only tilts the gripper
# (position tracking + stability are unaffected).
_ORIENTATION_OFFSET_XYZW: tuple[float, float, float, float] | None = None


def _build_so101_ik_abs_pipeline():
    """Upstream-style clutch IsaacTeleop pipeline for single-arm SO101 + jaw.

    Controller-driven: :class:`SO101ClutchRetargeter` emits a smooth, engage-rebased
    absolute 7-D pose (no snap, no solver flipping), and :class:`SO101GripperRetargeter`
    an analog jaw closedness. Flattened to the 8-D action.
    """
    from isacteleop.retargeters import SO101ClutchRetargeter, SO101GripperRetargeter, TensorReorderer
    from isacteleop.retargeters.SO101.gripper_retargeter import GRIPPER_COMMAND_KEY, GRIPPER_ELEMENT_LABEL
    from isacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
    from isacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isacteleop.retargeting_engine.tensor_types import TransformMatrix

    controllers = ControllersSource(name="controllers")
    # target_frame_prim_path folds base_T_world into this transform, so the
    # controller stream arrives already in the robot base frame.
    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_controllers = controllers.transformed(transform_input.output(ValueInput.VALUE))

    clutch = SO101ClutchRetargeter(
        name="ee_pose",
        input_device=ControllersSource.RIGHT,
        home_base_T_ee=_HOME_BASE_T_GRIPPER,
        orientation_offset=(
            np.array(_ORIENTATION_OFFSET_XYZW, dtype=np.float64)
            if _ORIENTATION_OFFSET_XYZW is not None
            else None
        ),
        # Log the controller grip orientation (base frame) at each engage so the
        # exact calibration offset R_off = quat_inv(q_grip) (x) q_G0 can be set in
        # _ORIENTATION_OFFSET_XYZW. Turn off once calibrated.
        debug_log_engage=True,
    )
    connected_clutch = clutch.connect(
        {ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT)}
    )

    gripper = SO101GripperRetargeter(name="gripper", input_device=ControllersSource.RIGHT)
    connected_gripper = gripper.connect(
        {ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT)}
    )

    ee_pose_elements = ["pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"]
    gripper_elements = [GRIPPER_ELEMENT_LABEL]
    reorderer = TensorReorderer(
        input_config={"ee_pose": ee_pose_elements, GRIPPER_COMMAND_KEY: gripper_elements},
        output_order=ee_pose_elements + gripper_elements,
        name="action_reorderer",
        input_types={"ee_pose": "array", GRIPPER_COMMAND_KEY: "scalar"},
    )
    connected_reorderer = reorderer.connect(
        {
            "ee_pose": connected_clutch.output("ee_pose"),
            GRIPPER_COMMAND_KEY: connected_gripper.output(GRIPPER_COMMAND_KEY),
        }
    )
    return OutputCombiner({"action": connected_reorderer.output("output")})


@configclass
class SO101IkActionsCfg:
    """Action terms ordered [arm(7 pose), jaw(1)] — positional contract with the pipeline."""

    arm_action: SO101PoseIKActionCfg = MISSING
    gripper_action: JointPositionActionCfg = MISSING


@configclass
class CentrifugeSO101IkAbsEnvCfg(CentrifugeSO101EnvCfg):
    """SO101 absolute task-space (IK) VR teleop via the stable upstream clutch retargeter."""

    xr: XrCfg = XrCfg(
        anchor_pos=_XR_ANCHOR_LOCAL_POS,
        anchor_rot=_XR_ANCHOR_LOCAL_ROT_XYZW,
        anchor_prim_path=_BASE_PRIM_PATH,
    )

    def __post_init__(self):
        super().__post_init__()

        # robot — mount SO101 on the table, yawed toward the props.
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

        # Full-pose SE3 IK over the 5 arm joints + analog jaw. Order [arm, gripper]
        # is the positional contract with the pipeline output.
        self.actions = SO101IkActionsCfg(
            arm_action=SO101PoseIKActionCfg(
                asset_name="robot",
                joint_names=list(ARM_JOINT_NAMES),
                body_name=END_EFFECTOR_LINK_NAME,
                scale=1.0,
                controller=SO101PoseIKControllerCfg(
                    command_type="pose",
                    use_relative_mode=False,
                    ik_method="adaptive_dls",
                    ik_params={"lambda_min": 0.05, "lambda_max": 0.2, "sigma_thresh": 0.02},
                    orientation_weight=0.5,
                    orientation_joint_names=("Wrist_Pitch", "Wrist_Roll"),
                    joint_limit_avoidance_gain=0.5,
                    joint_limit_avoidance_margin=0.3,
                ),
            ),
            # Analog jaw: closedness c in [0,1] -> joint = offset + scale*c
            # (offset == open, offset+scale == close). Matches SO101GripperRetargeter.
            gripper_action=JointPositionActionCfg(
                asset_name="robot",
                joint_names=[GRIPPER_JOINT_NAME],
                scale=JAW_CLOSE - JAW_OPEN,
                offset=JAW_OPEN,
                use_default_offset=False,
            ),
        )

        # IsaacTeleop clutch pipeline. ``target_frame_prim_path`` rebases the
        # controller stream into the robot base frame (the clutch/IK command frame).
        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=_build_so101_ik_abs_pipeline,
            sim_device=self.sim.device,
            xr_cfg=self.xr,
            target_frame_prim_path=_BASE_PRIM_PATH,
        )
