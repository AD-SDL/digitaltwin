# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual OpenArm + absolute differential-IK teleop variant (IsaacLab 3.0.0).

Recommended variant for VR teleoperation: the operator's wrist pose in the
XR-anchor frame maps directly to each arm's absolute end-effector target. No
delta integration, no drift.

Input-source mapping (matches the upstream reference tasks
``IsaacContrib-Stack-Cube-Franka-IK-Abs`` /
``IsaacContrib-Stack-Cube-SO101-IK-Abs``): teleop runs entirely through the
``isaacteleop`` pipeline on :attr:`isaac_teleop`. Each arm is driven by a
**dual-source** absolute-pose retargeter that accepts EITHER the VR controller
grip pose (joystick) OR the tracked wrist pose (hand-tracking) — controller has
priority, hand-tracking is the fallback (see :mod:`rpl_centrifuge.teleop`). Each
gripper is driven by the stock ``GripperRetargeter``, which likewise fuses the
controller trigger (priority) with the hand pinch (fallback). So one task
supports both modalities in a single session.

Base-frame rebase: ``IsaacTeleopCfg.target_frame_prim_path`` is set to the robot
body link, so the device folds ``base_T_world`` into the anchor transform and the
retargeters emit targets already in the robot base frame — the frame the
absolute differential-IK action consumes. This replaces the previous
hand-rolled ``_RootFrameSe3AbsRetargeter`` (which subtracted a hard-coded world
offset by poking retargeter internals).

There is no legacy ``teleop_devices`` path here: importing the old
``isaaclab.devices.openxr.*`` cfgs pulls in Kit's ``carb`` and breaks
headless import (gym listing, sim-free tests). The reference tasks dropped it
too.

Action tensor layout (16-D total), concatenated in action-manager term order:
  [ left_arm_pose (7)   # pos_x,pos_y,pos_z, quat_x,quat_y,quat_z,quat_w (base frame)
  , left_gripper  (1)   # scalar (-1 closed / +1 open)
  , right_arm_pose(7)
  , right_gripper (1) ]
"""

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,  # noqa: F401  (base class of the debug term)
)
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass

# ``isaaclab_teleop`` imports cleanly without the optional ``isaacteleop`` package
# (the heavy import is deferred to session start), so it is safe at module top —
# unlike the legacy ``isaaclab.devices.openxr.*`` cfgs.
from isaaclab_teleop import IsaacTeleopCfg
from isaaclab_teleop.xr_cfg import XrAnchorRotationMode, XrCfg

from isaaclab_assets.robots.openarm import OPENARM_BI_HIGH_PD_CFG

from .debug_ik_action import DebugDifferentialInverseKinematicsActionCfg
from .env_cfg import CentrifugeEnvCfg, _ROBOT_BASE_POS_W

# Concrete stage paths of prims we reference. Bimanual single-env task uses
# num_envs=1, so ``{ENV_REGEX_NS}`` always expands to ``/World/envs/env_0``.
_BODY_LINK_PRIM_PATH = "/World/envs/env_0/Robot/openarm_body_link"

# XR headset anchor placement (SeattleLabTable frame). Parented on the
# identity-rotated ``openarm_body_link`` so a plain local translate maps 1:1 to
# world coordinates. Kit XR adds the operator's real seated head height (~1.5 m)
# on top of the anchor, so the z is pushed well down to land the headset view
# above the table. With ``target_frame_prim_path`` doing the base rebase, the
# anchor governs only the operator's *view*. STARTING POINT — retune per operator
# while watching the --debug red (target) / green (current EE) markers.
_XR_ANCHOR_LOCAL_POS = (-0.42, 0.0, -0.02)
# -90° yaw about world Z (xyzw): (0, 0, sin(-45°), cos(-45°)).
_XR_ANCHOR_LOCAL_ROT_XYZW = (0.0, 0.0, -0.7071068, 0.7071068)


def _build_centrifuge_bimanual_pipeline():
    """Build the dual-source IsaacTeleop pipeline for bimanual OpenArm + grippers.

    Produces a single flattened 16-D action tensor (see module docstring). Each
    arm gets a dual-source :class:`Se3AbsRetargeter` (controller grip pose OR
    tracked wrist pose) and a :class:`GripperRetargeter` (controller trigger OR
    hand pinch). A :class:`TensorReorderer` flattens them.

    The world→anchor transform is applied to BOTH the controller and the hand
    streams via a :class:`ValueInput`, so poses arrive in the robot base frame
    (``target_frame_prim_path`` folds ``base_T_world`` into that transform).

    VR CONTROLLER / HAND LAYOUT (either modality drives the same targets):
      * Right controller grip pose  OR  right wrist  → right arm EE target.
      * Left  controller grip pose  OR  left  wrist  → left  arm EE target.
      * Right controller trigger    OR  right pinch  → right gripper.
      * Left  controller trigger    OR  left  pinch  → left  gripper.

    Returns ``(pipeline, tunable_retargeters)`` for the memoized cache pattern so
    the tuning UI edits the same retargeter instances driving the pipeline.
    """
    from isaacteleop.retargeters import (
        GripperRetargeter,
        GripperRetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource, HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    from ..teleop import make_dual_source_se3_abs_retargeter

    controllers = ControllersSource(name="controllers")
    hands = HandsSource(name="hands")
    world_T_anchor = ValueInput("world_T_anchor", TransformMatrix())
    transformed_controllers = controllers.transformed(world_T_anchor.output(ValueInput.VALUE))
    transformed_hands = hands.transformed(world_T_anchor.output(ValueInput.VALUE))

    # Wrist-rotation offsets (DEGREES, intrinsic XYZ) rotate the controller/wrist
    # frame into the robot hand frame. Informed by the upstream OpenArm bimanual
    # reach task (hand link points the gripper along +z; pitch -90, yaw 180 make a
    # forward-held controller face into the workspace; roll 90 is isaacteleop's
    # default). These target the CONTROLLER frame; hand-tracking may want
    # different values — tune live via the tuning UI (``rotation_offset_rpy``).
    _abs_kwargs = dict(
        zero_out_xy_rotation=False,
        use_wrist_rotation=True,
        use_wrist_position=True,
        target_offset_roll=90.0,
        target_offset_pitch=-90.0,
        target_offset_yaw=180.0,
    )

    left_se3 = make_dual_source_se3_abs_retargeter("left", "left_ee_pose", **_abs_kwargs)
    connected_left_se3 = left_se3.connect(
        {
            ControllersSource.LEFT: transformed_controllers.output(ControllersSource.LEFT),
            HandsSource.LEFT: transformed_hands.output(HandsSource.LEFT),
        }
    )
    right_se3 = make_dual_source_se3_abs_retargeter("right", "right_ee_pose", **_abs_kwargs)
    connected_right_se3 = right_se3.connect(
        {
            ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT),
            HandsSource.RIGHT: transformed_hands.output(HandsSource.RIGHT),
        }
    )

    # Gripper: controller trigger primary, hand pinch fallback (native to
    # GripperRetargeter). Connect BOTH sources so either modality works.
    left_gripper = GripperRetargeter(GripperRetargeterConfig(hand_side="left"), name="left_gripper")
    connected_left_gripper = left_gripper.connect(
        {
            ControllersSource.LEFT: controllers.output(ControllersSource.LEFT),
            HandsSource.LEFT: hands.output(HandsSource.LEFT),
        }
    )
    right_gripper = GripperRetargeter(GripperRetargeterConfig(hand_side="right"), name="right_gripper")
    connected_right_gripper = right_gripper.connect(
        {
            ControllersSource.RIGHT: controllers.output(ControllersSource.RIGHT),
            HandsSource.RIGHT: hands.output(HandsSource.RIGHT),
        }
    )

    # Flatten. Element name prefixes disambiguate the two arms in the reorderer.
    left_pose_elements = ["l_pos_x", "l_pos_y", "l_pos_z", "l_quat_x", "l_quat_y", "l_quat_z", "l_quat_w"]
    right_pose_elements = ["r_pos_x", "r_pos_y", "r_pos_z", "r_quat_x", "r_quat_y", "r_quat_z", "r_quat_w"]
    left_gripper_elements = ["l_gripper"]
    right_gripper_elements = ["r_gripper"]

    reorderer = TensorReorderer(
        input_config={
            "left_ee_pose": left_pose_elements,
            "left_gripper": left_gripper_elements,
            "right_ee_pose": right_pose_elements,
            "right_gripper": right_gripper_elements,
        },
        output_order=left_pose_elements + left_gripper_elements + right_pose_elements + right_gripper_elements,
        name="action_reorderer",
        input_types={
            "left_ee_pose": "array",
            "left_gripper": "scalar",
            "right_ee_pose": "array",
            "right_gripper": "scalar",
        },
    )
    connected_reorderer = reorderer.connect(
        {
            "left_ee_pose": connected_left_se3.output("ee_pose"),
            "left_gripper": connected_left_gripper.output("gripper_command"),
            "right_ee_pose": connected_right_se3.output("ee_pose"),
            "right_gripper": connected_right_gripper.output("gripper_command"),
        }
    )
    pipeline = OutputCombiner({"action": connected_reorderer.output("output")})
    tunable_retargeters = [left_se3, right_se3]
    return pipeline, tunable_retargeters


# Memoized wrapper so ``pipeline_builder`` and ``retargeters_to_tune`` return the
# SAME retargeter instances -- otherwise the tuning UI would edit a different
# copy from the one driving the pipeline.
_pipeline_cache: tuple | None = None


def _get_pipeline_and_retargeters():
    global _pipeline_cache
    if _pipeline_cache is None:
        _pipeline_cache = _build_centrifuge_bimanual_pipeline()
    return _pipeline_cache


@configclass
class CentrifugeBimanualIkAbsEnvCfg(CentrifugeEnvCfg):
    """Bimanual OpenArm with absolute differential-IK actions for VR teleop.

    Supports both hand-tracking and joystick (VR-controller) control in one
    session via the dual-source pipeline.
    """

    xr: XrCfg = XrCfg(
        anchor_pos=_XR_ANCHOR_LOCAL_POS,
        anchor_rot=_XR_ANCHOR_LOCAL_ROT_XYZW,
        anchor_prim_path=_BODY_LINK_PRIM_PATH,
        # FIXED (not FOLLOW_PRIM): FOLLOW_PRIM's per-frame synchronizer would
        # overwrite the anchor orientation with body_link's and cancel the baked
        # -90° yaw. FIXED locks it; the user's head still rotates freely on top.
        anchor_rotation_mode=XrAnchorRotationMode.FIXED,
    )

    def __post_init__(self):
        super().__post_init__()

        # robot — mount the bimanual OpenArm on top of the pedestal.
        self.scene.robot = OPENARM_BI_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = _ROBOT_BASE_POS_W

        # end-effector frame transformers (relative to robot body root)
        self.scene.left_ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_body_link",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/openarm_left_hand",
                    name="left_end_effector",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                ),
            ],
        )
        self.scene.right_ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_body_link",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/openarm_right_hand",
                    name="right_end_effector",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                ),
            ],
        )

        # IK action terms — one per arm. Absolute mode: action = target pose in
        # the robot base frame (no per-step scaling). The debug-capable subclass
        # renders the IK target (red) vs current EE (green) markers under
        # ``--debug``; behaves like the base term otherwise.
        self.actions.left_arm_action = DebugDifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["openarm_left_joint[1-7]"],
            body_name="openarm_left_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"
            ),
        )
        self.actions.right_arm_action = DebugDifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["openarm_right_joint[1-7]"],
            body_name="openarm_right_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"
            ),
        )

        # Binary gripper action terms — open/close per arm.
        self.actions.left_gripper_action = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_left_finger_joint.*"],
            open_command_expr={"openarm_left_finger_joint.*": 0.044},
            close_command_expr={"openarm_left_finger_joint.*": 0.0},
        )
        self.actions.right_gripper_action = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_right_finger_joint.*"],
            open_command_expr={"openarm_right_finger_joint.*": 0.044},
            close_command_expr={"openarm_right_finger_joint.*": 0.0},
        )

        # CloudXR / IsaacTeleop path (the only teleop path — no legacy
        # ``teleop_devices``). ``teleop_se3_agent.py`` / ``record_demos.py`` pick
        # this automatically when ``--teleop_device`` is NOT passed. The pipeline
        # emits the 16-D action tensor matching the ActionsCfg terms above.
        # ``target_frame_prim_path`` rebases teleop output into the robot base
        # frame (the absolute-IK command frame).
        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=lambda: _get_pipeline_and_retargeters()[0],
            retargeters_to_tune=lambda: _get_pipeline_and_retargeters()[1],
            sim_device=self.sim.device,
            xr_cfg=self.xr,
            target_frame_prim_path=_BODY_LINK_PRIM_PATH,
        )
