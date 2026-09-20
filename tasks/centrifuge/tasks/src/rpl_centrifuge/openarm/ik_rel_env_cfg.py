# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual OpenArm + relative differential-IK teleop variant (IsaacLab 3.0.0).

Uses dual-source :class:`Se3RelRetargeter` (isaacteleop) +
``DifferentialIKController(use_relative_mode=True)``. Both consume frame-agnostic
6-D pose deltas — no world-vs-base-frame mismatch — so relative mode needs no
``target_frame_prim_path`` rebase.

Input-source mapping: teleop runs entirely through the ``isaacteleop`` pipeline on
:attr:`isaac_teleop`. Each arm is driven by a **dual-source** relative-pose
retargeter that accepts EITHER the VR controller pose delta (joystick) OR the
tracked wrist pose delta (hand-tracking); each gripper by the stock
``GripperRetargeter`` (trigger primary, pinch fallback). One task supports both
modalities in a single session.

There is no legacy ``teleop_devices`` path: the old ``isaaclab.devices.openxr.*``
and ``isaaclab.devices.keyboard`` cfgs import Kit's ``carb`` and break headless
import. The reference tasks dropped that path too.

Action tensor layout (14-D total):
  [ left_arm_delta (6)   # dx, dy, dz, rx, ry, rz
  , left_gripper   (1)   # scalar (-1 closed / +1 open)
  , right_arm_delta(6)
  , right_gripper  (1) ]
"""

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,  # noqa: F401  (base class of the debug term)
)
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass

from isaaclab_teleop import IsaacTeleopCfg
from isaaclab_teleop.xr_cfg import XrAnchorRotationMode, XrCfg

from isaaclab_assets.robots.openarm import OPENARM_BI_HIGH_PD_CFG

from .debug_ik_action import DebugDifferentialInverseKinematicsActionCfg
from .env_cfg import CentrifugeEnvCfg, _ROBOT_BASE_POS_W

# Concrete stage paths. num_envs=1, so ``{ENV_REGEX_NS}`` = ``/World/envs/env_0``.
_BODY_LINK_PRIM_PATH = "/World/envs/env_0/Robot/openarm_body_link"

# XR headset anchor placement (SeattleLabTable frame) — see IK-Abs cfg for the
# tuning rationale. STARTING POINT; retune per operator in sim.
_XR_ANCHOR_LOCAL_POS = (-0.42, 0.0, -0.02)
_XR_ANCHOR_LOCAL_ROT_XYZW = (0.0, 0.0, -0.7071068, 0.7071068)


def _build_centrifuge_bimanual_rel_pipeline():
    """Build the dual-source IsaacTeleop pipeline for bimanual OpenArm (relative IK).

    Produces a flattened 14-D action tensor (see module docstring). Each arm gets
    a dual-source :class:`Se3RelRetargeter` (6-D delta from the live controller OR
    tracked wrist each frame) and a :class:`GripperRetargeter` (trigger primary,
    pinch fallback). Delta output means no world-vs-base frame issue: the DIK
    controller (``use_relative_mode=True``) integrates deltas from the arm's
    CURRENT ee pose.

    Returns ``(pipeline, tunable_retargeters)`` for the memoized cache pattern.
    """
    from isaacteleop.retargeters import (
        GripperRetargeter,
        GripperRetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource, HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    from ..teleop import make_dual_source_se3_rel_retargeter

    controllers = ControllersSource(name="controllers")
    hands = HandsSource(name="hands")
    world_T_anchor = ValueInput("world_T_anchor", TransformMatrix())
    transformed_controllers = controllers.transformed(world_T_anchor.output(ValueInput.VALUE))
    transformed_hands = hands.transformed(world_T_anchor.output(ValueInput.VALUE))

    # Delta scale factors amplify per-frame motion into the IK delta command.
    # 1.0 = 1:1; tune down if arm motion feels too fast, up if too sluggish.
    _rel_kwargs = dict(
        zero_out_xy_rotation=False,
        use_wrist_rotation=True,
        use_wrist_position=True,
        delta_pos_scale_factor=1.0,
        delta_rot_scale_factor=1.0,
    )

    left_se3 = make_dual_source_se3_rel_retargeter("left", "left_ee_delta", **_rel_kwargs)
    connected_left_se3 = left_se3.connect(
        {
            ControllersSource.LEFT: transformed_controllers.output(ControllersSource.LEFT),
            HandsSource.LEFT: transformed_hands.output(HandsSource.LEFT),
        }
    )
    right_se3 = make_dual_source_se3_rel_retargeter("right", "right_ee_delta", **_rel_kwargs)
    connected_right_se3 = right_se3.connect(
        {
            ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT),
            HandsSource.RIGHT: transformed_hands.output(HandsSource.RIGHT),
        }
    )

    # Gripper: trigger primary, hand-pinch fallback. Connect BOTH sources.
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

    left_delta_elements = ["l_dx", "l_dy", "l_dz", "l_rx", "l_ry", "l_rz"]
    right_delta_elements = ["r_dx", "r_dy", "r_dz", "r_rx", "r_ry", "r_rz"]
    left_gripper_elements = ["l_gripper"]
    right_gripper_elements = ["r_gripper"]

    reorderer = TensorReorderer(
        input_config={
            "left_ee_delta": left_delta_elements,
            "left_gripper": left_gripper_elements,
            "right_ee_delta": right_delta_elements,
            "right_gripper": right_gripper_elements,
        },
        output_order=left_delta_elements + left_gripper_elements + right_delta_elements + right_gripper_elements,
        name="action_reorderer",
        input_types={
            "left_ee_delta": "array",
            "left_gripper": "scalar",
            "right_ee_delta": "array",
            "right_gripper": "scalar",
        },
    )
    connected_reorderer = reorderer.connect(
        {
            "left_ee_delta": connected_left_se3.output("ee_delta"),
            "left_gripper": connected_left_gripper.output("gripper_command"),
            "right_ee_delta": connected_right_se3.output("ee_delta"),
            "right_gripper": connected_right_gripper.output("gripper_command"),
        }
    )
    pipeline = OutputCombiner({"action": connected_reorderer.output("output")})
    # Se3RelRetargeter does not register tunable UI parameters (only the Abs
    # variant does), so no ``retargeters_to_tune`` is returned. Tune via
    # ``delta_*_scale_factor`` here and ``scale`` on the DIK action term.
    return pipeline, []


_pipeline_cache: tuple | None = None


def _get_rel_pipeline_and_retargeters():
    global _pipeline_cache
    if _pipeline_cache is None:
        _pipeline_cache = _build_centrifuge_bimanual_rel_pipeline()
    return _pipeline_cache


@configclass
class CentrifugeBimanualIkRelEnvCfg(CentrifugeEnvCfg):
    """Bimanual OpenArm with relative differential-IK actions for VR teleop.

    Supports both hand-tracking and joystick (VR-controller) control in one
    session via the dual-source pipeline.
    """

    xr: XrCfg = XrCfg(
        anchor_pos=_XR_ANCHOR_LOCAL_POS,
        anchor_rot=_XR_ANCHOR_LOCAL_ROT_XYZW,
        anchor_prim_path=_BODY_LINK_PRIM_PATH,
        anchor_rotation_mode=XrAnchorRotationMode.FIXED,
    )

    def __post_init__(self):
        super().__post_init__()

        # robot — mount the bimanual OpenArm (shared base pos with env_cfg / abs).
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

        # IK action terms — one per arm. Relative mode: action = 6-D delta pose.
        # ``scale=0.5`` dampens the retargeter output; tune together with
        # ``delta_pos_scale_factor``. Debug-capable subclass renders --debug markers.
        self.actions.left_arm_action = DebugDifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["openarm_left_joint[1-7]"],
            body_name="openarm_left_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
            scale=0.5,
        )
        self.actions.right_arm_action = DebugDifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["openarm_right_joint[1-7]"],
            body_name="openarm_right_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
            scale=0.5,
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

        # IsaacTeleop pipeline (the only teleop path — no legacy teleop_devices).
        # Relative deltas are frame-invariant so no target_frame_prim_path needed.
        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=lambda: _get_rel_pipeline_and_retargeters()[0],
            sim_device=self.sim.device,
            xr_cfg=self.xr,
        )
