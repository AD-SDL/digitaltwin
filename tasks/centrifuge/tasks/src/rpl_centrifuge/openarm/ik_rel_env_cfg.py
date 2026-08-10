# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual OpenArm + relative differential-IK teleop variant.

Uses ``Se3RelRetargeter`` (isaacteleop) + ``DifferentialIKController(use_relative_mode=True)``.
Both consume frame-agnostic 6-D pose deltas — no world-vs-root-frame mismatch
(the pitfall that made IK-Abs command targets 0.6-1.5 m off in world Z).

Wires:
  * Robot: ``OPENARM_BI_HIGH_PD_CFG`` (stiffer PD for IK tracking).
  * Two relative-IK action terms (one per arm) on ``openarm_{left,right}_hand``.
  * Two binary gripper action terms on ``openarm_{left,right}_finger_joint.*``.
  * CloudXR/isaacteleop pipeline: two Se3Rel + two Gripper retargeters, VR
    controllers primary (grip pose + trigger), hand-pinch as gripper fallback.
  * Legacy ``teleop_devices`` retained so ``--teleop_device handtracking`` /
    ``--teleop_device keyboard`` still work without the isaacteleop stack.

Action tensor layout (14-D total):
  [ left_arm_delta (6)   # dx, dy, dz, rx, ry, rz
  , left_gripper   (1)   # scalar in {-1.0 closed, 1.0 open}
  , right_arm_delta(6)
  , right_gripper  (1) ]
"""

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DeviceBase, DevicesCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters.manipulator.gripper_retargeter import GripperRetargeterCfg
from isaaclab.devices.openxr.retargeters.manipulator.se3_rel_retargeter import Se3RelRetargeterCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass

# IsaacTeleop-based (CloudXR-compatible) teleop stack. See
# ``scripts/environments/teleoperation/teleop_se3_agent.py``: the ``--xr`` flag
# activates the Kit XR renderer + CloudXR pipeline, and the script picks the
# IsaacTeleop path whenever ``env_cfg.isaac_teleop`` is set AND
# ``--teleop_device`` is NOT passed. Keep the legacy ``teleop_devices`` field
# too so users can still choose the local-input path with ``--teleop_device
# handtracking``.
from isaaclab_teleop import IsaacTeleopCfg
from isaaclab_teleop.xr_cfg import XrAnchorRotationMode, XrCfg

# Concrete stage paths. num_envs=1, so ``{ENV_REGEX_NS}`` = ``/World/envs/env_0``.
_BODY_LINK_PRIM_PATH = "/World/envs/env_0/Robot/openarm_body_link"

# XR headset anchor placement — see IK-Abs cfg for the tuning rationale. Same
# values: parent on identity-rotated body_link, push z below ground because Kit
# XR adds the user's real seated height (~1.5 m) on top of the anchor, apply
# -90° yaw so 'look ahead' faces the workspace.
_XR_ANCHOR_LOCAL_POS = (0.08, 0.0, -0.77)
_XR_ANCHOR_LOCAL_ROT_XYZW = (0.0, 0.0, -0.7071068, 0.7071068)

from isaaclab_assets.robots.openarm import OPENARM_BI_HIGH_PD_CFG

from .env_cfg import CentrifugeEnvCfg
from .devices import BimanualKeyboardRightArmCfg


def _build_centrifuge_bimanual_rel_pipeline():
    """Build the IsaacTeleop retargeting pipeline for bimanual OpenArm (relative IK).

    Produces a flattened 14-D action tensor matching the env action manager:

        [ left_arm_delta (6)    # dx, dy, dz, rx, ry, rz  (frame-invariant)
        , left_gripper   (1)    # scalar in {-1 closed, 1 open}
        , right_arm_delta(6)
        , right_gripper  (1) ]

    Each arm gets an :class:`Se3RelRetargeter` (6-D delta from VR controller
    pose deltas each frame) and a :class:`GripperRetargeter` (trigger primary,
    hand-pinch fallback). Delta output means no world-vs-root frame issue —
    the DIK controller (``use_relative_mode=True``) integrates deltas from
    the arm's CURRENT ee pose, so both sides operate in the same implicit frame.

    Returns ``(pipeline, tunable_retargeters)`` for the memoized cache pattern
    below (both callbacks of ``IsaacTeleopCfg`` need the SAME retargeter
    instances or the tuning UI edits a phantom copy).
    """
    from isaacteleop.retargeters import (
        GripperRetargeter,
        GripperRetargeterConfig,
        Se3RelRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource, HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    controllers = ControllersSource(name="controllers")
    hands = HandsSource(name="hands")
    world_T_anchor = ValueInput("world_T_anchor", TransformMatrix())
    transformed_controllers = controllers.transformed(world_T_anchor.output(ValueInput.VALUE))

    def _se3_rel_cfg(side: str) -> Se3RetargeterConfig:
        # Delta scale factors amplify per-frame controller motion into a larger
        # IK delta command. 10x is the value the legacy hand-tracking cfg used;
        # tune down if arm motion feels too fast, up if too sluggish.
        return Se3RetargeterConfig(
            input_device=ControllersSource.LEFT if side == "left" else ControllersSource.RIGHT,
            zero_out_xy_rotation=False,
            use_wrist_rotation=True,
            use_wrist_position=True,
            delta_pos_scale_factor=1.0,
            delta_rot_scale_factor=1.0,
        )

    left_se3 = Se3RelRetargeter(_se3_rel_cfg("left"), name="left_ee_delta")
    connected_left_se3 = left_se3.connect(
        {ControllersSource.LEFT: transformed_controllers.output(ControllersSource.LEFT)}
    )
    right_se3 = Se3RelRetargeter(_se3_rel_cfg("right"), name="right_ee_delta")
    connected_right_se3 = right_se3.connect(
        {ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT)}
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

    # Flatten. Element name prefixes disambiguate the two arms in the reorderer.
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
    tunable_retargeters = [left_se3, right_se3]
    return pipeline, tunable_retargeters


# Memoized wrapper so ``pipeline_builder`` and ``retargeters_to_tune`` in the
# ``IsaacTeleopCfg`` below return the SAME retargeter instances -- otherwise
# the tuning UI would edit a different copy from the one driving the pipeline.
_pipeline_cache: tuple | None = None


def _get_rel_pipeline_and_retargeters():
    global _pipeline_cache
    if _pipeline_cache is None:
        _pipeline_cache = _build_centrifuge_bimanual_rel_pipeline()
    return _pipeline_cache


@configclass
class CentrifugeBimanualIkRelEnvCfg(CentrifugeEnvCfg):
    """Bimanual OpenArm with relative differential-IK actions for VR-controller teleop."""

    xr: XrCfg = XrCfg(
        anchor_pos=_XR_ANCHOR_LOCAL_POS,
        anchor_rot=_XR_ANCHOR_LOCAL_ROT_XYZW,
        anchor_prim_path=_BODY_LINK_PRIM_PATH,
        # FIXED (not FOLLOW_PRIM): the per-frame synchronizer under FOLLOW_PRIM
        # overwrites the anchor's orientation with body_link's, cancelling the
        # -90° yaw we bake into ``anchor_rot``. FIXED locks orientation at
        # the initial value; the user's head still rotates freely on top.
        anchor_rotation_mode=XrAnchorRotationMode.FIXED,
    )

    def __post_init__(self):
        super().__post_init__()

        # robot — mount the bimanual OpenArm on top of the pedestal (matches aiet_scene.usd)
        self.scene.robot = OPENARM_BI_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (-0.62, 0.0, 0.5786)

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

        # IK action terms — one per arm. Relative mode: action = 6-D delta pose
        # in the ee body frame (or a scaled Cartesian delta depending on
        # DIK controller settings). ``scale=0.5`` dampens the retargeter output;
        # tune together with ``delta_pos_scale_factor`` on Se3RelRetargeter.
        self.actions.left_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["openarm_left_joint[1-7]"],
            body_name="openarm_left_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
            scale=0.5,
        )
        self.actions.right_arm_action = DifferentialInverseKinematicsActionCfg(
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

        # IsaacTeleop pipeline: teleop_se3_agent.py auto-picks this when --xr is
        # passed AND --teleop_device is NOT set. Delta output + relative DIK =
        # frame-agnostic control that avoids the IK-Abs world/root confusion.
        # NOTE: ``retargeters_to_tune`` is intentionally omitted. Se3RelRetargeter
        # doesn't register tunable parameters (only Se3AbsRetargeter does), and
        # ``MultiRetargeterTuningUIImGui`` raises "No retargeters with tunable
        # parameters found" if we hand it an all-Rel list. Tune the relative
        # arm via ``delta_pos_scale_factor`` / ``delta_rot_scale_factor`` in
        # ``_se3_rel_cfg`` and ``scale`` on the DIK action term at cfg-load
        # time; live retargeter tuning is not supported for the Rel variant.
        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=lambda: _get_rel_pipeline_and_retargeters()[0],
            sim_device=self.sim.device,
            xr_cfg=self.xr,
        )

        # Legacy teleop devices retained so users without CloudXR/VR can still
        # test via ``--teleop_device keyboard`` or ``--teleop_device handtracking``.
        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": BimanualKeyboardRightArmCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        Se3RelRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_LEFT,
                            zero_out_xy_rotation=True,
                            use_wrist_rotation=False,
                            use_wrist_position=True,
                            delta_pos_scale_factor=10.0,
                            delta_rot_scale_factor=10.0,
                            sim_device=self.sim.device,
                        ),
                        GripperRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_LEFT,
                            sim_device=self.sim.device,
                        ),
                        Se3RelRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                            zero_out_xy_rotation=True,
                            use_wrist_rotation=False,
                            use_wrist_position=True,
                            delta_pos_scale_factor=10.0,
                            delta_rot_scale_factor=10.0,
                            sim_device=self.sim.device,
                        ),
                        GripperRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                            sim_device=self.sim.device,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )
