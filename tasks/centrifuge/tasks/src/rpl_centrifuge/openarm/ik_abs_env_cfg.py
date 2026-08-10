# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual OpenArm + absolute differential-IK teleop variant.

This is the recommended variant for hand-tracking: the user's wrist pose in the
XR-anchor frame maps directly to the robot end-effector's absolute target pose.
No delta integration, no drift.

Wires:
  * Robot: ``OPENARM_BI_HIGH_PD_CFG`` (stiffer PD for IK tracking).
  * Two absolute-IK action terms (one per arm) on ``openarm_{left,right}_hand``.
  * Two binary gripper action terms on ``openarm_{left,right}_finger_joint.*``.
  * Bimanual hand-tracking teleop via OpenXR (one Se3Abs + Gripper retargeter per hand).

The XR anchor (``self.xr``) defines where the user is *physically* standing in
the env frame; tune it so the user's natural arm-extension pose lines up with
the robot's reachable workspace above the table.
"""

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DeviceBase, DevicesCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters.manipulator.gripper_retargeter import GripperRetargeterCfg
from isaaclab.devices.openxr.retargeters.manipulator.se3_abs_retargeter import Se3AbsRetargeterCfg
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

# Concrete stage paths of prims we need to reference. Bimanual-single-env task
# uses num_envs=1, so ``{ENV_REGEX_NS}`` always expands to ``/World/envs/env_0``;
# fine to hard-code. Change if the env count or scene hierarchy changes.
_BODY_LINK_PRIM_PATH = "/World/envs/env_0/Robot/openarm_body_link"
_CHEST_CAMERA_PRIM_PATH = f"{_BODY_LINK_PRIM_PATH}/chest_camera"

# XR headset anchor placement, tuned live in a seated session (see
# revert-isaaclab-env memory). We parent the anchor on ``openarm_body_link``
# (identity-rotated) rather than on ``chest_camera`` (which has a 45° pitch
# + 180° yaw) so a plain local translate maps 1:1 to world coordinates —
# no need to unwind the camera rotation. The z is pushed well below the
# robot because Kit XR adds the operator's real seated head height (~1.5 m)
# on top of the anchor position; setting the anchor to world z ≈ −0.19
# lands the actual headset view around z=1.33 (about 20 cm above the chest
# camera at z=1.13). Retune ``_XR_ANCHOR_LOCAL_POS.z`` per operator if the
# seated height differs materially.
_XR_ANCHOR_LOCAL_POS = (0.08, 0.0, -0.77)
# -90° yaw about world Z. Kit XR eye-forward doesn't align with the anchor's
# +X even under an identity-rot parent; empirically -90° puts 'look ahead' on
# the workspace (+90° pointed us backward, -90° = -90 mod 360 lands correct).
# xyzw quaternion:
#   -90° Z: (x=0, y=0, z=sin(-45°)=-0.7071068, w=cos(-45°)=0.7071068)
_XR_ANCHOR_LOCAL_ROT_XYZW = (0.0, 0.0, -0.7071068, 0.7071068)

from isaaclab_assets.robots.openarm import OPENARM_BI_HIGH_PD_CFG

from .env_cfg import CentrifugeEnvCfg


def _build_centrifuge_bimanual_pipeline():
    """Build the IsaacTeleop retargeting pipeline for bimanual OpenArm + grippers.

    Produces a single flattened 16-D action tensor matching the env action manager:

        [ left_arm_pose (7)      # pos_x,pos_y,pos_z, quat_x,quat_y,quat_z,quat_w
        , left_gripper    (1)    # scalar in [0, 1]
        , right_arm_pose  (7)
        , right_gripper   (1) ]

    Each arm gets an :class:`Se3AbsRetargeter` (7-D pose from the paired VR
    controller) and a :class:`GripperRetargeter` (scalar from the controller
    trigger, with hand-pinch fallback when the trigger is not present).
    A :class:`TensorReorderer` glues them into the layout above. The
    world→anchor transform is applied to the controller/hand poses via a
    :class:`ValueInput` so absolute poses arrive in the sim world frame.

    VR CONTROLLER LAYOUT (this is what the operator drives):
      * Right controller grip pose → right arm end-effector target.
      * Left  controller grip pose → left  arm end-effector target.
      * Right controller trigger   → right gripper open/close.
      * Left  controller trigger   → left  gripper open/close.
      * Thumbsticks / face buttons currently unmapped — see the G1
        locomanipulation task for a ``LocomotionRootCmdRetargeter`` example if
        you later want to bind thumbsticks to base velocity, etc.

    Modeled on the Franka stack IK-abs pipeline
    (``source/isaaclab_tasks/isaaclab_tasks/contrib/stack/config/franka/stack_ik_abs_env_cfg.py``)
    for controller-driven wrist pose + trigger gripper, doubled up for two
    arms. The wrist rotation offsets below are placeholders; tune them once
    you can compare the physical controller pose against the on-screen arm
    pose. See G1 locomanipulation cfg for a working example
    (``roll=45,pitch=180,yaw=-90`` for left, ``roll=-135,pitch=0,yaw=90`` for
    right) — those are G1-specific but illustrate the ballpark magnitudes.
    """
    import numpy as np
    from isaacteleop.retargeters import (
        GripperRetargeter,
        GripperRetargeterConfig,
        Se3AbsRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource, HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    # Static robot root world position. DifferentialInverseKinematicsAction in
    # absolute mode expects target pose in the robot ROOT frame, but the
    # retargeter emits world-frame targets (from world_T_anchor @ controller).
    # Subtracting this constant offset in ``_RootFrameSe3AbsRetargeter`` below
    # converts world → root. Assumes robot base is fixed (which it is — mounted
    # on the pedestal) AND has identity world rotation (verified via prims).
    # If the robot pose changes, recompute or read from the scene at runtime.
    _ROBOT_ROOT_POS_W = np.array([-0.62, 0.0, 0.5786], dtype=np.float32)

    class _RootFrameSe3AbsRetargeter(Se3AbsRetargeter):
        """Se3AbsRetargeter that emits targets in the robot ROOT frame.

        The base class writes world-frame targets via ``ee_pose[0] = final_pose``
        after computing them. We override ``_compute_fn`` to shift the position
        by ``-_ROBOT_ROOT_POS_W`` immediately after the base class runs, so the
        emitted target lands in root frame — the frame DifferentialInverse
        KinematicsAction (absolute mode) actually consumes.
        Rotation is untouched (root has identity world rotation).
        """

        def _compute_fn(self, inputs, outputs, context):
            super()._compute_fn(inputs, outputs, context)
            # Read the world-frame pose the parent just wrote, shift XYZ, write back.
            world_pose = np.asarray(self._last_pose, dtype=np.float32).copy()
            if world_pose[:3].any():  # skip default (0,0,0,identity) initial state
                world_pose[:3] -= _ROBOT_ROOT_POS_W
                self._last_pose = world_pose
                outputs["ee_pose"][0] = world_pose

    controllers = ControllersSource(name="controllers")
    hands = HandsSource(name="hands")
    world_T_anchor = ValueInput("world_T_anchor", TransformMatrix())
    # Only the controller pose feeds Se3AbsRetargeter (via ``input_device``);
    # hands still flow through to GripperRetargeter as a fallback pinch source.
    transformed_controllers = controllers.transformed(world_T_anchor.output(ValueInput.VALUE))

    def _se3_cfg(side: str) -> Se3RetargeterConfig:
        # ``input_device`` picks the tracker whose 6-DOF pose drives the arm.
        # ``target_offset_{roll,pitch,yaw}`` rotate the tracker frame into the
        # robot's wrist frame — set to identity here as a starting point;
        # retune from an XR session by comparing controller pose to arm pose.
        return Se3RetargeterConfig(
            input_device=ControllersSource.LEFT if side == "left" else ControllersSource.RIGHT,
            zero_out_xy_rotation=False,
            use_wrist_rotation=True,
            use_wrist_position=True,
            target_offset_roll=0.0,
            target_offset_pitch=0.0,
            target_offset_yaw=0.0,
        )

    left_se3 = _RootFrameSe3AbsRetargeter(_se3_cfg("left"), name="left_ee_pose")
    connected_left_se3 = left_se3.connect(
        {ControllersSource.LEFT: transformed_controllers.output(ControllersSource.LEFT)}
    )
    right_se3 = _RootFrameSe3AbsRetargeter(_se3_cfg("right"), name="right_ee_pose")
    connected_right_se3 = right_se3.connect(
        {ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT)}
    )

    # Gripper: trigger is primary (GripperRetargeter picks controller over hand
    # per its ``_compute_fn`` priority order), hand pinch is a graceful fallback
    # for pure hand-tracking sessions. Connect BOTH sources so either works.
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
    # Return both the OutputCombiner AND the Se3AbsRetargeter instances so the
    # tuning UI (enabled via ``IsaacTeleopCfg.retargeters_to_tune``) can adjust
    # target_offset_{roll,pitch,yaw} live during an XR session. Grippers are
    # threshold-based and don't need tuning.
    pipeline = OutputCombiner({"action": connected_reorderer.output("output")})
    tunable_retargeters = [left_se3, right_se3]
    return pipeline, tunable_retargeters


# Memoized wrapper so ``pipeline_builder`` and ``retargeters_to_tune`` in the
# ``IsaacTeleopCfg`` below return the SAME retargeter instances -- otherwise
# the tuning UI would edit a different copy from the one driving the pipeline.
_pipeline_cache: tuple | None = None


def _get_pipeline_and_retargeters():
    global _pipeline_cache
    if _pipeline_cache is None:
        _pipeline_cache = _build_centrifuge_bimanual_pipeline()
    return _pipeline_cache


@configclass
class CentrifugeBimanualIkAbsEnvCfg(CentrifugeEnvCfg):
    """Bimanual OpenArm with absolute differential-IK actions for hand-tracking teleop."""

    xr: XrCfg = XrCfg(
        # isaacteleop's XrAnchorManager creates a child ``XRAnchor`` Xform prim
        # under ``anchor_prim_path`` at these local (anchor_pos, anchor_rot)
        # values. Parenting on body_link (identity-rotated) means local
        # translate == world offset from body_link. See _XR_ANCHOR_LOCAL_POS
        # for the tuned values and rationale.
        anchor_pos=_XR_ANCHOR_LOCAL_POS,
        anchor_rot=_XR_ANCHOR_LOCAL_ROT_XYZW,
        anchor_prim_path=_BODY_LINK_PRIM_PATH,
        # FIXED (not FOLLOW_PRIM): the per-frame synchronizer under FOLLOW_PRIM
        # overwrites the anchor's orientation with body_link's, which cancels
        # the +90° yaw we bake into ``anchor_rot``. FIXED locks orientation at
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

        # IK action terms — one per arm. Absolute mode: action = 6-D target pose
        # in the robot base frame (no per-step scaling).
        self.actions.left_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["openarm_left_joint[1-7]"],
            body_name="openarm_left_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"
            ),
        )
        self.actions.right_arm_action = DifferentialInverseKinematicsActionCfg(
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

        # Legacy local-input teleop path. Selected when the user passes
        # ``--teleop_device handtracking`` on ``teleop_se3_agent.py``; not used
        # with ``--xr`` / CloudXR (that goes through ``self.isaac_teleop`` below).
        # Each hand owns one Se3Abs retargeter (7-D pose) and one gripper
        # retargeter (1-D scalar). Concatenated in action-manager term order:
        #   [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)] = 16
        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        Se3AbsRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_LEFT,
                            zero_out_xy_rotation=True,
                            use_wrist_rotation=False,
                            use_wrist_position=True,
                            sim_device=self.sim.device,
                        ),
                        GripperRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_LEFT,
                            sim_device=self.sim.device,
                        ),
                        Se3AbsRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                            zero_out_xy_rotation=True,
                            use_wrist_rotation=False,
                            use_wrist_position=True,
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

        # CloudXR / IsaacTeleop path. Selected automatically by
        # ``teleop_se3_agent.py`` when ``--teleop_device`` is NOT passed AND this
        # attribute is set. The pipeline builder emits the same 16-D action
        # tensor as the legacy path above, so both stacks drive the same
        # ActionsCfg terms.
        # Memoized helper returns (pipeline, retargeters) — both callbacks read
        # the same cached tuple so the tuning UI edits the retargeter instances
        # actually driving the pipeline. See _pipeline_cache docstring.
        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=lambda: _get_pipeline_and_retargeters()[0],
            retargeters_to_tune=lambda: _get_pipeline_and_retargeters()[1],
            sim_device=self.sim.device,
            xr_cfg=self.xr,
        )
