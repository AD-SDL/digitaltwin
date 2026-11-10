# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visuomotor (camera) variant of the OpenArm cabinet IK-Abs task.

Takes the camera-free IK-Abs task and adds cameras + RGB image observations,
for **headless dataset generation** (``generate_dataset`` on the Mimic
variant). This is NOT for live XR teleop -- rendering a camera observation
under the Kit XR pipeline crashes Kit; the canonical flow records demos with
the camera-free teleop task and renders images here, headless.

Cameras: ``left_wrist_cam`` / ``right_wrist_cam`` on each OpenArm hand, plus a
``chest_cam`` on the robot body link that frames the cabinet door.

The chest camera is aimed in sim: it looks along the body link's +X (toward
the cabinet) with a 10 deg downward pitch, which centres both lever handles in
the 848x480 frame at the task's robot/cabinet standoff. Verify after any change
to the robot base pose or the cabinet standoff with::

    cd <isaaclab_root>
    ./isaaclab.sh -p <task_package_root>/scripts/snapshot_cameras.py --out-dir /tmp/cams

The wrist mount offsets are still STARTING POINTS fitted by eye against the
real footage (see the ``cabinet_sim/openarm_cameras.py`` prototype carried into
``scripts/openarm_cameras.py``); they look down the forearm with the gripper in
frame, but the exact pose has not been measured off the physical robot. Tune in
sim before publishing a synthetic dataset.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils.presets import set_isaac_rtx_global_settings

from .. import mdp
from .env_cfg import ObservationsCfg
from .ik_abs_env_cfg import OpenDoorBimanualIkAbsEnvCfg


@configclass
class _VisuomotorPolicyCfg(ObservationsCfg.PolicyCfg):
    """Low-dim policy observations + the RGB image terms.

    The images go in the **policy** group, not a separate ``rgb_camera`` group:
    ``generate_dataset.py`` falls back to ``ActionStateRecorderManagerCfg``,
    whose only observation recorder is ``PreStepFlatPolicyObservationsRecorder``,
    which records ``obs_buf["policy"]`` and nothing else. See the centrifuge
    pack's visuomotor cfg for the full discussion.
    """

    chest_cam = ObsTerm(
        func=mdp.image,
        params={"sensor_cfg": SceneEntityCfg("chest_cam"), "data_type": "rgb", "normalize": False},
    )
    left_wrist_cam = ObsTerm(
        func=mdp.image,
        params={"sensor_cfg": SceneEntityCfg("left_wrist_cam"), "data_type": "rgb", "normalize": False},
    )
    right_wrist_cam = ObsTerm(
        func=mdp.image,
        params={"sensor_cfg": SceneEntityCfg("right_wrist_cam"), "data_type": "rgb", "normalize": False},
    )


@configclass
class _VisuomotorObservationsCfg(ObservationsCfg):
    """Base observation groups, with the image terms folded into ``policy``."""

    policy: _VisuomotorPolicyCfg = _VisuomotorPolicyCfg()


# -- camera specs -----------------------------------------------------------
#
# Resolutions match the real LeRobot dataset's feature spec so renders are
# directly comparable to the real observations (per the HANDOFF notes):
#   chest                848 x 480  RGB (16:9, as recorded)
#   wrist_left_uvc       640 x 480  RGB (Innomaker U20CAM-1080P)
#   wrist_right_uvc      640 x 480  RGB (Innomaker U20CAM-1080P)
#
# Isaac's default horizontal aperture is 20.955 mm (NOT a 36 mm full-frame
# sensor), so focal_length values here are derived from that aperture and the
# physical cameras' horizontal FOV:
#   ZED Mini         102   deg H  -> focal_length ~=  8.4845 mm  (chest)
#   U20CAM-1080P     ~119   deg H -> focal_length ~=  6.0 mm (approx; see
#                                     HANDOFF for the dfov-to-hfov derivation)
#
# Chest optics follow the OpenArm reference robot, whose body camera is a
# **ZED Mini**, not a RealSense: ``openarm_description`` declares the body
# sensor as ``product: zed`` / ``camera_model: zedm`` on a ``zed_mount``
# accessory at the chest (``assets/robot/openarm_v2.0/config/robot_presets/
# default_bimanual.yaml``). Stereolabs quote the ZED Mini at 102 deg H x
# 57 deg V x 118 deg D. The upstream ``openarm_isaac_lab`` pack has no camera
# config at all -- its ``openarm_bimanual_sensor.usd`` is still an empty
# ``Xform`` stub -- so the hardware description is the only source for this.
#
# Caveat: the ZED Mini's f/2.0 2.8 mm lens is not rectilinear, so its quoted
# 102 x 57 cannot be reproduced by a pinhole at one aspect ratio
# (tan(51)/tan(28.5) = 2.27, but the 848x480 frame is 1.77). We match the
# headline HORIZONTAL figure; the resulting vertical FOV is 69.9 deg, i.e.
# the sim sees more above/below than the real camera does. Match the other
# axis instead by setting focal_length = 11.60 mm (57 deg V, 87.7 deg H) if
# vertical fidelity ever matters more.


def _pinhole(focal_length: float):
    return sim_utils.PinholeCameraCfg(
        focal_length=focal_length,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.05, 3.0),
    )


@configclass
class OpenDoorBimanualIkAbsVisuomotorEnvCfg(OpenDoorBimanualIkAbsEnvCfg):
    """IK-Abs bimanual OpenArm + cameras + image obs (for headless dataset gen)."""

    observations: _VisuomotorObservationsCfg = _VisuomotorObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Wrist cameras (one per hand). Offset fitted by eye against the real
        # wrist footage -- known gap (see ``scripts/openarm_cameras.py`` and
        # the HANDOFF). Tune in sim; the mount geometry is the current error
        # dominant over the optics.
        self.scene.left_wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_left_hand/left_wrist_cam",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=_pinhole(focal_length=6.0),
            offset=CameraCfg.OffsetCfg(
                pos=(0.05, 0.0, -0.05),
                rot=(0.03701, 0.03701, -0.70614, -0.70614),
                convention="ros",
            ),
        )
        self.scene.right_wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_right_hand/right_wrist_cam",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=_pinhole(focal_length=6.0),
            offset=CameraCfg.OffsetCfg(
                pos=(0.05, 0.0, -0.05),
                rot=(0.03701, 0.03701, -0.70614, -0.70614),
                convention="ros",
            ),
        )
        # Chest camera on the robot body link, framing the cabinet doors and
        # their lever handles.
        #
        # ``openarm_body_link`` is identity-rotated on the stage, so the offset
        # is effectively in world axes: +X forward (toward the cabinet), +Y
        # left, +Z up. With pos=(0.08, 0, 0.55) the lens sits at world
        # (-0.27, 0, 0.03); the two handle bodies sit at world
        # (0.082, +0.065, -0.03) and (0.082, -0.088, -0.03), i.e. 0.35 m ahead
        # and 0.06 m below. A 10 deg downward pitch puts the optical axis
        # within 2 mm of handle height, so both handles land in the middle of
        # the frame.
        #
        # We deliberately do NOT copy the upstream mount angle. OpenArm's
        # own chest mount (``zed_mount`` -> ``camera_mount_point``) sits at
        # body-base (0.056, 0.009, 0.809) with a 1.08210 rad = 62 deg
        # DOWNWARD pitch, which frames a table directly in front of the
        # robot. This task's target is a vertical cabinet door at chest
        # height 0.35 m away; at 62 deg the handles fall far above the
        # frame. Optics are matched to the real camera, aim is matched to
        # the task.
        #
        # NOTE: ``OffsetCfg.rot`` is **(x, y, z, w)** in Isaac Lab 3.0, NOT
        # (w, x, y, z). The previous value here was authored as a wxyz
        # "pitch -20 deg" quaternion; read as xyzw it became a 180 deg twist
        # that aimed the camera up at the ceiling with the image rolled 90 deg.
        self.scene.chest_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_body_link/chest_cam",
            update_period=0.0,
            height=480,
            width=848,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=_pinhole(focal_length=8.4845),  # 102 deg H x 69.9 deg V
            offset=CameraCfg.OffsetCfg(
                pos=(0.08, 0.0, 0.55),
                # xyzw, ROS convention (+X right, +Y down, +Z forward):
                # look along body-link +X, pitched 10 deg down, no roll/yaw.
                rot=(-0.54168, 0.54168, -0.45452, 0.45452),
                convention="ros",
            ),
        )

        # Camera-rendering settings (match the reference visuomotor cfg).
        self.num_rerenders_on_reset = 3
        for camera_cfg in (self.scene.chest_cam, self.scene.left_wrist_cam, self.scene.right_wrist_cam):
            set_isaac_rtx_global_settings(camera_cfg.renderer_cfg, antialiasing_mode="DLAA")

        self.image_obs_list = ["chest_cam", "left_wrist_cam", "right_wrist_cam"]
