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
``chest_cam`` on the robot body link that frames the cabinet door. Mount
offsets are STARTING POINTS fitted by eye against the real wrist/chest footage
(see the ``cabinet_sim/openarm_cameras.py`` prototype carried into
``scripts/openarm_cameras.py``). The HANDOFF.md "known gaps" explicitly flags
these as the biggest open item: ``CHEST_POS_LOCALISH`` / ``CHEST_PITCH_DOWN_DEG``
need measuring off the physical robot. Tune in sim before publishing a
synthetic dataset.
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
#   chest_d435i_color    848 x 480  RGB (D435i colour imager)
#   wrist_left_uvc       640 x 480  RGB (Innomaker U20CAM-1080P)
#   wrist_right_uvc      640 x 480  RGB (Innomaker U20CAM-1080P)
#
# Isaac's default horizontal aperture is 20.955 mm (NOT a 36 mm full-frame
# sensor), so focal_length values here are derived from that aperture and the
# physical cameras' horizontal FOV:
#   D435 colour       69.4 deg H  -> focal_length ~= 15.0 mm
#   U20CAM-1080P     ~119   deg H -> focal_length ~=  6.0 mm (approx; see
#                                     HANDOFF for the dfov-to-hfov derivation)


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
        # Chest camera on the robot body link. The real D435i points slightly
        # down toward the workspace -- the offset here is approximate (see the
        # package HANDOFF for how to match the real framing).
        self.scene.chest_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_body_link/chest_cam",
            update_period=0.0,
            height=480,
            width=848,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=_pinhole(focal_length=15.0),
            offset=CameraCfg.OffsetCfg(
                # +X on the body link points forward (OpenArm faces +X); drop
                # a few cm down from the top of the torso with a slight pitch
                # down toward the handle (~20 deg).
                pos=(0.08, 0.0, 0.55),
                rot=(0.17365, 0.0, 0.98481, 0.0),  # pitch -20 deg (wxyz)
                convention="ros",
            ),
        )

        # Camera-rendering settings (match the reference visuomotor cfg).
        self.num_rerenders_on_reset = 3
        for camera_cfg in (self.scene.chest_cam, self.scene.left_wrist_cam, self.scene.right_wrist_cam):
            set_isaac_rtx_global_settings(camera_cfg.renderer_cfg, antialiasing_mode="DLAA")

        self.image_obs_list = ["chest_cam", "left_wrist_cam", "right_wrist_cam"]
