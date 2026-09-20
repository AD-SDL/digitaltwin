# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visuomotor (camera) variant of the bimanual OpenArm centrifuge IK-Abs task.

Mirrors the upstream ``stack_ik_rel_visuomotor_env_cfg.py`` pattern: it takes the
camera-free IK-Abs task and adds cameras + RGB image observations, for **headless
dataset generation** (``generate_dataset`` on the Mimic variant). This is NOT for
live XR teleop — rendering a camera observation under the Kit XR pipeline crashes
Kit; the canonical flow records demos with the camera-free teleop task and renders
images here, headless.

Cameras (mirroring the reference wrist_cam/table_cam):
  * ``left_wrist_cam`` / ``right_wrist_cam`` — on each OpenArm hand.
  * ``table_cam`` — fixed 3rd-person view of the work area.
Camera poses are STARTING POINTS — tune to frame the rack/tube/bucket.
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
from .ik_abs_env_cfg import CentrifugeBimanualIkAbsEnvCfg


@configclass
class _VisuomotorPolicyCfg(ObservationsCfg.PolicyCfg):
    """Low-dim policy observations + the RGB image terms.

    The images go in the **policy** group, not a separate ``rgb_camera`` group.
    That is not cosmetic: ``generate_dataset.py`` discards the env's own recorder
    config (``setup_env_config`` replaces ``env_cfg.recorders`` with
    ``mimic_recorder_config``, which is ``None`` here, falling back to
    ``ActionStateRecorderManagerCfg``), and that manager's only observation
    recorder is ``PreStepFlatPolicyObservationsRecorder`` — which records
    ``obs_buf["policy"]`` and nothing else. With the cameras in their own group
    a generation run pays the full RTX cost per step and still writes a low-dim
    dataset. Upstream's ``FrankaCubeStackVisuomotorEnvCfg`` puts ``table_cam`` /
    ``wrist_cam`` directly in ``PolicyCfg`` for exactly this reason.
    """

    table_cam = ObsTerm(
        func=mdp.image,
        params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False},
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


def _pinhole():
    return sim_utils.PinholeCameraCfg(
        focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2.0)
    )


@configclass
class CentrifugeBimanualIkAbsVisuomotorEnvCfg(CentrifugeBimanualIkAbsEnvCfg):
    """IK-Abs bimanual OpenArm + cameras + image obs (for headless dataset gen)."""

    observations: _VisuomotorObservationsCfg = _VisuomotorObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Wrist cameras (one per hand). Offset mirrors the reference wrist_cam.
        self.scene.left_wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_left_hand/left_wrist_cam",
            update_period=0.0,
            height=200,
            width=200,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=_pinhole(),
            offset=CameraCfg.OffsetCfg(
                pos=(0.05, 0.0, -0.05), rot=(0.03701, 0.03701, -0.70614, -0.70614), convention="ros"
            ),
        )
        self.scene.right_wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_right_hand/right_wrist_cam",
            update_period=0.0,
            height=200,
            width=200,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=_pinhole(),
            offset=CameraCfg.OffsetCfg(
                pos=(0.05, 0.0, -0.05), rot=(0.03701, 0.03701, -0.70614, -0.70614), convention="ros"
            ),
        )
        # Fixed 3rd-person table camera framing the work area (props at x≈0.3).
        # STARTING POINT — tune to frame the rack + bucket.
        self.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=0.0,
            height=200,
            width=200,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=_pinhole(),
            offset=CameraCfg.OffsetCfg(
                pos=(0.9, 0.0, 0.45), rot=(-0.61237, -0.61237, 0.35355, 0.35355), convention="ros"
            ),
        )

        # Camera-rendering settings (match the reference visuomotor cfg).
        self.num_rerenders_on_reset = 3
        for camera_cfg in (self.scene.table_cam, self.scene.left_wrist_cam, self.scene.right_wrist_cam):
            set_isaac_rtx_global_settings(camera_cfg.renderer_cfg, antialiasing_mode="DLAA")

        # List of image observations (consumed by downstream IL / GR00T tooling).
        self.image_obs_list = ["table_cam", "left_wrist_cam", "right_wrist_cam"]
