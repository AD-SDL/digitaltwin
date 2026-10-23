# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visuomotor cfg for the SO101 IK-Abs task — HEADLESS dataset rendering.

The IK-Abs / Mimic env is low-dim (joint-pos + eef pose + prop pose). This
variant adds table and wrist RGB cameras in the **policy** observation group
(not a separate ``rgb_camera`` group) so IsaacLab's generation path picks them
up via its default ``PreStepFlatPolicyObservationsRecorder`` — see the
OpenArm visuomotor cfg for the rationale.

Not for live teleop. Use ``lerobot_leader_env_cfg`` for the live-camera leader
teleop; use this to re-render a recorded demo HDF5 into an image dataset.
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
from .ik_abs_env_cfg import CentrifugeSO101IkAbsEnvCfg
from .so101_cfg import END_EFFECTOR_LINK_NAME

_CAMERA_HEIGHT = 240
_CAMERA_WIDTH = 320


def _pinhole():
    return sim_utils.PinholeCameraCfg(
        focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2.0)
    )


@configclass
class _VisuomotorPolicyCfg(ObservationsCfg.PolicyCfg):
    """Low-dim policy observations + the RGB image terms (in-policy, not visual).

    The images go in the **policy** group, not a separate ``rgb_camera`` group:
    IsaacLab's generation path uses the default ``ActionStateRecorderManagerCfg``
    whose only observation recorder is ``PreStepFlatPolicyObservationsRecorder``,
    which records ``obs_buf["policy"]`` and nothing else. Putting cameras here
    guarantees they land in the generated dataset. Mirrors upstream
    ``FrankaCubeStackVisuomotorEnvCfg``.
    """

    table_cam = ObsTerm(
        func=mdp.image,
        params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False},
    )
    wrist_cam = ObsTerm(
        func=mdp.image,
        params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
    )


@configclass
class _VisuomotorObservationsCfg(ObservationsCfg):
    policy: _VisuomotorPolicyCfg = _VisuomotorPolicyCfg()


@configclass
class CentrifugeSO101IkAbsVisuomotorEnvCfg(CentrifugeSO101IkAbsEnvCfg):
    """IK-Abs cfg + cameras + camera obs terms, for headless dataset rendering."""

    observations: _VisuomotorObservationsCfg = _VisuomotorObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=0.0,
            height=_CAMERA_HEIGHT,
            width=_CAMERA_WIDTH,
            data_types=["rgb"],
            spawn=_pinhole(),
            offset=CameraCfg.OffsetCfg(
                pos=(0.66, 0.0, 0.35), rot=(-0.61237, -0.61237, 0.35355, 0.35355), convention="ros"
            ),
        )
        self.scene.wrist_cam = CameraCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{END_EFFECTOR_LINK_NAME}/wrist_cam",
            update_period=0.0,
            height=_CAMERA_HEIGHT,
            width=_CAMERA_WIDTH,
            data_types=["rgb"],
            spawn=_pinhole(),
            offset=CameraCfg.OffsetCfg(
                pos=(0.02, 0.0, -0.04), rot=(0.03701, 0.03701, -0.70614, -0.70614), convention="ros"
            ),
        )

        self.num_rerenders_on_reset = 3
        for camera_cfg in (self.scene.table_cam, self.scene.wrist_cam):
            set_isaac_rtx_global_settings(camera_cfg.renderer_cfg, antialiasing_mode="DLAA")
