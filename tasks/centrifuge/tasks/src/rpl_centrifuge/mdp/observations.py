# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_env_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Position of a rigid object in the per-env origin frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


def object_orientation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Quaternion (w, x, y, z) of a rigid object in world frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_quat_w


def ee_frame_position_in_env_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Position of the first target frame of a FrameTransformer in the env frame."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins


def ee_frame_orientation(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Quaternion of the first target frame of a FrameTransformer in world frame."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_quat_w[:, 0, :]
