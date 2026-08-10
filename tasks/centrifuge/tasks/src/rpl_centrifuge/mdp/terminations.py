# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def tube_inside_bucket(
    env: ManagerBasedRLEnv,
    tube_cfg: SceneEntityCfg = SceneEntityCfg("tube"),
    bucket_cfg: SceneEntityCfg = SceneEntityCfg("bucket"),
    xy_radius: float = 0.04,
    z_below: float = 0.05,
    z_above: float = 0.20,
    max_lin_speed: float = 0.05,
) -> torch.Tensor:
    """Tube is inside the bucket and roughly at rest.

    True per-env iff:
      - tube xy is within ``xy_radius`` of bucket xy, and
      - tube z is between ``bucket_z - z_below`` and ``bucket_z + z_above``, and
      - tube linear speed is below ``max_lin_speed``.
    """
    tube: RigidObject = env.scene[tube_cfg.name]
    bucket: RigidObject = env.scene[bucket_cfg.name]

    diff = tube.data.root_pos_w - bucket.data.root_pos_w
    xy_dist = torch.linalg.vector_norm(diff[:, :2], dim=1)
    z_offset = diff[:, 2]
    z_in = (z_offset > -z_below) & (z_offset < z_above)
    xy_in = xy_dist < xy_radius

    speed = torch.linalg.vector_norm(tube.data.root_lin_vel_w, dim=1)
    at_rest = speed < max_lin_speed

    return xy_in & z_in & at_rest
