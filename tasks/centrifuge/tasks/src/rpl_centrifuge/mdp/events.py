# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_object_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Reset a rigid object to a uniformly-sampled pose, expressed in the env origin frame.

    ``pose_range`` keys: x, y, z, roll, pitch, yaw. Missing keys default to (0.0, 0.0).
    The translation is added to the env origin; the orientation is composed from Euler XYZ.
    """
    if env_ids is None or len(env_ids) == 0:
        return

    asset: RigidObject = env.scene[asset_cfg.name]
    device = env.device
    n = len(env_ids)

    def _u(key: str) -> torch.Tensor:
        lo, hi = pose_range.get(key, (0.0, 0.0))
        return torch.empty(n, device=device).uniform_(lo, hi)

    dx, dy, dz = _u("x"), _u("y"), _u("z")
    droll, dpitch, dyaw = _u("roll"), _u("pitch"), _u("yaw")

    default_state = asset.data.default_root_state[env_ids].clone()
    positions = default_state[:, 0:3] + torch.stack([dx, dy, dz], dim=1) + env.scene.env_origins[env_ids]
    base_quat = default_state[:, 3:7]
    delta_quat = math_utils.quat_from_euler_xyz(droll, dpitch, dyaw)
    orientations = math_utils.quat_mul(base_quat, delta_quat)

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)
