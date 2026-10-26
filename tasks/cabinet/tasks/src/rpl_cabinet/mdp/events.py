# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events for the cabinet door-opening task.

Two kinds of randomization at reset:

* The cabinet's base pose is shifted by a small in-plane delta (so the policy
  doesn't just memorize one absolute cabinet position), and
* The cabinet articulation state is reset: all four joints zeroed (doors
  closed, levers neutral) and joint-position targets re-set to zero so the
  default-stiff position drive doesn't immediately try to track a stale target.

The second half is why we need a bespoke event rather than just dropping
``mdp.reset_scene_to_default`` in: the upstream helper writes a velocity to
every rigid object on reset, which PhysX rejects for kinematic bodies -- same
reason the centrifuge pack ships ``reset_scene_to_default_safe``. We reuse
that pattern but add the joint-target reset for the cabinet articulation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _is_kinematic(asset: RigidObject) -> bool:
    """True if the asset was spawned as a kinematic rigid body."""
    rigid_props = getattr(getattr(asset.cfg, "spawn", None), "rigid_props", None)
    return bool(rigid_props is not None and getattr(rigid_props, "kinematic_enabled", False))


def reset_scene_to_default_safe(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    reset_joint_targets: bool = True,
):
    """Kinematic-safe drop-in for ``mdp.reset_scene_to_default``.

    Mirrors the centrifuge pack's helper but defaults ``reset_joint_targets``
    to ``True``: the cabinet's revolute joints carry a stiff position drive
    (from the authored USD), so if the targets are left at their last-step
    values the cabinet re-opens the moment the sim un-pauses the reset. Zeroing
    the targets alongside the positions keeps the doors shut.
    """
    for rigid_object in env.scene.rigid_objects.values():
        default_root_pose = rigid_object.data.default_root_pose.torch[env_ids].clone()
        default_root_pose[:, :3] += env.scene.env_origins[env_ids]
        rigid_object.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=env_ids)
        if not _is_kinematic(rigid_object):
            default_root_vel = rigid_object.data.default_root_vel.torch[env_ids].clone()
            rigid_object.write_root_velocity_to_sim_index(root_velocity=default_root_vel, env_ids=env_ids)

    for articulation_asset in env.scene.articulations.values():
        default_root_pose = articulation_asset.data.default_root_pose.torch[env_ids].clone()
        default_root_vel = articulation_asset.data.default_root_vel.torch[env_ids].clone()
        default_root_pose[:, :3] += env.scene.env_origins[env_ids]
        articulation_asset.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=env_ids)
        articulation_asset.write_root_velocity_to_sim_index(root_velocity=default_root_vel, env_ids=env_ids)
        default_joint_pos = articulation_asset.data.default_joint_pos.torch[env_ids].clone()
        default_joint_vel = articulation_asset.data.default_joint_vel.torch[env_ids].clone()
        articulation_asset.write_joint_position_to_sim_index(position=default_joint_pos, env_ids=env_ids)
        articulation_asset.write_joint_velocity_to_sim_index(velocity=default_joint_vel, env_ids=env_ids)
        if reset_joint_targets:
            articulation_asset.set_joint_position_target_index(target=default_joint_pos, env_ids=env_ids)
            articulation_asset.set_joint_velocity_target_index(target=default_joint_vel, env_ids=env_ids)


def reset_cabinet_pose(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    cabinet_cfg: SceneEntityCfg = SceneEntityCfg("cabinet"),
):
    """Shift the cabinet root pose by a uniformly-sampled delta on reset.

    The cabinet is an articulation with a fixed-base root (authored into the
    USD as a world-fixed joint, see ``body`` at mass 60 kg) -- shifting the
    *root* writes the articulation's base pose and the doors/handles follow
    rigidly. ``pose_range`` keys: ``x``, ``y``, ``z``, ``roll``, ``pitch``,
    ``yaw``. Missing keys default to ``(0.0, 0.0)``. Keep the ranges tight so
    the handle stays inside the OpenArm's reach envelope.
    """
    if env_ids is None or len(env_ids) == 0:
        return

    asset: Articulation = env.scene[cabinet_cfg.name]
    device = env.device
    n = len(env_ids)

    def _u(key: str) -> torch.Tensor:
        lo, hi = pose_range.get(key, (0.0, 0.0))
        return torch.empty(n, device=device).uniform_(lo, hi)

    delta_pos = torch.stack([_u("x"), _u("y"), _u("z")], dim=1)
    delta_quat = math_utils.quat_from_euler_xyz(_u("roll"), _u("pitch"), _u("yaw"))

    default_root_state = asset.data.default_root_state[env_ids].clone()
    positions = default_root_state[:, 0:3] + delta_pos + env.scene.env_origins[env_ids]
    orientations = math_utils.quat_mul(default_root_state[:, 3:7], delta_quat)
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)
