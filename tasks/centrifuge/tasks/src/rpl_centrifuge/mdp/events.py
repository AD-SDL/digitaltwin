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


def _is_kinematic(asset: RigidObject) -> bool:
    """True if the asset was spawned as a kinematic rigid body.

    PhysX rejects ``setLinear/AngularVelocity`` on kinematic bodies, so the reset
    helpers below must skip the velocity write for them (only the pose is set).
    """
    rigid_props = getattr(getattr(asset.cfg, "spawn", None), "rigid_props", None)
    return bool(rigid_props is not None and getattr(rigid_props, "kinematic_enabled", False))


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
    # Only dynamic bodies accept a velocity write (PhysX errors on kinematic bodies).
    if not _is_kinematic(asset):
        asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)


def reset_objects_together(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    asset_cfgs: list[SceneEntityCfg],
):
    """Reset several rigid objects by the SAME sampled pose delta (they move as a group).

    Samples one ``(dx, dy, dz, droll, dpitch, dyaw)`` per env from ``pose_range`` and
    applies it to every asset in ``asset_cfgs`` relative to that asset's own default
    root state. Use this to keep an object and its holder aligned — e.g. the
    centrifuge ``tube`` and its ``rack`` (+ ``rack_floor``) shift together so the tube
    stays in the rack well while the whole pickup station is randomized.

    All listed assets must be rigid objects (static ``AssetBaseCfg`` colliders cannot
    be moved at runtime — make the rack/floor kinematic ``RigidObjectCfg`` first).
    ``pose_range`` keys: x, y, z, roll, pitch, yaw (missing → (0, 0)).
    """
    if env_ids is None or len(env_ids) == 0:
        return

    device = env.device
    n = len(env_ids)

    def _u(key: str) -> torch.Tensor:
        lo, hi = pose_range.get(key, (0.0, 0.0))
        return torch.empty(n, device=device).uniform_(lo, hi)

    delta_pos = torch.stack([_u("x"), _u("y"), _u("z")], dim=1)
    delta_quat = math_utils.quat_from_euler_xyz(_u("roll"), _u("pitch"), _u("yaw"))

    for asset_cfg in asset_cfgs:
        asset: RigidObject = env.scene[asset_cfg.name]
        default_state = asset.data.default_root_state[env_ids].clone()
        positions = default_state[:, 0:3] + delta_pos + env.scene.env_origins[env_ids]
        orientations = math_utils.quat_mul(default_state[:, 3:7], delta_quat)
        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        # Only dynamic bodies accept a velocity write (PhysX errors on kinematic bodies).
        if not _is_kinematic(asset):
            asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)


def reset_scene_to_default_safe(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    reset_joint_targets: bool = False,
):
    """Kinematic-safe drop-in for ``mdp.reset_scene_to_default``.

    The upstream ``reset_scene_to_default`` writes a velocity to EVERY rigid object
    on reset, but PhysX rejects ``setLinear/AngularVelocity`` on kinematic bodies
    (our rack / rack_floor / bucket) → spurious "Body must be non-kinematic!"
    errors. This mirrors the upstream behavior (rigid bodies + articulations reset
    to their default state) but skips the velocity write for kinematic rigid
    bodies. Poses are still reset for everything.
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


def reset_tube_in_random_well(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    station_pose_range: dict[str, tuple[float, float]],
    well_offsets: list[tuple[float, float]],
    tube_drop_height: float,
    rack_cfg: SceneEntityCfg,
    tube_cfg: SceneEntityCfg,
    rack_floor_cfg: SceneEntityCfg | None = None,
):
    """Randomize the pickup station and drop the tube into a RANDOM rack well.

    Each reset:
      1. samples one in-plane station delta ``(dx, dy, dyaw)`` from
         ``station_pose_range`` and shifts the (kinematic) rack + optional floor by
         it (so the whole holder moves as a unit);
      2. picks one of the ``well_offsets`` (rack-local xy, from the rack origin) per
         env, rotates it by the station yaw, and places the tube above that well
         (``tube_drop_height`` above the rack) so it drops in.

    ``well_offsets`` are measured rack-local well centers (see the 2x2 vial rack).
    ``station_pose_range`` keys: x, y, z, roll, pitch, yaw (missing → (0, 0)).
    """
    if env_ids is None or len(env_ids) == 0:
        return

    device = env.device
    n = len(env_ids)
    origins = env.scene.env_origins[env_ids]

    def _u(key: str) -> torch.Tensor:
        lo, hi = station_pose_range.get(key, (0.0, 0.0))
        return torch.empty(n, device=device).uniform_(lo, hi)

    dx, dy, dz = _u("x"), _u("y"), _u("z")
    droll, dpitch, dyaw = _u("roll"), _u("pitch"), _u("yaw")
    delta_pos = torch.stack([dx, dy, dz], dim=1)
    delta_quat = math_utils.quat_from_euler_xyz(droll, dpitch, dyaw)

    # 1) shift the rack (+ floor) by the station delta (kinematic → pose only).
    rack: RigidObject = env.scene[rack_cfg.name]
    station_cfgs = [rack_cfg] + ([rack_floor_cfg] if rack_floor_cfg is not None else [])
    for cfg in station_cfgs:
        asset: RigidObject = env.scene[cfg.name]
        ds = asset.data.default_root_state[env_ids].clone()
        pos = ds[:, 0:3] + delta_pos + origins
        quat = math_utils.quat_mul(ds[:, 3:7], delta_quat)
        asset.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
        if not _is_kinematic(asset):
            asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)

    # 2) drop the tube into a randomly chosen well of the (shifted) rack.
    tube: RigidObject = env.scene[tube_cfg.name]
    rack_ds = rack.data.default_root_state[env_ids]
    rack_world = rack_ds[:, 0:3] + delta_pos + origins  # (n, 3)

    wells = torch.tensor(well_offsets, device=device, dtype=rack_world.dtype)  # (num_wells, 2)
    idx = torch.randint(0, wells.shape[0], (n,), device=device)
    chosen = wells[idx]  # (n, 2)
    cos_y, sin_y = torch.cos(dyaw), torch.sin(dyaw)
    ox, oy = chosen[:, 0], chosen[:, 1]
    rot_ox = cos_y * ox - sin_y * oy
    rot_oy = sin_y * ox + cos_y * oy

    tube_pos = torch.stack(
        [
            rack_world[:, 0] + rot_ox,
            rack_world[:, 1] + rot_oy,
            rack_world[:, 2] + float(tube_drop_height),
        ],
        dim=1,
    )
    tube_ds = tube.data.default_root_state[env_ids]
    tube_quat = math_utils.quat_mul(tube_ds[:, 3:7], delta_quat)
    tube.write_root_pose_to_sim(torch.cat([tube_pos, tube_quat], dim=-1), env_ids=env_ids)
    tube.write_root_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)
