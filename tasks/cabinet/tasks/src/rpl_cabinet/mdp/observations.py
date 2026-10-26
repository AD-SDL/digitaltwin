# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the cabinet door-opening task.

Covers (a) the generic rigid-object / EE pose helpers (identical to the
centrifuge pack's -- kept local so the MDP module is self-contained), and (b)
cabinet-specific joint-angle readouts used by the policy and the success term.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
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


# ---------------------------------------------------------------------------
# Cabinet joint readouts.
#
# The cabinet articulation (see ``src/rpl_cabinet/assets/rpl_cabinet.usda``)
# carries four DOFs. Joint names set at authoring time:
#
#   hinge_left   revolute Z about the left stile,   -110..0  deg,  9 kg
#   hinge_right  revolute Z about the right stile,    0..110 deg,  9 kg
#   latch_left   revolute Y at the pivot,             0..90  deg,  0.15 kg
#   latch_right  revolute Y at the pivot,            -90..0  deg,  0.15 kg
#
# Isaac Lab's ``find_joints`` matches full joint names as regex patterns by
# default, so passing an exact list of names (e.g. ``["hinge_left"]``)
# avoids any accidental catches like ``hinge_right`` matching ``hinge_le``. The ``find``
# method on the articulation data returns a 1-D int tensor of DOF indices.
# ---------------------------------------------------------------------------


def _cabinet_joint_angles(env: "ManagerBasedRLEnv", cabinet_cfg: SceneEntityCfg, joint_names: list[str]) -> torch.Tensor:
    """Return the (num_envs, len(joint_names)) joint positions of a cabinet articulation."""
    cab: Articulation = env.scene[cabinet_cfg.name]
    idx, _ = cab.find_joints(joint_names)
    return cab.data.joint_pos[:, idx]


def cabinet_door_angles(
    env: ManagerBasedRLEnv,
    cabinet_cfg: SceneEntityCfg = SceneEntityCfg("cabinet"),
) -> torch.Tensor:
    """Door joint positions in radians: columns are (door_left, door_right)."""
    return _cabinet_joint_angles(env, cabinet_cfg, ["hinge_left", "hinge_right"])


def cabinet_handle_angles(
    env: ManagerBasedRLEnv,
    cabinet_cfg: SceneEntityCfg = SceneEntityCfg("cabinet"),
) -> torch.Tensor:
    """Handle joint positions in radians: columns are (handle_left, handle_right)."""
    return _cabinet_joint_angles(env, cabinet_cfg, ["latch_left", "latch_right"])
