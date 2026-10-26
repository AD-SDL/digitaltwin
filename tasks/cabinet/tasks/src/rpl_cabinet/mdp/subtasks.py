# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Subtask termination-signal observations for the cabinet door-opening Mimic env.

Each function returns a boolean tensor of shape ``(num_envs,)`` that is 1 when
its subtask is complete and 0 otherwise. They are consumed by
:class:`isaaclab_mimic`-style dataset annotation
(``annotate_demos.py --auto``): the annotator scans the recorded signal for a
rising edge and marks that step as the subtask boundary.

Design notes
------------
* Signals are **state-only** -- no dependence on env history -- so replaying an
  episode reproduces them exactly.
* The subtask set is deliberately small (one grasp signal + the terminal
  subtask whose end coincides with ``door_opened``). The real demonstrations
  show no distinct lever-turn phase -- the operator pre-closes the gripper at
  home and hooks the lever on approach (see the package README's "What the
  real data says about the task"), so we don't add a ``handle_turned``
  subtask that the demos don't contain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _eef_near_body(
    env: "ManagerBasedRLEnv",
    ee_frame_cfg: SceneEntityCfg,
    cabinet_cfg: SceneEntityCfg,
    body_name: str,
    max_distance: float,
) -> torch.Tensor:
    """``True`` per env where the EE frame sits within ``max_distance`` metres
    of the named cabinet body (world frame)."""
    cab: Articulation = env.scene[cabinet_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    body_idx, _ = cab.find_bodies([body_name])
    body_pos_w = cab.data.body_pos_w[:, body_idx[0], :]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    return torch.linalg.vector_norm(ee_pos_w - body_pos_w, dim=-1) < max_distance


def left_grasp_handle(
    env: "ManagerBasedRLEnv",
    left_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("left_ee_frame"),
    cabinet_cfg: SceneEntityCfg = SceneEntityCfg("cabinet"),
    proximity_gate: float = 0.12,
    engage_angle_rad: float = 0.17,  # ~10 deg
) -> torch.Tensor:
    """Left arm has engaged the left-door handle: EE close to the handle hub AND
    the door has started to swing open.

    Two independent gates, ANDed together:

    * **Left EE nearby**: left EE frame within ``proximity_gate`` metres of
      the ``handle_left`` body (the chrome hub, authored at the handle pivot).
      0.12 m is loose enough to tolerate finger-span and offset uncertainty
      without catching passing hands.
    * **Door started to swing**: ``abs(hinge_left) > engage_angle_rad``.
      ~10 deg confirms the hook is pulling the door, not just resting at the
      lever face. Avoids a signal that fires the instant the EE enters the
      proximity sphere.

    Thresholds are STARTING POINTS -- tune once teleop traces exist. The real
    dataset doesn't have a canonical "grasp" moment either, so this is the
    subtask boundary rather than a measured one.
    """
    near = _eef_near_body(env, left_ee_frame_cfg, cabinet_cfg, "handle_left", proximity_gate)
    cab: Articulation = env.scene[cabinet_cfg.name]
    door_idx, _ = cab.find_joints(["hinge_left"])
    swung = torch.abs(cab.data.joint_pos[:, door_idx[0]]) > engage_angle_rad
    return near & swung
