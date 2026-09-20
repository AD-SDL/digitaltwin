# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Subtask termination-signal observations for the bimanual centrifuge Mimic env.

Each function returns a boolean tensor of shape ``(num_envs,)`` that is 1 when
its subtask is complete (or being held) and 0 otherwise. They are consumed by
:class:`isaaclab_mimic`-style dataset annotation (``scripts/imitation_learning/
isaaclab_mimic/annotate_demos.py --auto``): the annotator scans the recorded
signal for a rising edge and marks that step as the subtask boundary.

Design notes
------------
* Signals are **state-only** — no dependence on env history — so replaying an
  episode reproduces them exactly.
* Heights are measured **relative to the rack root**, never as absolute world
  ``z``. An earlier revision hard-coded ``tube z > 0.86`` against a scene whose
  work surface sat at ~0.785 m; when the scene was re-based to
  ``_SURFACE_Z = 0.0`` the gate silently became unreachable (the tube now lives
  between 0.029 m and 0.190 m) and every demo failed annotation with "did not
  detect completion". A rack-relative offset cannot go stale that way, and it
  is env-origin-invariant for free, which matters once data generation runs
  multiple envs.
* Calibrated against the 20 shipped demos (``centrifuge_openarm.hdf5``), all of
  which are successful right-arm pick-and-place:

  =============================================  ==================
  quantity                                       measured
  =============================================  ==================
  rack root world z (never randomized)           0.011 m
  tube seated in a rack well                     0.0352 m (exact, all 20)
  tube during sustained carry (95th pct)         0.157 – 0.190 m
  tube peak *before* the real lift, EE within
  0.15 m (failed-grasp fumbling)                 0.0719 – 0.0796 m
  right EE ↔ tube distance at the true lift      0.090 – 0.111 m
  right EE ↔ tube distance during the reset
  drop transient (the only frames above any
  plausible lift threshold)                      0.233 – 0.329 m
  =============================================  ==================

  So the usable lift window is ``[0.0796, 0.1569]`` world → the gate sits at its
  midpoint, ``0.104`` above the rack root (= 0.115 world). Sweeping the
  threshold from 0.08 to 0.13 moves the detected edge by only 4–14 steps
  (0.2–0.7 s), which is the real evidence the separation is clean.
* The **proximity gate does the drop-transient rejection**, not the height gate.
  ``reset_tube_in_random_well`` drops the tube from ``tube_drop_height=0.15``
  above the rack, so frames 0–1 of *every* episode sit at z ≈ 0.149 / 0.112 —
  higher than the tube ever gets while carried in six of the demos. No height
  threshold can exclude that. 0.15 m of proximity does, with 36 % margin on both
  sides.
* Do NOT gate on the finger joint position, in **either** direction. The obvious
  test — "fingers stopped short of ``closed_command_expr`` (0.0), therefore they
  are closed on the tube" — fails both ways: a failed close on empty air reaches
  exactly 0.0, but during genuine carries the aperture also dips to 0.0001
  (demo_16) and 0.0032 (demo_12) as the tube shifts in the grasp. Measured right
  finger aperture is 0.044 open / ~0.010 closed-on-tube / 0.000 closed-on-air,
  with the carry distribution overlapping the empty one. The tube-lift check
  side-steps this entirely.

Add new signal functions here and re-export them via
:file:`mdp/__init__.py` — the Mimic env cfg wires them into the ``subtask_terms``
observation group.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _eef_near_object(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    max_distance: float,
) -> torch.Tensor:
    """Return ``True`` per env where the first target frame of ``ee_frame_cfg``
    is within ``max_distance`` metres of the object's root position (world frame)."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    diff = ee_pos_w - obj.data.root_pos_w
    return torch.linalg.vector_norm(diff, dim=-1) < max_distance


def _tube_lifted_off_rack(
    env: ManagerBasedRLEnv,
    tube_cfg: SceneEntityCfg,
    rack_cfg: SceneEntityCfg,
    lift_height: float,
) -> torch.Tensor:
    """Return ``True`` per env where the tube sits more than ``lift_height``
    metres above the rack root.

    Both positions are in the world frame, so the subtraction cancels the env
    origin — the same constant is correct for every env in a multi-env run.
    """
    tube: RigidObject = env.scene[tube_cfg.name]
    rack: RigidObject = env.scene[rack_cfg.name]
    return (tube.data.root_pos_w[:, 2] - rack.data.root_pos_w[:, 2]) > lift_height


def right_grasp_tube(
    env: ManagerBasedRLEnv,
    tube_cfg: SceneEntityCfg = SceneEntityCfg("tube"),
    right_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("right_ee_frame"),
    rack_cfg: SceneEntityCfg = SceneEntityCfg("rack"),
    tube_lift_above_rack: float = 0.104,
    proximity_gate: float = 0.15,
) -> torch.Tensor:
    """Right arm has lifted the tube clear of the rack.

    Two independent gates, ANDed together:

    * **Tube lifted**: tube root more than ``tube_lift_above_rack`` above the
      rack root. Default 0.104 m (= 0.115 m world) is the midpoint of the
      measured window ``[0.0796, 0.1569]`` — above every pre-lift fumble in the
      shipped demos, below every sustained carry. See the module docstring for
      the calibration table and for why this is rack-relative rather than an
      absolute world height.
    * **Right EE nearby**: right EE frame within ``proximity_gate`` metres of
      the tube. Default 0.15 m sits between the true-grasp distances
      (0.090–0.111 m) and the reset drop transient (0.233–0.329 m), so it both
      attributes the lift to the right arm and suppresses the frame-0/1 drop
      that would otherwise be detected as the subtask boundary.

    Returns:
        Bool tensor of shape ``(num_envs,)``.
    """
    tube_lifted = _tube_lifted_off_rack(env, tube_cfg, rack_cfg, tube_lift_above_rack)
    right_near = _eef_near_object(env, tube_cfg, right_ee_frame_cfg, proximity_gate)
    return tube_lifted & right_near


def left_grasp_tube(
    env: ManagerBasedRLEnv,
    tube_cfg: SceneEntityCfg = SceneEntityCfg("tube"),
    left_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("left_ee_frame"),
    rack_cfg: SceneEntityCfg = SceneEntityCfg("rack"),
    tube_lift_above_rack: float = 0.104,
    proximity_gate: float = 0.15,
) -> torch.Tensor:
    """Mirror of :func:`right_grasp_tube` for the left arm.

    Not wired into any observation group today (the left arm is passive in the
    shipped demos); kept in sync so a future left-handed or bimanual variant
    starts from the same calibration rather than the stale one.
    """
    tube_lifted = _tube_lifted_off_rack(env, tube_cfg, rack_cfg, tube_lift_above_rack)
    left_near = _eef_near_object(env, tube_cfg, left_ee_frame_cfg, proximity_gate)
    return tube_lifted & left_near
