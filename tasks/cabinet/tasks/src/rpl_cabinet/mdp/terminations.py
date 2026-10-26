# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the cabinet door-opening task.

The success condition mirrors the structure of the centrifuge pack's
``tube_inside_bucket`` -- the geometric gate must hold **continuously** for
``settle_time`` seconds so a transient door swing past threshold doesn't count,
and the dwell counter is advanced idempotently (``annotate_demos`` /
``generate_dataset`` both call the success term once per step through the
observation group; this guards against double-counting). See
:mod:`rpl_cabinet.mdp.subtasks` for the module docstring explaining why the
counter is advanced from an observation term, not from the termination path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_door_settle_counter(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None) -> None:
    """Zero the :func:`door_opened` dwell counter for the given envs.

    Wired as an ``EventTerm(mode="reset")``. The counter also self-zeroes on any
    step where the door is not past the open threshold, so this is
    belt-and-braces -- but relying solely on that self-healing is the kind of
    implicit invariant that quietly stops holding when another manager touches
    the step order, so make it explicit.
    """
    counter = getattr(env, "_door_settle_steps", None)
    if counter is None:
        return
    if env_ids is None:
        counter.zero_()
    else:
        counter[env_ids] = 0


def _advance_settle_counter(env: ManagerBasedRLEnv, opened: torch.Tensor) -> torch.Tensor:
    """Advance (at most once per step) the "opened for N consecutive steps" counter.

    See :func:`rpl_centrifuge.mdp.terminations._advance_settle_counter` -- same
    idempotence contract, same reasoning. ``common_step_counter`` increments
    once at the top of :meth:`ManagerBasedRLEnv.step`, before any manager runs,
    so every caller within one step observes the same value.
    """
    counter = getattr(env, "_door_settle_steps", None)
    if counter is None or counter.shape[0] != env.num_envs:
        counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._door_settle_steps = counter
        env._door_settle_step_id = None

    if getattr(env, "_door_settle_step_id", None) == env.common_step_counter:
        return counter  # already advanced by the other manager this step

    counter = torch.where(opened, counter + 1, torch.zeros_like(counter))
    env._door_settle_steps = counter
    env._door_settle_step_id = env.common_step_counter
    return counter


def door_opened(
    env: ManagerBasedRLEnv,
    cabinet_cfg: SceneEntityCfg = SceneEntityCfg("cabinet"),
    door_joint_names: tuple[str, ...] = ("hinge_left", "hinge_right"),
    open_angle_rad: float = 1.40,  # ~80 deg; door approaching 'almost fully open'
    settle_time: float = 1.0,
) -> torch.Tensor:
    """Success once ANY named door has been held open past ``open_angle_rad``.

    The angle is read as ``abs(joint_pos)`` -- Isaac Lab reports joint
    positions in radians, so the authored ±110 deg door range becomes roughly
    ±1.92 rad in sim. The default ``open_angle_rad=1.40`` is ~80 deg: shy of
    the mechanical stop so the operator does not have to muscle the door into
    the limit, but far enough past perpendicular (90 deg = 1.57 rad) that
    'almost fully open' is the right description. A left door (authored range
    ``[-110, 0]`` deg) and a right door (``[0, +110]`` deg) are treated
    symmetrically -- magnitude past threshold counts as "open" regardless of
    sign. The dwell is enforced for ``settle_time`` seconds so a transient
    swing-through on the way to re-closing does not count.

    Multiple joints by default so the operator can pull EITHER door in a given
    demo and the recorder still exports -- the real ``openarm_open_lab_door``
    dataset is single-arm left-arm/left-door, but sim demos vary and locking
    the task to one side would silently drop right-handed demos. Pass a
    1-tuple like ``("hinge_left",)`` to lock the task back to one door.

    Implementation note: this is a plain function (not a stateful
    ``ManagerTermBase``) because ``scripts/tools/record_demos.py`` evaluates
    the success term by calling ``success_term.func(env, **params)`` directly
    -- a class-based term would be (wrongly) *constructed* there. The per-env
    dwell counter is therefore stashed on the ``env`` object by
    :func:`_advance_settle_counter`, and cleared on reset by
    :func:`reset_door_settle_counter`. Expose this function as an observation
    term in the ``subtask_terms`` group too (see
    :class:`~rpl_cabinet.openarm.env_cfg.ObservationsCfg`) so the counter
    stays live in ``annotate_demos.py`` / ``generate_dataset.py``, which set
    ``env_cfg.terminations = None`` and would otherwise only tick the counter
    through the manual once-per-episode success-term call.
    """
    cab: Articulation = env.scene[cabinet_cfg.name]
    idx, _ = cab.find_joints(list(door_joint_names))
    # (num_envs, len(joint_names)) -> max |angle| across the listed doors per env.
    angles = cab.data.joint_pos[:, idx]
    max_abs_angle = torch.abs(angles).amax(dim=1)
    opened = max_abs_angle >= open_angle_rad

    counter = _advance_settle_counter(env, opened)
    required_steps = max(1, int(round(settle_time / env.step_dt)))
    return counter >= required_steps
