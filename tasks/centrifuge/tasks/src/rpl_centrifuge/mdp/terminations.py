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


def reset_tube_settle_counter(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None) -> None:
    """Zero the :func:`tube_inside_bucket` dwell counter for the given envs.

    Wired as an ``EventTerm(mode="reset")``. The counter also self-zeroes on any
    step where the tube is not seated, so this is belt-and-braces — but the
    original implementation relied *solely* on that self-healing, which is the
    kind of implicit invariant that stops holding quietly.
    """
    counter = getattr(env, "_tube_settle_steps", None)
    if counter is None:
        return
    if env_ids is None:
        counter.zero_()
    else:
        counter[env_ids] = 0


def _advance_settle_counter(env: ManagerBasedRLEnv, seated: torch.Tensor) -> torch.Tensor:
    """Advance the per-env "seated for N consecutive steps" counter, at most once
    per simulation step, and return it.

    Idempotence matters in two places:

    * :file:`annotate_demos.py` and :file:`generate_dataset.py` both set
      ``env_cfg.terminations = None`` and then call ``success_term.func(env, ...)``
      by hand. The observation term keeps the counter live; the manual call must
      *read* it rather than advance it a second time.
    * ``MultiWaypoint.execute`` (isaaclab_mimic) calls the success term once per
      env per step. Without this guard an ``N``-env generation run would advance
      the shared counter ``N`` times per step and reach ``settle_time`` in
      ``1/N`` of the intended dwell.

    ``common_step_counter`` is incremented once at the top of
    :meth:`ManagerBasedRLEnv.step`, before either manager runs, so every caller
    within one step observes the same value.
    """
    counter = getattr(env, "_tube_settle_steps", None)
    if counter is None or counter.shape[0] != env.num_envs:
        counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._tube_settle_steps = counter
        env._tube_settle_step_id = None

    if getattr(env, "_tube_settle_step_id", None) == env.common_step_counter:
        return counter  # already advanced by the other manager this step

    counter = torch.where(seated, counter + 1, torch.zeros_like(counter))
    env._tube_settle_steps = counter
    env._tube_settle_step_id = env.common_step_counter
    return counter


def tube_inside_bucket(
    env: ManagerBasedRLEnv,
    tube_cfg: SceneEntityCfg = SceneEntityCfg("tube"),
    bucket_cfg: SceneEntityCfg = SceneEntityCfg("bucket"),
    xy_radius: float = 0.05,
    z_min: float = -0.02,
    z_max: float = 0.09,
    max_lin_speed: float = 0.3,
    max_ang_speed: float = 10.0,
    settle_time: float = 2.0,
) -> torch.Tensor:
    """Success once the tube has dropped into the bucket well and settled there.

    A purely instantaneous check fires the moment the tube enters the acceptance
    region, which -- with a generous ``z`` window -- happens while the tube is
    still being lowered toward (or held above) the hole. Instead we require the
    "seated" condition to hold *continuously* for ``settle_time`` seconds, so the
    tube must drop into the well and come to rest rather than merely pass through:

      - tube xy within ``xy_radius`` of the bucket axis, and
      - tube z-offset above the bucket root within ``[z_min, z_max]`` -- ``z_max``
        sits below the bucket rim so a tube balanced on the rim or held at the
        entrance does not qualify, and
      - tube linear *and* angular speed below ``max_lin_speed`` / ``max_ang_speed``,

    all true for ``ceil(settle_time / step_dt)`` consecutive steps.

    ALL openings, not just the centre: ``xy_radius=0.05`` covers the bucket's
    entire ~0.10 m footprint, so a tube dropped into any well qualifies. The
    z-offset gate is what makes this safe -- the tube can only sit low enough
    (in ``[z_min, z_max]``, below the rim) if it actually descended into a well;
    resting on the solid top between wells keeps it high and is rejected. So we
    don't need per-well centres, just the footprint radius + the descent check.

    The velocity gate is deliberately LENIENT (0.3 m/s, 10 rad/s): the bucket is
    a dynamic rigid body and the tube fits its well tightly, so a correctly
    placed tube jitters well above a few cm/s and never fully "rests" -- strict
    thresholds (0.03 / 0.5) blocked success entirely. The geometric dwell does
    the real work: a tube must stay within the sub-rim z-window for
    ``settle_time``, which a fast pass-through cannot do, so the velocity gate
    only has to exclude a tube being actively flung.

    Implementation note: this is a plain function (not a stateful
    ``ManagerTermBase``) because ``scripts/tools/record_demos.py`` evaluates the
    success term by calling ``success_term.func(env, **params)`` directly -- a
    class-based term would be (wrongly) *constructed* there. The per-env dwell
    counter is therefore stashed on the ``env`` object by
    :func:`_advance_settle_counter`, and cleared on reset by
    :func:`reset_tube_settle_counter`.

    **A dwell needs per-step evaluation, which the TerminationManager does not
    always provide.** Both ``annotate_demos.py`` and ``generate_dataset.py`` set
    ``env_cfg.terminations = None`` and evaluate the extracted success term by
    hand -- and ``annotate_demos`` calls it exactly *once*, after replaying the
    whole episode. A counter advanced only by that call reaches 1, needs
    ``settle_time / step_dt`` (60 steps at the shipped 3.0 s), and reports
    failure for every episode, which is why every annotated dataset in
    ``datasets/`` came out empty. The fix is to also expose this function as an
    observation term in the ``subtask_terms`` group (see
    :class:`~rpl_centrifuge.openarm.env_cfg.ObservationsCfg`): the
    ObservationManager runs every step in all three scripts, so the counter stays
    live, and the guard in :func:`_advance_settle_counter` keeps the extra
    termination-path call from double-counting.

    Threshold defaults are grounded in the bundled asset geometry
    (``centrifuge_tube_big.usd`` is ~0.122 m tall; ``centrifuge_bucket_big.usd``
    is ~0.137 m tall with its root at the base, so the rim is ~0.137 m above the
    bucket root). All are exposed as params so callers can retune per task.
    """
    tube: RigidObject = env.scene[tube_cfg.name]
    bucket: RigidObject = env.scene[bucket_cfg.name]

    diff = tube.data.root_pos_w - bucket.data.root_pos_w
    xy_dist = torch.linalg.vector_norm(diff[:, :2], dim=1)
    z_offset = diff[:, 2]
    xy_in = xy_dist < xy_radius
    z_in = (z_offset > z_min) & (z_offset < z_max)

    lin_speed = torch.linalg.vector_norm(tube.data.root_lin_vel_w, dim=1)
    ang_speed = torch.linalg.vector_norm(tube.data.root_ang_vel_w, dim=1)
    at_rest = (lin_speed < max_lin_speed) & (ang_speed < max_ang_speed)

    seated = xy_in & z_in & at_rest

    counter = _advance_settle_counter(env, seated)

    required_steps = max(1, int(round(settle_time / env.step_dt)))
    return counter >= required_steps
