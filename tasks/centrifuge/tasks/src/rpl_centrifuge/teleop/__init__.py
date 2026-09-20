# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared IsaacTeleop retargeting helpers for the centrifuge task pack.

Everything here that touches the optional ``isaacteleop`` package is imported
*lazily* inside functions, mirroring the upstream reference tasks
(``contrib/stack/config/franka/stack_ik_abs_env_cfg.py`` etc.). This keeps the
env-cfg modules that import this package **headless-importable** — the heavy
``isaacteleop`` import is deferred to teleop session start, so ``gym`` listing,
the sim-free test suite, and dataset tooling can import the cfgs without Kit /
``carb`` present.
"""

from .retargeters import (
    make_dual_source_dex_hand_retargeter,
    make_dual_source_se3_abs_retargeter,
    make_dual_source_se3_rel_retargeter,
    make_trigger_grasp_retargeter,
)

__all__ = [
    "make_dual_source_dex_hand_retargeter",
    "make_dual_source_se3_abs_retargeter",
    "make_dual_source_se3_rel_retargeter",
    "make_trigger_grasp_retargeter",
]
