# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mimic cfg for the SO101 IK-Abs task (data-gen input).

Thin wrapper on :class:`CentrifugeSO101IkAbsEnvCfg` that wires in the SubTask
schedule IsaacLab Mimic expects — a single ``"right"`` arm with one passive
subtask so the recorder can query the EE pose; the actual MDP success is
already defined on the base env (``TerminationsCfg.success``).
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass
from isaaclab_mimic.envs.mimic_env_cfg import SubTaskConfig
from isaaclab_mimic.managers.mimic_env_cfg import MimicEnvCfg

from .ik_abs_env_cfg import CentrifugeSO101IkAbsEnvCfg


@configclass
class CentrifugeSO101IkAbsMimicEnvCfg(CentrifugeSO101IkAbsEnvCfg, MimicEnvCfg):
    """IK-Abs cfg + Mimic SubTask schedule for the SO101 centrifuge task."""

    def __post_init__(self):
        CentrifugeSO101IkAbsEnvCfg.__post_init__(self)
        MimicEnvCfg.__post_init__(self)

        cfg = self.subtask_configs
        # Single-arm SO101: Mimic still needs an arm key — use ``"right"`` to
        # match the OpenArm single-arm convention.
        cfg["right"] = [
            SubTaskConfig(
                object_ref="tube",
                subtask_term_signal="grasp_tube",
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
            SubTaskConfig(
                object_ref="bucket",
                subtask_term_signal=None,  # final subtask -> terminates on env success
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
        ]
