# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab Mimic env cfg for the OpenArm cabinet door-opening IK-Abs task.

Low-dimensional variant (no cameras). Wraps :class:`OpenDoorBimanualIkAbsEnvCfg`
with :class:`MimicEnvCfg` and declares the per-arm subtask sequence. The base
env already exposes the ``subtask_terms`` observation group
(:mod:`rpl_cabinet.openarm.env_cfg`), which the Mimic env wrapper
(:mod:`rpl_cabinet.openarm.mimic_env`) reads for the ``left_grasp_handle``
signal.

For an IMAGE-bearing Mimic dataset, use
:class:`~rpl_cabinet.openarm.visuomotor_mimic_env_cfg.OpenDoorBimanualIkAbsVisuomotorMimicEnvCfg`.
"""

from __future__ import annotations

import os

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils.configclass import configclass

from .ik_abs_env_cfg import OpenDoorBimanualIkAbsEnvCfg


def _apply_opendoor_mimic_cfg(cfg) -> None:
    """Fill datagen + per-arm subtask configs (shared by low-dim and visuomotor Mimic cfgs)."""
    cfg.datagen_config.name = "opendoor_openarm_bimanual_ik_abs_D0"
    cfg.datagen_config.generation_guarantee = True
    cfg.datagen_config.generation_keep_failed = False
    cfg.datagen_config.generation_num_trials = 100
    cfg.datagen_config.generation_select_src_per_subtask = False
    cfg.datagen_config.generation_select_src_per_arm = False
    cfg.datagen_config.generation_relative = False
    cfg.datagen_config.generation_joint_pos = False
    cfg.datagen_config.generation_transform_first_robot_pose = False
    cfg.datagen_config.generation_interpolate_from_last_target_pose = True
    # Door-pull is an under-constrained hook, not a precise insertion, so the
    # per-attempt success rate should be noticeably higher than the centrifuge
    # task's ~10%. Keep a 2000-cap runaway guard anyway; it only triggers if the
    # scene has drifted far enough that generation can't recover.
    cfg.datagen_config.max_num_failures = 2000
    cfg.datagen_config.num_demo_to_render = 10
    cfg.datagen_config.num_fail_demo_to_render = 25
    # generate_dataset.py seeds random/numpy/torch from this value. Override per
    # shard when fanning out across GPUs:
    #   OPENDOOR_DATAGEN_SEED=2 ... --device cuda:1 --output_file shard1.hdf5
    cfg.datagen_config.seed = int(os.environ.get("OPENDOOR_DATAGEN_SEED") or 1)

    # Left arm: 1) approach + engage the left-door handle (term = left_grasp_handle),
    #           2) pull door open (term = None -> last subtask).
    cfg.subtask_configs["left"] = [
        SubTaskConfig(
            object_ref="cabinet",
            subtask_term_signal="left_grasp_handle",
            subtask_term_offset_range=(0, 0),
            first_subtask_start_offset_range=(0, 0),
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs={"nn_k": 3},
            action_noise=0.003,
            num_interpolation_steps=0,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        ),
        SubTaskConfig(
            object_ref="cabinet",
            subtask_term_signal=None,
            subtask_term_offset_range=(0, 0),
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs={"nn_k": 3},
            action_noise=0.003,
            num_interpolation_steps=3,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        ),
    ]

    # Right arm: single passive subtask so the recorder can query the right eef
    # (the real demos park the right arm near home; keep that convention).
    cfg.subtask_configs["right"] = [
        SubTaskConfig(
            object_ref="cabinet",
            subtask_term_signal=None,
            subtask_term_offset_range=(0, 0),
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs={"nn_k": 3},
            action_noise=0.003,
            num_interpolation_steps=0,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        ),
    ]


@configclass
class OpenDoorBimanualIkAbsMimicEnvCfg(OpenDoorBimanualIkAbsEnvCfg, MimicEnvCfg):
    """Low-dim Mimic-annotatable variant of the OpenArm cabinet IK-Abs task."""

    def __post_init__(self):
        super().__post_init__()
        _apply_opendoor_mimic_cfg(self)
