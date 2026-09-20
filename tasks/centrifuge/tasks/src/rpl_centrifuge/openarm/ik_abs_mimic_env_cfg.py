# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab Mimic env cfg for the bimanual OpenArm centrifuge IK-Abs task.

Low-dimensional Mimic variant (no cameras). Wraps
:class:`CentrifugeBimanualIkAbsEnvCfg` with :class:`MimicEnvCfg` and declares the
per-arm subtask sequence. The base env already exposes the ``subtask_terms``
observation group (see :mod:`rpl_centrifuge.openarm.env_cfg`), which the Mimic
env wrapper (:mod:`rpl_centrifuge.openarm.mimic_env`) reads for the
``right_grasp_tube`` signal.

For an IMAGE-bearing Mimic dataset, use
:class:`~rpl_centrifuge.openarm.visuomotor_mimic_env_cfg.CentrifugeBimanualIkAbsVisuomotorMimicEnvCfg`
instead (adds cameras + image obs; rendered headless by ``generate_dataset``).
"""

from __future__ import annotations

import os

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils.configclass import configclass

from .ik_abs_env_cfg import CentrifugeBimanualIkAbsEnvCfg


def _apply_centrifuge_mimic_cfg(cfg) -> None:
    """Fill datagen + per-arm subtask configs (shared by the low-dim and visuomotor Mimic cfgs)."""
    cfg.datagen_config.name = "centrifuge_openarm_bimanual_ik_abs_D0"
    cfg.datagen_config.generation_guarantee = True
    cfg.datagen_config.generation_keep_failed = False
    cfg.datagen_config.generation_num_trials = 100
    cfg.datagen_config.generation_select_src_per_subtask = False
    cfg.datagen_config.generation_select_src_per_arm = False
    cfg.datagen_config.generation_relative = False
    cfg.datagen_config.generation_joint_pos = False
    cfg.datagen_config.generation_transform_first_robot_pose = False
    cfg.datagen_config.generation_interpolate_from_last_target_pose = True
    # Open-loop replay of this insertion is lossy: the gripper closing on the
    # tube amplifies a ~2 mm pose error into ~13 mm (median, 53 mm worst), which
    # is enough to miss the bucket well. Replaying the ORIGINAL actions from the
    # ORIGINAL recorded state succeeds only 3/20, and generation is strictly
    # harder. At that yield the old cap of 25 aborted a run almost immediately;
    # with generation_guarantee=True the failures are expected, not pathological.
    # Sized for a 100-demo target at the measured ~10% yield (~900 failures
    # expected); the cap is a runaway guard, not an expected limit. Too low and
    # the run aborts mid-way with a partial dataset: at 10% yield the old cap of
    # 25 died after ~3 demos and 200 would die after ~23.
    cfg.datagen_config.max_num_failures = 3000
    cfg.datagen_config.num_demo_to_render = 10
    cfg.datagen_config.num_fail_demo_to_render = 25
    # generate_dataset.py seeds random / numpy / torch from this value and
    # exposes no --seed flag, so two concurrently launched runs with the default
    # produce BYTE-IDENTICAL episodes -- duplicated data, not more data. Override
    # per shard when fanning out across GPUs:
    #   CENTRIFUGE_DATAGEN_SEED=2 ... --device cuda:1 --output_file shard1.hdf5
    # An exported-but-empty var is treated as unset rather than crashing on int("").
    cfg.datagen_config.seed = int(os.environ.get("CENTRIFUGE_DATAGEN_SEED") or 1)

    # Right arm: 1) approach + grasp tube on rack (term = right_grasp_tube),
    #            2) transport + drop into bucket (term = None -> last subtask).
    cfg.subtask_configs["right"] = [
        SubTaskConfig(
            object_ref="tube",
            subtask_term_signal="right_grasp_tube",
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
            object_ref="bucket",
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

    # Left arm: single passive subtask so the recorder can query the left eef.
    cfg.subtask_configs["left"] = [
        SubTaskConfig(
            object_ref="tube",
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
class CentrifugeBimanualIkAbsMimicEnvCfg(CentrifugeBimanualIkAbsEnvCfg, MimicEnvCfg):
    """Low-dim Mimic-annotatable variant of the bimanual OpenArm IK-Abs task."""

    def __post_init__(self):
        super().__post_init__()
        _apply_centrifuge_mimic_cfg(self)
