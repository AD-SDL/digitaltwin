# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cabinet door-opening task pack for Isaac Lab (OpenArm).

A full-height black storage cabinet (RPL "Global Industrial 603599BK",
articulated as USD: two revolute doors + two revolute lever handles) stands in
front of the bimanual OpenArm. The goal matches the real
``openarm_open_lab_door`` LeRobot dataset: hook the left-door lever with a
pre-closed left-hand gripper, pull the door open through its full swing. See
``src/rpl_cabinet/README.md`` for the geometry, the dataset analysis, and the
known gaps carried over from the ``cabinet_sim/`` prototype.

Teleop
------
All VR teleop tasks use the IsaacLab 3.0.0 ``isaacteleop`` pipeline
(:attr:`isaac_teleop`) and support **both hand-tracking and joystick
(VR-controller) control** in a single session: arm pose is driven by a
dual-source retargeter (controller grip pose has priority, tracked wrist is
the fallback) and the gripper by the stock ``GripperRetargeter`` (trigger vs.
pinch). See :mod:`rpl_cabinet.teleop`. There is no legacy ``teleop_devices``
path (the old ``isaaclab.devices.openxr.*`` cfgs pull in Kit's ``carb`` and
break headless import).

Package self-registers these gym IDs on import:
  * ``Isaac-OpenDoor-Bimanual-OpenArm-IK-Rel-v0`` -- bimanual OpenArm,
    relative differential-IK. VR hand-tracking + controllers.
  * ``Isaac-OpenDoor-Bimanual-OpenArm-IK-Abs-v0`` -- bimanual OpenArm,
    absolute differential-IK. VR hand-tracking + controllers. Recommended.
  * ``Isaac-OpenDoor-Bimanual-OpenArm-IK-Abs-Mimic-v0`` -- the IK-Abs task
    wrapped in :class:`~isaaclab.envs.ManagerBasedRLMimicEnv` so
    ``annotate_demos.py`` / ``generate_dataset.py`` from
    ``scripts/imitation_learning/isaaclab_mimic`` can run on recorded demos.
  * ``Isaac-OpenDoor-Bimanual-OpenArm-IK-Abs-Visuomotor-v0`` -- IK-Abs +
    wrist/table cameras + image obs. For HEADLESS dataset generation; XR
    teleop with a camera observation crashes Kit.
  * ``Isaac-OpenDoor-Bimanual-OpenArm-IK-Abs-Visuomotor-Mimic-v0`` -- the
    canonical image-dataset generator for this task.

A ``.pth`` file shipped with the wheel runs ``import rpl_cabinet`` at every
Python startup. As a fallback, explicitly ``import rpl_cabinet  # noqa: F401``
before ``gym.make``.
"""

import gymnasium as gym

from . import agents  # noqa: F401

gym.register(
    id="Isaac-OpenDoor-Bimanual-OpenArm-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.openarm.ik_rel_env_cfg:OpenDoorBimanualIkRelEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-OpenDoor-Bimanual-OpenArm-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.openarm.ik_abs_env_cfg:OpenDoorBimanualIkAbsEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-OpenDoor-Bimanual-OpenArm-IK-Abs-Mimic-v0",
    entry_point=f"{__name__}.openarm.mimic_env:OpenDoorBimanualOpenArmIKAbsMimicEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.openarm.ik_abs_mimic_env_cfg:OpenDoorBimanualIkAbsMimicEnvCfg"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-OpenDoor-Bimanual-OpenArm-IK-Abs-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.openarm.visuomotor_env_cfg:OpenDoorBimanualIkAbsVisuomotorEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-OpenDoor-Bimanual-OpenArm-IK-Abs-Visuomotor-Mimic-v0",
    entry_point=f"{__name__}.openarm.mimic_env:OpenDoorBimanualOpenArmIKAbsMimicEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.openarm.visuomotor_mimic_env_cfg:OpenDoorBimanualIkAbsVisuomotorMimicEnvCfg"
        ),
    },
    disable_env_checker=True,
)
