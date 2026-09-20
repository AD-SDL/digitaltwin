# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Centrifuge task pack for Isaac Lab.

Contains multiple robot embodiments for the same centrifuge workflow. A plastic
centrifuge tube starts on the table/rack; the goal is to drop it inside the
centrifuge bucket. Designed for human teleoperation + imitation learning via
Isaac Lab's ``scripts/tools/record_demos.py``.

All VR teleop tasks use the IsaacLab 3.0.0 ``isaacteleop`` pipeline
(:attr:`isaac_teleop`) and support **both hand-tracking and joystick
(VR-controller) control** in a single session: the arm/wrist pose is driven by a
dual-source retargeter (controller grip pose has priority, tracked wrist is the
fallback) and the gripper by the stock ``GripperRetargeter`` (controller trigger
vs. hand pinch). See :mod:`rpl_centrifuge.teleop`. There is no legacy
``teleop_devices`` path (the old ``isaaclab.devices.openxr.*`` cfgs pull in Kit's
``carb`` and break headless import).

This package self-registers its gym IDs on import:
  * ``Isaac-Centrifuge-Bimanual-OpenArm-IK-Rel-v0`` — bimanual OpenArm, relative
    differential-IK. VR hand-tracking + controllers.
  * ``Isaac-Centrifuge-Bimanual-OpenArm-IK-Abs-v0`` — bimanual OpenArm, absolute
    differential-IK. VR hand-tracking + controllers. Recommended.
  * ``Isaac-Centrifuge-Bimanual-OpenArm-IK-Abs-Mimic-v0`` — same scene + control
    as the IK-Abs task, wrapped in
    :class:`~isaaclab.envs.ManagerBasedRLMimicEnv` so ``annotate_demos.py`` and
    ``generate_dataset.py`` from ``scripts/imitation_learning/isaaclab_mimic``
    can run on demos recorded from the IK-Abs task.
  * ``Isaac-Centrifuge-GR1T2-WaistEnabled-Abs-v0`` — GR1T2 waist-enabled variant
    derived from Isaac Lab's GR1T2 pick-place task, with the centrifuge rack /
    tube / bucket scene swapped in. VR hand-tracking (dexterous fingers) +
    controller wrists with a trigger-driven power grasp.
  * ``Isaac-Centrifuge-SO101-IK-Abs-v0`` — single-arm SO101, absolute task-space
    IK. VR hand-tracking + controllers (reuses the upstream SO101 pose-IK action).
  * ``Isaac-Centrifuge-SO101-Leader-v0`` — single-arm SO101 driven by a physical
    SO101 leader arm over USB, via the native IsaacTeleop joint-space pipeline
    (``so101_leader`` device); same scene/props/randomization as the IK-Abs task.
  * ``Isaac-Centrifuge-SO101-LeRobot-Leader-v0`` — single-arm SO101 driven by a
    physical SO101 leader arm over USB via **LeRobot's** driver ("data transfer"),
    with live cameras + a ``visual`` obs group; demos record straight to a LeRobot
    dataset. Run with ``scripts/lerobot_leader_agent.py`` (not ``record_demos.py``).
    ``Isaac-Centrifuge-SO101-LeRobot-Leader-DR-v0`` widens the reset randomization.

A companion ``.pth`` file shipped with the wheel runs ``import
rpl_centrifuge`` at every Python startup, so the IDs are visible to
``gym.make()`` without any user code changes. As a fallback, explicitly
``import rpl_centrifuge  # noqa: F401`` before ``gym.make``.
"""

import gymnasium as gym

from . import agents  # noqa: F401

gym.register(
    id="Isaac-Centrifuge-Bimanual-OpenArm-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.openarm.ik_rel_env_cfg:CentrifugeBimanualIkRelEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Centrifuge-Bimanual-OpenArm-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.openarm.ik_abs_env_cfg:CentrifugeBimanualIkAbsEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Centrifuge-Bimanual-OpenArm-IK-Abs-Mimic-v0",
    entry_point=f"{__name__}.openarm.mimic_env:CentrifugeBimanualOpenArmIKAbsMimicEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.openarm.ik_abs_mimic_env_cfg:CentrifugeBimanualIkAbsMimicEnvCfg"
        ),
    },
    disable_env_checker=True,
)

# OpenArm Visuomotor (cameras + image obs) — for HEADLESS dataset generation.
gym.register(
    id="Isaac-Centrifuge-Bimanual-OpenArm-IK-Abs-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.openarm.visuomotor_env_cfg:CentrifugeBimanualIkAbsVisuomotorEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Centrifuge-Bimanual-OpenArm-IK-Abs-Visuomotor-Mimic-v0",
    entry_point=f"{__name__}.openarm.mimic_env:CentrifugeBimanualOpenArmIKAbsMimicEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.openarm.visuomotor_mimic_env_cfg:CentrifugeBimanualIkAbsVisuomotorMimicEnvCfg"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Centrifuge-GR1T2-WaistEnabled-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gr1t2.waist_enabled_env_cfg:GR1T2CentrifugeWaistEnabledEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Centrifuge-SO101-Leader-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.so101.leader_teleop_env_cfg:CentrifugeSO101LeaderTeleopEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Centrifuge-SO101-LeRobot-Leader-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.so101.lerobot_leader_env_cfg:CentrifugeSO101LeRobotLeaderEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Centrifuge-SO101-LeRobot-Leader-DR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.so101.lerobot_leader_env_cfg:CentrifugeSO101LeRobotLeaderDREnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Centrifuge-SO101-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.so101.ik_abs_env_cfg:CentrifugeSO101IkAbsEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Centrifuge-SO101-IK-Abs-Mimic-v0",
    entry_point=f"{__name__}.so101.mimic_env:CentrifugeSO101IKAbsMimicEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.so101.ik_abs_mimic_env_cfg:CentrifugeSO101IkAbsMimicEnvCfg",
    },
    disable_env_checker=True,
)

# SO101 Visuomotor (cameras + image obs) — for HEADLESS dataset generation.
gym.register(
    id="Isaac-Centrifuge-SO101-IK-Abs-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.so101.visuomotor_env_cfg:CentrifugeSO101IkAbsVisuomotorEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Centrifuge-SO101-IK-Abs-Visuomotor-Mimic-v0",
    entry_point=f"{__name__}.so101.mimic_env:CentrifugeSO101IKAbsMimicEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.so101.visuomotor_mimic_env_cfg:CentrifugeSO101IkAbsVisuomotorMimicEnvCfg"
        ),
    },
    disable_env_checker=True,
)
