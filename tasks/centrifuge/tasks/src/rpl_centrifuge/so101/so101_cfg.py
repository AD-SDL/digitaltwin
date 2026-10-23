# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SO-ARM101 (SO101) articulation config — 5-DoF arm + 1-DoF jaw gripper.

Ported from the Sim-to-Real-SO-101-Workshop IsaacLab config (same USD, joint
order, and actuator gains), with the USD bundled inside this package. Control is
6-DoF joint position (the ``Jaw`` is the 6th joint), matching the workshop's
lerobot action space so a trajectory teleoperated from the physical SO101 leader
maps directly onto the real arm.
"""

from importlib.resources import files as _pkg_files

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_ASSET_DIR = _pkg_files("rpl_centrifuge") / "assets"
SO101_USD_PATH = str(_ASSET_DIR / "SO-ARM101-USD.usd")

# Joint order matches the workshop / lerobot convention.
ARM_JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
GRIPPER_JOINT_NAME = "Jaw"
ALL_JOINT_NAMES = [*ARM_JOINT_NAMES, GRIPPER_JOINT_NAME]

# Jaw revolute limits are [-10 deg, 100 deg]. The calibration home sits near the
# lower limit (open); driving toward the upper limit closes the fingers.
JAW_OPEN = -0.10
JAW_CLOSE = 1.20

# FrameTransformer base / EE target link names within the SO101 USD.
BASE_LINK_NAME = "base"
END_EFFECTOR_LINK_NAME = "gripper"

# Fixed-base SO101. ``prim_path`` is a placeholder; concrete env cfgs call
# ``SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")``.
SO101_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=SO101_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # pos / rot are set per-scene by the env cfg (the arm is mounted on the
        # table). Joint home is the workshop calibration pose.
        joint_pos={
            "Rotation": -0.2736,
            "Pitch": -0.6109,
            "Elbow": -0.0745,
            "Wrist_Pitch": 1.5148,
            "Wrist_Roll": -1.6034,
            "Jaw": -0.1465,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        # Gains ported verbatim from the workshop SO101_CFG.
        "rotation": ImplicitActuatorCfg(
            joint_names_expr=["Rotation"], effort_limit_sim=30, stiffness=55, damping=0.7
        ),
        "pitch": ImplicitActuatorCfg(
            joint_names_expr=["Pitch"], effort_limit_sim=30, stiffness=30, damping=0.8
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["Elbow"], effort_limit_sim=30, stiffness=25, damping=0.7
        ),
        "wrist_pitch": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Pitch"], effort_limit_sim=30, stiffness=12, damping=0.5
        ),
        "wrist_roll": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Roll"], effort_limit_sim=30, stiffness=7, damping=0.5
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[GRIPPER_JOINT_NAME], effort_limit_sim=30, stiffness=4, damping=0.3
        ),
    },
)
