# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-torch SO101 leader <-> sim joint mapping (NO lerobot / carb imports).

The in-process agent (``scripts/so101_lerobot_agent.py``) reads the physical
leader via lerobot's ``so_leader`` driver and passes the raw joint vector
through :class:`SO101LeaderMapping` to get absolute sim-SO101 joint targets
(radians); for dataset recording it maps the sim joint positions back to
real-arm space with the inverse method.

This module itself imports nothing from lerobot, so it is safe to reuse from
any lerobot-free context (headless tests, dataset conversion helpers, etc.).

The mapping mirrors the Sim-to-Real-SO-101-Workshop convention: leader values
arrive as normalized percent (``[-100, 100]`` for the 5 arm joints, ``[0, 100]``
for the gripper — i.e. lerobot ``use_degrees=False``), are scaled into the USD
joint ranges (degrees), then converted to radians.
"""

from __future__ import annotations

import torch

# USD articulation joint ranges (degrees) for the bundled SO-ARM101-USD.usd.
SO101_USD_MAPPING = {
    "shoulder_pan": {"joint_min": -110, "joint_max": 110},
    "shoulder_lift": {"joint_min": -100, "joint_max": 100},
    "elbow_flex": {"joint_min": -100, "joint_max": 90},
    "wrist_flex": {"joint_min": -95, "joint_max": 95},
    "wrist_roll": {"joint_min": -160, "joint_max": 160},
    "gripper": {"joint_min": -10, "joint_max": 100},
}

# Leader stream order. Positionally aligned with the env action term's joint
# order [Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw] (preserve_order).
SO101_JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
NUM_JOINTS = len(SO101_JOINT_ORDER)


class SO101LeaderMapping:
    """Vectorized leader<->sim joint conversions on a given device."""

    def __init__(self, device: str):
        self.device = device
        self.joint_mins = torch.tensor(
            [SO101_USD_MAPPING[n]["joint_min"] for n in SO101_JOINT_ORDER],
            dtype=torch.float32, device=device,
        )
        self.joint_maxs = torch.tensor(
            [SO101_USD_MAPPING[n]["joint_max"] for n in SO101_JOINT_ORDER],
            dtype=torch.float32, device=device,
        )

    def leader_to_sim_radians(self, raw: torch.Tensor) -> torch.Tensor:
        """Normalized leader percent -> USD joint range (deg) -> radians."""
        normalized = torch.zeros_like(raw)
        normalized[:-1] = (raw[:-1] + 100) / 200.0  # arm: [-100,100] -> [0,1]
        normalized[-1] = raw[-1] / 100.0            # gripper: [0,100] -> [0,1]
        mapped_deg = self.joint_mins + normalized * (self.joint_maxs - self.joint_mins)
        return mapped_deg * torch.pi / 180

    def sim_radians_to_real(self, rad: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`leader_to_sim_radians` (radians -> leader percent)."""
        mapped_deg = rad * 180 / torch.pi
        normalized = (mapped_deg - self.joint_mins) / (self.joint_maxs - self.joint_mins)
        raw = torch.zeros_like(normalized)
        raw[:-1] = normalized[:-1] * 200.0 - 100
        raw[-1] = normalized[-1] * 100.0
        return raw
