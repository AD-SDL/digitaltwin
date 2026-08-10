# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual-action keyboard wrapper.

``Se3Keyboard.advance()`` returns a 7-dim tensor (6-D delta pose + 1-D gripper)
that drives a single arm. The centrifuge env's action manager publishes a
14-dim action vector with term order:

    [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]

This wrapper pads the keyboard output so it slots into the right-arm half of
that vector while the left arm holds its default joint pose and the left
gripper stays closed. Lets you smoke-test the bimanual env from a keyboard
without authoring a separate unimanual env cfg.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from isaaclab.devices.device_base import DeviceBase, DeviceCfg
from isaaclab.devices.keyboard import Se3Keyboard, Se3KeyboardCfg


class BimanualKeyboardRightArm(DeviceBase):
    """``Se3Keyboard`` mapped onto the right arm of the bimanual centrifuge env.

    Output layout (matches the env's action term order):
        idx 0-5  : left arm delta pose      (zeros — hold)
        idx 6    : left gripper             (0.0 — closed; below BinaryJoint threshold 0.5)
        idx 7-12 : right arm delta pose     (keyboard delta)
        idx 13   : right gripper            (+1.0 open / -1.0 close, toggled with "K")
    """

    def __init__(self, cfg: BimanualKeyboardRightArmCfg):
        super().__init__()
        self._sim_device = cfg.sim_device
        self._keyboard = Se3Keyboard(
            Se3KeyboardCfg(
                pos_sensitivity=cfg.pos_sensitivity,
                rot_sensitivity=cfg.rot_sensitivity,
                sim_device=cfg.sim_device,
            )
        )

    def __str__(self) -> str:
        return "BimanualKeyboardRightArm(wrapping Se3Keyboard; keyboard drives right arm only)"

    def reset(self) -> None:
        self._keyboard.reset()

    def add_callback(self, key: Any, func: Callable) -> None:
        self._keyboard.add_callback(key, func)

    def advance(self) -> torch.Tensor:
        kb = self._keyboard.advance()  # shape: (7,)
        out = torch.zeros(14, dtype=kb.dtype, device=self._sim_device)
        # left arm: zero delta (hold pose), left gripper: 0.0 < 0.5 → closed
        out[7:13] = kb[:6]
        out[13] = kb[6]
        return out


@dataclass
class BimanualKeyboardRightArmCfg(DeviceCfg):
    """Configuration for :class:`BimanualKeyboardRightArm`."""

    pos_sensitivity: float = 0.05
    rot_sensitivity: float = 0.05
    class_type: type[DeviceBase] = BimanualKeyboardRightArm
    # No retargeters — advance() returns the final 14-dim action directly.
    retargeters: list = field(default_factory=list)
