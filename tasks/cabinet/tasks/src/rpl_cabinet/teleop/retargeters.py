# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dual-source (hand-tracking **and** VR-controller) retargeters for CloudXR teleop.

Why this module exists
----------------------
IsaacLab 3.0.0 drives teleop through an ``isaacteleop`` retargeting pipeline
(see the reference tasks ``IsaacContrib-Stack-Cube-Franka-IK-Abs``,
``IsaacContrib-Stack-Cube-SO101-IK-Abs`` and
``IsaacContrib-PickPlace-GR1T2-WaistEnabled-Abs``). In that pipeline a
``Se3AbsRetargeter`` / ``Se3RelRetargeter`` binds to a **single** input source
via ``Se3RetargeterConfig.input_device`` — either a controller
(``controller_left`` / ``controller_right``) or a tracked hand
(``hand_left`` / ``hand_right``). The Franka reference therefore only works with
controllers; the GR1T2 reference only works with tracked hands.

Our requirement is that every centrifuge robot be teleoperable over a VR headset
with **both** hand-tracking and joystick (Touch-controller) control. Both
``ControllersSource`` and ``HandsSource`` are polled every frame at runtime
(the operator may hold the controllers or track bare hands), and the built-in
``GripperRetargeter`` already fuses them (controller trigger has priority, hand
pinch is the fallback). We extend that exact pattern to the 6-DOF pose channel:
``DualSourceSe3AbsRetargeter`` / ``DualSourceSe3RelRetargeter`` declare **both**
optional inputs and, each frame, pick the controller when it reports a valid
grip pose, otherwise fall back to the tracked hand — then delegate to the stock
retargeter compute so all the tuned math (offsets, smoothing, wrist selection)
is reused unchanged.

Everything is defined inside factory functions so ``isaacteleop`` is imported
lazily (this module stays headless-importable; see :mod:`rpl_centrifuge.teleop`).

Frame note
----------
Connect the SAME (anchor-transformed) stream you feed the stock retargeters:
``controllers.transformed(world_T_anchor)`` for the controller key and
``hands.transformed(world_T_anchor)`` for the hand key. When the owning task sets
``IsaacTeleopCfg.target_frame_prim_path`` the device folds ``base_T_world`` into
that transform, so both streams arrive already in the robot base frame.

Tuning caveat
-------------
A controller grip frame and a tracked wrist frame differ by a roughly fixed
rotation, so a single ``target_offset_{roll,pitch,yaw}`` cannot be optimal for
both modalities at once. The defaults chosen by each task target the
controller; hand-tracking may need a different offset. These values require
live tuning on a headset (via the retargeter tuning UI) and cannot be validated
without VR hardware.
"""

from __future__ import annotations

from typing import Any


def _side_keys(side: str) -> tuple[str, str]:
    """Return the ``(controller_key, hand_key)`` deviceio source names for a side."""
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got: {side!r}")
    return f"controller_{side}", f"hand_{side}"


def make_dual_source_se3_abs_retargeter(side: str, name: str, **config_kwargs: Any):
    """Build a :class:`Se3AbsRetargeter` that accepts controller **or** hand pose.

    Args:
        side: ``"left"`` or ``"right"``.
        name: retargeter node name (also the ``ee_pose`` output owner).
        **config_kwargs: forwarded to ``Se3RetargeterConfig`` (e.g.
            ``target_offset_roll``, ``use_wrist_rotation``, ...). ``input_device``
            is set internally and must not be passed.

    Returns:
        A retargeter instance emitting a 7-D ``ee_pose`` (pos xyz + quat xyzw),
        identical in contract to the stock ``Se3AbsRetargeter`` but driven by
        whichever of the two sources is live (controller priority).
    """
    from isaacteleop.retargeters import Se3AbsRetargeter, Se3RetargeterConfig
    from isaacteleop.retargeting_engine.interface.tensor_group_type import OptionalType
    from isaacteleop.retargeting_engine.tensor_types import (
        ControllerInput,
        ControllerInputIndex,
        HandInput,
    )

    controller_key, hand_key = _side_keys(side)

    class DualSourceSe3AbsRetargeter(Se3AbsRetargeter):
        """Se3AbsRetargeter reading controller (priority) or tracked hand (fallback)."""

        def input_spec(self):
            return {
                controller_key: OptionalType(ControllerInput()),
                hand_key: OptionalType(HandInput()),
            }

        def _compute_fn(self, inputs, outputs, context) -> None:
            controller = inputs[controller_key]
            chosen = None
            # Controller wins when present AND its grip pose is localizable this
            # frame (matching the stock retargeter's GRIP_IS_VALID guard). A
            # connected-but-untracked controller reports is_none=False with an
            # invalid grip, so check both before selecting it.
            if not controller.is_none and bool(controller[ControllerInputIndex.GRIP_IS_VALID]):
                chosen = controller_key
            elif not inputs[hand_key].is_none:
                chosen = hand_key

            if chosen is None:
                # Neither source live: hold the last pose (same as the stock
                # retargeter's is_none path).
                outputs["ee_pose"][0] = self._last_pose
                return

            # Delegate to the stock compute with input_device pointed at the live
            # source. The parent reads inputs[self._config.input_device] and
            # branches on "hand" vs controller; both keys are present in ``inputs``.
            saved = self._config.input_device
            self._config.input_device = chosen
            try:
                super()._compute_fn(inputs, outputs, context)
            finally:
                self._config.input_device = saved

    cfg = Se3RetargeterConfig(input_device=controller_key, **config_kwargs)
    return DualSourceSe3AbsRetargeter(cfg, name=name)


def make_dual_source_se3_rel_retargeter(side: str, name: str, **config_kwargs: Any):
    """Build a :class:`Se3RelRetargeter` that accepts controller **or** hand deltas.

    Same selection logic as :func:`make_dual_source_se3_abs_retargeter`. Emits a
    6-D ``ee_delta`` (pos delta + rotvec). When the live source changes between
    frames the delta baseline is re-armed (one zero-delta frame) so switching
    modalities mid-session does not produce a jump.
    """
    from isaacteleop.retargeters import Se3RelRetargeter, Se3RetargeterConfig
    from isaacteleop.retargeting_engine.interface.tensor_group_type import OptionalType
    from isaacteleop.retargeting_engine.tensor_types import (
        ControllerInput,
        ControllerInputIndex,
        HandInput,
    )

    controller_key, hand_key = _side_keys(side)

    class DualSourceSe3RelRetargeter(Se3RelRetargeter):
        """Se3RelRetargeter reading controller (priority) or tracked hand (fallback)."""

        def __init__(self, config, name: str) -> None:
            super().__init__(config, name)
            self._active_device: str | None = None

        def input_spec(self):
            return {
                controller_key: OptionalType(ControllerInput()),
                hand_key: OptionalType(HandInput()),
            }

        def _compute_fn(self, inputs, outputs, context) -> None:
            controller = inputs[controller_key]
            chosen = None
            if not controller.is_none and bool(controller[ControllerInputIndex.GRIP_IS_VALID]):
                chosen = controller_key
            elif not inputs[hand_key].is_none:
                chosen = hand_key

            if chosen is None:
                import numpy as np

                outputs["ee_delta"][0] = np.zeros(6, dtype=np.float32)
                return

            # Re-baseline when the live source switches so the first frame on the
            # new source emits a zero delta instead of a stale-baseline jump.
            if chosen != self._active_device:
                self._first_frame = True
                self._active_device = chosen

            saved = self._config.input_device
            self._config.input_device = chosen
            try:
                super()._compute_fn(inputs, outputs, context)
            finally:
                self._config.input_device = saved

    cfg = Se3RetargeterConfig(input_device=controller_key, **config_kwargs)
    return DualSourceSe3RelRetargeter(cfg, name=name)


def make_trigger_grasp_retargeter(
    side: str,
    name: str,
    joint_names: list[str],
    open_value: float = 0.0,
    close_value: float = 1.0,
    trigger_threshold: float = 0.5,
):
    """Build a dexterous-hand grasp retargeter driven by the controller trigger.

    This gives GR1T2's dexterous hand a usable **joystick** grasp: when the
    operator holds the Touch controllers (so hand-tracking is unavailable and the
    ``DexHandRetargeter`` has no data), squeezing the trigger past
    ``trigger_threshold`` drives every named finger joint to ``close_value``,
    releasing drives them to ``open_value`` — a simple power grasp. When the
    controller is absent it emits ``None`` so a downstream selector can prefer the
    hand-tracking finger solution instead.

    Output is a scalar-per-joint tensor group named ``hand_joints`` with the
    given ``joint_names`` order, matching the ``DexHandRetargeter`` contract so it
    can be swapped into the same reorderer slot.
    """
    from isaacteleop.retargeting_engine.interface import BaseRetargeter, RetargeterIOType
    from isaacteleop.retargeting_engine.interface.retargeter_core_types import RetargeterIO
    from isaacteleop.retargeting_engine.interface.tensor_group_type import (
        OptionalType,
        TensorGroupType,
    )
    from isaacteleop.retargeting_engine.tensor_types import (
        ControllerInput,
        ControllerInputIndex,
        FloatType,
    )

    controller_key, _ = _side_keys(side)
    _joint_names = list(joint_names)

    class TriggerGraspRetargeter(BaseRetargeter):
        """Maps a controller trigger to an open/close posture over ``joint_names``."""

        def input_spec(self) -> RetargeterIOType:
            return {controller_key: OptionalType(ControllerInput())}

        def output_spec(self) -> RetargeterIOType:
            return {
                "hand_joints": TensorGroupType(
                    "hand_joints", [FloatType(jn) for jn in _joint_names]
                )
            }

        def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
            controller = inputs[controller_key]
            hand_joints = outputs["hand_joints"]
            if controller.is_none:
                # No controller: emit the open posture. (In practice the pipeline
                # prefers the hand-tracking solution when hands are live.)
                for i in range(len(_joint_names)):
                    hand_joints[i] = float(open_value)
                return
            trigger = float(controller[ControllerInputIndex.TRIGGER_VALUE])
            target = close_value if trigger > trigger_threshold else open_value
            for i in range(len(_joint_names)):
                hand_joints[i] = float(target)

    return TriggerGraspRetargeter(name=name)


def make_dual_source_dex_hand_retargeter(
    side: str,
    name: str,
    grasp_close_value: float = 1.0,
    grasp_open_value: float = 0.0,
    trigger_threshold: float = 0.5,
    **config_kwargs: Any,
):
    """Build a :class:`DexHandRetargeter` with a controller-trigger grasp fallback.

    Dexterous finger retargeting fundamentally needs tracked hands, so this keeps
    the stock ``DexHandRetargeter`` solution whenever the hand is tracked (so the
    hand-tracking path is byte-for-byte the upstream behavior). When only the VR
    controller is present (hand-tracking unavailable because the operator is
    holding the Touch controllers), it drives every finger joint to a simple
    open/close power grasp from the controller trigger — so joystick mode can
    still grasp the tube. Output contract (node output ``hand_joints`` over
    ``hand_joint_names``) is identical to the stock retargeter, so it drops into
    the same reorderer slot.

    Args:
        side: ``"left"`` or ``"right"``.
        name: retargeter node name.
        grasp_close_value / grasp_open_value: finger joint targets [rad] for the
            controller power grasp (closed / open). Uniform across joints —
            STARTING POINT, needs on-headset tuning.
        trigger_threshold: controller trigger fraction above which to close.
        **config_kwargs: forwarded to ``DexHandRetargeterConfig`` (must include
            ``hand_retargeting_config``, ``hand_urdf``, ``hand_joint_names``, and
            optionally ``handtracking_to_baselink_frame_transform``). ``hand_side``
            is set internally.

    Returns:
        A retargeter instance emitting ``hand_joints`` over ``hand_joint_names``.
    """
    from isaacteleop.retargeters import DexHandRetargeter, DexHandRetargeterConfig
    from isaacteleop.retargeting_engine.interface.tensor_group_type import OptionalType
    from isaacteleop.retargeting_engine.tensor_types import ControllerInput, ControllerInputIndex, HandInput

    controller_key, hand_key = _side_keys(side)

    class DualSourceDexHandRetargeter(DexHandRetargeter):
        """DexHandRetargeter that falls back to a controller-trigger power grasp."""

        def input_spec(self):
            return {
                hand_key: OptionalType(HandInput()),
                controller_key: OptionalType(ControllerInput()),
            }

        def _compute_fn(self, inputs, outputs, context) -> None:
            hand = inputs[hand_key]
            if not hand.is_none:
                # Tracked hand available: use the full dexterous solution
                # (identical to upstream). The parent only reads the hand key.
                super()._compute_fn(inputs, outputs, context)
                return
            controller = inputs[controller_key]
            hand_joints = outputs["hand_joints"]
            n = len(self._hand_joint_names)
            if controller.is_none:
                for i in range(n):
                    hand_joints[i] = float(grasp_open_value)
                return
            trigger = float(controller[ControllerInputIndex.TRIGGER_VALUE])
            target = grasp_close_value if trigger > trigger_threshold else grasp_open_value
            for i in range(n):
                hand_joints[i] = float(target)

    cfg = DexHandRetargeterConfig(hand_side=side, **config_kwargs)
    return DualSourceDexHandRetargeter(cfg, name=name)
