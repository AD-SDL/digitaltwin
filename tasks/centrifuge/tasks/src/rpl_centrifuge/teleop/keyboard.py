# SPDX-License-Identifier: Apache-2.0
#
# Derived from NVIDIA's "Sim-to-Real-SO-101-Workshop"
# (source/sim_to_real_so101/utils/keyboard.py), Copyright (c) 2026 NVIDIA
# CORPORATION & AFFILIATES, licensed under the Apache License, Version 2.0.

"""Keyboard control for LeRobot leader teleop + recording.

Key bindings (in the Kit viewport):
  * ``S`` — start / stop the current episode.
  * ``R`` — reset the world (also stops recording).
  * ``C`` — cancel the current episode (discard buffered frames).

Start/stop/cancel are broadcast as carb app events consumed by
:class:`~rpl_centrifuge.teleop.lerobot_recorder.LeRobotRecorder`. The event names
must match that recorder's constants.
"""

from __future__ import annotations


class KeyboardControl:

    START_RECORDING_EVENT: str = "rpl_centrifuge_so101_teleop.start_recording"
    STOP_RECORDING_EVENT: str = "rpl_centrifuge_so101_teleop.stop_recording"
    CANCEL_RECORDING_EVENT: str = "rpl_centrifuge_so101_teleop.cancel_recording"

    def __init__(self):
        import carb
        import omni.appwindow

        self.reset_world = False
        self.recording = False

        self._window = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._window.get_keyboard()
        self._key_press = carb.input.KeyboardEventType.KEY_PRESS
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_keyboard_event
        )

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type != self._key_press:
            return False

        if event.input.name == "R":
            self.reset_world = True
            self.stop_recording()
            print("[INFO]: Reset world...")
            return True
        if event.input.name == "S":
            if self.recording:
                self.stop_recording()
            else:
                self.start_recording()
            return True
        if event.input.name == "C":
            if self.recording:
                self.cancel_recording()
            return True
        return False

    def cleanup(self):
        if self._sub_keyboard:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_keyboard)
            self._sub_keyboard = None

    def start_recording(self):
        import omni.kit.app

        if not self.recording:
            self.recording = True
            print("[INFO]: Started recording.")
            omni.kit.app.queue_event(self.START_RECORDING_EVENT, payload={})

    def stop_recording(self):
        import omni.kit.app

        if self.recording:
            self.recording = False
            print("[INFO]: Stopped recording.")
            omni.kit.app.queue_event(self.STOP_RECORDING_EVENT, payload={})

    def cancel_recording(self):
        import omni.kit.app

        if self.recording:
            self.recording = False
            print("[INFO]: Cancelled recording.")
            omni.kit.app.queue_event(self.CANCEL_RECORDING_EVENT, payload={})
