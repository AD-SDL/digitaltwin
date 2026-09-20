# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record SO101 leader-teleop episodes to an IsaacLab-style HDF5 file.

Runs entirely in the IsaacLab venv — **no lerobot import** (see
:mod:`rpl_centrifuge.teleop.so101_mapping` for why). The resulting HDF5 uses the
same ``data/demo_<i>/{actions, obs/<key>}`` layout the repo's other tools expect,
so ``scripts/convert_to_lerobot.py`` (run later in the lerobot venv) turns it into
a LeRobot dataset.

Per-frame the sim pushes real-arm-space state + action and one RGB image per
camera. Frames are buffered on-device during an episode and flushed to disk by a
background thread when recording stops (``S``/``R``); ``C`` cancels.

Layout written::

    data/                       (attrs: fps, task)
      demo_0/
        actions                 (T, 6)  float32  real-arm space
        obs/joint_pos           (T, 6)  float32  real-arm space (state)
        obs/table_cam           (T, H, W, 3) uint8
        obs/wrist_cam           (T, H, W, 3) uint8
      demo_1/ ...
"""

from __future__ import annotations

import os
import queue
import threading

import h5py
import numpy as np
import torch
from tqdm import tqdm


class HDF5Recorder:

    STOP_RECORDING_EVENT: str = "rpl_centrifuge_so101_teleop.stop_recording"
    CANCEL_RECORDING_EVENT: str = "rpl_centrifuge_so101_teleop.cancel_recording"

    def __init__(
        self,
        output_file: str,
        fps: int,
        device: str,
        cameras: dict,
        task: str,
        max_episode_seconds: int = 120,
    ):
        from carb.eventdispatcher import get_eventdispatcher

        self.output_file = output_file
        self.fps = fps
        self.device = device
        self.cameras = cameras
        self.task = task

        self.capacity = max_episode_seconds * self.fps
        self.current_frame = 0
        self.num_recorded_episodes = 0

        self.action_buffer = None
        self.state_buffer = None
        self.rgb_buffers: dict = {}

        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        # Open (or append to) the HDF5 file up front and note existing demos.
        mode = "a" if os.path.exists(output_file) else "w"
        self._h5 = h5py.File(output_file, mode)
        self._data_grp = self._h5.require_group("data")
        self._data_grp.attrs["fps"] = self.fps
        self._data_grp.attrs["task"] = self.task
        self._next_demo_index = len(self._data_grp.keys())
        if self._next_demo_index:
            print(f"[INFO]: Appending to existing HDF5 ({self._next_demo_index} demos) - {output_file}")
        else:
            print(f"[INFO]: New HDF5 dataset - {output_file}")

        self._file_lock = threading.Lock()
        self.episode_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()

        self._stop_sub = get_eventdispatcher().observe_event(
            observer_name="rpl_centrifuge_hdf5_stop_observer",
            event_name=self.STOP_RECORDING_EVENT,
            on_event=self.save_episode,
        )
        self._cancel_sub = get_eventdispatcher().observe_event(
            observer_name="rpl_centrifuge_hdf5_cancel_observer",
            event_name=self.CANCEL_RECORDING_EVENT,
            on_event=self.cancel_recording,
        )

        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

    # ------------------------------------------------------------------
    # Buffering (called from the sim loop)
    # ------------------------------------------------------------------
    def allocate_buffers(self):
        self.action_buffer = torch.zeros((self.capacity, 6), dtype=torch.float32, device=self.device)
        self.state_buffer = torch.zeros((self.capacity, 6), dtype=torch.float32, device=self.device)
        for name, cam in self.cameras.items():
            self.rgb_buffers[name] = torch.zeros(
                (self.capacity, cam["height"], cam["width"], 3), dtype=torch.uint8, device=self.device
            )

    def push_frame_to_buffer(self, action_real: torch.Tensor, state_real: torch.Tensor, rgb: dict):
        if self.state_buffer is None:
            self.allocate_buffers()
        if self.current_frame >= self.capacity:
            print(f"[INFO]: Episode buffer full ({self.capacity} frames); dropping frame.")
            return
        self.action_buffer[self.current_frame] = action_real.clone()
        self.state_buffer[self.current_frame] = state_real.clone()
        for name in self.cameras.keys():
            self.rgb_buffers[name][self.current_frame] = rgb[name].clone()
        self.current_frame += 1

    # ------------------------------------------------------------------
    # Event handlers (carb dispatcher)
    # ------------------------------------------------------------------
    def save_episode(self, event):
        if event.event_name != self.STOP_RECORDING_EVENT:
            return
        if self.state_buffer is None or self.current_frame == 0:
            print("[INFO]: Nothing recorded; skipping save.")
            self._reset_buffers()
            return
        n = self.current_frame
        episode = {
            "actions": self.action_buffer[:n].to("cpu").numpy().copy(),
            "joint_pos": self.state_buffer[:n].to("cpu").numpy().copy(),
            "rgb": {name: self.rgb_buffers[name][:n].to("cpu").numpy().copy() for name in self.cameras},
            "total_frames": n,
        }
        self.episode_queue.put(episode)
        print(f"[INFO]: Episode queued for writing ({n} frames).")
        self._reset_buffers()

    def cancel_recording(self, event):
        if event.event_name != self.CANCEL_RECORDING_EVENT:
            return
        print("[INFO]: Recording cancelled; buffers cleared.")
        self._reset_buffers()

    def _reset_buffers(self):
        self.action_buffer = None
        self.state_buffer = None
        self.rgb_buffers = {}
        self.current_frame = 0

    # ------------------------------------------------------------------
    # Background writer
    # ------------------------------------------------------------------
    def _writer_loop(self):
        while not self._stop_event.is_set():
            try:
                episode = self.episode_queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                self._write_episode(episode)
                self.num_recorded_episodes += 1
                print(f"[INFO]: Episode {self.num_recorded_episodes} written.")
                self.episode_queue.task_done()
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR]: HDF5 episode write failed: {exc}")

    def _write_episode(self, episode):
        with self._file_lock:
            name = f"demo_{self._next_demo_index}"
            self._next_demo_index += 1
            grp = self._data_grp.create_group(name)
            grp.attrs["num_samples"] = int(episode["total_frames"])
            grp.create_dataset("actions", data=episode["actions"], compression="gzip")
            obs = grp.create_group("obs")
            obs.create_dataset("joint_pos", data=episode["joint_pos"], compression="gzip")
            for cam_name, frames in episode["rgb"].items():
                for _ in tqdm(range(1), desc=f"write {name}/{cam_name}", leave=False):
                    obs.create_dataset(f"{cam_name}_cam", data=frames, compression="gzip")
            self._h5.flush()

    # ------------------------------------------------------------------
    def close(self):
        # Drain the queue, then stop the writer and close the file.
        self.episode_queue.join()
        self._stop_event.set()
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=2.0)
            self._writer_thread = None
        with self._file_lock:
            self._h5.flush()
            self._h5.close()
        print(f"[INFO]: HDF5 closed ({self.num_recorded_episodes} episode(s)) - {self.output_file}")
