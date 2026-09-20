# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render camera images for recorded centrifuge demos via exact state playback.

Why this exists
---------------
The VR teleop tasks are camera-free (a live camera observation crashes Kit under
the XR renderer), and the IsaacLab Mimic ``annotate``/``generate_dataset`` path
does not work for this contact-rich insertion (open-loop action re-simulation
diverges — the tube never re-inserts, so 0 episodes are exported). So we produce
images a third way that is exact and robust: **state playback**.

For each recorded demo we set the full recorded state at every timestep (no
physics re-simulation → no divergence) inside the camera-enabled *Visuomotor*
env, render headless, and copy the RGB frames into a new dataset alongside the
original states / actions / low-dim observations. The result is a 1:1 image
dataset for your exact demos.

Render sync note: ``env.reset_to(state)`` writes to PhysX, but the RTX camera
reads scene transforms from Fabric, which only syncs on a physics *step* (a bare
``sim.render()`` shows a stale frame). So after setting each state we pin the
robot joint targets to that pose (zero drift) and do a single
``sim.step(render=True)`` to flush transforms, then read the cameras.

Usage (headless; run inside the IsaacLab venv):

    HEADLESS=1 ./isaaclab.sh -p <task_package_root>/scripts/render_images_from_demos.py \
        --input_file ./datasets/centrifuge_openarm.hdf5 \
        --output_file ./datasets/centrifuge_images.hdf5

Requires the ``rpl_centrifuge`` package importable (installed, or on PYTHONPATH).
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Render camera images for recorded demos via state playback.")
parser.add_argument("--input_file", type=str, required=True, help="Recorded (camera-free) demo dataset.")
parser.add_argument("--output_file", type=str, required=True, help="Output dataset with camera images added.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Centrifuge-Bimanual-OpenArm-IK-Abs-Visuomotor-v0",
    help="Camera-enabled (Visuomotor) task whose scene matches the recorded demos.",
)
parser.add_argument(
    "--cameras",
    type=str,
    default="table_cam,left_wrist_cam,right_wrist_cam",
    help="Comma-separated camera sensor names to render into obs/<name>.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
# Cameras must render; force it on regardless of how the launcher was invoked.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import shutil

import gymnasium as gym
import h5py
import numpy as np
import torch

import rpl_centrifuge  # noqa: F401  (registers the gym IDs)
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_tasks.utils import parse_env_cfg


def main():
    camera_names = [c.strip() for c in args_cli.cameras.split(",") if c.strip()]

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    robot = env.scene["robot"]

    for cam in camera_names:
        if cam not in env.scene.sensors:
            raise KeyError(f"Camera '{cam}' not in task '{args_cli.task}' sensors: {list(env.scene.sensors.keys())}")

    # Read handler for the recorded demos.
    reader = HDF5DatasetFileHandler()
    reader.open(args_cli.input_file)
    episode_names = list(reader.get_episode_names())
    print(f"[render] {len(episode_names)} episodes in {args_cli.input_file}; cameras={camera_names}")

    # Start the output as a copy of the input (preserves states/actions/obs + all
    # metadata), then add the rendered image observations into each demo's obs group.
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.output_file)) or ".", exist_ok=True)
    shutil.copyfile(args_cli.input_file, args_cli.output_file)

    def render_state(state) -> dict[str, np.ndarray]:
        env.reset_to(state, None, is_relative=True)
        # pin joint targets to the set pose so the flush-step doesn't move the arm
        robot.set_joint_position_target(robot.data.joint_pos.clone())
        robot.write_data_to_sim()
        env.sim.step(render=True)  # PhysX -> Fabric -> RTX render
        out = {}
        for cam in camera_names:
            sensor = env.scene[cam]
            sensor.update(0.0, force_recompute=True)
            out[cam] = sensor.data.output["rgb"][0].detach().cpu().numpy().astype(np.uint8)
        return out

    with h5py.File(args_cli.output_file, "a") as fout:
        data_grp = fout["data"]
        for i, ep_name in enumerate(episode_names):
            episode = reader.load_episode(ep_name, env.device)
            env.sim.reset()
            env.reset_to(episode.get_initial_state(), None, is_relative=True)

            # collect one frame per recorded state
            frames: dict[str, list[np.ndarray]] = {cam: [] for cam in camera_names}
            n = 0
            while True:
                state = episode.get_next_state()
                if state is None:
                    break
                imgs = render_state(state)
                for cam in camera_names:
                    frames[cam].append(imgs[cam])
                n += 1

            # align to the recorded obs length (defensive: trim/skip if mismatched)
            obs_grp = data_grp[ep_name]["obs"]
            ref_len = obs_grp["joint_pos"].shape[0] if "joint_pos" in obs_grp else n
            for cam in camera_names:
                arr = np.asarray(frames[cam])  # (n, H, W, 3)
                if arr.shape[0] != ref_len:
                    print(f"[render] {ep_name}: {cam} frames={arr.shape[0]} vs obs_len={ref_len} — aligning")
                    arr = arr[:ref_len]
                obs_grp.create_dataset(cam, data=arr, compression="gzip", compression_opts=4)
            print(f"[render] {ep_name} ({i + 1}/{len(episode_names)}): wrote {ref_len} frames for {camera_names}")

    reader.close()
    print(f"[render] done -> {args_cli.output_file}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
