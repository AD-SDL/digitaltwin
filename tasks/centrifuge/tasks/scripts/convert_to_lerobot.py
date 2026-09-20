# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert a rendered centrifuge HDF5 dataset to a LeRobot dataset (v3.0).

Reads the image-augmented dataset produced by ``render_images_from_demos.py``
(states/actions/low-dim obs + ``obs/<camera>`` RGB streams) and writes a LeRobot
dataset via the official ``LeRobotDataset`` API (parquet for low-dim, MP4 for the
cameras, ``meta/*`` metadata) — the idiomatic Hugging Face robotics format.

Run this in an environment with ``lerobot`` (+ ``h5py``) — NOT the IsaacLab venv,
e.g. ``~/repo/lerobot/.venv`` (``pip install h5py`` there). It needs no Isaac Sim.

    <lerobot_venv>/bin/python <task_pkg>/scripts/convert_to_lerobot.py \
        --input_file ./datasets/centrifuge_images.hdf5 \
        --repo_id argonne-rpl/centrifuge-quest3-episodes \
        --root ./datasets/centrifuge_lerobot

For GR00T-N1.7 (which wants LeRobot v2), afterwards run Isaac-GR00T's
``scripts/lerobot_conversion/convert_v3_to_v2.py`` and add ``meta/modality.json``.

Note: LeRobot's ``LeRobotDataset`` in ``~/repo/lerobot`` is v3.0; GR00T needs v2.
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 16-D OpenArm IK-Abs action layout (see openarm/ik_abs_env_cfg.py).
_OPENARM_ACTION_NAMES = [
    "left_pos_x", "left_pos_y", "left_pos_z", "left_quat_x", "left_quat_y", "left_quat_z", "left_quat_w", "left_gripper",
    "right_pos_x", "right_pos_y", "right_pos_z", "right_quat_x", "right_quat_y", "right_quat_z", "right_quat_w", "right_gripper",
]
# camera obs key in the HDF5 -> LeRobot feature name
_CAMERA_MAP = {
    "table_cam": "observation.images.table",
    "left_wrist_cam": "observation.images.left_wrist",
    "right_wrist_cam": "observation.images.right_wrist",
}


def _sorted_demos(data_grp):
    return sorted(data_grp.keys(), key=lambda k: int(k.split("_")[1]))


def main():
    p = argparse.ArgumentParser(description="Convert rendered centrifuge HDF5 -> LeRobot v3.0 dataset.")
    p.add_argument("--input_file", required=True, help="Image-augmented HDF5 (from render_images_from_demos.py).")
    p.add_argument("--repo_id", required=True, help="LeRobot repo id, e.g. 'argonne-rpl/centrifuge-quest3-episodes'.")
    p.add_argument("--root", default=None, help="Local output dir (default $HF_LEROBOT_HOME/<repo_id>).")
    p.add_argument("--fps", type=int, default=20, help="Control fps (OpenArm: 1/(sim.dt*decimation)=1/0.05=20).")
    p.add_argument("--state_key", default="joint_pos", help="obs key to store as observation.state.")
    p.add_argument("--task", default="Pick the centrifuge tube from the rack and place it in the bucket.",
                   help="Language task description stored per frame.")
    p.add_argument("--robot_type", default="openarm_bimanual")
    p.add_argument("--push_to_hub", action="store_true", help="Push to the HF hub after building.")
    args = p.parse_args()

    fin = h5py.File(args.input_file, "r")
    data = fin["data"]
    demos = _sorted_demos(data)
    cams = [c for c in _CAMERA_MAP if c in data[demos[0]]["obs"]]
    if not cams:
        raise SystemExit(f"No camera obs {list(_CAMERA_MAP)} found in {args.input_file} — run render_images_from_demos.py first.")

    # Derive dims from the first demo.
    d0 = data[demos[0]]
    action_dim = d0["actions"].shape[1]
    state_dim = d0["obs"][args.state_key].shape[1]
    h, w, c = d0["obs"][cams[0]].shape[1:]
    action_names = _OPENARM_ACTION_NAMES if action_dim == len(_OPENARM_ACTION_NAMES) else [f"action_{i}" for i in range(action_dim)]

    features = {
        "action": {"dtype": "float32", "shape": (action_dim,), "names": action_names},
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": [f"{args.state_key}_{i}" for i in range(state_dim)]},
    }
    for cam in cams:
        features[_CAMERA_MAP[cam]] = {"dtype": "video", "shape": (h, w, c), "names": ["height", "width", "channels"]}

    print(f"[lerobot] repo={args.repo_id} fps={args.fps} demos={len(demos)} "
          f"action_dim={action_dim} state_dim={state_dim} cams={cams} img={(h, w, c)}")

    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=args.root,
        robot_type=args.robot_type,
        use_videos=True,
    )

    for i, name in enumerate(demos):
        ep = data[name]
        actions = ep["actions"][:].astype(np.float32)
        state = ep["obs"][args.state_key][:].astype(np.float32)
        cam_arrs = {cam: ep["obs"][cam] for cam in cams}
        T = actions.shape[0]
        for t in range(T):
            frame = {"action": actions[t], "observation.state": state[t], "task": args.task}
            for cam in cams:
                frame[_CAMERA_MAP[cam]] = cam_arrs[cam][t]  # HWC uint8
            ds.add_frame(frame)
        ds.save_episode()
        print(f"[lerobot] episode {i + 1}/{len(demos)} ({name}): {T} frames")

    ds.finalize()
    fin.close()
    print(f"[lerobot] wrote dataset to: {ds.root}")

    if args.push_to_hub:
        ds.push_to_hub()
        print(f"[lerobot] pushed to hub: {args.repo_id}")


if __name__ == "__main__":
    main()
