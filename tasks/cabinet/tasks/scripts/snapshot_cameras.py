# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render one still frame from each camera of the OpenArm cabinet visuomotor task.

Use this to check camera framing (chest / left wrist / right wrist) after
editing the mount offsets in ``visuomotor_env_cfg.py`` -- it is a lot cheaper
than running a full dataset generation pass just to look at a picture.

Runs headless with cameras enabled, resets the env, lets the RTX renderer
converge on the reset pose, then writes one PNG per camera plus a
side-by-side contact sheet. Also prints the world pose of the
robot body link and of every camera, which is what you need to reason about
a wrong rotation.

Example::

    cd <isaaclab_root>
    ./isaaclab.sh -p <guide_repo_root>/tasks/cabinet/tasks/scripts/snapshot_cameras.py \
        --out-dir /tmp/cabinet_cams
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Snapshot the cabinet task cameras.")
parser.add_argument("--task", default="Isaac-OpenDoor-Bimanual-OpenArm-IK-Abs-Visuomotor-v0")
parser.add_argument("--out-dir", default="/tmp/cabinet_cams")
parser.add_argument("--settle-steps", type=int, default=5,
                    help="Extra render passes after reset before grabbing the frame.")
parser.add_argument("--tag", default="", help="Suffix appended to every output filename.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.num_envs = 1
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

import rpl_cabinet  # noqa: F401,E402  (registers the gym ids)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

CAMERAS = ("chest_cam", "left_wrist_cam", "right_wrist_cam")


def _quat_to_rpy_deg(q):
    """xyzw quaternion -> roll/pitch/yaw degrees (ZYX intrinsic).

    Isaac Lab 3.0 quaternions are (x, y, z, w) -- including
    ``CameraCfg.OffsetCfg.rot`` and ``Camera.data.quat_w_world``.
    """
    x, y, z, w = (float(v) for v in q)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return tuple(round(float(np.degrees(v)), 2) for v in (roll, pitch, yaw))


def main():
    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args_cli.tag}" if args_cli.tag else ""

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    env.reset()
    # Don't step the policy: a zero IK-abs action would command both EEs to the
    # base-frame origin and fold the arms. Just advance the renderer in place so
    # the RTX path converges on the reset pose, then update the sensors.
    for _ in range(args_cli.settle_steps):
        env.sim.render()
    env.scene.update(dt=env.physics_dt)

    body = env.scene["robot"].data
    body_idx = env.scene["robot"].body_names.index("openarm_body_link")
    print("\n=== poses (world) ===")
    print(f"openarm_body_link pos={np.round(body.body_pos_w[0, body_idx].cpu().numpy(), 4)} "
          f"quat_xyzw={np.round(body.body_quat_w[0, body_idx].cpu().numpy(), 4)} "
          f"rpy_deg={_quat_to_rpy_deg(body.body_quat_w[0, body_idx].cpu().numpy())}")

    from PIL import Image

    panes = []
    for name in CAMERAS:
        cam = env.scene[name]
        pos = cam.data.pos_w[0].cpu().numpy()
        quat = cam.data.quat_w_world[0].cpu().numpy()
        print(f"{name:16s} pos={np.round(pos, 4)} quat_xyzw={np.round(quat, 4)} "
              f"rpy_deg={_quat_to_rpy_deg(quat)}")

        rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        path = out_dir / f"{name}{suffix}.png"
        Image.fromarray(rgb).save(path)
        print(f"  -> {path}  shape={rgb.shape}")
        panes.append((name, rgb))

    # contact sheet: all three scaled to a common height, laid out left to right
    height = max(p.shape[0] for _, p in panes)
    scaled = []
    for name, rgb in panes:
        img = Image.fromarray(rgb)
        if img.height != height:
            img = img.resize((round(img.width * height / img.height), height))
        scaled.append((name, img))
    sheet = Image.new("RGB", (sum(i.width for _, i in scaled), height), (20, 20, 20))
    x = 0
    for _, img in scaled:
        sheet.paste(img, (x, 0))
        x += img.width
    sheet_path = out_dir / f"contact_sheet{suffix}.png"
    sheet.save(sheet_path)
    print(f"\ncontact sheet -> {sheet_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
