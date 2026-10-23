# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Derived in part from NVIDIA's "Sim-to-Real-SO-101-Workshop"
# (source/sim_to_real_so101/scripts/lerobot_agent.py), Copyright (c) 2026 NVIDIA
# CORPORATION & AFFILIATES, Apache-2.0.

"""SO101 centrifuge teleop + HDF5 recording — single process, lerobot in-venv.

Replaces the earlier ``so101_leader_bridge.py`` + ``so101_leader_agent.py`` UDP
split: the physical SO101 leader is read **in-process** via lerobot's
``so_leader`` module, mapped into the sim SO101's joint radians, and stepped as
an absolute joint-position command. Observations/actions are recorded in
real-arm space (deg-normalized) so the HDF5 can be post-converted to a LeRobot
dataset by ``scripts/convert_to_lerobot.py``.

Why no UDP: the lerobot leader surface (``SO101LeaderConfig`` +
``make_teleoperator_from_config``) does **not** pull in ``transformers`` (I
guarded the import chain to confirm), so leader teleop runs cleanly in the
IsaacLab venv despite the ``huggingface-hub`` version conflict with
``transformers`` that would otherwise bite the policy-eval path.

Viewport keys: ``S`` start/stop episode, ``R`` reset (also stops), ``C`` cancel,
``P`` dump a snapshot of each scene camera to /tmp/sim_cam_<name>.png.

Example:

    export ISAACLAB_ROOT=~/repo/IsaacLab
    export TASK_PACKAGE_DIR=~/repo/digitaltwin/tasks/centrifuge/tasks
    cd "${ISAACLAB_ROOT}"
    ./isaaclab.sh -p "${TASK_PACKAGE_DIR}/scripts/so101_lerobot_agent.py" \
        --task Isaac-Centrifuge-SO101-LeRobot-Leader-v0 \
        --port /dev/ttyACM0 --robot_id leader_arm_1 \
        --output_file ./datasets/centrifuge_so101_leader.hdf5

Prerequisite: the leader arm must be calibrated once per ``--robot_id``. See
``docs/04b_collect_episodes_so101_leader.md``.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SO101 centrifuge leader teleop (in-process) + HDF5 record.")
parser.add_argument("--task", type=str, default="Isaac-Centrifuge-SO101-LeRobot-Leader-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--port", type=str, default="/dev/ttyACM0",
                    help="Serial port of the physical SO101 leader arm.")
parser.add_argument("--robot_id", type=str, default="leader_arm_1",
                    help="LeRobot calibration id (see docs/04b for the one-time calibrate step).")
parser.add_argument("--output_file", type=str, default=None,
                    help="HDF5 output path (enables recording). Omit for teleop-only.")
parser.add_argument("--task_name", type=str,
                    default="Pick the centrifuge tube from the rack and place it in the bucket.",
                    help="Language task string stored in the HDF5 (carried into the LeRobot dataset).")
parser.add_argument("--fps", type=int, default=20,
                    help="Record fps. SO101 base: 1/(sim.dt*decimation) = 1/0.05 = 20.")
parser.add_argument("--seed", type=int, default=101)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Interactive teleop: default to the Omniverse Kit GUI so the viewport + keyboard
# (S/R/C/P) work. IsaacLab 3.0 is headless unless a visualizer is requested.
if args_cli.visualizer is None:
    args_cli.visualizer = ["kit"]

# `--enable_cameras` was dropped as a CLI flag in IsaacLab 3.0; opt in via the
# AppLauncher kwarg (the LeRobot env expects live cameras).
app_launcher = AppLauncher(args_cli, enable_cameras=True)
simulation_app = app_launcher.app

"""Rest of the imports (after the app is live)."""

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import rpl_centrifuge  # noqa: F401  (self-registers gym ids)
from rpl_centrifuge.teleop.keyboard import KeyboardControl
from rpl_centrifuge.teleop.so101_mapping import SO101LeaderMapping, SO101_JOINT_ORDER

# Leader stream order used by lerobot's SO101 driver — positionally aligned with
# the env action term's joint order [Rotation, Pitch, Elbow, Wrist_Pitch,
# Wrist_Roll, Jaw] (via ``preserve_order=True`` in the env cfg).
_LEADER_ACTION_KEYS = [f"{j}.pos" for j in SO101_JOINT_ORDER]


def _connect_leader(port: str, robot_id: str):
    """Open the lerobot SO101 leader driver on the given serial port.

    Imported lazily so the AppLauncher setup is unaffected if lerobot is absent.
    This path uses only ``lerobot.teleoperators.so_leader`` and
    ``make_teleoperator_from_config`` — both verified not to touch ``transformers``.
    """
    from lerobot.teleoperators.so_leader import SO101LeaderConfig
    from lerobot.teleoperators.utils import make_teleoperator_from_config

    cfg = SO101LeaderConfig(port=port, id=robot_id, use_degrees=False)
    leader = make_teleoperator_from_config(cfg)
    leader.connect()  # requires prior `lerobot-calibrate` for this id
    print(f"[INFO]: Connected to SO101 leader on {port} (id={robot_id})")
    return leader


def _read_leader_action(leader, device: str) -> torch.Tensor:
    """Return the leader's current 6-joint action as a float32 tensor on ``device``.

    lerobot's SO101 ``get_action`` returns a dict ``{<motor>.pos: float}`` in
    normalized percent (``use_degrees=False``: arm ±100, gripper 0-100).
    """
    action = leader.get_action()
    return torch.tensor(
        [float(action[k]) for k in _LEADER_ACTION_KEYS],
        dtype=torch.float32,
        device=device,
    )


def main():
    keyboard_control = KeyboardControl()

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[INFO]: Action space: {env.action_space}")
    print("[INFO]: 'S' start/stop recording | 'R' reset (also stops) | 'C' cancel episode | 'P' snapshot")

    obs, _ = env.reset()
    device = env.unwrapped.device
    mapping = SO101LeaderMapping(device)

    # Point the Kit viewport at a scene camera on startup. In this Kit app
    # the default ``/OmniverseKit_Persp`` rendering is broken (gizmos draw but
    # mesh prims never render — a known RTX quirk), so we switch to a scene
    # camera that renders normally. Priority: ``camera_table`` (top-down
    # overview parked over the pickup zone) if present, else the first
    # ``camera_*`` scene key. Swallow if the viewport API isn't available
    # (headless, pre-viewport, etc.).
    try:
        from omni.kit.viewport.utility import get_active_viewport

        vp = get_active_viewport()
        if vp is not None:
            scene_cam_keys = [k for k in env.unwrapped.scene.keys() if k.startswith("camera_")]
            preferred = "camera_table" if "camera_table" in scene_cam_keys else (
                scene_cam_keys[0] if scene_cam_keys else None
            )
            if preferred is not None:
                vp.camera_path = f"/World/envs/env_0/{preferred}"
                print(f"[INFO]: Kit viewport switched to scene camera: {preferred}")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN]: could not switch viewport camera ({exc})")

    # Discover cameras (scene keys prefixed 'camera_').
    cameras = {}
    for key in env.unwrapped.scene.keys():
        if key.startswith("camera_"):
            cam_cfg = getattr(env.unwrapped.scene.cfg, key)
            cameras[key.replace("camera_", "")] = {"height": cam_cfg.height, "width": cam_cfg.width}
            print(f"[INFO]: Found camera: {key.replace('camera_', '')}")

    # Connect the physical leader (fail fast — nothing to record without it).
    leader = _connect_leader(args_cli.port, args_cli.robot_id)

    # Hold at the current sim pose until the first leader read lands (avoids a
    # snap on connect). ``obs["policy"]["joint_pos_obs"]`` is the 6-joint vector
    # emitted by the LeRobot env cfg.
    action_rad = obs["policy"]["joint_pos_obs"][0].clone()
    last_raw = mapping.sim_radians_to_real(action_rad).clone()
    actions = torch.zeros(env.action_space.shape, device=device)

    recorder = None
    if args_cli.output_file:
        if not cameras:
            print("[WARNING]: No cameras found; recording disabled.")
        else:
            from rpl_centrifuge.teleop.hdf5_recorder import HDF5Recorder

            try:
                recorder = HDF5Recorder(
                    output_file=args_cli.output_file,
                    fps=args_cli.fps,
                    device=device,
                    cameras=cameras,
                    task=args_cli.task_name,
                )
            except OSError as exc:
                print(f"[ERROR]: {exc}")
                try:
                    leader.disconnect()
                except Exception:  # noqa: BLE001
                    pass
                env.close()
                simulation_app.close()
                return
    else:
        print("[INFO]: No --output_file; teleop only (no recording).")

    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                # Read leader + map to sim radians. On a transient read failure
                # (occasional Feetech "Incorrect status packet"), hold the last
                # action so the arm doesn't flinch.
                try:
                    last_raw = _read_leader_action(leader, device)
                    action_rad = mapping.leader_to_sim_radians(last_raw)
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN]: leader read failed, holding last action ({exc})")

                actions[:] = action_rad
                obs, _, terminated, truncated, _ = env.step(actions)

                if keyboard_control.reset_world:
                    keyboard_control.reset_world = False
                    obs, _ = env.reset()
                    action_rad = obs["policy"]["joint_pos_obs"][0].clone()
                    continue

                # Auto save/discard on env termination (which also triggers an
                # auto-reset in the next step). The ``TerminationManager`` holds
                # per-term dones as a 2-D tensor and exposes ``get_term(name)``;
                # the 'success' term fires only when the tube has settled in a
                # bucket well for ``settle_time`` seconds (see env_cfg). Non-
                # success terminations (tube_dropped / time_out) are discarded.
                # Re-arm recording so the operator can chain demos back-to-back.
                if (terminated.any() or truncated.any()) and recorder is not None and keyboard_control.recording:
                    tm = env.unwrapped.termination_manager
                    active = set(tm.active_terms)
                    was_success = "success" in active and bool(tm.get_term("success")[0])
                    if was_success:
                        print(f"[INFO]: Task success -> auto-saving demo (total so far: {recorder.num_recorded_episodes + 1}).")
                        keyboard_control.stop_recording()
                    else:
                        fired = [n for n in active if bool(tm.get_term(n)[0])]
                        print(f"[INFO]: Episode ended without success ({', '.join(fired) or '?'}) -> discarding.")
                        keyboard_control.cancel_recording()
                    keyboard_control.start_recording()
                    # Skip pushing this frame — env has already auto-reset so
                    # obs is the FIRST frame of the next episode, not the last
                    # of the one that just ended.
                    action_rad = obs["policy"]["joint_pos_obs"][0].clone()
                    continue

                # 'P' key — dump each scene camera's current RGB to a known path.
                # Uses the obs tensors the renderer already populates each tick,
                # so no Kit capture extension is involved (which is broken in
                # IsaacLab 3.0's base kit app, see earlier debugging).
                if keyboard_control.snapshot_requested:
                    keyboard_control.snapshot_requested = False
                    visual_obs = obs.get("visual")
                    if visual_obs is None or not cameras:
                        print("[WARN]: no 'visual' obs group / no cameras — snapshot skipped")
                    else:
                        try:
                            from PIL import Image

                            for cam_name in cameras:
                                rgb = (
                                    visual_obs[f"rgb_{cam_name}"][0]
                                    .to(torch.uint8)
                                    .cpu()
                                    .numpy()
                                )
                                path = f"/tmp/sim_cam_{cam_name}.png"
                                Image.fromarray(rgb[..., :3]).save(path)
                                print(f"[INFO]: wrote {path}  shape={rgb.shape}")
                        except Exception as exc:  # noqa: BLE001
                            print(f"[WARN]: snapshot failed: {exc}")

                if recorder is not None and keyboard_control.recording:
                    visual_obs = obs.get("visual", None)
                    if visual_obs is None:
                        print("[WARNING]: No 'visual' obs group; cannot record.")
                        keyboard_control.recording = False
                        continue
                    state_real = mapping.sim_radians_to_real(obs["policy"]["joint_pos_obs"][0])
                    rgb = {name: visual_obs[f"rgb_{name}"][0].to(torch.uint8) for name in cameras}
                    recorder.push_frame_to_buffer(last_raw, state_real, rgb)
    finally:
        if recorder is not None:
            recorder.close()
        try:
            leader.disconnect()
        except Exception:  # noqa: BLE001
            pass
        env.close()


if __name__ == "__main__":
    main()
    while simulation_app.is_running():
        simulation_app.update()
    simulation_app.close()
