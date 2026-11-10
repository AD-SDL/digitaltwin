# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Roll a trained VLA policy out on the OpenArm cabinet task and score it.

Runs in the **IsaacLab venv**, headless with cameras on. The policy itself
lives behind ``policy_server.py`` in the lerobot venv (the two venvs cannot be
merged -- see ``eval_wire``).

Writes one JSON-lines record per episode plus a run manifest; use
``summarize_eval.py`` to turn those into a report.

IMPORTANT -- episode horizon. The task cfg ships ``episode_length_s = 40.0``
(800 steps at 20 Hz) but the recorded demos reach the success condition at
step ~1018 (p50) and ~1384 (max), i.e. 51-69 s. Evaluating at the cfg default
would truncate every episode before the door could be opened and report 0%
regardless of policy quality. This script therefore defaults to
``--episode-length-s 80`` and records the value it used.

Example::

    cd <isaaclab_root>
    HEADLESS=1 ./isaaclab.sh -p \
      <guide_repo_root>/evaluation/scripts/eval_rollouts.py \
      --num-episodes 50 --out-dir <guide_repo_root>/evaluation/artifacts/pi05_242000
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# CLI + AppLauncher must come before any isaaclab / omni import.
# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Rollout eval for the OpenArm cabinet task.")
parser.add_argument("--task", default="Isaac-OpenDoor-Bimanual-OpenArm-IK-Abs-Visuomotor-v0")
parser.add_argument("--num-episodes", type=int, default=50)
parser.add_argument("--episode-length-s", type=float, default=80.0,
                    help="Per-episode horizon. Demos succeed at 51 s (p50) / 69 s (max); "
                         "the task cfg default of 40 s truncates every episode. Keep >= 70.")
parser.add_argument("--seed", type=int, default=10000, help="Base seed; episode i uses seed+i.")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=5557)
parser.add_argument("--out-dir", required=True)
parser.add_argument("--task-prompt",
                    default="Open the cabinet door by pulling its handle with the right gripper.",
                    help="Language instruction. MUST match the dataset's task string.")
parser.add_argument("--robot-type", default="openarm_bimanual")
parser.add_argument("--no-normalize-quat", action="store_true",
                    help="Feed the policy's raw quaternion blocks to IK without renormalizing.")
parser.add_argument("--cabinet-jitter", default=None, metavar="X,Y,YAW",
                    help="Override reset pose jitter half-ranges, e.g. '0.05,0.05,0.12'. "
                         "Default keeps the task cfg's +/-0.02 m, +/-0.05 rad.")
parser.add_argument("--gpu-sample-hz", type=float, default=2.0,
                    help="nvidia-smi sampling rate for the utilization/power trace (0 disables).")
parser.add_argument("--video-episodes", type=int, default=0,
                    help="Write an mp4 of the chest camera for the first N episodes.")
parser.add_argument("--door-trace-every", type=int, default=20,
                    help="Sample the door angle every N control steps into the episode record "
                         "(20 = 1 Hz). 0 disables. Lets you tell a stalled episode from one "
                         "still making progress when the horizon ran out.")
parser.add_argument("--video-layout", choices=("composite", "separate"), default="composite",
                    help="composite: one 1280x960 mp4 tiling chest + both wrists. "
                         "separate: one mp4 per camera at native resolution.")
parser.add_argument("--no-video-labels", action="store_true",
                    help="Skip the per-pane camera captions in composite videos.")
parser.add_argument("--shutdown-server", action="store_true",
                    help="Tell the policy server to exit when the run finishes.")

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
# Rollout bookkeeping below assumes one episode at a time: the policy holds a
# single action-chunk queue, so parallel envs would share (and corrupt) each
# other's plan. See the README for what a batched variant would need.
args_cli.num_envs = 1

# enable_cameras is set on args_cli above; passing it as a kwarg too makes
# AppLauncher raise on the duplicate.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import rpl_cabinet  # noqa: F401,E402  -- registers the gym ids
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_wire import recv_msg, send_msg  # noqa: E402
from gpu_trace import GpuSampler  # noqa: E402

STATE_TERM = "joint_pos"
CAMERA_TERMS = {
    "chest_cam": "observation.images.chest",
    "left_wrist_cam": "observation.images.left_wrist",
    "right_wrist_cam": "observation.images.right_wrist",
}
QUAT_SLICES = (slice(3, 7), slice(11, 15))  # left / right arm quaternion blocks


def _to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _pack_observation(obs: dict) -> dict[str, np.ndarray]:
    """Pull the four policy inputs out of the IsaacLab observation dict.

    ``concatenate_terms=False`` on the policy group, so ``obs["policy"]`` is a
    dict of named terms rather than one flat tensor.
    """
    policy_obs = obs["policy"]

    state = _to_numpy(policy_obs[STATE_TERM])[0].astype(np.float32)
    packed = {"observation.state": state}

    for term, wire_key in CAMERA_TERMS.items():
        img = _to_numpy(policy_obs[term])[0]
        if img.dtype != np.uint8:
            # Isaac returns float RGB in [0,1] if the obs term normalizes;
            # the dataset stored uint8, so match that.
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        packed[wire_key] = np.ascontiguousarray(img)

    return packed


def _normalize_quats(action: np.ndarray) -> np.ndarray:
    """Renormalize the two quaternion blocks in place-ish.

    The recorded demo actions have |q| in [0.991, 1.010] and the policy
    regresses them without a unit constraint, so the raw output drifts further.
    Normalization is order-agnostic, which matters here: the LeRobot feature
    names say ``quat_x..quat_w`` but the values are IsaacLab's ``wxyz``
    (demo frame 0 is [0.9975, -0.009, -0.001, 0.0001] = identity in wxyz).
    The 16-D vector is otherwise passed through verbatim so it round-trips
    into the same action space the demos were recorded in.
    """
    action = action.copy()
    for sl in QUAT_SLICES:
        q = action[sl]
        n = np.linalg.norm(q)
        if n > 1e-6:
            action[sl] = q / n
    return action


class PolicyClient:
    """Thin blocking client for ``policy_server.py``."""

    def __init__(self, host: str, port: int, task: str, robot_type: str):
        self.sock = socket.create_connection((host, port), timeout=600)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.task = task
        self.robot_type = robot_type

    def ping(self) -> dict:
        send_msg(self.sock, {"kind": "ping"})
        meta, _ = recv_msg(self.sock)
        return meta

    def reset(self, seed: int | None = None) -> None:
        """Clear the action-chunk queue and reseed the policy's noise stream."""
        send_msg(self.sock, {"kind": "reset", "seed": seed})
        recv_msg(self.sock)

    def act(self, packed: dict[str, np.ndarray]) -> tuple[np.ndarray, dict, float]:
        t0 = time.perf_counter()
        send_msg(self.sock, {"kind": "observe", "task": self.task, "robot_type": self.robot_type}, packed)
        meta, arrays = recv_msg(self.sock)
        rtt_ms = 1e3 * (time.perf_counter() - t0)
        return arrays["action"].reshape(-1), meta, rtt_ms

    def close(self, shutdown_server: bool = False) -> None:
        """Close the socket, and only stop the server if explicitly asked.

        Leaving the server up by default means a second eval (a different
        jitter, a different seed block) reuses the already-resident weights
        instead of paying the ~70 s load again. ``run_eval.sh`` owns the
        server lifetime via its EXIT trap.
        """
        try:
            if shutdown_server:
                send_msg(self.sock, {"kind": "shutdown"})
                recv_msg(self.sock)
        except Exception:
            pass
        self.sock.close()


def main() -> int:
    out_dir = Path(args_cli.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- env ------------------------------------------------------------
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.episode_length_s = float(args_cli.episode_length_s)
    env_cfg.seed = int(args_cli.seed)

    if args_cli.cabinet_jitter:
        jx, jy, jyaw = (float(v) for v in args_cli.cabinet_jitter.split(","))
        env_cfg.events.randomize_cabinet.params["pose_range"] = {
            "x": (-jx, jx), "y": (-jy, jy), "yaw": (-jyaw, jyaw),
        }

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    step_dt = float(unwrapped.step_dt)
    max_steps = int(round(env_cfg.episode_length_s / step_dt))
    print(f"[eval] task={args_cli.task} step_dt={step_dt:.4f}s "
          f"({1 / step_dt:.1f} Hz) horizon={args_cli.episode_length_s}s = {max_steps} steps", flush=True)

    cabinet = unwrapped.scene["cabinet"]
    door_idx, _ = cabinet.find_joints(["hinge_left", "hinge_right"])

    # -- policy ---------------------------------------------------------
    client = PolicyClient(args_cli.host, args_cli.port, args_cli.task_prompt, args_cli.robot_type)
    server_info = client.ping()
    print(f"[eval] policy server: {server_info}", flush=True)

    sampler = GpuSampler(hz=args_cli.gpu_sample_hz, out_path=out_dir / "gpu_trace.csv")
    sampler.start()

    manifest = {
        "task": args_cli.task,
        "task_prompt": args_cli.task_prompt,
        "robot_type": args_cli.robot_type,
        "num_episodes": args_cli.num_episodes,
        "episode_length_s": args_cli.episode_length_s,
        "max_steps": max_steps,
        "control_hz": 1.0 / step_dt,
        "base_seed": args_cli.seed,
        "normalize_quat": not args_cli.no_normalize_quat,
        "cabinet_jitter": args_cli.cabinet_jitter or "task-default (+/-0.02 m, +/-0.05 rad)",
        "sim_device": str(args_cli.device),
        "server": server_info,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    episodes_path = out_dir / "episodes.jsonl"
    episodes_path.unlink(missing_ok=True)

    video_dir = out_dir / "videos"
    if args_cli.video_episodes > 0:
        video_dir.mkdir(exist_ok=True)

    run_t0 = time.time()
    n_success = 0

    for ep in range(args_cli.num_episodes):
        # reset(seed=) seeds torch/numpy right before _reset_idx runs the
        # reset events, so the cabinet pose jitter is reproducible per episode.
        seed = args_cli.seed + ep
        obs, _ = env.reset(seed=seed)
        # Same seed to both sides: the env seeds its reset events, the server
        # seeds flow-matching noise. Fixing only the env leaves the rollout
        # non-reproducible.
        client.reset(seed=seed)

        ep_t0 = time.time()
        rtts, server_ms, gpu_ms, forward_ms, pre_ms, post_ms = [], [], [], [], [], []
        replan_rtt, cached_rtt = [], []
        env_ms, pack_ms = [], []
        max_door = 0.0
        max_door_step = 0
        door_trace: list[float] = []
        grasped = False
        success = False
        success_step = -1
        recorder = (
            VideoRecorder(
                video_dir / f"episode_{ep:03d}.mp4",
                fps=1.0 / step_dt,
                separate=args_cli.video_layout == "separate",
                label=not args_cli.no_video_labels,
            )
            if ep < args_cli.video_episodes
            else None
        )

        for step in range(max_steps):
            t_pack = time.perf_counter()
            packed = _pack_observation(obs)
            pack_ms.append(1e3 * (time.perf_counter() - t_pack))

            if recorder is not None:
                recorder.add(packed)

            action_np, timing, rtt = client.act(packed)
            rtts.append(rtt)
            server_ms.append(timing["server_ms"])
            forward_ms.append(timing["forward_ms"])
            pre_ms.append(timing["pre_ms"])
            post_ms.append(timing["post_ms"])
            if timing.get("gpu_ms", -1) >= 0:
                gpu_ms.append(timing["gpu_ms"])
            (replan_rtt if timing["replan"] else cached_rtt).append(rtt)

            if not args_cli.no_normalize_quat:
                action_np = _normalize_quats(action_np)

            action = torch.as_tensor(action_np, dtype=torch.float32, device=unwrapped.device).unsqueeze(0)

            t_env = time.perf_counter()
            obs, _reward, terminated, truncated, _info = env.step(action)
            env_ms.append(1e3 * (time.perf_counter() - t_env))

            # Partial-progress signals. The success threshold (1.40 rad) sits
            # only ~0.16 rad below what the demos actually reach, so "how far
            # did the door get" separates near-misses from no-contact runs in
            # a way a binary success rate cannot.
            angles = cabinet.data.joint_pos[:, door_idx]
            door_now = float(torch.abs(angles).amax().item())
            if door_now > max_door:
                max_door = door_now
                max_door_step = step + 1
            if args_cli.door_trace_every and (step % args_cli.door_trace_every == 0):
                door_trace.append(round(np.degrees(door_now), 2))
            if "subtask_terms" in obs and "grasp_handle" in obs["subtask_terms"]:
                grasped = grasped or bool(_to_numpy(obs["subtask_terms"]["grasp_handle"]).any())

            done_success = bool(_to_numpy(terminated).any())
            if done_success and not success:
                success = True
                success_step = step + 1

            if done_success or bool(_to_numpy(truncated).any()):
                break

        steps_taken = step + 1
        n_success += int(success)
        wall_s = time.time() - ep_t0

        record = {
            "episode": ep,
            "seed": seed,
            "success": success,
            "success_step": success_step,
            "success_sim_s": success_step * step_dt if success_step > 0 else None,
            "steps": steps_taken,
            "sim_s": steps_taken * step_dt,
            "wall_s": wall_s,
            "realtime_factor": (steps_taken * step_dt) / wall_s if wall_s > 0 else None,
            "max_door_angle_rad": max_door,
            "max_door_angle_deg": float(np.degrees(max_door)),
            "max_door_angle_step": max_door_step,
            # Fraction of the episode that elapsed after the door stopped
            # improving. Near 0 => still progressing when time ran out (more
            # horizon might help). Near 1 => stalled long ago (it will not).
            "stalled_fraction": (1.0 - max_door_step / steps_taken) if steps_taken else None,
            "door_angle_trace_deg": door_trace,
            "door_trace_hz": (1.0 / step_dt / args_cli.door_trace_every) if args_cli.door_trace_every else None,
            "grasped_handle": grasped,
            "latency": {
                "rtt_ms": _stats(rtts),
                "rtt_replan_ms": _stats(replan_rtt),
                "rtt_cached_ms": _stats(cached_rtt),
                "server_ms": _stats(server_ms),
                "forward_ms": _stats(forward_ms),
                "pre_ms": _stats(pre_ms),
                "post_ms": _stats(post_ms),
                "gpu_ms": _stats(gpu_ms),
                "env_step_ms": _stats(env_ms),
                "obs_pack_ms": _stats(pack_ms),
                "n_replans": len(replan_rtt),
            },
        }
        with episodes_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

        if recorder is not None:
            recorder.close()

        print(f"[eval] ep {ep + 1}/{args_cli.num_episodes} "
              f"{'SUCCESS' if success else 'fail   '} "
              f"steps={steps_taken:4d} door={np.degrees(max_door):5.1f}deg "
              f"grasp={'y' if grasped else 'n'} "
              f"wall={wall_s:5.1f}s rtt_p50={record['latency']['rtt_ms']['p50']:.1f}ms "
              f"| running SR {n_success}/{ep + 1} = {100 * n_success / (ep + 1):.1f}%", flush=True)

    sampler.stop()
    client.close(shutdown_server=args_cli.shutdown_server)
    env.close()

    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest["total_wall_s"] = time.time() - run_t0
    manifest["success_rate"] = n_success / args_cli.num_episodes
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n[eval] {n_success}/{args_cli.num_episodes} successes "
          f"= {100 * n_success / args_cli.num_episodes:.1f}% in {manifest['total_wall_s'] / 60:.1f} min")
    print(f"[eval] wrote {episodes_path}")
    return 0


def _stats(values) -> dict:
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p90": None, "p99": None, "max": None}
    a = np.asarray(values, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


_LABEL_COLOR = (255, 255, 255)
_LABEL_BG = (0, 0, 0)


def _compose_frame(packed: dict[str, np.ndarray], label: bool = True) -> np.ndarray:
    """Tile the three camera views into one 1280x960 frame.

    Layout, chosen so nothing is resampled -- every pixel is the pixel the
    policy saw::

        +--------------------------------+
        |  pad |   chest 848x848 |  pad   |   1280 x 480
        +---------------+----------------+
        |  left_wrist   |  right_wrist   |   1280 x 480
        +---------------+----------------+

    The chest camera is 848 wide against the wrists' combined 1280, so it is
    letterboxed rather than stretched. Scaling it up would invent pixels and
    make the chest view look sharper than it is.
    """
    chest = packed["observation.images.chest"]
    left = packed["observation.images.left_wrist"]
    right = packed["observation.images.right_wrist"]

    bottom = np.concatenate([left, right], axis=1)          # 480 x 1280
    width = bottom.shape[1]

    pad_total = width - chest.shape[1]
    pad_l = max(0, pad_total // 2)
    pad_r = max(0, pad_total - pad_l)
    top = np.pad(chest, ((0, 0), (pad_l, pad_r), (0, 0)), mode="constant")
    if top.shape[1] != width:                                # chest wider than the pair
        top = top[:, :width]

    frame = np.concatenate([top, bottom], axis=0)           # 960 x 1280
    return _label_frame(frame, chest.shape[1], pad_l) if label else frame


def _label_frame(frame: np.ndarray, chest_w: int, pad_l: int) -> np.ndarray:
    """Caption each pane. Silently returns the frame unchanged without PIL."""
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return frame

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    h_top = frame.shape[0] // 2
    for text, (x, y) in (
        ("chest", (pad_l + 6, 6)),
        ("left_wrist", (6, h_top + 6)),
        ("right_wrist", (frame.shape[1] // 2 + 6, h_top + 6)),
    ):
        box = draw.textbbox((x, y), text)
        draw.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=_LABEL_BG)
        draw.text((x, y), text, fill=_LABEL_COLOR)
    return np.asarray(img)


class VideoRecorder:
    """Streaming mp4 writer -- one composite frame at a time.

    Buffering frames and encoding at the end would hold
    1600 steps x 1280 x 960 x 3 B = ~5.9 GB of RAM per episode, so frames go
    straight to the encoder instead.
    """

    def __init__(self, path: Path, fps: float, separate: bool = False, label: bool = True):
        self.path = Path(path)
        self.fps = fps
        self.separate = separate
        self.label = label
        self._writers: dict[str, object] = {}
        self._failed = False

    def _writer_for(self, key: str, shape):
        if key not in self._writers:
            import imageio.v3 as iio

            target = self.path if key == "_composite" else self.path.with_name(
                f"{self.path.stem}_{key}{self.path.suffix}"
            )
            w = iio.imopen(str(target), "w", plugin="pyav")
            w.init_video_stream("libx264", fps=self.fps)
            self._writers[key] = (w, target)
        return self._writers[key][0]

    def add(self, packed: dict[str, np.ndarray]) -> None:
        if self._failed:
            return
        try:
            if self.separate:
                for wire_key in CAMERA_TERMS.values():
                    short = wire_key.rsplit(".", 1)[-1]
                    self._writer_for(short, None).write_frame(
                        np.ascontiguousarray(packed[wire_key])
                    )
            else:
                frame = _compose_frame(packed, label=self.label)
                self._writer_for("_composite", frame.shape).write_frame(
                    np.ascontiguousarray(frame)
                )
        except Exception as exc:  # pragma: no cover - video is a convenience
            print(f"[eval] video capture disabled ({type(exc).__name__}: {exc})", flush=True)
            self._failed = True

    def close(self) -> None:
        for key, (writer, target) in self._writers.items():
            try:
                writer.close()
                print(f"[eval] wrote {target}", flush=True)
            except Exception as exc:  # pragma: no cover
                print(f"[eval] video close failed ({type(exc).__name__}: {exc})", flush=True)
        self._writers.clear()


if __name__ == "__main__":
    try:
        code = main()
    finally:
        simulation_app.close()
    raise SystemExit(code)
