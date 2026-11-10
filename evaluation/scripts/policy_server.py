# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Serve a trained LeRobot policy (pi05 / smolvla / groot) over a loopback socket.

Runs in the **lerobot venv** -- it needs transformers, which the IsaacLab venv
cannot import (see ``eval_wire`` for the why). The sim-side client is
``eval_rollouts.py``.

The server owns all policy-side timing. It reports, per request:

``forward_ms``     wall time of ``policy.select_action`` alone
``pre_ms``         observation -> tensors -> preprocessor pipeline
``post_ms``        postprocessor pipeline (unnormalize etc.)
``gpu_ms``         device time for the whole inference, via CUDA events
``replan``         True when this step ran a fresh action-chunk inference
                   rather than popping a cached step from the chunk queue

That ``replan`` flag matters for action-chunking policies: pi05 with
``n_action_steps=50`` pays a ~160 ms chunk inference once every 50 control
steps and ~5 ms on the other 49, so a single mean latency would describe
neither. Reporting the two populations separately is the whole point.

Example::

    ~/repo/lerobot/.venv/bin/python -I \
      evaluation/scripts/policy_server.py \
      --checkpoint /data1/yongho/models/242000/pretrained_model \
      --device cuda:1 --port 5557
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_wire import DEFAULT_PORT, recv_msg, send_msg  # noqa: E402

# Matches the LeRobot feature names written by convert_to_lerobot_parallel.py.
STATE_KEY = "observation.state"
IMAGE_KEYS = (
    "observation.images.chest",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)


def _build_policy(args):
    """Load the checkpoint plus its pre/post-processor pipelines."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    ckpt = str(Path(args.checkpoint).expanduser())

    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.device = args.device
    # Replan horizon and flow-matching denoise steps are the two knobs worth
    # sweeping at eval time; both change latency AND behaviour, so they are
    # recorded into the run manifest by the client.
    if args.n_action_steps is not None:
        cfg.n_action_steps = args.n_action_steps
    if args.num_inference_steps is not None and hasattr(cfg, "num_inference_steps"):
        cfg.num_inference_steps = args.num_inference_steps

    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(ckpt, config=cfg)
    policy.eval()
    policy.to(args.device)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=ckpt,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    return cfg, policy, preprocessor, postprocessor


def _handle_conn(conn: socket.socket, state: dict, args) -> None:
    from lerobot.policies.utils import prepare_observation_for_inference

    cfg = state["cfg"]
    policy = state["policy"]
    pre = state["pre"]
    post = state["post"]
    device = torch.device(args.device)
    is_cuda = device.type == "cuda"

    while True:
        try:
            meta, arrays = recv_msg(conn)
        except (ConnectionError, OSError):
            return

        kind = meta.get("kind")

        if kind == "ping":
            send_msg(conn, {"kind": "pong", "policy_type": cfg.type,
                            "checkpoint": str(args.checkpoint),
                            "n_action_steps": int(cfg.n_action_steps),
                            "chunk_size": int(getattr(cfg, "chunk_size", cfg.n_action_steps)),
                            "num_inference_steps": int(getattr(cfg, "num_inference_steps", -1)),
                            "device": str(args.device)})
            continue

        if kind == "reset":
            # Drop the cached action chunk so no episode inherits the previous
            # episode's plan. Without this the first ~50 steps of every episode
            # after the first replay a stale chunk.
            policy.reset()

            # Seed the global torch RNG per episode. Flow matching draws its
            # x_1 sample through `torch.normal` with no explicit generator
            # (lerobot/policies/common/flow_matching.py::sample_noise), so
            # without this two runs over identical scene seeds still diverge --
            # observed as the same episode reaching 42.6 deg on one run and
            # 51.6 deg on the next. Seeding here makes episode i reproducible
            # while still giving each episode a different noise draw.
            seed = meta.get("seed")
            if seed is not None and not args.no_policy_seed:
                torch.manual_seed(int(seed))
                if is_cuda:
                    torch.cuda.manual_seed_all(int(seed))

            if is_cuda:
                torch.cuda.reset_peak_memory_stats(device)
            send_msg(conn, {"kind": "reset_ok", "seeded": seed if not args.no_policy_seed else None})
            continue

        if kind == "shutdown":
            send_msg(conn, {"kind": "bye"})
            state["stop"].set()
            return

        if kind != "observe":
            send_msg(conn, {"kind": "error", "error": f"unknown kind {kind!r}"})
            continue

        # ---- inference -------------------------------------------------
        obs = {STATE_KEY: arrays[STATE_KEY].astype(np.float32)}
        for key in IMAGE_KEYS:
            if key in arrays:
                obs[key] = arrays[key]

        # A fresh chunk is computed exactly when the queue is empty.
        queue = getattr(policy, "_action_queue", None)
        replan = queue is None or len(queue) == 0

        if is_cuda:
            ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        t_start = time.perf_counter()
        with torch.inference_mode():
            if is_cuda:
                ev0.record()
            tensor_obs = prepare_observation_for_inference(
                obs, device, meta.get("task", ""), meta.get("robot_type", "")
            )
            tensor_obs = pre(tensor_obs)
            t_pre = time.perf_counter()

            action = policy.select_action(tensor_obs)
            t_fwd = time.perf_counter()

            action = post(action)
            if is_cuda:
                ev1.record()
        action_np = action.detach().to("cpu", torch.float32).numpy()
        t_end = time.perf_counter()

        gpu_ms = -1.0
        if is_cuda:
            torch.cuda.synchronize(device)
            gpu_ms = ev0.elapsed_time(ev1)

        timings = {
            "kind": "action",
            "replan": bool(replan),
            "pre_ms": 1e3 * (t_pre - t_start),
            "forward_ms": 1e3 * (t_fwd - t_pre),
            "post_ms": 1e3 * (t_end - t_fwd),
            "server_ms": 1e3 * (t_end - t_start),
            "gpu_ms": gpu_ms,
        }
        if is_cuda:
            timings["vram_alloc_gib"] = torch.cuda.memory_allocated(device) / 2**30
            timings["vram_peak_gib"] = torch.cuda.max_memory_allocated(device) / 2**30

        send_msg(conn, timings, {"action": action_np})


def main() -> int:
    p = argparse.ArgumentParser(description="Serve a LeRobot policy over loopback TCP.")
    p.add_argument("--checkpoint", required=True,
                   help="Path to the checkpoint's pretrained_model/ directory.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--n-action-steps", type=int, default=None,
                   help="Override the replan horizon (chunk steps consumed before re-inferring).")
    p.add_argument("--num-inference-steps", type=int, default=None,
                   help="Override flow-matching denoise steps (pi0/pi05 only).")
    p.add_argument("--no-policy-seed", action="store_true",
                   help="Do not seed the torch RNG per episode. Flow-matching noise then "
                        "varies run to run, so rollouts stop being reproducible.")
    p.add_argument("--ready-file", default=None,
                   help="Touch this path once the model is loaded and the socket is listening.")
    args = p.parse_args()

    print(f"[server] loading {args.checkpoint} onto {args.device} ...", flush=True)
    t0 = time.time()
    cfg, policy, pre, post = _build_policy(args)
    load_s = time.time() - t0
    print(f"[server] loaded {cfg.type} in {load_s:.1f}s "
          f"(n_action_steps={cfg.n_action_steps}, "
          f"num_inference_steps={getattr(cfg, 'num_inference_steps', 'n/a')})", flush=True)

    state = {"cfg": cfg, "policy": policy, "pre": pre, "post": post, "stop": threading.Event()}

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    srv.settimeout(1.0)
    print(f"[server] listening on {args.host}:{args.port}", flush=True)

    if args.ready_file:
        ready = Path(args.ready_file)
        ready.parent.mkdir(parents=True, exist_ok=True)
        ready.write_text(json.dumps({
            "policy_type": cfg.type,
            "device": args.device,
            "checkpoint": str(args.checkpoint),
            "load_s": load_s,
            "n_action_steps": int(cfg.n_action_steps),
            "num_inference_steps": int(getattr(cfg, "num_inference_steps", -1)),
        }))

    try:
        while not state["stop"].is_set():
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"[server] client {addr} connected", flush=True)
            _handle_conn(conn, state, args)
            conn.close()
            print("[server] client disconnected", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        srv.close()
        if args.ready_file:
            Path(args.ready_file).unlink(missing_ok=True)
    print("[server] exit", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
