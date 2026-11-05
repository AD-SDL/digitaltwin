# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parallel HDF5 -> LeRobot v3.0 converter for the OpenArm cabinet dataset.

Fans episodes out across ``--num_workers`` processes (each builds a per-shard
LeRobot dataset), then calls ``lerobot.datasets.aggregate.aggregate_datasets``
to stitch them into a single dataset under ``--root``. With
``--num_workers 1`` it degrades to a straight single-process converter, which
is useful for small datasets or when debugging.

Run in the lerobot venv (NOT the IsaacLab venv; no Isaac Sim needed). Example::

    <lerobot_root>/.venv/bin/python -u \
      <guide_repo_root>/tasks/cabinet/tasks/scripts/convert_to_lerobot_parallel.py \
      --input_file <isaaclab_root>/datasets/cabinet/openarm_vr_cabinet_images.hdf5 \
      --repo_id argonne-rpl/openarm-cabinet-vr-episodes \
      --root     <isaaclab_root>/datasets/cabinet/openarm_vr_cabinet_lerobot \
      --num_workers 8

Progress goes to stdout as ``[shard K] episode N/T ...`` lines.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np

# Delay LeRobot imports until child processes, so the parent doesn't pre-load
# ffmpeg/AV1 shared libs it never uses.

# 16-D OpenArm IK-Abs action layout.
_OPENARM_ACTION_NAMES = [
    "left_pos_x", "left_pos_y", "left_pos_z", "left_quat_x", "left_quat_y", "left_quat_z", "left_quat_w", "left_gripper",
    "right_pos_x", "right_pos_y", "right_pos_z", "right_quat_x", "right_quat_y", "right_quat_z", "right_quat_w", "right_gripper",
]
_CAMERA_MAP = {
    "chest_cam": "observation.images.chest",
    "left_wrist_cam": "observation.images.left_wrist",
    "right_wrist_cam": "observation.images.right_wrist",
}


def _sorted_demos(data_grp):
    return sorted(data_grp.keys(), key=lambda k: int(k.split("_")[1]))


def _shard_demos(demos: list[str], num_shards: int) -> list[list[str]]:
    """Even-ish round-robin split so slow/fast demos are spread across shards."""
    shards: list[list[str]] = [[] for _ in range(num_shards)]
    for i, name in enumerate(demos):
        shards[i % num_shards].append(name)
    return [s for s in shards if s]


def _worker(args):
    """Build a LeRobot dataset for one shard. Runs in a child process."""
    (
        shard_idx,
        shard_demos,
        input_file,
        shard_root,
        repo_id,
        fps,
        state_key,
        task,
        robot_type,
        vcodec,
        crf,
    ) = args

    from lerobot.configs import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    tag = f"shard {shard_idx}"
    t0 = time.time()
    print(f"[{tag}] start -- {len(shard_demos)} demos -> {shard_root}", flush=True)

    with h5py.File(input_file, "r") as fin:
        data = fin["data"]
        d0 = data[shard_demos[0]]
        action_dim = d0["actions"].shape[1]
        state_dim = d0["obs"][state_key].shape[1]
        cams = [c for c in _CAMERA_MAP if c in d0["obs"]]
        action_names = (
            _OPENARM_ACTION_NAMES if action_dim == len(_OPENARM_ACTION_NAMES)
            else [f"action_{i}" for i in range(action_dim)]
        )

        features = {
            "action": {"dtype": "float32", "shape": (action_dim,), "names": action_names},
            "observation.state": {
                "dtype": "float32", "shape": (state_dim,),
                "names": [f"{state_key}_{i}" for i in range(state_dim)],
            },
        }
        for cam in cams:
            ch, cw, cc = d0["obs"][cam].shape[1:]
            features[_CAMERA_MAP[cam]] = {
                "dtype": "video", "shape": (ch, cw, cc),
                "names": ["height", "width", "channels"],
            }

        rgb_encoder = RGBEncoderConfig(vcodec=vcodec, crf=crf)
        ds = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=shard_root,
            robot_type=robot_type,
            use_videos=True,
            streaming_encoding=True,
            rgb_encoder=rgb_encoder,
        )

        for i, name in enumerate(shard_demos):
            ep = data[name]
            actions = ep["actions"][:].astype(np.float32)
            state = ep["obs"][state_key][:].astype(np.float32)
            cam_arrs = {cam: ep["obs"][cam] for cam in cams}
            T = actions.shape[0]
            for t in range(T):
                frame = {"action": actions[t], "observation.state": state[t], "task": task}
                for cam in cams:
                    frame[_CAMERA_MAP[cam]] = cam_arrs[cam][t]
                ds.add_frame(frame)
            ds.save_episode()
            print(
                f"[{tag}] episode {i + 1}/{len(shard_demos)} ({name}): {T} frames "
                f"({time.time() - t0:.1f}s elapsed)",
                flush=True,
            )

        ds.finalize()

    print(f"[{tag}] done in {time.time() - t0:.1f}s -> {ds.root}", flush=True)
    return shard_idx, str(ds.root)


def main():
    p = argparse.ArgumentParser(description="Parallel OpenArm cabinet HDF5 -> LeRobot v3.0.")
    p.add_argument("--input_file", required=True)
    p.add_argument("--repo_id", required=True)
    p.add_argument("--root", default=None, help="Final aggregated dataset dir.")
    p.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 8) // 4),
                   help="Number of parallel shards (default: cpu_count/4).")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--state_key", default="joint_pos")
    p.add_argument("--task", default="Open the cabinet door by pulling its handle with the right gripper.")
    p.add_argument("--robot_type", default="openarm_bimanual")
    p.add_argument("--vcodec", default="h264")
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--shard_root", default=None,
                   help="Where to keep per-shard datasets (default: <root>/_shards).")
    p.add_argument("--keep_shards", action="store_true",
                   help="Do not delete per-shard directories after aggregation.")
    args = p.parse_args()

    input_file = str(Path(args.input_file).expanduser())
    with h5py.File(input_file, "r") as fin:
        demos = _sorted_demos(fin["data"])
    print(f"[parallel] found {len(demos)} demos in {input_file}", flush=True)

    num_workers = max(1, min(args.num_workers, len(demos)))
    shards = _shard_demos(demos, num_workers)
    num_workers = len(shards)
    print(f"[parallel] using {num_workers} workers "
          f"(shard sizes: {[len(s) for s in shards]})", flush=True)

    final_root = Path(args.root).expanduser() if args.root else None
    # Keep shards OUTSIDE final_root: aggregate_datasets() creates final_root
    # with mkdir(exist_ok=False), which would fail if we nested shards inside it.
    shard_root = Path(args.shard_root).expanduser() if args.shard_root else (
        (final_root.parent / f".{final_root.name}_shards") if final_root
        else Path(tempfile.mkdtemp(prefix="lerobot_shards_"))
    )
    if final_root is not None and final_root.exists():
        raise SystemExit(
            f"[parallel] --root {final_root} already exists -- remove it first "
            "(aggregation step needs a fresh dir)."
        )
    shard_root.mkdir(parents=True, exist_ok=True)
    print(f"[parallel] shard root: {shard_root}", flush=True)
    print(f"[parallel] final root: {final_root}", flush=True)

    job_args = []
    shard_repo_ids = []
    shard_roots = []
    for k, s in enumerate(shards):
        sid = f"{args.repo_id}__shard{k:02d}"
        sroot = shard_root / f"shard_{k:02d}"
        if sroot.exists():
            shutil.rmtree(sroot)
        shard_repo_ids.append(sid)
        shard_roots.append(sroot)
        job_args.append((
            k, s, input_file, str(sroot), sid, args.fps,
            args.state_key, args.task, args.robot_type,
            args.vcodec, args.crf,
        ))

    t0 = time.time()
    # 'spawn' so each worker starts with a fresh Python / no fork-inherited state
    # (safer with ffmpeg/PyAV).
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        results = pool.map(_worker, job_args)
    print(f"[parallel] all shards done in {time.time() - t0:.1f}s", flush=True)
    results.sort()

    # Aggregate.
    from lerobot.datasets.aggregate import aggregate_datasets
    print("[parallel] aggregating shards ->", final_root, flush=True)
    aggregate_datasets(
        repo_ids=shard_repo_ids,
        aggr_repo_id=args.repo_id,
        roots=shard_roots,
        aggr_root=final_root,
    )
    print(f"[parallel] aggregation done in {time.time() - t0:.1f}s total", flush=True)

    if not args.keep_shards:
        print("[parallel] removing per-shard directories", flush=True)
        for sroot in shard_roots:
            shutil.rmtree(sroot, ignore_errors=True)
        # remove _shards parent only if we created it under final_root
        try:
            shard_root.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()
