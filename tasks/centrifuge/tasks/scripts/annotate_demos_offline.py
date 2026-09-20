# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Annotate recorded demos with Mimic ``datagen_info`` **without re-simulating**.

Why this exists
---------------
``scripts/imitation_learning/isaaclab_mimic/annotate_demos.py --auto`` replays
each recorded episode open-loop and keeps it only if the replay reaches the
success condition. For this contact-rich insertion the replay does not
reproduce: measured against the 20 shipped demos, the replayed tube pose tracks
the recording to ~1-2 mm all the way to the grasp, then jumps to a **13 mm
median (53 mm worst) deviation at the moment the gripper closes** — the fingers
close on a tube that is ~2 mm off and it seats differently in the hand. The arm
is then commanded to the recorded hand poses, which are correct, while the tube
hangs 13 mm from where it should be, and the release over a tight bucket well
misses. Only 3 of 20 episodes survive; the other 17 end with the tube on the
bucket rim, tipped over on the table, or on the floor.

Those 17 demos are not bad data. They were recorded with ``success=True`` and
every one of them satisfies the seated-in-bucket condition on its **recorded**
states. They are rejected by an artifact of the annotator's own re-simulation.

Mimic never needs the source episode to be replayable — ``DataGenInfoPool``
reads only waypoints:

    obs/datagen_info/eef_pose              per eef, (T, 4, 4)
    obs/datagen_info/object_pose           per rigid object, (T, 4, 4)
    obs/datagen_info/target_eef_pose       per eef, (T, 4, 4)
    obs/datagen_info/subtask_term_signals  per signal, (T,)

Every one of those is computable directly from what was recorded. This script
does that arithmetic and copies everything else through unchanged, so the output
is a drop-in replacement for the ``annotate_demos.py`` output with the full
source pool instead of the replay survivors.

What it does NOT do
-------------------
It does not validate that the demos are good — it trusts the recorded
``success`` flag and re-derives the subtask signal from recorded state. Run it
on datasets you recorded and believe. It also cannot help the *generation* side:
``generate_dataset.py`` still has to physically execute the transformed
waypoints, and it inherits the same grasp-error amplification.

Usage (no Isaac Sim / GPU needed — pure CPU, runs in seconds)::

    <isaaclab_root>/.venv/bin/python <task_package_root>/scripts/annotate_demos_offline.py \
        --input_file ./datasets/centrifuge_openarm.hdf5 \
        --output_file ./datasets/centrifuge_openarm_annotated.hdf5
"""

from __future__ import annotations

import argparse
import json
import pathlib

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

# Must match ``CentrifugeBimanualOpenArmIKAbsMimicEnv`` — the eef names used as
# keys in ``subtask_configs`` and in every per-eef dict Mimic reads.
EEF_NAMES = ("left", "right")

# Action-tensor slices, mirroring ``mimic_env.action_to_target_eef_pose``.
EEF_ACTION_SLICES = {"left": (slice(0, 3), slice(3, 7)), "right": (slice(8, 11), slice(11, 15))}

# Calibrated in ``rpl_centrifuge.mdp.subtasks`` — see that module for the
# measurement table behind these two constants.
TUBE_LIFT_ABOVE_RACK = 0.104
PROXIMITY_GATE = 0.15

DEFAULT_TASK = "Isaac-Centrifuge-Bimanual-OpenArm-IK-Abs-Mimic-v0"


def _pose_from_pos_quat(pos: torch.Tensor, quat_xyzw: torch.Tensor) -> torch.Tensor:
    """(T,3) + (T,4) XYZW -> (T,4,4) homogeneous transforms.

    Isaac Lab settled on XYZW across ``matrix_from_quat`` and the ``root_pose``
    ``[pos, quat]`` layout (PR #4437), and recorded datasets carry
    ``format_version >= 1``, so no reordering is needed. Datasets written before
    that change declare ``format_version 0`` and are rejected in :func:`main`
    rather than silently misinterpreted.
    """
    return PoseUtils.make_pose(pos, PoseUtils.matrix_from_quat(quat_xyzw))


def _right_grasp_tube(states: dict, obs: dict) -> torch.Tensor:
    """Recompute the right-arm grasp signal from recorded state.

    Mirrors :func:`rpl_centrifuge.mdp.subtasks.right_grasp_tube` exactly: tube
    lifted more than ``TUBE_LIFT_ABOVE_RACK`` above the rack root, AND the right
    eef within ``PROXIMITY_GATE`` of the tube.

    The env function works in world frame; the recorded states and eef
    observations are both env-frame. The lift test is a difference of two
    heights so the origin cancels, and the proximity test is a difference of two
    positions in the same frame, so both agree with the live signal.
    """
    tube_pose = states["rigid_object"]["tube"]["root_pose"]
    rack_pose = states["rigid_object"]["rack"]["root_pose"]
    lifted = (tube_pose[:, 2] - rack_pose[:, 2]) > TUBE_LIFT_ABOVE_RACK
    near = torch.linalg.vector_norm(obs["right_eef_pos"] - tube_pose[:, :3], dim=-1) < PROXIMITY_GATE
    return lifted & near


def _build_datagen_info(episode: EpisodeData) -> dict:
    data = episode.data
    obs, states, actions = data["obs"], data["states"], data["actions"]

    eef_pose = {
        name: _pose_from_pos_quat(obs[f"{name}_eef_pos"], obs[f"{name}_eef_quat"]) for name in EEF_NAMES
    }
    target_eef_pose = {
        name: _pose_from_pos_quat(actions[:, pos_sl], actions[:, quat_sl])
        for name, (pos_sl, quat_sl) in EEF_ACTION_SLICES.items()
    }
    # Mimic's ManagerBasedRLMimicEnv.get_object_poses() returns every rigid
    # object in the scene, so mirror that rather than only the referenced ones.
    object_pose = {
        name: _pose_from_pos_quat(obj["root_pose"][:, :3], obj["root_pose"][:, 3:7])
        for name, obj in states["rigid_object"].items()
    }
    return {
        "eef_pose": eef_pose,
        "object_pose": object_pose,
        "target_eef_pose": target_eef_pose,
        "subtask_term_signals": {"right_grasp_tube": _right_grasp_tube(states, obs)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_file", type=str, required=True, help="Recorded demo dataset.")
    parser.add_argument("--output_file", type=str, required=True, help="Annotated output dataset.")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Mimic env name to stamp into env_args.")
    parser.add_argument(
        "--keep_unsuccessful",
        action="store_true",
        help="Also annotate episodes whose recorded success flag is not True (default: skip them).",
    )
    args = parser.parse_args()

    in_handler = HDF5DatasetFileHandler()
    in_handler.open(args.input_file)

    if in_handler.is_legacy_quaternion_format():
        raise SystemExit(
            f"{args.input_file} declares a legacy (WXYZ) quaternion format. This script assumes the XYZW "
            "layout used from format_version 1 onward; re-record or convert it first."
        )

    out_path = pathlib.Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_handler = HDF5DatasetFileHandler()
    out_handler.create(str(out_path), env_name=args.task)

    written = skipped = no_signal = 0
    for name in in_handler.get_episode_names():
        episode = in_handler.load_episode(name, device="cpu")

        # NB: compare by value, not identity -- the handler hands back numpy
        # scalars (``np.True_``), which fail an ``is True`` check.
        if not bool(episode.success) and not args.keep_unsuccessful:
            print(f"{name}: recorded success flag is {episode.success!r} -- skipped")
            skipped += 1
            continue

        datagen_info = _build_datagen_info(episode)
        signal = datagen_info["subtask_term_signals"]["right_grasp_tube"]
        if not bool(signal.any()):
            # Mimic needs a 0->1 edge to split the trajectory; without one the
            # episode would blow up inside DataGenInfoPool rather than here.
            print(f"{name}: no right_grasp_tube edge -- skipped")
            no_signal += 1
            continue

        episode.data["obs"]["datagen_info"] = datagen_info
        episode.success = True
        out_handler.write_episode(episode)

        edge = int(torch.argmax(signal.int()))
        total = len(episode.data["actions"])
        print(f"{name}: T={total:4d}  grasp edge @ {edge:4d}  subtasks [0:{edge + 2}] [{edge + 2}:{total}]")
        written += 1

    out_handler.flush()
    out_handler.close()
    in_handler.close()

    print(f"\nWrote {written} annotated episode(s) to {out_path}")
    if skipped:
        print(f"  skipped {skipped} without a recorded success flag")
    if no_signal:
        print(f"  skipped {no_signal} without a right_grasp_tube edge")
    print(f"  env_args.env_name = {json.dumps(args.task)}")


if __name__ == "__main__":
    main()
