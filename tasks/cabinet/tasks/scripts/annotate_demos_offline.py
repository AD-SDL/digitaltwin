# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Annotate recorded cabinet demos with Mimic ``datagen_info`` **without re-simulating**.

Why this exists
---------------
``scripts/imitation_learning/isaaclab_mimic/annotate_demos.py --auto`` replays
each recorded episode open-loop and keeps it only if the replay reaches the
success condition. For this task the replay does not reproduce: measured on the
13-demo cabinet batch, upstream auto-annotation accepted 0/13 even though every
demo was recorded with ``success=True`` and had opened the door past the ~80 deg
threshold. The failure is PhysX nondeterminism during the gripper close + handle
hook: the finger closes on a lever that is a few mm off, the hook slips, the
pull does not engage, and the replayed door swings only part-way. This matches
the centrifuge pack's experience with the same tool (3/20 upstream vs 20/20
offline), so the fix follows that pack's precedent.

Mimic never needs the source episode to be replayable -- ``DataGenInfoPool``
reads only waypoints:

    obs/datagen_info/eef_pose              per eef, (T, 4, 4)
    obs/datagen_info/object_pose           per rigid object / articulation, (T, 4, 4)
    obs/datagen_info/target_eef_pose       per eef, (T, 4, 4)
    obs/datagen_info/subtask_term_signals  per signal, (T,)

Every one of those is computable directly from what was recorded. This script
does that arithmetic and copies everything else through unchanged, so the output
is a drop-in replacement for the ``annotate_demos.py`` output with the full
source pool instead of the replay survivors.

What the offline grasp signal is
--------------------------------
``grasp_handle`` here is computed as ``max(|cabinet_door_angles|) >
engage_angle_rad``, i.e. either door has swung past ~10 deg. This differs from
the LIVE :func:`~rpl_cabinet.mdp.subtasks.grasp_handle`, which also requires
the EE-to-handle proximity gate. The simplification is safe because in sim a
vertical-hinge door only swings if an EE is pulling it (no gravity on the Z
axis of the hinge), so the door-angle gate implicitly implies grasp. Avoiding
the proximity term also avoids forward-kinematicking the cabinet articulation
offline to derive handle-hub world positions.

What it does NOT do
-------------------
It does not validate that the demos are good -- it trusts the recorded
``success`` flag and re-derives the subtask signal from recorded state. Run it
on datasets you recorded and believe. It also cannot help the *generation*
side: ``generate_dataset.py`` still has to physically execute the transformed
waypoints, and it inherits the same grasp-error amplification. See
``rpl_cabinet.openarm.ik_abs_mimic_env_cfg._apply_opendoor_mimic_cfg``, which
sets ``max_num_failures=2000`` for exactly that reason.

Usage (no Isaac Sim / GPU needed -- pure CPU, runs in seconds)::

    <isaaclab_root>/.venv/bin/python <task_package_root>/scripts/annotate_demos_offline.py \\
        --input_file  ./datasets/cabinet/openarm_vr_cabinet_dataset.hdf5 \\
        --output_file ./datasets/cabinet/openarm_vr_cabinet_annotated.hdf5
"""

from __future__ import annotations

import argparse
import json
import pathlib

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

# Must match ``OpenDoorBimanualOpenArmIKAbsMimicEnv`` -- the eef names used as
# keys in ``subtask_configs`` and in every per-eef dict Mimic reads.
EEF_NAMES = ("left", "right")

# Action-tensor slices, mirroring ``mimic_env.action_to_target_eef_pose``.
# Shared with centrifuge because both tasks use the 16-D bimanual IK-Abs layout.
EEF_ACTION_SLICES = {"left": (slice(0, 3), slice(3, 7)), "right": (slice(8, 11), slice(11, 15))}

# Mirrors ``rpl_cabinet.mdp.subtasks._ipsilateral_grasp``'s engage threshold and
# ``rpl_cabinet.mdp.terminations.door_opened``'s treatment of magnitude -- 10 deg
# past rest confirms the door is being pulled, not just resting.
ENGAGE_ANGLE_RAD = 0.17  # ~10 deg

DEFAULT_TASK = "Isaac-OpenDoor-Bimanual-OpenArm-IK-Abs-Mimic-v0"


def _pose_from_pos_quat(pos: torch.Tensor, quat_xyzw: torch.Tensor) -> torch.Tensor:
    """(T,3) + (T,4) XYZW -> (T,4,4) homogeneous transforms.

    Isaac Lab settled on XYZW across ``matrix_from_quat`` and the ``root_pose``
    ``[pos, quat]`` layout (PR #4437), and recorded datasets carry
    ``format_version >= 1``, so no reordering is needed. Datasets written before
    that change declare ``format_version 0`` and are rejected in :func:`main`
    rather than silently misinterpreted.
    """
    return PoseUtils.make_pose(pos, PoseUtils.matrix_from_quat(quat_xyzw))


def _grasp_handle_offline(obs: dict) -> torch.Tensor:
    """Recompute the ipsilateral grasp subtask signal from recorded obs.

    See the module docstring for why this is door-angle-only and not the live
    function's door-AND-proximity gate. The live obs term records both door
    angles into ``obs/cabinet_door_angles`` (shape ``(T, 2)``) in radians --
    column order follows the articulation DOF order, so take ``amax`` over
    axis 1 rather than indexing by position.
    """
    door_angles = obs["cabinet_door_angles"]  # (T, 2)
    max_abs = torch.abs(door_angles).amax(dim=1)
    return max_abs > ENGAGE_ANGLE_RAD


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

    # Mimic's ManagerBasedRLMimicEnv.get_object_poses() returns one pose per
    # scene asset (the cabinet-task override whitelists articulations too). The
    # cabinet lives under ``states["articulation"]`` -- centrifuge iterated
    # ``states["rigid_object"]`` because its tube/bucket/rack are rigid objects,
    # not articulations. Include every articulation except "robot".
    object_pose: dict[str, torch.Tensor] = {}
    for name, art in states.get("articulation", {}).items():
        if name == "robot":
            continue
        rp = art["root_pose"]
        object_pose[name] = _pose_from_pos_quat(rp[:, :3], rp[:, 3:7])
    # Rigid objects (if any are ever added) -- also included for completeness.
    for name, obj in states.get("rigid_object", {}).items():
        rp = obj["root_pose"]
        object_pose[name] = _pose_from_pos_quat(rp[:, :3], rp[:, 3:7])

    return {
        "eef_pose": eef_pose,
        "object_pose": object_pose,
        "target_eef_pose": target_eef_pose,
        "subtask_term_signals": {"grasp_handle": _grasp_handle_offline(obs)},
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
        signal = datagen_info["subtask_term_signals"]["grasp_handle"]
        if not bool(signal.any()):
            # Mimic needs a 0->1 edge to split the trajectory; without one the
            # episode would blow up inside DataGenInfoPool rather than here.
            print(f"{name}: no grasp_handle edge -- skipped")
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
        print(f"  skipped {no_signal} without a grasp_handle edge")
    print(f"  env_args.env_name = {json.dumps(args.task)}")


if __name__ == "__main__":
    main()
