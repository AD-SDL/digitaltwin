# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab Mimic env wrapper for the OpenArm cabinet IK-Abs task.

Implements the five :class:`~isaaclab.envs.ManagerBasedRLMimicEnv` methods
required by :mod:`isaaclab_mimic` -- dataset annotation
(:file:`scripts/imitation_learning/isaaclab_mimic/annotate_demos.py`) and data
generation (:file:`generate_dataset.py`).

Action-tensor layout (16 dim) matches the env's action-manager term order
(see :class:`~rpl_cabinet.openarm.ik_abs_env_cfg.OpenDoorBimanualIkAbsEnvCfg.actions`)::

    [  0: 3]  left_arm_pose_pos    (x, y, z)      in robot ROOT frame
    [  3: 7]  left_arm_pose_quat   (qx, qy, qz, qw)
    [  7: 8]  left_gripper         scalar
    [  8:11]  right_arm_pose_pos   (x, y, z)      in robot ROOT frame
    [ 11:15]  right_arm_pose_quat  (qx, qy, qz, qw)
    [ 15:16]  right_gripper        scalar

Quaternion convention
---------------------
Isaac Lab's XYZW convention is used throughout (after PR #4437). Matches the
differential-IK controller's ``action_dim = 7`` layout, the FrameTransformer's
``target_quat_w`` output, and :func:`matrix_from_quat` / :func:`quat_from_matrix`.
No re-ordering is needed anywhere in this file.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv


class OpenDoorBimanualOpenArmIKAbsMimicEnv(ManagerBasedRLMimicEnv):
    """Mimic wrapper for ``Isaac-OpenDoor-Bimanual-OpenArm-IK-Abs-Mimic-v0``.

    Uses ``"left"`` and ``"right"`` as EE names throughout; every per-eef dict
    is keyed on those two strings, matching :attr:`subtask_configs` in
    :class:`~rpl_cabinet.openarm.ik_abs_mimic_env_cfg.OpenDoorBimanualIkAbsMimicEnvCfg`.
    """

    def get_robot_eef_pose(
        self, eef_name: str, env_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        if env_ids is None:
            env_ids = slice(None)
        pos = self.obs_buf["policy"][f"{eef_name}_eef_pos"][env_ids]
        quat_xyzw = self.obs_buf["policy"][f"{eef_name}_eef_quat"][env_ids]
        return PoseUtils.make_pose(pos, PoseUtils.matrix_from_quat(quat_xyzw))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Pack per-arm target poses + gripper commands into a 16-D env action."""
        left_pos, left_rot = PoseUtils.unmake_pose(target_eef_pose_dict["left"])
        right_pos, right_rot = PoseUtils.unmake_pose(target_eef_pose_dict["right"])
        # quat_unique() keeps the real part non-negative so a numerical flip of
        # sign in the source pose (q vs -q represent the same rotation) does
        # not add a spurious 180-degree tumble on interpolation.
        left_quat = PoseUtils.quat_unique(PoseUtils.quat_from_matrix(left_rot))
        right_quat = PoseUtils.quat_unique(PoseUtils.quat_from_matrix(right_rot))

        left_pose = torch.cat([left_pos, left_quat], dim=-1)
        right_pose = torch.cat([right_pos, right_quat], dim=-1)

        if action_noise_dict is not None:
            left_noise = action_noise_dict.get("left")
            right_noise = action_noise_dict.get("right")
            if left_noise is not None:
                left_pose = left_pose + left_noise * torch.randn_like(left_pose)
            if right_noise is not None:
                right_pose = right_pose + right_noise * torch.randn_like(right_pose)

        left_gripper = gripper_action_dict["left"]
        right_gripper = gripper_action_dict["right"]

        # Order MUST match the env's ActionsCfg term declaration order in
        # ``openarm/env_cfg.py`` -> ``ActionsCfg``:
        #   left_arm_action, left_gripper_action, right_arm_action, right_gripper_action
        return torch.cat(
            [left_pose, left_gripper, right_pose, right_gripper], dim=-1
        ).unsqueeze(0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Inverse of :meth:`target_eef_pose_to_action` -- (num_envs, 4, 4) per arm."""
        left_pos = action[:, 0:3]
        left_quat = action[:, 3:7]
        right_pos = action[:, 8:11]
        right_quat = action[:, 11:15]
        return {
            "left": PoseUtils.make_pose(left_pos, PoseUtils.matrix_from_quat(left_quat)),
            "right": PoseUtils.make_pose(right_pos, PoseUtils.matrix_from_quat(right_quat)),
        }

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Slice the gripper channels out of a demo action stream."""
        return {"left": actions[..., 7:8], "right": actions[..., 15:16]}

    def get_object_poses(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:
        """Return the cabinet root pose as a 4x4 matrix, keyed on ``"cabinet"``.

        The base-class default iterates ``scene.get_state()["rigid_object"]``,
        which is correct for the centrifuge pack (tube / bucket / rack are all
        ``RigidObjectCfg``) but EMPTY for us: the cabinet is an
        ``ArticulationCfg`` (fixed base + four revolute DOFs), so its state
        lives under ``["articulation"]`` and would never show up. SubTaskConfigs
        in :mod:`rpl_cabinet.openarm.ik_abs_mimic_env_cfg` set
        ``object_ref="cabinet"``; Mimic datagen resolves that reference against
        this dict, so an empty dict would silently break the nearest-neighbor
        source-waypoint lookup during generation.

        Pose is the ARTICULATION ROOT pose in the per-env origin frame
        (``is_relative=True``) -- the same frame the base class uses for rigid
        objects, so SubTaskConfig matching stays consistent across the two
        asset kinds. We deliberately return only the root (not every door /
        handle body): the Mimic ``nearest_neighbor_object`` strategy picks the
        source demo whose object pose is closest to the current scene's at
        subtask boundary time, and the root captures the only pose we
        randomize at reset (the door / handle angles start at zero and are
        state the policy must drive, not state we match on).
        """
        if env_ids is None:
            env_ids = slice(None)

        art_states = self.scene.get_state(is_relative=True)["articulation"]
        poses: dict[str, torch.Tensor] = {}
        for name, state in art_states.items():
            if name == "robot":
                # Mimic only cares about scene objects, not the robot itself
                # (EE pose is served separately by get_robot_eef_pose).
                continue
            root_pose = state["root_pose"][env_ids]
            poses[name] = PoseUtils.make_pose(
                root_pose[:, :3], PoseUtils.matrix_from_quat(root_pose[:, 3:7])
            )
        return poses

    def get_subtask_term_signals(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:
        """Read per-subtask boolean flags from the ``subtask_terms`` obs group.

        Keys must match the ``subtask_term_signal`` fields in
        :attr:`subtask_configs`. Only non-terminal subtasks appear here -- the
        last subtask uses ``subtask_term_signal=None`` and the annotator infers
        its termination from the episode's success signal.
        """
        if env_ids is None:
            env_ids = slice(None)
        subtask_terms = self.obs_buf["subtask_terms"]
        return {
            "left_grasp_handle": subtask_terms["left_grasp_handle"][env_ids],
        }
