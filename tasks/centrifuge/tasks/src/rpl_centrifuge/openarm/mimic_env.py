# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab Mimic environment wrapper for the bimanual OpenArm centrifuge IK-Abs task.

This class implements the five :class:`~isaaclab.envs.ManagerBasedRLMimicEnv`
methods required by :mod:`isaaclab_mimic` — dataset annotation
(:file:`scripts/imitation_learning/isaaclab_mimic/annotate_demos.py`) and data
generation (:file:`scripts/imitation_learning/isaaclab_mimic/generate_dataset.py`).

Action-tensor layout (16 dim) matches the env's action-manager term order
(see :class:`~rpl_centrifuge.openarm.ik_abs_env_cfg.CentrifugeBimanualIkAbsEnvCfg.actions`)::

    [  0: 3]  left_arm_pose_pos    (x, y, z)      in robot ROOT frame
    [  3: 7]  left_arm_pose_quat   (qx, qy, qz, qw)  (see Quaternion convention below)
    [  7: 8]  left_gripper         scalar in [0, 1] (BinaryJointPositionAction threshold)
    [  8:11]  right_arm_pose_pos   (x, y, z)      in robot ROOT frame
    [ 11:15]  right_arm_pose_quat  (qx, qy, qz, qw)
    [ 15:16]  right_gripper        scalar in [0, 1]

Quaternion convention
---------------------
Isaac Lab settled on **(x, y, z, w)** across math utilities and sensors after
PR #4437 ("Changes Quaternion convention from WXYZ to XYZW"). That matches
the differential-IK controller's ``action_dim = 7`` layout
(:file:`source/isaaclab/isaaclab/controllers/differential_ik.py`), the
FrameTransformer's ``target_quat_w`` output (Warp ``wp.quatf``), and
:func:`isaaclab.utils.math.matrix_from_quat` / :func:`quat_from_matrix`.
No re-ordering is needed anywhere in this file.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv


class CentrifugeBimanualOpenArmIKAbsMimicEnv(ManagerBasedRLMimicEnv):
    """Mimic wrapper for ``Isaac-Centrifuge-Bimanual-OpenArm-IK-Abs-Mimic-v0``.

    Uses ``"left"`` and ``"right"`` as end-effector names throughout; every
    method that returns a per-eef dict is keyed on those two strings, matching
    :attr:`subtask_configs` in
    :class:`~rpl_centrifuge.openarm.ik_abs_mimic_env_cfg.CentrifugeBimanualIkAbsMimicEnvCfg`.
    """

    # ------------------------------------------------------------------
    # End-effector pose accessors
    # ------------------------------------------------------------------

    def get_robot_eef_pose(
        self, eef_name: str, env_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        """Return the current EE pose as a 4x4 homogeneous transform.

        Reads position and orientation from the policy observation group, which
        for this task is populated by
        :func:`~rpl_centrifuge.mdp.observations.ee_frame_position_in_env_frame`
        and :func:`~rpl_centrifuge.mdp.observations.ee_frame_orientation`.

        Args:
            eef_name: ``"left"`` or ``"right"``.
            env_ids: Environment indices; ``None`` selects all envs.

        Returns:
            Pose matrix of shape ``(len(env_ids), 4, 4)``.
        """
        if env_ids is None:
            env_ids = slice(None)

        pos = self.obs_buf["policy"][f"{eef_name}_eef_pos"][env_ids]
        quat_xyzw = self.obs_buf["policy"][f"{eef_name}_eef_quat"][env_ids]
        return PoseUtils.make_pose(pos, PoseUtils.matrix_from_quat(quat_xyzw))

    # ------------------------------------------------------------------
    # Action encoding / decoding
    # ------------------------------------------------------------------

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,  # unused; interface requires it for the datagen waypoint executor
    ) -> torch.Tensor:
        """Pack per-arm target poses + gripper commands into a single 16-D env action.

        Args:
            target_eef_pose_dict: ``{"left": pose_4x4, "right": pose_4x4}``.
            gripper_action_dict: ``{"left": tensor(1,), "right": tensor(1,)}``
                — scalar gripper command per arm, in the range consumed by
                :class:`BinaryJointPositionActionCfg`.
            action_noise_dict: Optional per-arm noise magnitude to add to the
                7-D pose sub-action. Missing keys apply no noise for that arm.
            env_id: Unused; kept for interface compatibility.

        Returns:
            Action tensor of shape ``(1, 16)`` (single-env). The waypoint
            executor (:file:`source/isaaclab_mimic/isaaclab_mimic/datagen/waypoint.py`)
            unsqueezes a 1-D return to add the env dim, so either shape works.
        """
        left_pos, left_rot = PoseUtils.unmake_pose(target_eef_pose_dict["left"])
        right_pos, right_rot = PoseUtils.unmake_pose(target_eef_pose_dict["right"])

        # quat_unique() keeps the real part non-negative so a numerical flip of
        # sign in the source pose (q vs -q represent the same rotation) does not
        # add a spurious 180-degree tumble when we interpolate between waypoints.
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
        # Isaac Lab concatenates action-manager terms in declared order.
        return torch.cat(
            [left_pose, left_gripper, right_pose, right_gripper], dim=-1
        ).unsqueeze(0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Inverse of :meth:`target_eef_pose_to_action` — unpack a batch of env
        actions into per-arm 4x4 target poses.

        Args:
            action: Env actions, shape ``(num_envs, 16)``.

        Returns:
            ``{"left": (num_envs, 4, 4), "right": (num_envs, 4, 4)}``.
        """
        left_pos = action[:, 0:3]
        left_quat = action[:, 3:7]
        right_pos = action[:, 8:11]
        right_quat = action[:, 11:15]
        return {
            "left": PoseUtils.make_pose(left_pos, PoseUtils.matrix_from_quat(left_quat)),
            "right": PoseUtils.make_pose(right_pos, PoseUtils.matrix_from_quat(right_quat)),
        }

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Slice the gripper channels out of a demo action stream.

        Args:
            actions: A sequence of env actions, shape ``(num_envs, T, 16)`` or
                ``(T, 16)`` — the mimic pipeline uses both shapes at different
                stages and standard indexing handles either.

        Returns:
            ``{"left": (..., 1), "right": (..., 1)}``.
        """
        return {"left": actions[..., 7:8], "right": actions[..., 15:16]}

    # ------------------------------------------------------------------
    # Subtask signals
    # ------------------------------------------------------------------

    def get_subtask_term_signals(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:
        """Read per-subtask boolean flags from the ``subtask_terms`` obs group.

        The dict keys must match the ``subtask_term_signal`` fields declared in
        :attr:`subtask_configs` (see the Mimic env cfg). Only non-terminal
        subtasks appear here — the last subtask in each arm's list uses
        ``subtask_term_signal=None`` and the annotator infers its termination
        from the episode's success signal instead.
        """
        if env_ids is None:
            env_ids = slice(None)

        subtask_terms = self.obs_buf["subtask_terms"]
        return {
            "right_grasp_tube": subtask_terms["right_grasp_tube"][env_ids],
        }
