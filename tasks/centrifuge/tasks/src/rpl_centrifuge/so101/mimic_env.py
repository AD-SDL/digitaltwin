# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IsaacLab-Mimic env class for SO101 centrifuge (data-generation / replay).

Mirrors ``openarm.mimic_env`` structure (single-arm variant). The env class
itself is a thin wrapper around ``ManagerBasedRLMimicEnv`` that exposes the
EE-pose / gripper queries IsaacLab Mimic needs to splice recorded demos.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.utils.math import matrix_from_quat


class CentrifugeSO101IKAbsMimicEnv(ManagerBasedRLMimicEnv):
    """SO101 single-arm Mimic env for the centrifuge task.

    Exposes the end-effector pose / gripper queries IsaacLab Mimic requires to
    chain subtask trajectories from recorded demos. We follow the OpenArm
    pattern: a single arm key ``"right"`` (there is no left) so the Mimic
    scheduler finds exactly one arm.
    """

    def get_robot_eef_pose(self, eef_name: str, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Return the EE pose as a 4x4 transform (per env) in the world frame.

        ``eef_name`` is the arm key — always ``"right"`` for the single-arm SO101.
        """
        if env_ids is None:
            env_ids = slice(None)
        ee_frame = self.scene["ee_frame"]
        pos = ee_frame.data.target_pos_w[env_ids, 0, :]
        quat = ee_frame.data.target_quat_w[env_ids, 0, :]  # wxyz
        rot = matrix_from_quat(quat)
        n = pos.shape[0]
        pose = torch.eye(4, device=pos.device).unsqueeze(0).repeat(n, 1, 1)
        pose[:, :3, :3] = rot
        pose[:, :3, 3] = pos
        return pose

    def get_object_poses(self, env_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Return a dict of object_name -> 4x4 world pose tensor."""
        if env_ids is None:
            env_ids = slice(None)
        out: dict[str, torch.Tensor] = {}
        for name in ("tube", "bucket", "rack"):
            obj = self.scene[name]
            pos = obj.data.root_pos_w[env_ids]
            quat = obj.data.root_quat_w[env_ids]  # wxyz
            rot = matrix_from_quat(quat)
            n = pos.shape[0]
            pose = torch.eye(4, device=pos.device).unsqueeze(0).repeat(n, 1, 1)
            pose[:, :3, :3] = rot
            pose[:, :3, 3] = pos
            out[name] = pose
        return out

    def get_subtask_term_signals(self, env_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Return bool tensors for the subtask signals used by the Mimic scheduler."""
        if env_ids is None:
            env_ids = slice(None)
        policy = self.obs_buf["subtask_terms"]
        return {"grasp_tube": policy["grasp_tube"][env_ids]}
