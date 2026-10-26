# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Differential-IK action term with debug visualization of the IK target vs current EE.

The stock :class:`DifferentialInverseKinematicsAction` computes a target end-effector
pose each step but renders nothing. This subclass adds Isaac Lab's standard
``debug_vis`` hooks so that, when ``cfg.debug_vis`` is enabled, two markers are
drawn per action term (i.e. per arm):

  * **target** (red sphere) — the commanded IK goal pose (``ee_pos_des``), and
  * **current** (green sphere) — the arm's live end-effector pose.

The gap between the two markers is the instantaneous IK tracking error, which
makes it easy to see whether the solver is keeping up with the teleop command.

Enable it by setting ``debug_vis=True`` on the action term cfg (see the
``--debug`` flag wired into ``teleop_se3_agent.py``). With ``debug_vis=False``
(the default) this term behaves exactly like the base class and creates no
markers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DebugDifferentialInverseKinematicsAction(DifferentialInverseKinematicsAction):
    """Differential-IK action term that can draw its IK target and current EE pose."""

    cfg: DebugDifferentialInverseKinematicsActionCfg

    def __init__(self, cfg: DebugDifferentialInverseKinematicsActionCfg, env: ManagerBasedEnv):
        # base __init__ calls set_debug_vis(cfg.debug_vis), which routes to our
        # _set_debug_vis_impl below once the subclass is fully constructed.
        super().__init__(cfg, env)

    # -- debug visualization -------------------------------------------------

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_target_visualizer"):
                # Namespace the marker prims by body name so the left/right arm
                # terms don't clash on the same stage path. Use ``cfg.body_name``
                # (not the resolved ``self._body_name``): the base ActionTerm
                # __init__ calls set_debug_vis() before the IK subclass resolves
                # the body, so ``self._body_name`` does not exist yet here.
                body = self.cfg.body_name
                target_cfg = self.cfg.target_visualizer_cfg.copy()
                target_cfg.prim_path = f"/Visuals/IKDebug/{body}/target"
                current_cfg = self.cfg.current_visualizer_cfg.copy()
                current_cfg.prim_path = f"/Visuals/IKDebug/{body}/current"
                self._target_visualizer = VisualizationMarkers(target_cfg)
                self._current_visualizer = VisualizationMarkers(current_cfg)
            self._target_visualizer.set_visibility(True)
            self._current_visualizer.set_visibility(True)
        else:
            if hasattr(self, "_target_visualizer"):
                self._target_visualizer.set_visibility(False)
                self._current_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # Skip until the sim has produced valid body poses (root pose all-zero
        # before the first physics step would put markers at the origin).
        if not self._asset.is_initialized:
            return

        root_pos_w = self._asset.data.root_pos_w.torch
        root_quat_w = self._asset.data.root_quat_w.torch

        # IK target: controller stores the desired EE pose in the robot ROOT
        # frame (matching _compute_frame_pose's convention, offset included).
        target_pos_w, target_quat_w = math_utils.combine_frame_transforms(
            root_pos_w, root_quat_w, self._ik_controller.ee_pos_des, self._ik_controller.ee_quat_des
        )

        # Current EE: same root-frame convention, transformed to world.
        ee_pos_b, ee_quat_b = self._compute_frame_pose()
        current_pos_w, current_quat_w = math_utils.combine_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_b, ee_quat_b
        )

        self._target_visualizer.visualize(target_pos_w, target_quat_w)
        self._current_visualizer.visualize(current_pos_w, current_quat_w)


def _default_marker_cfg(color: tuple[float, float, float]) -> VisualizationMarkersCfg:
    """A single small sphere marker in the given RGB color (prim_path set per-arm at runtime)."""
    return VisualizationMarkersCfg(
        prim_path="/Visuals/IKDebug/placeholder",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            ),
        },
    )


@configclass
class DebugDifferentialInverseKinematicsActionCfg(DifferentialInverseKinematicsActionCfg):
    """Cfg for :class:`DebugDifferentialInverseKinematicsAction`.

    Identical to the base IK action cfg, plus two marker configs used only when
    ``debug_vis=True``. The ``prim_path`` on each is overwritten per-arm at
    runtime (namespaced by body name), so the value here is just a placeholder.
    """

    class_type: type = DebugDifferentialInverseKinematicsAction

    target_visualizer_cfg: VisualizationMarkersCfg = _default_marker_cfg((1.0, 0.1, 0.1))
    """Marker for the commanded IK target pose (default: red sphere)."""
    current_visualizer_cfg: VisualizationMarkersCfg = _default_marker_cfg((0.1, 1.0, 0.1))
    """Marker for the live end-effector pose (default: green sphere)."""
