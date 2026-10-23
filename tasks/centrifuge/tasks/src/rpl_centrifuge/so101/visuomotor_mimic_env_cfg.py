# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mimic + Visuomotor cfg for the SO101 IK-Abs task (headless image dataset gen)."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from .ik_abs_mimic_env_cfg import CentrifugeSO101IkAbsMimicEnvCfg
from .visuomotor_env_cfg import CentrifugeSO101IkAbsVisuomotorEnvCfg


@configclass
class CentrifugeSO101IkAbsVisuomotorMimicEnvCfg(
    CentrifugeSO101IkAbsVisuomotorEnvCfg, CentrifugeSO101IkAbsMimicEnvCfg
):
    """Diamond MRO combines the Visuomotor camera obs with the Mimic SubTask schedule."""

    pass
