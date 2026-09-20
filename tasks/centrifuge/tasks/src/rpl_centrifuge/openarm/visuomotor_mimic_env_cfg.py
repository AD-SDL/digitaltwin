# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Image-bearing Mimic env cfg for the bimanual OpenArm centrifuge task.

Extends the Visuomotor variant (cameras + RGB image obs) with
:class:`MimicEnvCfg`, so ``generate_dataset`` produces a dataset **with images**
— rendered headless. This is the canonical path for GR00T-style visuomotor data.
Same subtask sequence as the low-dim Mimic variant.
"""

from __future__ import annotations

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg
from isaaclab.utils.configclass import configclass

from .ik_abs_mimic_env_cfg import _apply_centrifuge_mimic_cfg
from .visuomotor_env_cfg import CentrifugeBimanualIkAbsVisuomotorEnvCfg


@configclass
class CentrifugeBimanualIkAbsVisuomotorMimicEnvCfg(CentrifugeBimanualIkAbsVisuomotorEnvCfg, MimicEnvCfg):
    """Visuomotor (image) Mimic variant of the bimanual OpenArm IK-Abs task."""

    def __post_init__(self):
        super().__post_init__()
        _apply_centrifuge_mimic_cfg(self)
        self.datagen_config.name = "centrifuge_openarm_bimanual_ik_abs_visuomotor_D0"
