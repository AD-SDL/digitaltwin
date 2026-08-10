# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GR1T2 centrifuge task derived from Isaac Lab's waist-enabled GR1T2 pick-place env.

This keeps the upstream GR1T2 robot, Pink IK action space, and Isaac Teleop
hand-tracking pipeline, but swaps the scene from steering-wheel-on-table to a
centrifuge workflow:

* ``object`` -> centrifuge tube
* add a static centrifuge rack as the pickup source
* add a heavy centrifuge bucket as the placement receptacle

The upstream observations and 36-D hand-tracking action layout are preserved so
VR teleoperation continues to work as expected.
"""

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_gr1t2_env_cfg import (
    EventCfg as UpstreamEventCfg,
)
from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_gr1t2_env_cfg import (
    ObjectTableSceneCfg as UpstreamObjectTableSceneCfg,
)
from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_gr1t2_env_cfg import (
    TerminationsCfg as UpstreamTerminationsCfg,
)
from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_gr1t2_waist_enabled_env_cfg import (
    PickPlaceGR1T2WaistEnabledEnvCfg,
)

from .. import mdp
from ..openarm.env_cfg import CENTRIFUGE_DATASET_DIR

_RACK_POS = (-0.35, 0.40, 0.986)
_RACK_ROT = (0.0, 0.0, 0.7071068, 0.7071068)
_TUBE_POS = (-0.332, 0.4075, 1.011)
_TUBE_ROT = (0.0, 0.0, 0.7071068, 0.7071068)
_BUCKET_POS = (-0.35, 0.55, 0.986)


@configclass
class GR1T2CentrifugeSceneCfg(UpstreamObjectTableSceneCfg):
    """GR1T2 pick-place scene with centrifuge props replacing the original object setup."""

    rack = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Rack",
        init_state=AssetBaseCfg.InitialStateCfg(pos=_RACK_POS, rot=_RACK_ROT),
        spawn=UsdFileCfg(
            usd_path=f"{CENTRIFUGE_DATASET_DIR}/centrifuge_tube_rack.usd",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05)),
        ),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_TUBE_POS, rot=_TUBE_ROT),
        spawn=UsdFileCfg(
            usd_path=f"{CENTRIFUGE_DATASET_DIR}/centrifuge_tube_big.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=-0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.82, 0.82, 0.82)),
        ),
    )

    bucket = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bucket",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_BUCKET_POS, rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=UsdFileCfg(
            usd_path=f"{CENTRIFUGE_DATASET_DIR}/centrifuge_bucket_big.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.35),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.16, 0.55, 0.22)),
        ),
    )


@configclass
class GR1T2CentrifugeTerminationsCfg(UpstreamTerminationsCfg):
    """Termination settings for the GR1T2 centrifuge task."""

    object_dropping = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={"minimum_height": 0.75, "asset_cfg": SceneEntityCfg("object")},
    )
    success = DoneTerm(
        func=mdp.tube_inside_bucket,
        params={
            "tube_cfg": SceneEntityCfg("object"),
            "bucket_cfg": SceneEntityCfg("bucket"),
        },
    )


@configclass
class GR1T2CentrifugeEventCfg(UpstreamEventCfg):
    """Reset logic for the GR1T2 centrifuge task."""

    reset_object = EventTerm(
        func=mdp.reset_object_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.005, 0.005),
                "y": (-0.005, 0.005),
                "yaw": (-0.15, 0.15),
            },
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class GR1T2CentrifugeWaistEnabledEnvCfg(PickPlaceGR1T2WaistEnabledEnvCfg):
    """Waist-enabled GR1T2 hand-tracking env with centrifuge props."""

    scene: GR1T2CentrifugeSceneCfg = GR1T2CentrifugeSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    terminations: GR1T2CentrifugeTerminationsCfg = GR1T2CentrifugeTerminationsCfg()
    events: GR1T2CentrifugeEventCfg = GR1T2CentrifugeEventCfg()
