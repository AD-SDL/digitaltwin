# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive debug scaffold for the centrifuge scene.

Bypasses the manager-based env and its MDP wiring so we can isolate which
combination of props is blowing up the physics pipeline. Every prop is
spawned via its Isaac Lab spawner ``.func`` directly, using the SAME USD
paths and the SAME z placements (``TABLE_TOP_Z``, ``PROP_CLEARANCE`` etc.)
that :mod:`rpl_centrifuge.openarm.env_cfg` uses -- so if we can
reproduce the crash here we've narrowed it to scene geometry rather than
task cfg or robot articulation.

Usage
-----
Run from the IsaacLab repo root::

    ./isaaclab.sh -p <task_package_root>/scripts/debug_scene.py [flags]

Flags
-----
``--only NAMES``        Spawn only the listed props (comma-separated).
                        Defaults: everything (rack, rack_floor, tube, bucket).
``--skip NAMES``        Spawn everything except the listed props.
``--no-physics``        Freeze the timeline after setup (SimulationApp.update()
                        still renders, but PhysX doesn't advance). Use to
                        inspect the *initial* placement before gravity /
                        contact resolution touch anything.
``--camera X,Y,Z``      Camera eye. Default (2.0, 0.0, 1.5).
``--target X,Y,Z``      Camera lookat. Default (-0.3, 0.0, 0.85).

Examples
--------
Baseline (ground + table + pedestal only, physics off)::

    ./isaaclab.sh -p tasks/scripts/debug_scene.py --only base --no-physics

Add just the rack::

    ./isaaclab.sh -p tasks/scripts/debug_scene.py --only base,rack

Everything except the bucket::

    ./isaaclab.sh -p tasks/scripts/debug_scene.py --skip bucket
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI + AppLauncher (must run before any isaaclab.sim imports)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument(
    "--only",
    default=None,
    help="Comma-separated prop names to include (base,rack,rack_floor,tube,bucket).",
)
parser.add_argument(
    "--skip",
    default=None,
    help="Comma-separated prop names to exclude.",
)
parser.add_argument(
    "--no-physics",
    action="store_true",
    help="Do not play the timeline; render only. Preserves the initial placement.",
)
parser.add_argument(
    "--camera",
    default="2.0,0.0,1.5",
    help="Camera eye X,Y,Z (default 2.0,0.0,1.5).",
)
parser.add_argument(
    "--target",
    default="-0.3,0.0,0.85",
    help="Camera lookat X,Y,Z (default -0.3,0.0,0.85, roughly the tabletop centre).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# ---------------------------------------------------------------------------
# Isaac Lab imports (only safe after AppLauncher has spun up Kit)
# ---------------------------------------------------------------------------
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg  # noqa: E402
from isaaclab.sim.spawners.meshes.meshes_cfg import MeshCuboidCfg  # noqa: E402

# Reuse the same constants + asset dir as the real task cfg so we're
# testing the exact numbers that failed. Importing the module also runs
# gym.register() side effects; harmless here.
from rpl_centrifuge.openarm.env_cfg import (  # noqa: E402
    CENTRIFUGE_DATASET_DIR,
    PROP_CLEARANCE,
    TABLE_HEIGHT,
    TABLE_TOP_Z,
    _RACK_FLOOR_BOTTOM_Z,
    _RACK_FLOOR_THICKNESS,
    _TUBE_ORIGIN_Z,
)
from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg  # noqa: E402


# ---------------------------------------------------------------------------
# Prop selection
# ---------------------------------------------------------------------------
ALL_PROPS = ["base", "rack", "rack_floor", "tube", "bucket"]


def _parse_prop_list(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    names = {n.strip() for n in raw.split(",") if n.strip()}
    unknown = names - set(ALL_PROPS)
    if unknown:
        parser.error(f"unknown prop(s): {sorted(unknown)}. Valid: {ALL_PROPS}")
    return names


_only = _parse_prop_list(args_cli.only)
_skip = _parse_prop_list(args_cli.skip) or set()

if _only is None:
    ENABLED = set(ALL_PROPS) - _skip
else:
    ENABLED = _only - _skip
print(f"[debug_scene] enabled props: {sorted(ENABLED)}")


# ---------------------------------------------------------------------------
# Scene design
# ---------------------------------------------------------------------------
def design_scene() -> None:
    """Spawn ground, light, and whichever centrifuge props are enabled."""
    # Ground plane + dome light are unconditional; without them Kit boots to
    # a black stage that's hard to interpret visually.
    GroundPlaneCfg().func("/World/GroundPlane", GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0)
    light_cfg.func("/World/Light", light_cfg)

    sim_utils.create_prim("/World/Debug", "Xform")

    if "base" in ENABLED:
        _spawn_base()
    if "rack" in ENABLED:
        _spawn_rack()
    if "rack_floor" in ENABLED:
        _spawn_rack_floor()
    if "tube" in ENABLED:
        _spawn_tube()
    if "bucket" in ENABLED:
        _spawn_bucket()


def _spawn_base() -> None:
    """Robot pedestal + yellow table -- the two static AssetBaseCfg cuboids."""
    pedestal_cfg = MeshCuboidCfg(
        size=(0.413, 0.413, 0.579),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
    )
    pedestal_cfg.func(
        "/World/Debug/Pedestal", pedestal_cfg, translation=(-0.62, 0.0, 0.579 / 2)
    )

    table_cfg = MeshCuboidCfg(
        size=(1.0, 0.6, TABLE_HEIGHT),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.2)),
    )
    table_cfg.func(
        "/World/Debug/Table", table_cfg, translation=(0.1, 0.0, TABLE_HEIGHT / 2)
    )
    print(f"[debug_scene] table top at z={TABLE_TOP_Z}")


def _spawn_rack() -> None:
    rack_cfg = UsdFileCfg(
        usd_path=f"{CENTRIFUGE_DATASET_DIR}/centrifuge_tube_rack.usd",
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05)),
    )
    rack_cfg.func(
        "/World/Debug/Rack",
        rack_cfg,
        translation=(-0.35, -0.15, TABLE_TOP_Z + PROP_CLEARANCE),
        # spawner-func orientation is (x, y, z, w). 90 deg CCW about +Z.
        orientation=(0.0, 0.0, 0.7071068, 0.7071068),
    )
    print(f"[debug_scene] rack bottom at z={TABLE_TOP_Z + PROP_CLEARANCE}")


def _spawn_rack_floor() -> None:
    center_z = _RACK_FLOOR_BOTTOM_Z + _RACK_FLOOR_THICKNESS / 2
    floor_cfg = MeshCuboidCfg(
        size=(0.082, 0.125, _RACK_FLOOR_THICKNESS),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05)),
    )
    floor_cfg.func(
        "/World/Debug/RackFloor", floor_cfg, translation=(-0.35, -0.15, center_z)
    )
    print(f"[debug_scene] rack floor: bottom z={_RACK_FLOOR_BOTTOM_Z} top z={_RACK_FLOOR_BOTTOM_Z + _RACK_FLOOR_THICKNESS}")


def _spawn_tube() -> None:
    tube_cfg = UsdFileCfg(
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
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.005,
            rest_offset=-0.001,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.82, 0.82, 0.82)),
    )
    tube_cfg.func(
        "/World/Debug/Tube",
        tube_cfg,
        translation=(-0.332, -0.1425, _TUBE_ORIGIN_Z),
        # (x, y, z, w) — 90 deg CCW about +Z, same as the rack.
        orientation=(0.0, 0.0, 0.7071068, 0.7071068),
    )
    print(f"[debug_scene] tube origin z={_TUBE_ORIGIN_Z} (mesh bottom ~ {_TUBE_ORIGIN_Z - 0.020})")


def _spawn_bucket() -> None:
    bucket_cfg = UsdFileCfg(
        usd_path=f"{CENTRIFUGE_DATASET_DIR}/centrifuge_bucket_big.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.7, 0.25)),
    )
    bucket_cfg.func(
        "/World/Debug/Bucket",
        bucket_cfg,
        translation=(-0.35, 0.0, TABLE_TOP_Z + PROP_CLEARANCE),
    )
    print(f"[debug_scene] bucket bottom at z={TABLE_TOP_Z + PROP_CLEARANCE}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def _parse_xyz(s: str) -> tuple[float, float, float]:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 3:
        parser.error(f"expected X,Y,Z (got {s!r})")
    return (parts[0], parts[1], parts[2])


def main() -> None:
    # Use the same PhysX tuning as the task so we're testing the real setup.
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.01,
        render_interval=2,
        physics=PhysxCfg(
            bounce_threshold_velocity=0.01,
            friction_correlation_distance=0.00625,
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(_parse_xyz(args_cli.camera), _parse_xyz(args_cli.target))
    design_scene()
    sim.reset()
    print("[debug_scene] setup complete. Ctrl-C in this terminal to quit.")
    if args_cli.no_physics:
        print("[debug_scene] --no-physics: timeline stopped, rendering only.")
        sim.stop()
        while simulation_app.is_running():
            simulation_app.update()
    else:
        while simulation_app.is_running():
            sim.step()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
