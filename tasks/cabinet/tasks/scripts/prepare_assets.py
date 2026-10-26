# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify the bundled cabinet USD opens and expose its articulation summary.

Unlike the centrifuge pack, the cabinet asset is **authored directly as USD**
(see ``scripts/build_cabinet.py``, carried from the ``cabinet_sim/``
prototype), so every ``PhysicsCollisionAPI`` / ``PhysicsRigidBodyAPI`` /
``PhysicsArticulationRootAPI`` is already applied at author time -- there is
no raw geometry to prep. This script is still useful as:

  * a shelf-check that the bundled USD loads under the current Isaac/USD
    version (regenerate it via ``build_cabinet.py`` if it does not), and
  * a cheat sheet of the articulation DOFs / limits for anyone wiring new
    MDP terms.

Examples:
    ./isaaclab.sh -p scripts/prepare_assets.py
    ./isaaclab.sh -p scripts/prepare_assets.py <path-to-alternate-rpl_cabinet.usda>
"""

from __future__ import annotations

import argparse
import os
import sys
from importlib.resources import files as _pkg_files


def _bundled_cabinet_usd() -> str:
    return str(_pkg_files("rpl_cabinet") / "assets" / "rpl_cabinet.usda")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "usd_path",
        nargs="?",
        default=None,
        help="Cabinet USD to inspect. Default: the bundled one.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    usd_path = args.usd_path or _bundled_cabinet_usd()

    if not os.path.isfile(usd_path):
        sys.exit(f"error: no such file: {usd_path}")

    # usd-core is enough to open the stage; no SimulationApp needed for an
    # inspect-only pass. If usd-core is not available, fall through to the
    # Isaac Sim path. The centrifuge pack's prepare_assets launches
    # SimulationApp because it writes PhysX schemas; we only read.
    try:
        from pxr import Usd, UsdPhysics  # noqa: PLC0415
    except ModuleNotFoundError:
        sys.exit(
            "error: no USD runtime available. Install `usd-core`, or run this "
            "inside the Isaac Lab venv so pxr is on the path."
        )

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        sys.exit(f"error: could not open stage at {usd_path}")

    root = stage.GetDefaultPrim()
    if not root or not root.IsValid():
        sys.exit(f"error: {usd_path} has no defaultPrim")

    has_art = root.HasAPI(UsdPhysics.ArticulationRootAPI)
    print(f"stage: {usd_path}")
    print(f"root prim: {root.GetPath()}  (articulation root: {has_art})")

    joints = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joints.append(prim)

    if not joints:
        print("  no UsdPhysics.Joint prims found -- is this an articulated USD?")
        return

    print(f"  {len(joints)} joint(s):")
    for j in joints:
        name = j.GetName()
        typ = j.GetTypeName()
        if j.IsA(UsdPhysics.RevoluteJoint):
            rj = UsdPhysics.RevoluteJoint(j)
            lo = rj.GetLowerLimitAttr().Get()
            hi = rj.GetUpperLimitAttr().Get()
            axis = rj.GetAxisAttr().Get()
            print(f"    revolute {name:>20}  axis={axis}  limits=[{lo:+6.1f}, {hi:+6.1f}] deg")
        else:
            print(f"    {typ:>8} {name}")


if __name__ == "__main__":
    main()
