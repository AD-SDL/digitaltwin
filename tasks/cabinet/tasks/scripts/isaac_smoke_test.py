#!/usr/bin/env python3
"""
Headless smoke test: load the rigged cabinet into Isaac Sim, step physics,
and drive each joint to its limit.  Proves the articulation is sane before
anyone builds a scene on top of it.

Run with Isaac Sim 5.1's python:
    <isaaclab_root>/isaaclab.sh -p isaac_smoke_test.py --usd rpl_cabinet.usda
"""

import argparse
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--usd", required=True)
ap.add_argument("--steps", type=int, default=90)
ap.add_argument("--result", default="smoke_result.txt")
args = ap.parse_args()

# sim_app.close() tears the process down hard, so buffered stdout is lost.
# Everything goes to a file that is flushed and fsynced as it is written.
import os                                                # noqa: E402
_RESULT = open(args.result, "w", buffering=1)


def say(msg=""):
    print(msg, flush=True)
    _RESULT.write(msg + "\n")
    _RESULT.flush()
    os.fsync(_RESULT.fileno())

from isaacsim import SimulationApp                      # noqa: E402
sim_app = SimulationApp({"headless": True})

import numpy as np                                       # noqa: E402

# Isaac Sim 5.x module layout, with a 4.x fallback.
try:
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.prims import SingleArticulation as Art
    API = "5.x"
except ImportError:                                      # pragma: no cover
    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.articulations import Articulation as Art
    API = "4.x"

FAIL = []


def check(cond, msg):
    say(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAIL.append(msg)


def set_targets(art, target):
    """Set joint position targets across Isaac API revisions."""
    if hasattr(art, "set_joint_position_targets"):
        art.set_joint_position_targets(target)
        return "set_joint_position_targets"
    try:
        from isaacsim.core.utils.types import ArticulationAction
    except ImportError:
        from omni.isaac.core.utils.types import ArticulationAction
    art.apply_action(ArticulationAction(joint_positions=np.asarray(target)))
    return "apply_action(ArticulationAction)"


def dof_limits(art, ndof):
    """Joint limits in radians, across Isaac API revisions.

    SingleArticulation dropped get_dof_limits() somewhere before 5.1, so try
    the accessors in turn and report which one answered.
    """
    def norm(a):
        a = np.asarray(a, dtype=float)
        if a.ndim == 3:
            a = a[0]
        return a[:, 0], a[:, 1]

    attempts = [
        ("art.get_dof_limits", lambda: norm(art.get_dof_limits())),
        ("art._articulation_view.get_dof_limits",
         lambda: norm(art._articulation_view.get_dof_limits())),
        ("art.dof_properties",
         lambda: (np.asarray(art.dof_properties["lower"], dtype=float),
                  np.asarray(art.dof_properties["upper"], dtype=float))),
        ("art.get_articulation_body_count/physx view",
         lambda: norm(art._physics_view.get_dof_limits())),
    ]
    for name, fn in attempts:
        try:
            lo, hi = fn()
            if len(lo) == ndof:
                say(f"[limits via] {name}")
                return lo, hi
        except Exception as e:
            say(f"[limits  no] {name}: {type(e).__name__}")

    # last resort: read what we authored, straight off the stage
    import omni.usd
    from pxr import UsdPhysics
    stage = omni.usd.get_context().get_stage()
    lo = np.zeros(ndof)
    hi = np.zeros(ndof)
    for n in (art.dof_names or []):
        j = UsdPhysics.RevoluteJoint(
            stage.GetPrimAtPath(f"/World/Cabinet/joints/{n}"))
        i = art.get_dof_index(n)
        lo[i] = np.radians(j.GetLowerLimitAttr().Get())
        hi[i] = np.radians(j.GetUpperLimitAttr().Get())
    say("[limits via] USD stage fallback")
    return lo, hi


def main():
    say(f"[api] {API}")
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    add_reference_to_stage(usd_path=args.usd, prim_path="/World/Cabinet")
    art = Art("/World/Cabinet")
    world.reset()
    art.initialize()

    names = list(art.dof_names or [])
    say(f"[dofs] {len(names)}: {names}")
    for want in ("hinge_left", "hinge_right", "latch_left", "latch_right"):
        check(want in names, f"dof present: {want}")
    if FAIL:
        return

    idx = {n: art.get_dof_index(n) for n in names}
    lower, upper = dof_limits(art, len(names))
    for n in names:
        i = idx[n]
        say(f"  {n:14} limits {np.degrees(lower[i]):7.1f} .. "
              f"{np.degrees(upper[i]):7.1f} deg")

    # settle
    for _ in range(30):
        world.step(render=False)
    rest = art.get_joint_positions()
    check(np.all(np.isfinite(rest)), "rest pose finite")
    check(float(np.max(np.abs(np.degrees(rest)))) < 2.0,
          f"rest pose near zero (max {np.max(np.abs(np.degrees(rest))):.2f} deg)")

    # drive each joint to the far end of its own range
    target = np.zeros(len(names))
    for n in names:
        i = idx[n]
        target[i] = upper[i] if abs(upper[i]) > abs(lower[i]) else lower[i]
    say(f"[targets via] {set_targets(art, target)}")
    # a passive hinge has no stiffness, so push it there directly too
    art.set_joint_positions(target)

    for _ in range(args.steps):
        world.step(render=False)

    reached = art.get_joint_positions()
    say("[after drive]")
    for n in names:
        i = idx[n]
        say(f"  {n:14} {np.degrees(reached[i]):8.1f} deg  "
              f"(target {np.degrees(target[i]):7.1f})")

    check(np.all(np.isfinite(reached)), "driven pose finite (no explosion)")
    for n in names:
        i = idx[n]
        moved = abs(np.degrees(reached[i] - rest[i]))
        check(moved > 5.0, f"{n} actually moved ({moved:.1f} deg)")
        within = (np.degrees(reached[i]) >= np.degrees(lower[i]) - 2.0 and
                  np.degrees(reached[i]) <= np.degrees(upper[i]) + 2.0)
        check(within, f"{n} stayed inside its limits")

    root_pos, _ = art.get_world_pose()
    check(float(np.linalg.norm(np.asarray(root_pos))) < 0.05,
          f"carcass stayed fixed (drifted {np.linalg.norm(np.asarray(root_pos)):.4f} m)")


try:
    main()
except Exception:
    import traceback
    say("EXCEPTION:")
    say(traceback.format_exc())
    FAIL.append("exception")
finally:
    say("\n" + "=" * 52)
    say(f"{len(FAIL)} failures")
    sim_app.close()

sys.exit(1 if FAIL else 0)
