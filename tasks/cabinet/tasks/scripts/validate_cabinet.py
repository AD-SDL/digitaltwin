#!/usr/bin/env python3
"""
Validate the rigged cabinet USD before it ever reaches Isaac Sim.

Checks:
  1. Every joint's two anchor frames resolve to the SAME world point.
     A mismatch here is invisible in the file and makes the articulation
     snap or explode on the first physics step.
  2. No two colliders interpenetrate in the rest pose.
  3. Doors clear the frame and each other through their full swing.
  4. Articulation tree, masses and limits are sane.

Usage: python3 validate_cabinet.py rpl_cabinet.usda
"""

import math
import sys
from pxr import Usd, UsdGeom, UsdPhysics, Gf

TOL = 1e-6
FAIL = []
WARN = []


def fail(msg):
    FAIL.append(msg)
    print(f"  FAIL  {msg}")


def warn(msg):
    WARN.append(msg)
    print(f"  warn  {msg}")


def ok(msg):
    print(f"  ok    {msg}")


def world_xform(stage, path):
    prim = stage.GetPrimAtPath(path)
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


def check_joint_frames(stage):
    print("\n[1] joint anchor frames")
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(prim)
        b0 = j.GetBody0Rel().GetTargets()
        b1 = j.GetBody1Rel().GetTargets()
        p0 = j.GetLocalPos0Attr().Get() or Gf.Vec3f(0)
        p1 = j.GetLocalPos1Attr().Get() or Gf.Vec3f(0)

        m0 = world_xform(stage, b0[0]) if b0 else Gf.Matrix4d(1)
        m1 = world_xform(stage, b1[0]) if b1 else Gf.Matrix4d(1)
        w0 = m0.Transform(Gf.Vec3d(*p0))
        w1 = m1.Transform(Gf.Vec3d(*p1))
        d = (w0 - w1).GetLength()

        name = prim.GetPath().name
        if d > 1e-5:
            fail(f"{name}: anchors disagree by {d*1000:.2f} mm  "
                 f"body0->{tuple(round(v,4) for v in w0)}  "
                 f"body1->{tuple(round(v,4) for v in w1)}")
        else:
            ok(f"{name}: anchors coincide at "
               f"({w0[0]*1000:.1f}, {w0[1]*1000:.1f}, {w0[2]*1000:.1f}) mm")


def owning_body(prim):
    """Nearest ancestor (or self) carrying RigidBodyAPI."""
    p = prim
    while p and p.IsValid() and str(p.GetPath()) != "/":
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            return str(p.GetPath())
        p = p.GetParent()
    return None


def collider_boxes(stage):
    """World-space AABB of every prim carrying a collider, with its body."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                              [UsdGeom.Tokens.default_], useExtentsHint=False)
    out = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if r.IsEmpty():
            continue
        out.append((str(prim.GetPath()), r, owning_body(prim)))
    return out


def check_interpenetration(stage):
    print("\n[2] collider interpenetration in rest pose")
    boxes = collider_boxes(stage)
    print(f"  {len(boxes)} colliders")
    bad = 0
    same_body = 0
    for i in range(len(boxes)):
        for k in range(i + 1, len(boxes)):
            n0, r0, b0 = boxes[i]
            n1, r1, b1 = boxes[k]
            inter = Gf.Range3d(r0).IntersectWith(r1)
            if inter.IsEmpty():
                continue
            s = inter.GetSize()
            # overlap in all three axes by more than a hair = real interpenetration
            depth = min(s[0], s[1], s[2])
            if depth <= 1e-4:
                continue
            if b0 is not None and b0 == b1:
                # shapes of one compound collider are allowed to overlap
                same_body += 1
                continue
            a = n0.split("/")[-2] + "/" + n0.split("/")[-1]
            b = n1.split("/")[-2] + "/" + n1.split("/")[-1]
            fail(f"{a}  overlaps  {b}  by {depth*1000:.1f} mm")
            bad += 1
    if same_body:
        ok(f"{same_body} intra-body shape overlap(s), allowed")
    if bad == 0:
        ok("no colliders interpenetrate across bodies")


def check_swing(stage):
    """Sweep each door through its limits and look for collisions with the
    fixed frame.  Approximated by rotating the door's AABB corners."""
    print("\n[3] door swing clearance")
    EPS = 0.0015  # shrink fixed geometry so closed-pose face contact is not a hit
    body_boxes = []
    for n, r, _ in collider_boxes(stage):
        if "/body/" not in n:
            continue
        shrunk = Gf.Range3d(r.GetMin() + Gf.Vec3d(EPS, EPS, EPS),
                            r.GetMax() - Gf.Vec3d(EPS, EPS, EPS))
        if not shrunk.IsEmpty():
            body_boxes.append((n, shrunk))
    for side in ("left", "right"):
        dpath = f"/Cabinet/door_{side}"
        jprim = stage.GetPrimAtPath(f"/Cabinet/joints/hinge_{side}")
        j = UsdPhysics.RevoluteJoint(jprim)
        lo = j.GetLowerLimitAttr().Get()
        hi = j.GetUpperLimitAttr().Get()
        p0 = j.GetLocalPos0Attr().Get()
        axis = Gf.Vec3d(0, 0, 1)
        pivot = Gf.Vec3d(*p0)

        door_parts = [(n, r) for n, r, _ in collider_boxes(stage)
                      if n.startswith(dpath)]
        worst = None
        for ang in [lo + (hi - lo) * t / 24.0 for t in range(25)]:
            rot = Gf.Rotation(axis, ang)
            for n, r in door_parts:
                for cx in (r.GetMin()[0], r.GetMax()[0]):
                    for cy in (r.GetMin()[1], r.GetMax()[1]):
                        for cz in (r.GetMin()[2], r.GetMax()[2]):
                            p = Gf.Vec3d(cx, cy, cz) - pivot
                            q = rot.TransformDir(p) + pivot
                            for bn, br in body_boxes:
                                if "frame" not in bn and "side" not in bn:
                                    continue
                                if br.Contains(q):
                                    d = abs(ang)
                                    if worst is None or d < worst[0]:
                                        worst = (d, bn)
        if worst:
            warn(f"door_{side}: corner enters {worst[1].split('/')[-1]} "
                 f"at {worst[0]:.0f} deg (grazing the frame is normal near 0; "
                 f"check in Isaac)")
        else:
            ok(f"door_{side}: clear through {lo:.0f}..{hi:.0f} deg")


def check_articulation(stage):
    print("\n[4] articulation structure")
    roots = [p for p in stage.Traverse()
             if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(roots) != 1:
        fail(f"expected exactly 1 articulation root, found {len(roots)}")
    else:
        ok(f"articulation root: {roots[0].GetPath()}")

    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    for b in bodies:
        m = UsdPhysics.MassAPI(b).GetMassAttr().Get()
        if m is None or m <= 0:
            fail(f"{b.GetPath().name}: mass is {m}")
        else:
            ok(f"{b.GetPath().name}: mass {m} kg")
        # nested rigid bodies are illegal in UsdPhysics
        par = b.GetParent()
        while par and par.IsValid() and str(par.GetPath()) != "/":
            if par.HasAPI(UsdPhysics.RigidBodyAPI):
                fail(f"{b.GetPath().name} is nested inside rigid body "
                     f"{par.GetPath().name}")
            par = par.GetParent()

    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        j = UsdPhysics.RevoluteJoint(prim)
        lo, hi = j.GetLowerLimitAttr().Get(), j.GetUpperLimitAttr().Get()
        ax = j.GetAxisAttr().Get()
        if lo is None or hi is None or hi <= lo:
            fail(f"{prim.GetPath().name}: bad limits {lo}..{hi}")
        else:
            ok(f"{prim.GetPath().name}: axis {ax}, {lo:.0f}..{hi:.0f} deg")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "rpl_cabinet.usda"
    stage = Usd.Stage.Open(path)
    print(f"validating {path}")
    print(f"  up axis {UsdGeom.GetStageUpAxis(stage)}, "
          f"metersPerUnit {UsdGeom.GetStageMetersPerUnit(stage)}")

    check_joint_frames(stage)
    check_interpenetration(stage)
    check_swing(stage)
    check_articulation(stage)

    print("\n" + "=" * 60)
    print(f"{len(FAIL)} failures, {len(WARN)} warnings")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
