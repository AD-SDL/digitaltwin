#!/usr/bin/env python3
"""
Render the rigged cabinet USD to a PNG so the geometry can be eyeballed
without Isaac Sim.  Reads the actual stage, applies the joint rotations
itself, and draws every collider/visual primitive as a solid.

Usage: python3 preview_cabinet.py [rpl_cabinet.usda] [-o preview.png]
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pxr import Usd, UsdGeom, UsdPhysics, Gf

# unit cube faces (as vertex index quads)
FACES = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
         (2, 3, 7, 6), (0, 3, 7, 4), (1, 2, 6, 5)]
CORNERS = np.array([[-.5, -.5, -.5], [+.5, -.5, -.5], [+.5, +.5, -.5], [-.5, +.5, -.5],
                    [-.5, -.5, +.5], [+.5, -.5, +.5], [+.5, +.5, +.5], [-.5, +.5, +.5]])


def prim_color(path):
    if "keyway" in path:
        return "#2a2a2e"
    if "lever" in path or "hub" in path or "esc_" in path:
        return "#c9ccd1"
    if "shelf" in path or path.endswith("/floor"):
        return "#3a3a3e"
    if "slab" in path:
        return "#121214"
    return "#1b1b1e"


def collect(stage, door_frac=0.0, handle_frac=0.0):
    """Return (verts, color, path) for every drawable box/cylinder.

    door_frac / handle_frac are 0..1 along each joint's own limit range, so
    the preview follows whatever limits the asset actually declares rather
    than assuming a direction.
    """
    out = []

    # joint pivots and limits straight from the stage, so the preview cannot
    # drift from the asset it is previewing
    pivots, limits = {}, {}
    for jn in ("hinge_left", "hinge_right", "latch_left", "latch_right"):
        jp = stage.GetPrimAtPath(f"/Cabinet/joints/{jn}")
        if not jp.IsValid():
            continue
        j = UsdPhysics.RevoluteJoint(jp)
        b0 = j.GetBody0Rel().GetTargets()[0]
        m0 = UsdGeom.Xformable(stage.GetPrimAtPath(b0)).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default())
        pivots[jn] = np.array(m0.Transform(Gf.Vec3d(*j.GetLocalPos0Attr().Get())))
        lo, hi = j.GetLowerLimitAttr().Get(), j.GetUpperLimitAttr().Get()
        # the travelled end is whichever limit is not the closed/rest pose
        limits[jn] = hi if abs(hi) > abs(lo) else lo

    def rotz(p, c, deg):
        a = np.radians(deg)
        d = p - c
        return c + np.array([d[0] * np.cos(a) - d[1] * np.sin(a),
                             d[0] * np.sin(a) + d[1] * np.cos(a), d[2]])

    def roty(p, c, deg):
        a = np.radians(deg)
        d = p - c
        return c + np.array([d[0] * np.cos(a) + d[2] * np.sin(a), d[1],
                             -d[0] * np.sin(a) + d[2] * np.cos(a)])

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        xf = UsdGeom.Xformable(prim)
        if not xf or not (prim.IsA(UsdGeom.Cube) or prim.IsA(UsdGeom.Cylinder)):
            continue
        m = np.array(xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T

        if prim.IsA(UsdGeom.Cube):
            local = CORNERS.copy()
        else:
            cyl = UsdGeom.Cylinder(prim)
            r, h = cyl.GetRadiusAttr().Get(), cyl.GetHeightAttr().Get()
            ax = cyl.GetAxisAttr().Get()
            s = {"X": (h, 2 * r, 2 * r), "Y": (2 * r, h, 2 * r),
                 "Z": (2 * r, 2 * r, h)}[ax]
            local = CORNERS * np.array(s)

        pts = np.array([(m @ np.append(p, 1.0))[:3] for p in local])

        for side in ("left", "right"):
            if f"/door_{side}" not in path and f"/handle_{side}" not in path:
                continue
            if f"/handle_{side}" in path and handle_frac:
                c = pivots[f"latch_{side}"]
                pts = np.array([roty(p, c, limits[f"latch_{side}"] * handle_frac)
                                for p in pts])
            if door_frac:
                c = pivots[f"hinge_{side}"]
                pts = np.array([rotz(p, c, limits[f"hinge_{side}"] * door_frac)
                                for p in pts])

        out.append((pts, prim_color(path), path))
    return out


def draw(ax, solids, title, elev, azim, bounds=None, only=None):
    # matplotlib's 3D depth sort is per-polygon and crude, so a large slab
    # will paint over small parts in front of it.  `only` restricts a panel
    # to the prims that matter for it.
    for pts, col, path in solids:
        if only and not any(k in path for k in only):
            continue
        polys = [[pts[i] for i in f] for f in FACES]
        edge = "#8a8a92" if only else "#5a5a62"
        pc = Poly3DCollection(polys, facecolor=col, edgecolor=edge,
                              linewidths=0.3, alpha=1.0)
        ax.add_collection3d(pc)
    if bounds:
        (x0, x1), (y0, y1), (z0, z1) = bounds
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(z0, z1)
        ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))
    else:
        ax.set_xlim(-0.75, 0.75); ax.set_ylim(-0.75, 0.75); ax.set_zlim(0, 2.05)
        ax.set_box_aspect((1.5, 1.5, 2.05))
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=10, color="#222")
    ax.set_xlabel("X (m)", fontsize=7); ax.set_ylabel("Y (m)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(False)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.fill = False
        pane.pane.set_edgecolor("#dddddd")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("usd", nargs="?", default="rpl_cabinet.usda")
    ap.add_argument("-o", "--output", default="cabinet_preview.png")
    a = ap.parse_args()

    stage = Usd.Stage.Open(a.usd)
    fig = plt.figure(figsize=(19, 6), dpi=130)
    fig.patch.set_facecolor("white")

    LATCH = ("handle_", "esc_")
    ZOOM = ((-0.18, 0.18), (-0.30, 0.06), (0.90, 1.14))
    panels = [
        (0.0, 0.0, "closed, front", 8, -90, None, None),
        (1.0, 1.0, "both doors fully open", 16, -58, None, None),
        (0.0, 0.0, "latch pair at rest", 12, -72, ZOOM, LATCH),
        (0.0, 1.0, "latch pair fully turned", 12, -72, ZOOM, LATCH),
    ]
    for i, (dang, hang, title, elev, azim, bounds, only) in enumerate(panels):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        draw(ax, collect(stage, dang, hang), title, elev, azim, bounds, only)

    plt.tight_layout()
    plt.savefig(a.output, facecolor="white", bbox_inches="tight")
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
