#!/usr/bin/env python3
"""
RPL black storage cabinet -> rigged USD for Isaac Sim.

Global Industrial "Turn Handle" full-height storage cabinet,
36"W x 18"D x 78"H, black.  MPN 603599BK.

Builds an articulated asset:
    body (fixed to world)
      |- door_left   revolute about Z, hinged on the left stile
      |    |- handle_left   revolute about Y (the turn handle)
      |- door_right  revolute about Z, hinged on the right stile
           |- handle_right  revolute about Y

Every collider is an exact box or cylinder primitive.  Nothing is convex
hulled, so the interior stays hollow and the wrist cameras see real shelves.

Stage is metres, Z up, origin at the centre of the cabinet footprint on
the floor.  The cabinet front faces -Y.

Usage:
    python3 build_cabinet.py [-o OUTPUT.usda] [--no-shelves]

Requires: pip install usd-core
"""

import argparse
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf

MM = 0.001

# --------------------------------------------------------------------------
# MEASURED  (from the team's sketch, img_2094 - these are trusted)
# --------------------------------------------------------------------------
H             = 1980 * MM   # overall height
W             =  917 * MM   # overall width
D             =  458 * MM   # overall depth
DOOR_W        =  421 * MM   # each door leaf
FRAME_SIDE    =   36 * MM   # fixed border down each side of the front
HANDLE_Z      = 1020 * MM   # handle pivot centre above the floor
HANDLE_INSET  =   75 * MM   # handle pivot in from the door's free edge
LEVER_LEN     =  110 * MM   # lever arm length from pivot
PIVOT_OFF     =   40 * MM   # door face out to the lever, i.e. the pivot standoff
KEY_DIA       =   25 * MM   # the round keyway face on the handle head

# --------------------------------------------------------------------------
# ASSUMED  -- not on the sketch.  CONFIRM THESE, they are easy to change.
# --------------------------------------------------------------------------
DOOR_T        =   20 * MM   # door slab thickness
PANEL_T       =    2 * MM   # carcass sheet steel
TOE_KICK_H    =  100 * MM   # recessed base height
TOE_RECESS    =   25 * MM   # how far the toe kick sets back from the front
FRAME_TOP     =   36 * MM   # border above the doors (assumed = FRAME_SIDE)
N_SHELVES     =    4
SHELF_T       =    3 * MM
SHELF_BACK_GAP =  20 * MM   # shelf stops short of the back panel

HUB_LEN       =   22 * MM   # depth of the round head along the pivot axis
FLANGE_R      =   20 * MM   # escutcheon flange against the door
FLANGE_T      =    5 * MM
NECK_R        =   11 * MM   # neck from the flange out to the head
BOSS_GAP      =    2 * MM   # clearance between fixed boss and turning head

DOOR_OPEN_DEG   = 110.0     # max door swing
HANDLE_TURN_DEG =  90.0     # lever throw to release the latch
LEVER_REST_DEG  =   0.0     # 0 = lever hangs straight down

DOOR_MASS   = 9.0
HANDLE_MASS = 0.15
BODY_MASS   = 60.0

# --------------------------------------------------------------------------
# Derived
# --------------------------------------------------------------------------
CENTER_GAP  = W - 2 * FRAME_SIDE - 2 * DOOR_W      # 3 mm, falls out of the measurements
DOOR_BOT    = TOE_KICK_H
DOOR_TOP    = H - FRAME_TOP
DOOR_H      = DOOR_TOP - DOOR_BOT
DOOR_MID_Z  = 0.5 * (DOOR_BOT + DOOR_TOP)

FRONT_Y     = -D / 2.0                 # outer plane of the doors
BACK_Y      =  D / 2.0
DOOR_MID_Y  = FRONT_Y + DOOR_T / 2.0   # door slab centre plane

HINGE_L_X   = -W / 2.0 + FRAME_SIDE    # left  door hinge axis
HINGE_R_X   =  W / 2.0 - FRAME_SIDE    # right door hinge axis
FREE_L_X    = HINGE_L_X + DOOR_W       # left  door free edge
FREE_R_X    = HINGE_R_X - DOOR_W       # right door free edge

PIVOT_L_X   = FREE_L_X - HANDLE_INSET
PIVOT_R_X   = FREE_R_X + HANDLE_INSET
PIVOT_Y     = FRONT_Y - PIVOT_OFF      # in front of the door face

HUB_R       = KEY_DIA / 2.0            # the round head carries the keyway
BOSS_H      = PIVOT_OFF - HUB_LEN / 2.0 - BOSS_GAP   # fixed boss, door -> head


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _xform(stage, path, translate=(0, 0, 0)):
    x = UsdGeom.Xform.Define(stage, path)
    x.AddTranslateOp().Set(Gf.Vec3d(*translate))
    return x


def add_box(stage, path, size, center, material=None, collision=True):
    """Axis-aligned box of dimensions `size` centred at `center` (parent frame)."""
    sx, sy, sz = size
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateExtentAttr([Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)])
    cube.AddTranslateOp().Set(Gf.Vec3d(*center))
    cube.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    if material:
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    return cube


def add_cyl(stage, path, radius, height, center, axis="Z", material=None,
            collision=True):
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    cyl.CreateAxisAttr(axis)
    h, r = height / 2.0, radius
    cyl.CreateExtentAttr([Gf.Vec3f(-r, -r, -h), Gf.Vec3f(r, r, h)])
    cyl.AddTranslateOp().Set(Gf.Vec3d(*center))
    if collision:
        UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    if material:
        UsdShade.MaterialBindingAPI.Apply(cyl.GetPrim()).Bind(material)
    return cyl


def make_material(stage, path, color, roughness, metallic):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def rigid_body(stage, path, mass, translate=(0, 0, 0)):
    x = _xform(stage, path, translate)
    p = x.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(p)
    UsdPhysics.MassAPI.Apply(p).CreateMassAttr(mass)
    return x


def revolute(stage, path, body0, body1, axis, lo, hi,
             local_pos0, local_pos1, damping=2.0, max_force=500.0):
    j = UsdPhysics.RevoluteJoint.Define(stage, path)
    j.CreateBody0Rel().SetTargets([body0])
    j.CreateBody1Rel().SetTargets([body1])
    j.CreateAxisAttr(axis)
    j.CreateLocalPos0Attr(Gf.Vec3f(*local_pos0))
    j.CreateLocalPos1Attr(Gf.Vec3f(*local_pos1))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0))
    j.CreateLowerLimitAttr(lo)
    j.CreateUpperLimitAttr(hi)
    # passive hinge: no stiffness, a little damping to stand in for friction
    drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(damping)
    drive.CreateMaxForceAttr(max_force)
    drive.CreateTargetPositionAttr(0.0)
    return j


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------
def build(out_path, shelves=True):
    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/Cabinet")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

    looks = UsdGeom.Scope.Define(stage, "/Cabinet/Looks")
    # Painted steel is a DIELECTRIC, not a metal: the paint layer is what the
    # light hits.  Setting metallic high makes it a black void with no diffuse
    # response, which is exactly how the first render came out.  Chrome is the
    # only genuinely metallic surface here.
    steel  = make_material(stage, "/Cabinet/Looks/BlackSteel", (0.035, 0.035, 0.038), 0.45, 0.0)
    chrome = make_material(stage, "/Cabinet/Looks/Chrome",     (0.83, 0.85, 0.88),   0.08, 1.0)
    inner  = make_material(stage, "/Cabinet/Looks/InnerSteel", (0.075, 0.075, 0.080), 0.60, 0.0)

    # ---- body ------------------------------------------------------------
    body = rigid_body(stage, "/Cabinet/body", BODY_MASS)
    B = "/Cabinet/body"
    inner_w = W - 2 * PANEL_T

    add_box(stage, B + "/side_left",  (PANEL_T, D, H),
            (-W / 2 + PANEL_T / 2, 0, H / 2), steel)
    add_box(stage, B + "/side_right", (PANEL_T, D, H),
            ( W / 2 - PANEL_T / 2, 0, H / 2), steel)
    add_box(stage, B + "/top",        (W, D, PANEL_T),
            (0, 0, H - PANEL_T / 2), steel)
    add_box(stage, B + "/back",       (W, PANEL_T, H),
            (0, BACK_Y - PANEL_T / 2, H / 2), steel)
    # Storage floor sits at the top of the toe kick, behind the closed door.
    floor_d = D - DOOR_T
    add_box(stage, B + "/floor",      (inner_w, floor_d, PANEL_T),
            (0, FRONT_Y + DOOR_T + floor_d / 2, TOE_KICK_H + PANEL_T / 2), inner)
    add_box(stage, B + "/toe_kick",   (W, PANEL_T, TOE_KICK_H),
            (0, FRONT_Y + TOE_RECESS, TOE_KICK_H / 2), steel)

    # front frame: the fixed border the doors sit inside.
    # Stops below frame_top so the two do not double-model the corner.
    frame_h = DOOR_TOP - TOE_KICK_H
    add_box(stage, B + "/frame_left",  (FRAME_SIDE, PANEL_T, frame_h),
            (-W / 2 + FRAME_SIDE / 2, FRONT_Y + PANEL_T / 2, TOE_KICK_H + frame_h / 2), steel)
    add_box(stage, B + "/frame_right", (FRAME_SIDE, PANEL_T, frame_h),
            ( W / 2 - FRAME_SIDE / 2, FRONT_Y + PANEL_T / 2, TOE_KICK_H + frame_h / 2), steel)
    add_box(stage, B + "/frame_top",   (W, PANEL_T, FRAME_TOP),
            (0, FRONT_Y + PANEL_T / 2, H - FRAME_TOP / 2), steel)

    if shelves:
        # Shelves live behind the closed door and stop short of the back panel.
        inner_front = FRONT_Y + DOOR_T
        inner_back = BACK_Y - PANEL_T - SHELF_BACK_GAP
        shelf_d = inner_back - inner_front
        shelf_y = 0.5 * (inner_front + inner_back)
        span = DOOR_TOP - (TOE_KICK_H + PANEL_T)
        for i in range(N_SHELVES):
            z = TOE_KICK_H + PANEL_T + span * (i + 1) / (N_SHELVES + 1)
            add_box(stage, B + f"/shelf_{i+1}", (inner_w, shelf_d, SHELF_T),
                    (0, shelf_y, z), inner)

    # ---- doors -----------------------------------------------------------
    # Each door Xform sits ON its hinge axis, so the joint's child frame is
    # the origin and the slab is offset inside it.
    for side, hinge_x, direction, lo, hi in (
        ("left",  HINGE_L_X,  1.0, -DOOR_OPEN_DEG, 0.0),
        ("right", HINGE_R_X, -1.0, 0.0,  DOOR_OPEN_DEG),
    ):
        dpath = f"/Cabinet/door_{side}"
        rigid_body(stage, dpath, DOOR_MASS, (hinge_x, DOOR_MID_Y, 0.0))
        add_box(stage, dpath + "/slab", (DOOR_W, DOOR_T, DOOR_H),
                (direction * DOOR_W / 2.0, 0.0, DOOR_MID_Z), steel)
        # Escutcheon is fixed to the door; only the head and lever turn.
        # Visual only: the lever and hub carry the collision, and a collider
        # here would sit permanently against the rotating head.
        free_local = direction * DOOR_W
        pivot_local_x = free_local - direction * HANDLE_INSET
        add_cyl(stage, dpath + "/esc_flange", FLANGE_R, FLANGE_T,
                (pivot_local_x, -DOOR_T / 2 - FLANGE_T / 2, HANDLE_Z),
                axis="Y", material=chrome, collision=False)
        neck_h = BOSS_H - FLANGE_T
        add_cyl(stage, dpath + "/esc_neck", NECK_R, neck_h,
                (pivot_local_x, -DOOR_T / 2 - FLANGE_T - neck_h / 2, HANDLE_Z),
                axis="Y", material=chrome, collision=False)

        revolute(stage, f"/Cabinet/joints/hinge_{side}",
                 Sdf.Path("/Cabinet/body"), Sdf.Path(dpath),
                 "Z", lo, hi,
                 local_pos0=(hinge_x, DOOR_MID_Y, 0.0),
                 local_pos1=(0.0, 0.0, 0.0),
                 damping=3.0, max_force=800.0)

    # ---- handles ---------------------------------------------------------
    # The pair mirrors: each lever sweeps away from the centre seam, so they
    # never cross over one another.  CONFIRM the real release direction.
    for side, pivot_x in (("left", PIVOT_L_X), ("right", PIVOT_R_X)):
        hpath = f"/Cabinet/handle_{side}"
        rigid_body(stage, hpath, HANDLE_MASS, (pivot_x, PIVOT_Y, HANDLE_Z))
        # Round head at the pivot, lever hanging down from it.  The head's
        # outward face is the 25 mm keyway the sketch dimensions.
        add_cyl(stage, hpath + "/hub", HUB_R, HUB_LEN, (0, 0, 0),
                axis="Y", material=chrome)
        add_box(stage, hpath + "/keyway", (0.012, 0.002, 0.003),
                (0.0, -HUB_LEN / 2 - 0.001, 0.0), inner, collision=False)
        add_box(stage, hpath + "/lever", (0.022, 0.028, LEVER_LEN),
                (0.0, 0.0, -LEVER_LEN / 2.0), chrome)

        dpath = f"/Cabinet/door_{side}"
        direction = 1.0 if side == "left" else -1.0
        free_local = direction * DOOR_W
        pivot_local_x = free_local - direction * HANDLE_INSET
        # +Y rotation swings the lever toward -X, so the left latch takes the
        # positive range and the right latch the negative one.
        lo, hi = (0.0, HANDLE_TURN_DEG) if side == "left" else (-HANDLE_TURN_DEG, 0.0)
        # localPos0 is in the door's frame, whose origin lies on the door
        # centre plane, so the pivot is DOOR_T/2 + PIVOT_OFF out in front.
        revolute(stage, f"/Cabinet/joints/latch_{side}",
                 Sdf.Path(dpath), Sdf.Path(hpath),
                 "Y", lo, hi,
                 local_pos0=(pivot_local_x, -(DOOR_T / 2 + PIVOT_OFF), HANDLE_Z),
                 local_pos1=(0.0, 0.0, 0.0),
                 damping=0.05, max_force=50.0)

    # ---- pin the carcass to the world ------------------------------------
    fj = UsdPhysics.FixedJoint.Define(stage, "/Cabinet/joints/fix_base")
    fj.CreateBody1Rel().SetTargets([Sdf.Path("/Cabinet/body")])
    fj.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
    fj.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))

    stage.GetRootLayer().Save()
    return stage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="rpl_cabinet.usda")
    ap.add_argument("--no-shelves", action="store_true")
    a = ap.parse_args()

    import os
    if os.path.exists(a.output):
        os.remove(a.output)
    build(a.output, shelves=not a.no_shelves)

    print(f"wrote {a.output}")
    print(f"  overall      {W*1000:.0f} x {D*1000:.0f} x {H*1000:.0f} mm")
    print(f"  door leaf    {DOOR_W*1000:.0f} mm, centre gap {CENTER_GAP*1000:.1f} mm")
    print(f"  door span    z {DOOR_BOT*1000:.0f} -> {DOOR_TOP*1000:.0f} mm "
          f"(centre {DOOR_MID_Z*1000:.0f})")
    print(f"  handle pivot z {HANDLE_Z*1000:.0f} mm, x +/-{PIVOT_L_X*-1000:.1f} mm")


if __name__ == "__main__":
    main()
