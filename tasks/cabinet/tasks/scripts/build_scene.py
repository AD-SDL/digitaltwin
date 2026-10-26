#!/usr/bin/env python3
"""
Compose the RPL door-opening scene: the cabinet with the bimanual OpenArm
standing in front of it, framed like OpenArm_Door.mov.  Renders a still and
saves the composed stage so it can be opened in the Isaac GUI.

Run under the Isaac Lab venv:
    <isaaclab_root>/isaaclab.sh -p build_scene.py

Placement is computed from the assets' real bounding boxes rather than
hardcoded, so it survives either asset changing.
"""

import argparse
import os
import sys

# Everything resolves against the SCRIPT's directory, not the shell's cwd.
# Running this from anywhere else used to fail with
# "ModuleNotFoundError: No module named 'openarm_cameras'".
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _here(*p):
    return os.path.join(HERE, *p)


ap = argparse.ArgumentParser()
ap.add_argument("--cabinet", default=_here("rpl_cabinet.usda"))
ap.add_argument("--openarm", default=_here(
    "assets", "OpenArm", "openarm_bimanual", "openarm_bimanual.usd"),
    help="NVIDIA ships this under /tmp/Assets but owned 0600 by another "
         "user; we keep a readable copy fetched from the public S3 root")
ap.add_argument("--out-dir", default=_here("render"))
ap.add_argument("--scene-usd", default=_here("rpl_door_scene.usd"))
ap.add_argument("--result", default=_here("scene_result.txt"))
# arm placement relative to the cabinet front face
ap.add_argument("--standoff", type=float, default=0.55,
                help="metres from the cabinet front face to the arm base")
ap.add_argument("--lateral", type=float, default=0.40,
                help="metres the arm sits to +X of the cabinet centre")
ap.add_argument("--arm-yaw", type=float, default=-90.0,
                help="degrees. The asset's shoulder line runs along Y and it\n"
                     "faces -X, so -90 turns it to face +Y, i.e. the cabinet")
ap.add_argument("--arm-lift", type=float, default=0.95,
                help="metres to raise the arm base. The asset is only 773 mm\n"
                     "tall with no stand; the real rig is on a tall tripod so\n"
                     "the grippers can reach the 1020 mm handle")
# camera
ap.add_argument("--cam", default="1.95,-3.15,1.62")
ap.add_argument("--look", default="-0.02,-0.40,1.18")
ap.add_argument("--res", default="1920x1080")
ap.add_argument("--no-render", action="store_true")
ap.add_argument("--focal", type=float, default=24.0)
ap.add_argument("--cams", default="",
                help="sweep several viewpoints in one launch: "
                     "\"cx,cy,cz|lx,ly,lz;...\"")
ap.add_argument("--pose", default="",
                help="left-arm joint angles in DEGREES, j1..j7, taken from "
                     "the real episode")
ap.add_argument("--pose-arm", default="left")
ap.add_argument("--align-handle", default="",
                help="x,y,z of the lever to land the gripper on. The pose is\n"
                     "fixed, so this is a rigid base shift, not IK.")
ap.add_argument("--robot-cams", action="store_true",
                help="add the real chest D435i + two wrist UVC cameras and\n"
                     "render through them at their dataset resolutions")
ap.add_argument("--flatten", action="store_true",
                help="inline the OpenArm geometry into the scene file "
                     "(self-contained but ~89 MB); default keeps references")
ap.add_argument("--contact", action="store_true",
                help="with --cams, render each at 1/3 res for a contact sheet")
args = ap.parse_args()

_R = open(args.result, "w", buffering=1)


def say(m=""):
    print(m, flush=True)
    _R.write(m + "\n")
    _R.flush()
    os.fsync(_R.fileno())


from isaacsim import SimulationApp                                # noqa: E402
W, H = (int(v) for v in args.res.split("x"))
sim_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting",
                         "width": W, "height": H})

import numpy as np                                                 # noqa: E402
from pxr import Usd, UsdGeom, UsdLux, UsdPhysics, Gf, Sdf          # noqa: E402

from isaacsim.core.api import World                                # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage       # noqa: E402
import omni.usd                                                    # noqa: E402

FAIL = []

# cabinet bbox min-y includes the handles, which stand proud of the door;
# the door face itself is 40 mm behind that
DOOR_INSET = 0.054


def bbox(stage, path):
    c = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    return c.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()


def fmt(r):
    return (f"min({r.GetMin()[0]:.3f},{r.GetMin()[1]:.3f},{r.GetMin()[2]:.3f}) "
            f"max({r.GetMax()[0]:.3f},{r.GetMax()[1]:.3f},{r.GetMax()[2]:.3f})")


def set_pose(stage, path, translate=None, yaw_deg=None):
    x = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    x.ClearXformOpOrder()
    if translate is not None:
        x.AddTranslateOp().Set(Gf.Vec3d(*translate))
    if yaw_deg:
        x.AddRotateZOp().Set(float(yaw_deg))


def make_refs_relative(stage, scene_dir, log=print):
    """Rewrite absolute reference paths on the root layer to be relative to
    the scene file, so the bundle is portable.

    The stage is built with absolute paths because they have to resolve
    while we compute bounding boxes; this runs just before export.
    """
    layer = stage.GetRootLayer()
    n = 0
    for prim in stage.Traverse():
        spec = layer.GetPrimAtPath(prim.GetPath())
        if not spec:
            continue
        rl = spec.referenceList
        for field in ("prependedItems", "addedItems", "explicitItems"):
            items = list(getattr(rl, field, []))
            if not items:
                continue
            new = []
            changed = False
            for r in items:
                ap = r.assetPath
                if ap and os.path.isabs(ap) and not ap.startswith(("http", "omniverse")):
                    rel = os.path.relpath(ap, scene_dir)
                    new.append(Sdf.Reference(rel, r.primPath, r.layerOffset))
                    changed = True
                    n += 1
                else:
                    new.append(r)
            if changed:
                setattr(rl, field, new)
    log(f"[portable] rewrote {n} reference(s) to relative paths")
    return n


def main():
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    # NOTE: deliberately NOT world.scene.add_default_ground_plane().  That
    # adds a reference to an https:// asset that only Isaac can resolve, so
    # the saved scene breaks anywhere else, and it is the blue grid that our
    # own floor was covering up anyway.  We lay a real collidable floor below.

    # The default ground is the blue Isaac grid.  Lay a pale floor over it to
    # stand in for the RPL's white raised-tile floor, which also gives the
    # black cabinet something to bounce light off.
    from pxr import UsdShade
    floor = UsdGeom.Cube.Define(stage, "/World/Floor")
    floor.CreateSizeAttr(1.0)
    floor.AddTranslateOp().Set(Gf.Vec3d(0, -1.5, -0.01))
    floor.AddScaleOp().Set(Gf.Vec3f(14.0, 14.0, 0.02))
    fm = UsdShade.Material.Define(stage, "/World/Looks/Floor")
    fs = UsdShade.Shader.Define(stage, "/World/Looks/Floor/S")
    fs.CreateIdAttr("UsdPreviewSurface")
    fs.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.80, 0.80, 0.78))
    fs.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
    fs.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    fm.CreateSurfaceOutput().ConnectToSource(fs.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(fm)
    # this floor is now the only ground, so it has to collide
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())

    # a back wall, so the dome does not read as open sky behind the cabinet
    wall = UsdGeom.Cube.Define(stage, "/World/Wall")
    wall.CreateSizeAttr(1.0)
    wall.AddTranslateOp().Set(Gf.Vec3d(0, 0.40, 1.6))
    wall.AddScaleOp().Set(Gf.Vec3f(12.0, 0.05, 3.2))
    wm = UsdShade.Material.Define(stage, "/World/Looks/Wall")
    ws = UsdShade.Shader.Define(stage, "/World/Looks/Wall/S")
    ws.CreateIdAttr("UsdPreviewSurface")
    ws.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.62, 0.61, 0.57))
    ws.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
    wm.CreateSurfaceOutput().ConnectToSource(ws.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(wall.GetPrim()).Bind(wm)

    # ---- cabinet at the origin, front face toward -Y --------------------
    add_reference_to_stage(usd_path=os.path.abspath(args.cabinet),
                           prim_path="/World/Cabinet")
    cab = bbox(stage, "/World/Cabinet")
    if cab.IsEmpty():
        say(f"  FAIL  cabinet bbox is EMPTY -> {args.cabinet} did not load")
        FAIL.append("cabinet did not load")
        raise RuntimeError(f"cabinet asset produced no geometry: {args.cabinet}")
    say(f"[cabinet] {fmt(cab)}")
    front_y = cab.GetMin()[1]

    # ---- OpenArm --------------------------------------------------------
    add_reference_to_stage(usd_path=args.openarm, prim_path="/World/OpenArm")
    raw = bbox(stage, "/World/OpenArm")
    if raw.IsEmpty():
        # the previous run sailed past this and "placed" the arm at -3.4e38
        n = len(list(Usd.PrimRange(stage.GetPrimAtPath("/World/OpenArm"))))
        say(f"  FAIL  OpenArm bbox is EMPTY -> asset did not load "
            f"(descendants={n}, path={args.openarm})")
        FAIL.append("openarm did not load")
        raise RuntimeError("OpenArm asset produced no geometry")
    say(f"[openarm raw] {fmt(raw)}")

    roots = [str(p.GetPath()) for p in stage.Traverse()
             if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    say(f"[articulation roots] {roots}")

    # Drop the arm onto the floor, then stand it off in front of the cabinet.
    base_z = -raw.GetMin()[2] + args.arm_lift
    arm_pos = (args.lateral, front_y - args.standoff, base_z)
    set_pose(stage, "/World/OpenArm", arm_pos, args.arm_yaw)
    placed = bbox(stage, "/World/OpenArm")
    say(f"[openarm placed] at {tuple(round(v,3) for v in arm_pos)} "
        f"yaw {args.arm_yaw}  ->  {fmt(placed)}")

    gap = cab.GetMin()[1] - placed.GetMax()[1]
    say(f"[clearance] arm front to cabinet face: {gap*1000:.0f} mm")
    if gap < -0.02:
        FAIL.append(f"arm intersects the cabinet by {-gap*1000:.0f} mm")
        say(f"  FAIL  arm intersects the cabinet by {-gap*1000:.0f} mm")
    else:
        say("  ok    arm clears the cabinet")

    # ---- stand, so the torso is not floating ----------------------------
    if args.arm_lift > 0.05:
        stand_mat = UsdShade.Material.Define(stage, "/World/Looks/Stand")
        ss = UsdShade.Shader.Define(stage, "/World/Looks/Stand/S")
        ss.CreateIdAttr("UsdPreviewSurface")
        ss.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.07, 0.07, 0.075))
        ss.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        stand_mat.CreateSurfaceOutput().ConnectToSource(ss.ConnectableAPI(),
                                                        "surface")
        sx, sy = args.lateral, front_y - args.standoff
        col = UsdGeom.Cube.Define(stage, "/World/Stand/Column")
        col.CreateSizeAttr(1.0)
        col.AddTranslateOp().Set(Gf.Vec3d(sx, sy, base_z / 2.0))
        col.AddScaleOp().Set(Gf.Vec3f(0.09, 0.09, base_z))
        UsdShade.MaterialBindingAPI.Apply(col.GetPrim()).Bind(stand_mat)
        foot = UsdGeom.Cube.Define(stage, "/World/Stand/Foot")
        foot.CreateSizeAttr(1.0)
        foot.AddTranslateOp().Set(Gf.Vec3d(sx, sy, 0.015))
        foot.AddScaleOp().Set(Gf.Vec3f(0.52, 0.52, 0.03))
        UsdShade.MaterialBindingAPI.Apply(foot.GetPrim()).Bind(stand_mat)
        say(f"[stand] column to z={base_z:.3f} at ({sx:.2f}, {sy:.2f})")

    # handle the robot should be reaching for
    say(f"[handle targets] left (-0.0765,-0.254,1.020)  right (0.0765,-0.254,1.020)")
    say(f"[reach check] grippers at z~{placed.GetMin()[2]:.2f}-{placed.GetMax()[2]:.2f}, "
        f"handle at z=1.020")

    # ---- lighting: overhead lab, not a studio ---------------------------
    # A near-black dielectric needs a lot of light to read at all.
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(900.0)
    dome.CreateColorAttr(Gf.Vec3f(0.88, 0.91, 0.96))

    # overhead strip fixtures, pointing down
    for i, lx in enumerate((-1.4, 0.0, 1.4)):
        r = UsdLux.RectLight.Define(stage, f"/World/Lights/Panel_{i}")
        r.CreateIntensityAttr(28000.0)
        r.CreateWidthAttr(1.4)
        r.CreateHeightAttr(0.45)
        r.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.94))
        xf = UsdGeom.Xformable(r.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(lx, -1.1, 2.80))
        xf.AddRotateXOp().Set(-90.0)

    # key light from the camera side so the door faces and the chrome catch
    key = UsdLux.RectLight.Define(stage, "/World/Lights/Key")
    key.CreateIntensityAttr(42000.0)
    key.CreateWidthAttr(2.0)
    key.CreateHeightAttr(1.6)
    key.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.92))
    kx = UsdGeom.Xformable(key.GetPrim())
    kx.AddTranslateOp().Set(Gf.Vec3d(2.6, -2.9, 2.20))
    kx.AddRotateYOp().Set(-38.0)
    kx.AddRotateXOp().Set(-28.0)

    world.reset()
    for _ in range(40):
        world.step(render=False)
    say("[physics] settled 40 steps")

    # ---- pose the arm from the real demonstration -----------------------
    # Rather than invent a reaching pose, replay the recorded left-arm joint
    # angles from the actual episode.  Dataset values are DEGREES.
    if args.pose:
        try:
            from isaacsim.core.prims import SingleArticulation as Art
        except ImportError:
            from omni.isaac.core.articulations import Articulation as Art
        art = Art("/World/OpenArm")
        art.initialize()
        dofs = list(art.dof_names or [])
        say(f"[openarm dofs] {len(dofs)}: {dofs}")

        want = [float(v) for v in args.pose.split(",")]
        side = args.pose_arm
        q = np.asarray(art.get_joint_positions()).reshape(-1).copy()
        # Names are like 'openarm_left_joint4'.  Match exactly: a substring
        # test on 'joint4' would also hit 'openarm_left_finger_joint4'.
        matched = []
        for j in range(1, len(want) + 1):
            target = f"openarm_{side}_joint{j}"
            for n_i, nm in enumerate(dofs):
                if nm.lower() == target:
                    q[n_i] = np.radians(want[j - 1])
                    matched.append((nm, want[j - 1]))
                    break
        if matched:
            art.set_joint_positions(q)
            for _ in range(5):
                world.step(render=False)
            say(f"[pose] applied {len(matched)} joints: "
                + ", ".join(f"{n}={v:.1f}d" for n, v in matched))
        else:
            say(f"  warn  no '{side}' dof names matched j1..j{len(want)}; "
                f"left in the default pose")

        # ---- land the gripper on the handle -----------------------------
        # The pose is fixed, so this is a rigid translation of the base, not
        # an IK problem: measure where the hand ended up and shift by the
        # difference to the handle.
        if args.align_handle and matched:
            # Must be a LINK with real geometry.  Joint prims match the same
            # name patterns but carry no extent, and their bbox reads as the
            # origin, which silently teleports the robot.
            cands = []
            for p in Usd.PrimRange(stage.GetPrimAtPath("/World/OpenArm")):
                nm = p.GetName().lower()
                # must be THIS arm: a bare `side in nm` also matches
                # 'openarm_right_left_finger', which is a right-arm link
                if not nm.startswith(f"openarm_{side}_"):
                    continue
                if "joint" in nm:
                    continue
                if not any(k in nm for k in ("hand", "finger", "gripper")):
                    continue
                r = bbox(stage, str(p.GetPath()))
                if r.IsEmpty():
                    continue
                c = np.array([(r.GetMin()[i] + r.GetMax()[i]) / 2.0
                              for i in range(3)])
                if np.linalg.norm(c) < 1e-6:      # sitting exactly at origin
                    continue
                cands.append((p, c))
            if not cands:
                say("  warn  no hand/finger LINK with geometry found; "
                    "skipping alignment")
            else:
                say(f"[align] {len(cands)} hand link(s): "
                    + ", ".join(p.GetName() for p, _ in cands[:6]))
                hand = cands[0][0]
                tip = np.mean([c for _, c in cands], axis=0)
                tgt = np.array([float(v) for v in args.align_handle.split(",")])
                d = tgt - tip
                say(f"[align] hand prim {hand.GetName()} at "
                    f"({tip[0]:.3f},{tip[1]:.3f},{tip[2]:.3f}), target "
                    f"({tgt[0]:.3f},{tgt[1]:.3f},{tgt[2]:.3f}), delta "
                    f"({d[0]:.3f},{d[1]:.3f},{d[2]:.3f})")
                new_pos = (arm_pos[0] + d[0], arm_pos[1] + d[1],
                           arm_pos[2] + d[2])
                set_pose(stage, "/World/OpenArm", new_pos, args.arm_yaw)
                say(f"[align] base moved to "
                    f"({new_pos[0]:.3f},{new_pos[1]:.3f},{new_pos[2]:.3f})")
                # rebuild the stand under the new base
                for pth in ("/World/Stand/Column", "/World/Stand/Foot"):
                    if stage.GetPrimAtPath(pth):
                        stage.RemovePrim(pth)
                bz = new_pos[2]
                col = UsdGeom.Cube.Define(stage, "/World/Stand/Column")
                col.CreateSizeAttr(1.0)
                col.AddTranslateOp().Set(Gf.Vec3d(new_pos[0], new_pos[1], bz / 2))
                col.AddScaleOp().Set(Gf.Vec3f(0.09, 0.09, max(bz, 0.01)))
                UsdShade.MaterialBindingAPI.Apply(col.GetPrim()).Bind(stand_mat)
                ft = UsdGeom.Cube.Define(stage, "/World/Stand/Foot")
                ft.CreateSizeAttr(1.0)
                ft.AddTranslateOp().Set(Gf.Vec3d(new_pos[0], new_pos[1], 0.015))
                ft.AddScaleOp().Set(Gf.Vec3f(0.52, 0.52, 0.03))
                UsdShade.MaterialBindingAPI.Apply(ft.GetPrim()).Bind(stand_mat)
                for _ in range(5):
                    world.step(render=False)
                fin = bbox(stage, "/World/OpenArm")
                say(f"[align] arm now {fmt(fin)}")
                # The earlier clearance check ran BEFORE this move, so it was
                # stale.  Re-check against the door face, allowing the hand to
                # reach the lever, which stands proud of the door.
                lever_y = float(args.align_handle.split(",")[1])
                over = lever_y - fin.GetMax()[1]
                say(f"[align] arm front {fin.GetMax()[1]:.3f} vs lever plane "
                    f"{lever_y:.3f} -> {over*1000:+.0f} mm")
                if fin.GetMax()[1] > cab.GetMin()[1] + DOOR_INSET:
                    FAIL.append("arm penetrates the cabinet after alignment")
                    say(f"  FAIL  arm reaches y={fin.GetMax()[1]:.3f}, past the "
                        f"door face at {cab.GetMin()[1] + DOOR_INSET:.3f}")
                else:
                    say("  ok    arm stays clear of the door face")

    # ---- the robot's real cameras ---------------------------------------
    robot_cams = []
    if args.robot_cams:
        from openarm_cameras import add_openarm_cameras
        robot_cams = add_openarm_cameras(stage, "/World/OpenArm", log=say)

    # ---- save the composed stage for the GUI ----------------------------
    # stage.Export() FLATTENS, which inlines the 89 MB OpenArm geometry into
    # the scene file.  Exporting the root layer instead keeps the reference,
    # giving a few-KB scene that still opens, as long as assets/OpenArm/ sits
    # alongside it.  Use --flatten for a self-contained copy to hand out.
    out_usd = os.path.abspath(args.scene_usd)
    if args.flatten:
        stage.Export(out_usd)
    else:
        make_refs_relative(stage, os.path.dirname(out_usd), log=say)
        stage.GetRootLayer().Export(out_usd)
    sz = os.path.getsize(out_usd)
    say(f"[scene] {out_usd}  ({sz/1e6:.1f} MB, "
        f"{'flattened' if args.flatten else 'references preserved'})")

    if args.no_render:
        return

    # ---- render ---------------------------------------------------------
    import omni.replicator.core as rep

    # One run can sweep several viewpoints, which beats guessing one at a
    # time at ~3 min per launch.  --cams is "cx,cy,cz|lx,ly,lz;..." .
    if args.cams:
        specs = []
        for i, chunk in enumerate(args.cams.split(";")):
            parts = chunk.split("|")
            cs, ls = parts[0], parts[1]
            fl = float(parts[2]) if len(parts) > 2 else args.focal
            specs.append(([float(v) for v in cs.split(",")],
                          [float(v) for v in ls.split(",")], fl))
        rw, rh = (W // 3, H // 3) if args.contact else (W, H)
    else:
        specs = [([float(v) for v in args.cam.split(",")],
                  [float(v) for v in args.look.split(",")], args.focal)]
        rw, rh = W, H

    prods = []
    # Isaac's default horizontal aperture is 20.955 mm, not a 36 mm
    # full-frame sensor, so a "24 mm" lens here is ~47 deg, not ~74 deg.
    for i, (cp, lp, fl) in enumerate(specs):
        cam = rep.create.camera(position=tuple(cp), look_at=tuple(lp),
                                focal_length=fl)
        rp = rep.create.render_product(cam, (rw, rh))
        wr = rep.WriterRegistry.get("BasicWriter")
        od = os.path.join(os.path.abspath(args.out_dir), f"cam_{i:02d}")
        wr.initialize(output_dir=od, rgb=True)
        wr.attach([rp])
        prods.append(od)
        say(f"[camera {i:02d}] at {cp} look {lp} focal {fl} -> {od}")

    # render through the robot's own cameras, at their real resolutions,
    # so the output is directly comparable to the LeRobot observations
    for c in robot_cams:
        nm = c["path"].split("/")[-1]
        rp = rep.create.render_product(c["path"], (c["width"], c["height"]))
        wr = rep.WriterRegistry.get("BasicWriter")
        od = os.path.join(os.path.abspath(args.out_dir), nm)
        wr.initialize(output_dir=od, rgb=True)
        wr.attach([rp])
        say(f"[robotcam] {nm} -> {od}")

    for _ in range(12):            # let the ray-traced image converge
        rep.orchestrator.step()
    say(f"[render] {len(prods)} view(s) at {rw}x{rh}, "
        f"{len(robot_cams)} robot camera(s)")


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
