#!/usr/bin/env python3
"""
Add the three real OpenArm cameras to a referenced OpenArm USD.

The rig matches the physical robot in the RPL:

  chest   Intel RealSense D435i, colour stream, 848 x 480
          mounted on openarm_body_link, facing forward and tilted down
  wrists  two Innomaker U20CAM-1080P USB UVC modules, 640 x 480
          mounted on openarm_<side>_link7, aimed at that arm's ee_tcp

Resolutions are taken from the LeRobot dataset feature spec, so renders
from these cameras line up with the real observations:
    observation.images.left_chest          480 x 848
    observation.images.left_wrist_left     480 x 640
    observation.images.right_wrist_right   480 x 640

Field of view:
  D435i colour  69.4 x 42.5 deg (Intel tech spec, +/- 3 deg)
  U20CAM-1080P  130 deg wide-angle lens, quoted by Innomaker

Isaac's default horizontal aperture is 20.955 mm, NOT a 36 mm full-frame
sensor, so focal length is derived from that.  Setting the vertical
aperture by image aspect then reproduces Intel's stated 42.5 deg vertical
from the 69.4 deg horizontal, which is a useful check that the convention
is right.

Usage:
    from openarm_cameras import add_openarm_cameras
    cams = add_openarm_cameras(stage, "/World/OpenArm")
"""

import math
from pxr import Usd, UsdGeom, Gf, Sdf

H_APERTURE = 20.955          # Isaac / Kit default, mm

# ---- measured / sourced -------------------------------------------------
CHEST = dict(
    name="chest_d435i_color",
    width=848, height=480,
    hfov_deg=69.4,           # Intel D435i tech spec, RGB sensor
    parent="openarm_body_link",
)
WRIST = dict(
    width=640, height=480,
    # 130 deg is Innomaker's quoted wide-angle figure and is almost
    # certainly DIAGONAL.  Converted to horizontal for a 4:3 frame below.
    # This is the least certain number in the file; change it here.
    dfov_deg=130.0,
)

# ---- assumed placement, easy to adjust ---------------------------------
# The robot faces +X in its own frame (left shoulder at +Y, up +Z).
CHEST_POS_LOCALISH = (0.105, 0.0, 0.480)   # FITTED TO FOOTAGE, not measured. See note.
CHEST_PITCH_DOWN_DEG = 18.0
# NOTE: the chest mount pose above is fitted by eye against the real
# left_chest frames, NOT measured off the hardware. In the real footage
# the handles sit upper-left with floor visible lower-right, which puts
# the camera near handle height looking slightly down. Whoever has the
# physical robot should replace these two constants with real numbers.
# The OPTICS (resolution and FOV) are from the vendor specs and are solid.
WRIST_OFFSET = (0.075, 0.0, 0.035)         # out in front of link7, slightly above
WRIST_LOOK_PAST = 0.28                     # metres beyond the TCP to aim at. Aiming AT
                                           # the TCP just frames the gripper; the real
                                           # cameras look past it at the workpiece.


def _diag_to_hfov(dfov_deg, w, h):
    """Diagonal FOV -> horizontal FOV for a rectilinear lens."""
    d = math.hypot(w, h)
    return 2.0 * math.degrees(math.atan(math.tan(math.radians(dfov_deg) / 2) * w / d))


def _focal_from_hfov(hfov_deg, h_aperture=H_APERTURE):
    return (h_aperture / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def _world(stage, path):
    return UsdGeom.Xformable(
        stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


def _make_camera(stage, path, width, height, hfov_deg,
                 world_pos, world_target, parent_path, up=(0, 0, 1)):
    """Create a camera under parent_path, posed in world then converted
    to the parent's local frame."""
    cam = UsdGeom.Camera.Define(stage, path)

    focal = _focal_from_hfov(hfov_deg)
    v_aperture = H_APERTURE * (height / float(width))
    cam.CreateFocalLengthAttr(focal)
    cam.CreateHorizontalApertureAttr(H_APERTURE)
    cam.CreateVerticalApertureAttr(v_aperture)
    # default far plane is 1e6, which wrecks depth precision up close
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))

    # USD cameras look down local -Z.  SetLookAt builds a view matrix, so
    # its inverse is the camera's world transform.
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*world_pos),
                                   Gf.Vec3d(*world_target),
                                   Gf.Vec3d(*up))
    world_xf = view.GetInverse()
    local_xf = world_xf * _world(stage, parent_path).GetInverse()

    x = UsdGeom.Xformable(cam.GetPrim())
    x.ClearXformOpOrder()
    x.AddTransformOp().Set(local_xf)

    vfov = 2 * math.degrees(math.atan((v_aperture / 2.0) / focal))
    return dict(path=path, width=width, height=height,
                hfov=hfov_deg, vfov=vfov, focal=focal)


def add_openarm_cameras(stage, root="/World/OpenArm", log=print):
    """Add chest + two wrist cameras.  Returns a list of dicts describing
    each camera.  Safe to call once per OpenArm instance."""
    out = []

    # ---- chest ----------------------------------------------------------
    body = f"{root}/{CHEST['parent']}"
    if not stage.GetPrimAtPath(body):
        raise RuntimeError(f"no {body}; is this an OpenArm?")
    bw = _world(stage, body)
    # interpret CHEST_POS_LOCALISH in the body link's frame
    pos = bw.Transform(Gf.Vec3d(*CHEST_POS_LOCALISH))
    # aim forward (+X of the body link) and tilted down
    fwd_local = Gf.Vec3d(math.cos(math.radians(CHEST_PITCH_DOWN_DEG)), 0.0,
                         -math.sin(math.radians(CHEST_PITCH_DOWN_DEG)))
    tgt = bw.Transform(Gf.Vec3d(*CHEST_POS_LOCALISH) + fwd_local * 1.0)
    out.append(_make_camera(stage, f"{body}/{CHEST['name']}",
                            CHEST["width"], CHEST["height"], CHEST["hfov_deg"],
                            pos, tgt, body))

    # ---- wrists ---------------------------------------------------------
    whfov = _diag_to_hfov(WRIST["dfov_deg"], WRIST["width"], WRIST["height"])
    for side in ("left", "right"):
        link = f"{root}/openarm_{side}_link7"
        tcp = f"{root}/openarm_{side}_ee_tcp"
        if not stage.GetPrimAtPath(link):
            log(f"  warn  no {link}; skipping {side} wrist camera")
            continue
        lw = _world(stage, link)
        pos = lw.Transform(Gf.Vec3d(*WRIST_OFFSET))
        if stage.GetPrimAtPath(tcp):
            tcp_p = _world(stage, tcp).Transform(Gf.Vec3d(0, 0, 0))
            d = tcp_p - pos
            n = d.GetLength()
            tgt = tcp_p + (d / n) * WRIST_LOOK_PAST if n > 1e-6 else tcp_p
        else:
            log(f"  warn  no {tcp}; aiming {side} wrist camera straight down")
            tgt = lw.Transform(Gf.Vec3d(WRIST_OFFSET[0], 0, -0.25))
        # name matches the dataset stream it corresponds to
        nm = "wrist_left_uvc" if side == "left" else "wrist_right_uvc"
        out.append(_make_camera(stage, f"{link}/{nm}",
                                WRIST["width"], WRIST["height"], whfov,
                                pos, tgt, link))

    for c in out:
        log(f"[camera] {c['path'].split('/')[-1]:20} {c['width']}x{c['height']}  "
            f"hfov {c['hfov']:.1f} deg  vfov {c['vfov']:.1f} deg  "
            f"focal {c['focal']:.2f} mm")
    return out


if __name__ == "__main__":
    # sanity-check the optics maths without Isaac
    f = _focal_from_hfov(69.4)
    va = H_APERTURE * 480 / 848.0
    v = 2 * math.degrees(math.atan((va / 2) / f))
    print(f"D435i colour: focal {f:.2f} mm -> vfov {v:.1f} deg "
          f"(Intel says 42.5, so the convention checks out)")
    wh = _diag_to_hfov(130.0, 640, 480)
    fw = _focal_from_hfov(wh)
    vaw = H_APERTURE * 480 / 640.0
    vw = 2 * math.degrees(math.atan((vaw / 2) / fw))
    print(f"U20CAM 130 deg diag -> hfov {wh:.1f} deg, vfov {vw:.1f} deg, "
          f"focal {fw:.2f} mm")
