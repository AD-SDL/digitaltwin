#!/usr/bin/env python3
"""Why did the OpenArm reference produce no geometry?  Inspect it properly."""
import os

_R = open("openarm_inspect.txt", "w", buffering=1)


def say(m=""):
    print(m, flush=True)
    _R.write(m + "\n")
    _R.flush()
    os.fsync(_R.fileno())


from isaacsim import SimulationApp                               # noqa: E402
sim_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdPhysics                          # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage      # noqa: E402
import omni.usd                                                   # noqa: E402

P = ("/tmp/Assets/Isaac/6.0/Isaac/Robots/OpenArm/"
     "openarm_bimanual/openarm_bimanual.usd")

try:
    say(f"[file] exists={os.path.exists(P)} size={os.path.getsize(P)}")

    # 1. open it standalone, loading everything
    src = Usd.Stage.Open(P, Usd.Stage.LoadAll)
    dp = src.GetDefaultPrim()
    say(f"[standalone] defaultPrim={dp.GetPath() if dp else None}")
    say(f"[standalone] upAxis={UsdGeom.GetStageUpAxis(src)} "
        f"mpu={UsdGeom.GetStageMetersPerUnit(src)}")
    allp = list(src.Traverse())
    say(f"[standalone] prims={len(allp)}")
    say(f"[standalone] unloaded payloads={len(src.FindAllUnloadedPrimPaths() or [])}")
    meshes = [p for p in allp if p.IsA(UsdGeom.Mesh)]
    say(f"[standalone] meshes={len(meshes)}")
    roots = [str(p.GetPath()) for p in allp
             if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    say(f"[standalone] articulation roots={roots}")
    say("[standalone] top-level children:")
    if dp:
        for c in dp.GetChildren()[:12]:
            say(f"    {c.GetPath()}  <{c.GetTypeName()}>")
    bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_,
                                                    UsdGeom.Tokens.render,
                                                    UsdGeom.Tokens.proxy,
                                                    UsdGeom.Tokens.guide])
    if dp:
        r = bb.ComputeWorldBound(dp).ComputeAlignedRange()
        say(f"[standalone] bbox empty={r.IsEmpty()} {r.GetMin()} -> {r.GetMax()}")

    # 2. now reference it the way build_scene did
    stage = omni.usd.get_context().get_stage()
    add_reference_to_stage(usd_path=P, prim_path="/World/OpenArm")
    prim = stage.GetPrimAtPath("/World/OpenArm")
    say(f"[referenced] valid={prim.IsValid()} active={prim.IsActive()} "
        f"type={prim.GetTypeName()}")
    kids = list(prim.GetChildren())
    say(f"[referenced] children={len(kids)}: {[c.GetName() for c in kids[:10]]}")
    say(f"[referenced] descendants={len(list(Usd.PrimRange(prim)))}")
    unloaded = stage.FindAllUnloadedPrimPaths() or []
    say(f"[referenced] unloaded payload paths={[str(x) for x in unloaded][:8]}")
    if unloaded:
        say("[referenced] -> loading payloads")
        stage.Load()
        say(f"[referenced] descendants after Load()="
            f"{len(list(Usd.PrimRange(prim)))}")
    c2 = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    r2 = c2.ComputeWorldBound(prim).ComputeAlignedRange()
    say(f"[referenced] bbox empty={r2.IsEmpty()} {r2.GetMin()} -> {r2.GetMax()}")
except Exception:
    import traceback
    say("EXCEPTION:")
    say(traceback.format_exc())
finally:
    sim_app.close()
