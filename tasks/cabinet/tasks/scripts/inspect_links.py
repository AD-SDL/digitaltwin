#!/usr/bin/env python3
"""Dump the OpenArm link tree so cameras can be parented to the right prims."""
import os
_R=open("links.txt","w",buffering=1)
def say(m=""):
    print(m,flush=True); _R.write(m+"\n"); _R.flush(); os.fsync(_R.fileno())
from isaacsim import SimulationApp
app=SimulationApp({"headless":True})
from pxr import Usd, UsdGeom, UsdPhysics
from isaacsim.core.utils.stage import add_reference_to_stage
import omni.usd
P=os.path.expanduser("~/cabinet_sim/assets/OpenArm/openarm_bimanual/openarm_bimanual.usd")
try:
    stage=omni.usd.get_context().get_stage()
    add_reference_to_stage(usd_path=P, prim_path="/World/OpenArm")
    cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_])
    say("[rigid bodies / links with geometry]")
    for p in Usd.PrimRange(stage.GetPrimAtPath("/World/OpenArm")):
        if not p.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        r=cache.ComputeWorldBound(p).ComputeAlignedRange()
        if r.IsEmpty():
            say(f"  {str(p.GetPath()).replace('/World/OpenArm/',''):44} (no geom)")
            continue
        c=[(r.GetMin()[i]+r.GetMax()[i])/2 for i in range(3)]
        s=[r.GetMax()[i]-r.GetMin()[i] for i in range(3)]
        say(f"  {str(p.GetPath()).replace('/World/OpenArm/',''):44} "
            f"c({c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f}) size({s[0]:.3f},{s[1]:.3f},{s[2]:.3f})")
    say("")
    say("[top-level children]")
    for c in stage.GetPrimAtPath("/World/OpenArm").GetChildren():
        say(f"  {c.GetName():40} <{c.GetTypeName()}>")
except Exception:
    import traceback; say("EXC:"); say(traceback.format_exc())
finally:
    say("done"); app.close()
