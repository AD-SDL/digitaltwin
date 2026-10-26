#!/usr/bin/env python3
"""Which way does the OpenArm face?  Read it off the chest camera, not by eye."""
import os
_R=open("facing.txt","w",buffering=1)
def say(m=""):
    print(m,flush=True); _R.write(m+"\n"); _R.flush(); os.fsync(_R.fileno())

from isaacsim import SimulationApp
app=SimulationApp({"headless":True})
import numpy as np
from pxr import Usd, UsdGeom
from isaacsim.core.utils.stage import add_reference_to_stage
import omni.usd
P=os.path.expanduser("~/cabinet_sim/assets/OpenArm/openarm_bimanual/openarm_bimanual.usd")
try:
    stage=omni.usd.get_context().get_stage()
    add_reference_to_stage(usd_path=P, prim_path="/World/OpenArm")   # NO rotation
    hits=[]
    for p in Usd.PrimRange(stage.GetPrimAtPath("/World/OpenArm")):
        n=p.GetName().lower()
        if any(k in n for k in ("cam","chest","d435","d405","sensor","head")):
            hits.append(p)
    say(f"[camera-ish prims] {[p.GetName() for p in hits][:20]}")
    for p in hits[:10]:
        m=UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        a=np.array(m).T
        pos=a[:3,3]
        # USD camera looks down its local -Z
        fwd=-a[:3,2]
        say(f"  {p.GetName():34} pos({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})"
            f"  fwd({fwd[0]:+.2f},{fwd[1]:+.2f},{fwd[2]:+.2f})")
    # also: where are the two shoulders, to get the shoulder axis
    sh=[]
    for p in Usd.PrimRange(stage.GetPrimAtPath("/World/OpenArm")):
        n=p.GetName().lower()
        if "link1" in n or ("shoulder" in n and "joint" not in n):
            c=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_]
                                ).ComputeWorldBound(p).ComputeAlignedRange()
            if not c.IsEmpty():
                sh.append((p.GetName(),[ (c.GetMin()[i]+c.GetMax()[i])/2 for i in range(3)]))
    for n,c in sh[:6]:
        say(f"  shoulder-ish {n:28} ({c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f})")
except Exception:
    import traceback; say("EXC:"); say(traceback.format_exc())
finally:
    say("done"); app.close()
