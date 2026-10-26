"""
Open the door scene at Isaac Sim startup.

Kit runs --exec strings through wordexp(), so an inline Python one-liner
fails: the ';' and '()' are shell metacharacters and you get

    [omni.kit.app.plugin] wordexp error on: "import omni.usd; ..." rc: 2

A bare file path has no metacharacters and survives wordexp, so pass this
file instead:

    <isaaclab_root>/_isaac_sim/isaac-sim.sh isaacsim.exp.full \
        --exec <task_package_root>/scripts/open_scene.py

Override the stage with ISAAC_OPEN_STAGE=/path/to/other.usd.

The open is deferred by a few app ticks. Calling open_stage() during startup
races the renderer coming up.
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_STAGE = os.environ.get("ISAAC_OPEN_STAGE",
                        os.path.join(_HERE, "rpl_door_scene.usd"))
_LOG = os.path.join(_HERE, "open_scene.log")


def _note(msg):
    print(f"[open_scene] {msg}", flush=True)
    try:
        with open(_LOG, "a") as f:
            f.write(msg + "\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        pass


def _go():
    import asyncio
    import omni.kit.app
    import omni.usd

    async def _open():
        app = omni.kit.app.get_app()
        for _ in range(10):            # let the renderer finish coming up
            await app.next_update_async()
        if not os.path.exists(_STAGE):
            _note(f"FAIL stage not found: {_STAGE}")
            return
        ctx = omni.usd.get_context()
        ok, err = await ctx.open_stage_async(_STAGE)
        if not ok:
            _note(f"FAIL open_stage returned {ok}: {err}")
            return
        stage = ctx.get_stage()
        n = len(list(stage.Traverse())) if stage else 0
        cab = bool(stage and stage.GetPrimAtPath("/World/Cabinet"))
        arm = bool(stage and stage.GetPrimAtPath("/World/OpenArm"))
        _note(f"OK opened {_STAGE} prims={n} cabinet={cab} openarm={arm}")

    asyncio.ensure_future(_open())


_note(f"--exec ran, queuing open of {_STAGE}")
_go()
