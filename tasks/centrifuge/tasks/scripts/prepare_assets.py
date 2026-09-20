# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Apply physics schemas to the centrifuge tube/bucket/rack USDs.

The raw geometry-only USDs need rigid-body / collision schemas applied before
Isaac Lab will spawn them with physics (``UsdFileCfg.rigid_props`` only
*modifies* an existing API — it does not *apply* one).

Two preparation modes:
  * ``dynamic``: applies ``PhysicsRigidBodyAPI`` + ``MassAPI`` +
    ``PhysxRigidBodyAPI`` on the root, plus ``CollisionAPI`` +
    ``MeshCollisionAPI("convexHull")`` on every Mesh under it. Use for
    free-moving objects like the tube and bucket.
  * ``static``: applies ``CollisionAPI`` + ``MeshCollisionAPI("none")``
    only — no rigid body. Use for fixtures like the tube rack that should
    stay in place but need concave collision (wells, holes, etc.) preserved.
    The "none" token is USD's name for "raw triangle mesh, no convex
    approximation"; Isaac Sim rejects the older "triangleMesh" spelling.

If no source dir is given, the script prepares the package's bundled assets
in place (assets/ -> assets/). Useful for re-preparing after dropping a new
raw USD into the package.

Examples:
    ./isaaclab.sh -p scripts/prepare_assets.py                       # in-place
    ./isaaclab.sh -p scripts/prepare_assets.py /my/dataset/objects
    CENTRIFUGE_SRC_DIR=/my/dataset/objects ./isaaclab.sh -p scripts/prepare_assets.py
    ./isaaclab.sh -p scripts/prepare_assets.py /my/raw --dst-dir /tmp/prepared
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from importlib.resources import files as _pkg_files

from isaaclab.app import AppLauncher


@dataclass(frozen=True)
class _AssetSpec:
    filename: str
    mode: str  # "dynamic" or "static"


# Authoritative list of bundled assets and how each should be prepared.
ASSETS: tuple[_AssetSpec, ...] = (
    # SDF (not convexHull): a convex hull of the tube (slim body + wider cap)
    # bulges OUTSIDE the mesh into a frustum, so the collider caught the bucket
    # well rim, penetrated, then got ejected -> bouncing on insertion. SDF is
    # the only dynamic-valid approximation that follows the mesh exactly, so the
    # collider matches the true slim body and inserts cleanly.
    _AssetSpec("centrifuge_tube_big.usd", "dynamic_sdf"),
    # Bucket has wells, so convexHull would fill them and block the tube. SDF
    # is the only approximation valid for dynamic bodies that preserves
    # concave geometry, so we use the dedicated dynamic_sdf mode.
    _AssetSpec("centrifuge_bucket_big.usd", "dynamic_sdf"),
    # Rack is a static fixture; triangleMesh preserves the concave wells so
    # tubes can actually be inserted (a convex hull would fill them in).
    _AssetSpec("centrifuge_tube_rack.usd", "static"),
)


def _bundled_assets_dir() -> str:
    return str(_pkg_files("rpl_centrifuge") / "assets")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "src_dir",
        nargs="?",
        default=os.environ.get("CENTRIFUGE_SRC_DIR"),
        help=(
            "Directory holding the raw geometry-only USDs. If omitted (and "
            "CENTRIFUGE_SRC_DIR is not set), the script falls back to the "
            "package's bundled assets/ dir for *in-place* re-preparation."
        ),
    )
    parser.add_argument(
        "--dst-dir",
        default=os.environ.get("CENTRIFUGE_DST_DIR"),
        help=(
            "Destination dir for prepared USDs. Defaults to the package's "
            "bundled assets/ dir (resolved via importlib.resources). May also "
            "be set via CENTRIFUGE_DST_DIR."
        ),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _ensure_dynamic_root_apis(stage_root, usd_physics, physx_schema) -> None:
    if not stage_root.HasAPI(usd_physics.RigidBodyAPI):
        usd_physics.RigidBodyAPI.Apply(stage_root)
        print(f"  + RigidBodyAPI on {stage_root.GetPath()}")
    if not stage_root.HasAPI(usd_physics.MassAPI):
        usd_physics.MassAPI.Apply(stage_root)
        print(f"  + MassAPI on {stage_root.GetPath()}")
    if not stage_root.HasAPI(physx_schema.PhysxRigidBodyAPI):
        physx_schema.PhysxRigidBodyAPI.Apply(stage_root)
        print(f"  + PhysxRigidBodyAPI on {stage_root.GetPath()}")


def _set_mesh_collision(stage_root, usd_physics, approximation: str) -> int:
    """Apply CollisionAPI + MeshCollisionAPI to every Mesh under ``stage_root``
    and force the approximation to ``approximation``.

    Idempotent — always overwrites the approximation attribute so re-running
    the prep script after a change to the spec actually takes effect.
    """
    from pxr import Usd  # noqa: PLC0415

    n = 0
    for prim in Usd.PrimRange(stage_root):
        if prim.GetTypeName() != "Mesh":
            continue
        if not prim.HasAPI(usd_physics.CollisionAPI):
            usd_physics.CollisionAPI.Apply(prim)
        mca = (
            usd_physics.MeshCollisionAPI(prim)
            if prim.HasAPI(usd_physics.MeshCollisionAPI)
            else usd_physics.MeshCollisionAPI.Apply(prim)
        )
        attr = mca.GetApproximationAttr() or mca.CreateApproximationAttr()
        attr.Set(approximation)
        n += 1
    return n


def _apply_dynamic(stage_root, usd_physics, physx_schema) -> None:
    _ensure_dynamic_root_apis(stage_root, usd_physics, physx_schema)
    n = _set_mesh_collision(stage_root, usd_physics, "convexHull")
    print(f"  + CollisionAPI(convexHull) on {n} mesh(es)")


def _apply_dynamic_sdf(stage_root, usd_physics, physx_schema) -> None:
    """Same as dynamic, but uses SDF (signed distance field) collision so the
    mesh's concave features (e.g. bucket wells) are preserved. SDF is the only
    PhysX approximation valid for dynamic bodies that supports concavity.
    """
    from pxr import Usd  # noqa: PLC0415

    _ensure_dynamic_root_apis(stage_root, usd_physics, physx_schema)
    n = _set_mesh_collision(stage_root, usd_physics, "sdf")
    for prim in Usd.PrimRange(stage_root):
        if prim.GetTypeName() == "Mesh":
            physx_schema.PhysxSDFMeshCollisionAPI.Apply(prim)
    print(f"  + CollisionAPI(sdf) on {n} mesh(es)")


def _apply_static(stage_root, usd_physics, _physx_schema) -> None:
    # The USD token for "raw triangle mesh, no approximation" is "none"
    # (see PhysX MeshCollisionAPI schema). Isaac Sim rejects the older
    # "triangleMesh" string with:
    #   "Prim ... is using unknown value 'triangleMesh' for physics:approximation"
    # which leaves the collider malformed and cascades into GPU cooking failures.
    n = _set_mesh_collision(stage_root, usd_physics, "none")
    print(f"  + CollisionAPI(none [raw triangle mesh]) on {n} mesh(es)")


def _prepare_one(spec: _AssetSpec, src_dir: str, dst_dir: str, usd_physics, physx_schema) -> None:
    src = os.path.join(src_dir, spec.filename)
    dst = os.path.join(dst_dir, spec.filename)
    print(f"[{spec.mode:>7}] {src} -> {dst}")
    # If src == dst we're operating in-place; avoid the no-op self-copy that
    # would race on some filesystems.
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copyfile(src, dst)

    from pxr import Usd  # noqa: PLC0415

    stage = Usd.Stage.Open(dst)
    if stage is None:
        raise RuntimeError(f"Could not open stage at {dst}")
    root = stage.GetDefaultPrim()
    if not root or not root.IsValid():
        raise RuntimeError(f"{dst} has no defaultPrim")

    if spec.mode == "dynamic":
        _apply_dynamic(root, usd_physics, physx_schema)
    elif spec.mode == "dynamic_sdf":
        _apply_dynamic_sdf(root, usd_physics, physx_schema)
    elif spec.mode == "static":
        _apply_static(root, usd_physics, physx_schema)
    else:
        raise ValueError(f"unknown mode {spec.mode!r} for {spec.filename}")

    stage.GetRootLayer().Save()
    print("  saved.")


def main() -> None:
    args = _parse_args()

    bundled = _bundled_assets_dir()
    src_dir = args.src_dir or bundled
    dst_dir = args.dst_dir or bundled

    if not os.path.isdir(src_dir):
        sys.exit(f"error: source directory does not exist or is not a directory: {src_dir}")

    missing = [s.filename for s in ASSETS if not os.path.isfile(os.path.join(src_dir, s.filename))]
    if missing:
        sys.exit(f"error: source dir {src_dir} is missing required USD(s): {missing}")

    # Launch SimulationApp only after args + files are validated so --help is fast.
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        from pxr import PhysxSchema, UsdPhysics  # noqa: PLC0415

        os.makedirs(dst_dir, exist_ok=True)
        for spec in ASSETS:
            _prepare_one(spec, src_dir, dst_dir, UsdPhysics, PhysxSchema)
        print(f"done. prepared {len(ASSETS)} USD(s) into {dst_dir}")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
