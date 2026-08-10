# rpl_centrifuge

Centrifuge task pack for [Isaac Lab](https://github.com/isaac-sim/IsaacLab). It includes:

* a bimanual OpenArm centrifuge pick-and-place task, and
* a GR1T2 waist-enabled hand-tracking variant derived from Isaac Lab's GR1T2 pick-place task.

The package is an external Isaac Lab task extension: it bundles its USD assets, registers Gym IDs on import, and ships a `.pth` file so the registration happens automatically in any Python process running inside the Isaac Lab venv — no edits to Isaac Lab itself required.

## Registered tasks

| Gym ID | Action mode | Suited for |
|---|---|---|
| `Isaac-Centrifuge-Bimanual-OpenArm-IK-Rel-v0` | Relative differential IK (14-dim Δpose + grippers) | Keyboard / SpaceMouse / fallback; left-arm-padded keyboard wrapper included |
| `Isaac-Centrifuge-Bimanual-OpenArm-IK-Abs-v0` | Absolute differential IK | Hand-tracking via OpenXR (recommended) |
| `Isaac-Centrifuge-GR1T2-WaistEnabled-Abs-v0` | GR1T2 Pink IK + dexterous hand tracking | VR / XR hand tracking with the GR1T2 robot |

## Requirements

* Isaac Lab installed (provides `isaaclab`, `isaaclab.devices`, and `isaaclab_assets.robots.openarm.OPENARM_BI_HIGH_PD_CFG`)
* Python ≥ 3.10
* (Optional) An OpenXR-compatible headset + hand tracking for the IK-Abs / bimanual teleop path

## Install

From inside the Isaac Lab venv (so the Isaac Lab Python sees it):

```bash
export ISAACLAB_ROOT="<path-to-IsaacLab-root>"
export TASK_PACKAGE_ROOT="<path-to-rpl_centrifuge-task-package>"
cd "${TASK_PACKAGE_ROOT}"
"${ISAACLAB_ROOT}/isaaclab.sh" -p -m pip install -e .
```

The install drops a `rpl_centrifuge_autoregister.pth` file next to your site-packages. From then on, every Python process inside the Isaac Lab venv auto-imports `rpl_centrifuge` at startup, so `gym.make("Isaac-Centrifuge-...")` works without any script edits.

If the `.pth` mechanism doesn't fire (some environments strip them, especially when running scripts outside the venv), add `import rpl_centrifuge  # noqa: F401` to your script before the first `gym.make` call as a fallback.

## Run

Smoke-test (no teleop):

```bash
./isaaclab.sh -p scripts/environments/zero_agent.py \
  --task Isaac-Centrifuge-Bimanual-OpenArm-IK-Rel-v0 --num_envs 1
```

Keyboard teleop (right arm only — left arm holds its default pose):

```bash
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Centrifuge-Bimanual-OpenArm-IK-Rel-v0 --teleop_device keyboard
```

Key bindings: `WSAD`/`QE` translate, `ZX`/`TG`/`CV` rotate, `K` toggles gripper, `L` resets deltas.

Bimanual hand-tracking teleop:

```bash
LIVESTREAM=2 ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Centrifuge-Bimanual-OpenArm-IK-Abs-v0 --teleop_device handtracking --xr --visualize kit
```

With Claude AI skill:

```bash
# Add the following options for isaaclab.sh
--kit_args "--ext-folder <path-to-agent-sim-tools> --enable omni.claude.bridge"
```

Record demonstrations:

```bash
./isaaclab.sh -p scripts/tools/record_demos.py \
  --task Isaac-Centrifuge-Bimanual-OpenArm-IK-Abs-v0 \
  --teleop_device handtracking \
  --dataset_file ./datasets/centrifuge_demos.hdf5 \
  --num_demos 10
```

GR1T2 VR hand-tracking variant:

```bash
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Centrifuge-GR1T2-WaistEnabled-Abs-v0 \
  --viz kit \
  --xr
```

## Assets

The package bundles physics-ready USDs in `src/rpl_centrifuge/assets/`:

* `centrifuge_tube_big.usd` — the plastic tube (dynamic rigid body, convex-hull collision, mass)
* `centrifuge_bucket_big.usd` — the bucket (dynamic rigid body, convex-hull collision, mass)
* `centrifuge_tube_rack.usd` — static rack with convex-decomposition collision (the wells stay concave so the tube can be inserted)

These are what `gym.make()` actually loads. To regenerate them from raw geometry-only USDs (e.g. if you have new mesh authoring), use the included one-shot script:

```bash
# in-place re-preparation of the bundled assets (default when no src dir given)
./isaaclab.sh -p "${TASK_PACKAGE_ROOT}/scripts/prepare_assets.py"

# or, prepare from a separate source directory
./isaaclab.sh -p "${TASK_PACKAGE_ROOT}/scripts/prepare_assets.py" <path-to-raw-usds>
CENTRIFUGE_SRC_DIR=<path-to-raw-usds> \
  ./isaaclab.sh -p "${TASK_PACKAGE_ROOT}/scripts/prepare_assets.py"
```

The script knows per-asset whether to prep it as `dynamic` (tube, bucket — adds `RigidBodyAPI` + `MassAPI` + `MeshCollisionAPI(convexHull)`) or `static` (rack — adds `MeshCollisionAPI(triangleMesh)` only). It's idempotent: it never overwrites existing schemas, so re-running is safe.

## Scene layout

| Asset | Position (env frame) | Notes |
|---|---|---|
| Ground plane | z = 0 | |
| Robot pedestal | (-0.62, 0, 0.289), 0.413 × 0.413 × 0.579 m cube | Top at z = 0.578 |
| Robot base (OpenArm bimanual) | (-0.62, 0, 0.5786) | Mounted on pedestal top |
| Table | (0.1, 0, 0.40), 1.0 × 0.6 × 0.80 m cube | Top at z = 0.80; non-overlapping with pedestal |
| Tube rack | (-0.30, -0.20, 0.80) | Static fixture on the robot's right (-y) side; holds the tube |
| Tube | (-0.30, -0.20, 0.90) | Starts above the rack, settles into a well; randomized ±5 cm xy + ±0.3 rad yaw on reset |
| Bucket | (-0.20, 0.00, 0.85) | On the centre-line, reachable by either hand; receptacle for the task |
| Chest camera | offset (0.08, 0, 0.55) from `openarm_body_link` | Intel RealSense D435-style RGB + depth, 640×480 @ 30 Hz, ~69° H-FOV. Not in obs by default — see notes below. |

The OpenArm USD ships without a camera prim (the included `openarm_bimanual_sensor.usd` layer is an empty stub), so this package attaches a `CameraCfg` to `openarm_body_link` in the scene cfg. The camera rotates with the chest if the body link moves.

**Pixel data is not in the policy observation by default** — adding it changes the observation space and would break any downstream IL configs trained against the current obs shape. To opt in, add to `ObservationsCfg.PolicyCfg`:

```python
chest_rgb = ObsTerm(func=mdp.image,
                    params={"sensor_cfg": SceneEntityCfg("chest_camera"),
                            "data_type": "rgb"})
```

When using the camera in **headless runs without livestream**, the rendering pipeline must be on — pass `--enable_cameras` on the launch command (livestream and GUI modes have rendering on already).

## Success criterion

`tube_inside_bucket` — true per-env iff:
* tube xy is within 4 cm of bucket xy
* tube z is between (bucket_z − 5 cm) and (bucket_z + 20 cm)
* tube linear speed < 5 cm/s

The `record_demos.py` tooling picks this up by name and exports an episode once it holds for `--num_success_steps` consecutive frames.

## Package layout

```
rpl_centrifuge/
├── pyproject.toml
├── src/
│   ├── rpl_centrifuge_autoregister.pth             ← .pth auto-import shim
│   └── rpl_centrifuge/
│       ├── __init__.py                             ← gym.register
│       ├── openarm/
│       │   ├── env_cfg.py                          ← OpenArm base scene + MDP wiring
│       │   ├── ik_rel_env_cfg.py                   ← Relative-IK + keyboard + handtracking
│       │   ├── ik_abs_env_cfg.py                   ← Absolute-IK + handtracking
│       │   └── devices/                            ← OpenArm-specific teleop helpers
│       ├── gr1t2/
│       │   └── waist_enabled_env_cfg.py            ← GR1T2 hand-tracking variant
│       ├── mdp/                                    ← observations / events / terminations
│       ├── agents/                                 ← (empty, for robomimic JSONs later)
│       └── assets/                                 ← bundled USDs
└── scripts/
    └── prepare_assets.py                           ← raw → rigid-prepared USDs
```


## Dev notes

### ALVR with STEAMVR
Make sure to pass the following option in Launch option in STEAMVR ([Reference](https://www.reddit.com/r/virtualreality_linux/comments/1hlyur6/black_vr_display_alvr/))

```bash
~/.local/share/Steam/steamapps/common/SteamVR/bin/vrmonitor.sh %command%
```