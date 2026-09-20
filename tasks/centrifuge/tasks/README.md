# rpl_centrifuge

Centrifuge task pack for [Isaac Lab](https://github.com/isaac-sim/IsaacLab) **3.0.0**. It includes:

* a bimanual OpenArm centrifuge pick-and-place task (relative + absolute IK),
* a single-arm SO101 task (VR task-space IK, plus two physical leader-arm-over-USB
  variants: IsaacTeleop joint-space, and a **LeRobot**-driven variant that records
  straight to a LeRobot dataset — see `docs/04b_collect_episodes_so101_leader.md`), and
* a GR1T2 waist-enabled variant derived from Isaac Lab's GR1T2 pick-place task.

The package is an external Isaac Lab task extension: it bundles its USD assets, registers Gym IDs on import, and ships a `.pth` file so the registration happens automatically in any Python process running inside the Isaac Lab venv — no edits to Isaac Lab itself required.

### Structure (mirrors the 3.0.0 stack examples)

The tasks are rebuilt to match the canonical 3.0.0 manipulation-task layout
(`contrib/stack/config/{franka,so101}`). Each robot has a family:

* **base** — scene on the shared nucleus **SeattleLabTable**
  (`Props/Mounts/SeattleLabTable/table_instanceable.usd`) + ground + dome light,
  with our custom centrifuge `rack` / `tube` / `bucket` USDs standing in for the
  reference cubes. Observations use the three canonical groups: `policy`
  (low-dim), an (empty) `rgb_camera` placeholder, and `subtask_terms`.
* **teleop** (`IK-Abs` / `IK-Rel` / `JointPos`) — **camera-free**. This is the key
  fix: in 3.0.0 the teleop/record tasks carry no cameras, so nothing renders a
  camera observation under the live XR pipeline (which crashes Kit).
* **Visuomotor** — adds `wrist`/`table` cameras + RGB image observations, run
  **headless** by `generate_dataset` to produce image datasets (never during live
  teleop). This is exactly how the IsaacLab examples produce images.
* **Mimic** — `Visuomotor` (or base) + `MimicEnvCfg` for `annotate_demos` /
  `generate_dataset`.

> Scene geometry (robot mount + prop positions on the SeattleLabTable) and the XR
> anchor / retargeter offsets are STARTING POINTS that need in-sim tuning; they
> can't be validated without the GUI / a headset.

## Teleoperation: both hand-tracking and joystick (VR controllers)

All VR tasks drive teleop through the IsaacLab 3.0.0 **IsaacTeleop** pipeline
(`env_cfg.isaac_teleop`), matching the upstream reference tasks
(`IsaacContrib-Stack-Cube-Franka-IK-Abs`, `IsaacContrib-Stack-Cube-SO101-IK-Abs`,
`IsaacContrib-PickPlace-GR1T2-WaistEnabled-Abs`). Every VR task supports **both
modalities in a single session**:

* **Hand-tracking** — bare hands tracked by the headset drive the wrist/EE pose
  (and, for GR1T2, the full dexterous fingers), with a pinch closing the gripper.
* **Joystick** — holding the Touch controllers drives the EE pose from the grip
  pose and the gripper from the trigger (for GR1T2, the trigger is a power grasp).

This is implemented with dual-source retargeters (controller has priority, hand
tracking is the fallback — the same fusion the stock `GripperRetargeter` uses).
See `src/rpl_centrifuge/teleop/`. There is **no legacy `teleop_devices` path**:
the old `isaaclab.devices.openxr.*` cfgs pull in Kit's `carb` and break headless
import, so (like the reference tasks) we rely solely on `isaac_teleop`.

> Retargeter offsets and the controller power-grasp posture are STARTING POINTS
> that need live tuning on a headset (via the retargeter tuning UI); they cannot
> be validated without VR hardware.

## Registered tasks

| Gym ID | Kind | Cameras | Use |
|---|---|---|---|
| `Isaac-Centrifuge-Bimanual-OpenArm-IK-Rel-v0` | Bimanual OpenArm, relative IK | no | VR teleop (hand-tracking **or** controllers) |
| `Isaac-Centrifuge-Bimanual-OpenArm-IK-Abs-v0` | Bimanual OpenArm, absolute IK | no | VR teleop (recommended) |
| `Isaac-Centrifuge-Bimanual-OpenArm-IK-Abs-Mimic-v0` | + Mimic (low-dim) | no | annotate / generate (low-dim) |
| `Isaac-Centrifuge-Bimanual-OpenArm-IK-Abs-Visuomotor-v0` | + wrist/table cameras | yes | headless image rendering |
| `Isaac-Centrifuge-Bimanual-OpenArm-IK-Abs-Visuomotor-Mimic-v0` | Visuomotor + Mimic | yes | **generate image datasets** |
| `Isaac-Centrifuge-SO101-IK-Abs-v0` | Single-arm SO101, task-space IK | no | VR teleop (hand-tracking **or** controllers) |
| `Isaac-Centrifuge-SO101-IK-Abs-Mimic-v0` | + Mimic (low-dim) | no | annotate / generate (low-dim) |
| `Isaac-Centrifuge-SO101-IK-Abs-Visuomotor-v0` | + wrist/table cameras | yes | headless image rendering |
| `Isaac-Centrifuge-SO101-IK-Abs-Visuomotor-Mimic-v0` | Visuomotor + Mimic | yes | **generate image datasets** |
| `Isaac-Centrifuge-SO101-Leader-v0` | Single-arm SO101, joint mirror | no | physical SO101 leader arm over USB (native joint teleop) |
| `Isaac-Centrifuge-GR1T2-WaistEnabled-Abs-v0` | GR1T2 Pink IK + dexterous hands | no | VR teleop (upstream-derived; unchanged) |

**Getting images (the canonical flow):** teleop tasks are camera-free (so live XR
teleop never crashes). Record demos with a teleop task, then render images
**headless** on the matching `-Visuomotor-Mimic-v0` task via `generate_dataset`.

## Requirements

* Isaac Lab **3.0.0** installed (provides `isaaclab`, `isaaclab_teleop`, `isaacteleop`, and `isaaclab_assets.robots.{openarm,so101,fourier}`)
* Python ≥ 3.10
* An OpenXR / CloudXR-compatible headset (e.g. Meta Quest 3) with hand tracking **and** controllers for the VR teleop paths

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

VR teleop (hand-tracking **or** controllers — both work in the same session). Do
**not** pass `--teleop_device`: omitting it selects the IsaacTeleop pipeline; the
operator then chooses the modality simply by holding the Touch controllers or
tracking bare hands.

```bash
# Bimanual OpenArm, absolute IK (recommended)
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Centrifuge-Bimanual-OpenArm-IK-Abs-v0 --xr --visualize kit

# Single-arm SO101, task-space IK
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Centrifuge-SO101-IK-Abs-v0 --xr --visualize kit

# GR1T2 waist-enabled (dexterous hands via hand-tracking; controller wrists + trigger grasp)
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Centrifuge-GR1T2-WaistEnabled-Abs-v0 --xr --visualize kit
```

The `--cloudxr_env` flag defaults to `cloudxrjs` (Quest/Pico) with `--xr`; pass
`--enable_debug_visualization` to overlay the retargeted hand/controller aim and
the IK target (red) vs current EE (green) markers while tuning.

With the Claude AI skill:

```bash
# Add the following options for isaaclab.sh
--kit_args "--ext-folder <path-to-agent-sim-tools> --enable omni.claude.bridge"
```

Record demonstrations (camera-free teleop task; omit `--teleop_device` to use the VR pipeline):

```bash
isaaclab teleop record \
  --task Isaac-Centrifuge-Bimanual-OpenArm-IK-Abs-Mimic-v0 \
  --xr --visualizer kit \
  --dataset_file ./datasets/centrifuge_src.hdf5 \
  --num_demos 10 --reset_settle_seconds 5
```

### Generating image datasets (headless — this is how you get images)

Teleop tasks are **camera-free** (that is what keeps live XR teleop from
crashing). Camera images are rendered **headless** on the matching `Visuomotor`
Mimic task, exactly like the IsaacLab stack examples:

```bash
# 1) annotate the recorded demos with subtask signals -- OFFLINE.
#    Pure CPU, no Isaac Sim, runs in seconds. Use this instead of upstream's
#    annotate_demos.py; see "Why offline annotation" below.
<isaaclab_root>/.venv/bin/python <task_package_root>/scripts/annotate_demos_offline.py \
  --input_file ./datasets/centrifuge_src.hdf5 \
  --output_file ./datasets/centrifuge_annotated.hdf5

# 2) generate the final dataset WITH images (headless; cameras render here).
#    generate_dataset.py passes enable_cameras=True to the AppLauncher itself.
PYTHONUNBUFFERED=1 HEADLESS=1 ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --task Isaac-Centrifuge-Bimanual-OpenArm-IK-Abs-Visuomotor-Mimic-v0 \
  --input_file ./datasets/centrifuge_annotated.hdf5 \
  --output_file ./datasets/centrifuge_images.hdf5 \
  --num_envs 1
```

> `PYTHONUNBUFFERED=1` matters when redirecting to a log. Kit's C++ logging
> writes straight to the file descriptor, but Mimic's per-attempt progress
> banner is a Python `print()` sitting behind an 8 KB block buffer — at ~120
> bytes per banner that is ~65 attempts before a flush, so a redirected run can
> look completely frozen while it is in fact simulating at full tilt.

Watch progress with:

```bash
tail -f <logfile> | grep --line-buffered -E "successful demos generated|datagen info pool"
```

`Loaded N to datagen info pool` prints once before generation and should equal
your source demo count; the `M/N (x%) successful demos generated` banner prints
after every attempt.

#### Why offline annotation

Upstream's `annotate_demos.py --auto` replays each episode open-loop and keeps
it only if the replay reaches the success condition. That does not survive this
insertion. Measured over the 20 shipped demos, the replayed tube pose tracks the
recording to **1-2 mm all the way to the grasp**, then jumps to a **13 mm median
(53 mm worst) deviation the moment the gripper closes**: the fingers close on a
tube ~2 mm off and it seats differently in the hand. The arm is then driven to
the recorded hand poses, which are correct, while the tube hangs 13 mm from
where it should be, and the release over a tight bucket well misses. Result:

| annotator | episodes annotated |
| --------- | ------------------ |
| `annotate_demos.py --auto` (replay) | 3 / 20 |
| `annotate_demos_offline.py`         | 20 / 20 |

The 17 rejected demos are not bad data -- they were recorded `success=True` and
all 20 satisfy the seated-in-bucket condition on their *recorded* states. Mimic
only ever reads waypoints (`eef_pose`, `object_pose`, `target_eef_pose`,
`subtask_term_signals`) from the source, every one of which is computable from
what was recorded, so re-simulating to obtain them is both unnecessary and lossy.

This does **not** fix the generation side: `generate_dataset.py` still has to
physically execute the transformed waypoints and inherits the same grasp-error
amplification. Expect a low success rate and leave `generation_guarantee=True`
with a high `max_num_failures` (200 in `_apply_centrifuge_mimic_cfg`).

> IsaacLab 3.0.0 removed `--headless` and `--enable_cameras` from the
> `AppLauncher` CLI args — passing either is an `unrecognized arguments` error.
> Use the `HEADLESS=1` environment variable instead.

> Keep `--num_envs 1` for now. `MultiWaypoint.execute` evaluates the success
> term once per env per step, and the `tube_inside_bucket` dwell counter is
> shared across envs; it is guarded against double-counting within a step, but
> multi-env generation has not been validated end to end.

The SO101 equivalents are `Isaac-Centrifuge-SO101-IK-Abs-Mimic-v0` (record /
annotate) and `Isaac-Centrifuge-SO101-IK-Abs-Visuomotor-Mimic-v0` (generate).

### SO101 physical-leader teleop over USB (`Isaac-Centrifuge-SO101-Leader-v0`)

Driven by a **physical SO101 leader arm connected over USB**, using the **native
IsaacLab 3.0.0 joint-space leader pipeline** (no custom bridge). The
`so101_leader` device reads the leader arm and streams its joint encoders over
the IsaacTeleop tensor transport; a `JointStateSource` + `JointStateRetargeter`
mirror those DOFs 1:1 onto the follower's joint targets. The leader's standard
SO101 DOF names are remapped to our USD joint names in
`so101/leader_teleop_env_cfg.py` (`_LEADER_TO_ROBOT`).

Launch the `so101_leader` device (pointed at the USB arm) alongside the sim, then
run/record the task through the IsaacTeleop CLI (omit `--teleop_device`):

```bash
# terminal 1 — start the SO101 leader device on the USB-connected arm
#   (see the IsaacLab teleop docs for the so101_leader launcher; collection id "so101_leader")

# terminal 2 — run or record the follower task
isaaclab teleop record \
  --task Isaac-Centrifuge-SO101-Leader-v0 \
  --visualizer kit \
  --dataset_file ./datasets/so101_centrifuge_demos.hdf5 \
  --num_demos 10
```

The 6-DoF joint action `[Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw]`
is the leader's mapped joint targets, so recorded trajectories map back onto the
real SO101. If the leader/follower are miscalibrated, tune per-joint
`scale`/`offset`/`sign` on the `JointStateRetargeterConfig` in
`leader_teleop_env_cfg.py`. Smoke-test the scene without a leader:
`isaaclab zero_agent --task Isaac-Centrifuge-SO101-Leader-v0 --num_envs 1`.

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
│       ├── teleop/                                 ← dual-source (hand + controller) retargeters
│       │   └── retargeters.py                      ← Se3Abs/Se3Rel/DexHand dual-source factories
│       ├── openarm/
│       │   ├── env_cfg.py                          ← base scene (SeattleLabTable + props), camera-free obs
│       │   ├── ik_rel_env_cfg.py                   ← Relative-IK teleop (camera-free)
│       │   ├── ik_abs_env_cfg.py                   ← Absolute-IK teleop (camera-free)
│       │   ├── visuomotor_env_cfg.py               ← + wrist/table cameras + image obs (headless)
│       │   ├── ik_abs_mimic_env_cfg.py             ← low-dim Mimic env cfg
│       │   ├── visuomotor_mimic_env_cfg.py         ← image Mimic env cfg (generate_dataset)
│       │   └── mimic_env.py                        ← bimanual ManagerBasedRLMimicEnv impl
│       ├── so101/
│       │   ├── env_cfg.py                          ← base scene (SeattleLabTable + props), camera-free obs
│       │   ├── ik_abs_env_cfg.py                   ← task-space IK teleop (camera-free)
│       │   ├── visuomotor_env_cfg.py               ← + wrist/table cameras + image obs (headless)
│       │   ├── ik_abs_mimic_env_cfg.py             ← low-dim Mimic env cfg
│       │   ├── visuomotor_mimic_env_cfg.py         ← image Mimic env cfg (generate_dataset)
│       │   ├── mimic_env.py                        ← single-arm ManagerBasedRLMimicEnv impl
│       │   └── leader_teleop_env_cfg.py            ← physical SO101 leader over USB (native joint teleop)
│       ├── gr1t2/
│       │   └── waist_enabled_env_cfg.py            ← GR1T2 variant, dual-source VR teleop
│       ├── mdp/                                    ← observations / events / terminations / subtasks
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