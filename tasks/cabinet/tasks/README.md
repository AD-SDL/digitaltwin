# rpl_cabinet

Cabinet door-opening task pack for [Isaac Lab](https://github.com/isaac-sim/IsaacLab) **3.0.0**, using the bimanual OpenArm against the authored RPL storage cabinet (Global Industrial "Turn Handle", MPN 603599BK).

It includes:

* a bimanual OpenArm **cabinet door-opening** task (relative + absolute IK, VR teleop with both hand-tracking and controllers in one session),
* `Mimic` and `Visuomotor` variants for `annotate_demos` / `generate_dataset`, and
* the authored cabinet USD (`src/rpl_cabinet/assets/rpl_cabinet.usda`) with four revolute DOFs: two doors (±110 deg) and two lever handles (±90 deg).

Scope matches the real [`openarm_open_lab_door`](https://huggingface.co/datasets/) LeRobot dataset: left arm pulls the left door with a pre-closed gripper; the right arm stays near home. See `src/rpl_cabinet/README.md` for the geometry, dataset analysis, and known gaps carried from the `cabinet_sim/` prototype.

### Structure (mirrors `tasks/centrifuge/tasks/`)

* **base** (`openarm/env_cfg.py`) — ground + dome light + robot stand + bimanual OpenArm + the articulated cabinet. No cameras. Observations: proprioception, left/right EE pose, cabinet door angles, cabinet handle angles. Success: left-door angle magnitude past `~57 deg` for 1 s continuous.
* **teleop** (`IK-Abs` / `IK-Rel`) — camera-free (so live XR teleop never crashes). Dual-source retargeters: VR controller (priority) OR tracked hand (fallback), both the arm pose and the trigger-vs-pinch gripper.
* **Visuomotor** — adds wrist/chest cameras + RGB image obs, run **headless** by `generate_dataset` to produce image datasets (never during live teleop).
* **Mimic** — `Visuomotor` (or base) + `MimicEnvCfg` for `annotate_demos` / `generate_dataset`.

> Robot base pose, cabinet standoff, and the three camera mount poses are **STARTING POINTS**. The `cabinet_sim/HANDOFF.md` evidence (biggest known gap: the chest/wrist mount poses were fitted by eye, not measured off the real D435i / Innomaker, so the sim view does not yet match `compare_real_vs_sim.jpg`) is unchanged by the port — close them in-lab before publishing a synthetic dataset.

## Registered tasks

| Gym ID | Kind | Cameras | Use |
|---|---|---|---|
| `Isaac-OpenDoor-Bimanual-OpenArm-IK-Rel-v0` | Bimanual OpenArm, relative IK | no | VR teleop |
| `Isaac-OpenDoor-Bimanual-OpenArm-IK-Abs-v0` | Bimanual OpenArm, absolute IK | no | VR teleop (recommended) |
| `Isaac-OpenDoor-Bimanual-OpenArm-IK-Abs-Mimic-v0` | + Mimic (low-dim) | no | annotate / generate (low-dim) |
| `Isaac-OpenDoor-Bimanual-OpenArm-IK-Abs-Visuomotor-v0` | + wrist/chest cameras | yes | headless image rendering |
| `Isaac-OpenDoor-Bimanual-OpenArm-IK-Abs-Visuomotor-Mimic-v0` | Visuomotor + Mimic | yes | **generate image datasets** |

## Requirements

* Isaac Lab **3.0.0** installed (provides `isaaclab`, `isaaclab_teleop`, `isaacteleop`, `isaaclab_assets.robots.openarm`)
* Python ≥ 3.10
* OpenXR / CloudXR-compatible headset (Meta Quest 3, etc.) for VR teleop

## Install

From inside the Isaac Lab venv:

```bash
export ISAACLAB_ROOT="<path-to-IsaacLab-root>"
export TASK_PACKAGE_ROOT="<path-to-rpl_cabinet-task-package>"
cd "${TASK_PACKAGE_ROOT}"
"${ISAACLAB_ROOT}/isaaclab.sh" -p -m pip install -e .
```

The install drops a `rpl_cabinet_autoregister.pth` file into site-packages so every Python process inside the Isaac Lab venv auto-imports `rpl_cabinet` at startup — `gym.make("Isaac-OpenDoor-...")` works without any script edits.

If the `.pth` mechanism doesn't fire (some environments strip them), add `import rpl_cabinet  # noqa: F401` to your script before the first `gym.make` call.

## Run

Smoke-test (no teleop):

```bash
./isaaclab.sh -p scripts/environments/zero_agent.py \
  --task Isaac-OpenDoor-Bimanual-OpenArm-IK-Rel-v0 --num_envs 1
```

VR teleop (hand-tracking **or** controllers — both work in one session). Do **not** pass `--teleop_device`:

```bash
# Bimanual OpenArm, absolute IK (recommended)
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-OpenDoor-Bimanual-OpenArm-IK-Abs-v0 --xr --visualize kit
```

Record demonstrations (camera-free teleop task; omit `--teleop_device` to use the VR pipeline):

```bash
isaaclab teleop record \
  --task Isaac-OpenDoor-Bimanual-OpenArm-IK-Abs-Mimic-v0 \
  --xr --visualizer kit \
  --dataset_file ./datasets/opendoor_src.hdf5 \
  --num_demos 10 --reset_settle_seconds 5
```

### Generating image datasets (headless — this is how you get images)

Teleop tasks are **camera-free** (keeps live XR teleop from crashing). Camera images are rendered **headless** on the matching `Visuomotor` Mimic task:

```bash
# 1) annotate the recorded demos with subtask signals.
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --task Isaac-OpenDoor-Bimanual-OpenArm-IK-Abs-Mimic-v0 \
  --input_file  ./datasets/opendoor_src.hdf5 \
  --output_file ./datasets/opendoor_annotated.hdf5

# 2) generate the final dataset WITH images (headless; cameras render here).
PYTHONUNBUFFERED=1 HEADLESS=1 ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --task Isaac-OpenDoor-Bimanual-OpenArm-IK-Abs-Visuomotor-Mimic-v0 \
  --input_file  ./datasets/opendoor_annotated.hdf5 \
  --output_file ./datasets/opendoor_images.hdf5 \
  --num_envs 1
```

Same `PYTHONUNBUFFERED=1` + `HEADLESS=1` conventions as the centrifuge pack. See `tasks/centrifuge/tasks/README.md` for the full discussion of why offline annotation beats upstream's `--auto` replay for grasp-insertion tasks (open-door is less sensitive but shares the convention).

## Assets

* `src/rpl_cabinet/assets/rpl_cabinet.usda` — the rigged cabinet (two revolute doors, two revolute lever handles; nothing is convex-hulled, so the interior stays hollow and the wrist cameras see real shelves).

The USD is authored directly rather than converted from meshes, so every collider is an exact box or cylinder, and every dimension is a one-line edit in `scripts/build_cabinet.py` + a re-run. The `scripts/prepare_assets.py` helper only verifies and inspects — the physics schemas are applied at author time.

```bash
# Regenerate from the authoring script (requires usd-core):
./isaaclab.sh -p scripts/build_cabinet.py -o src/rpl_cabinet/assets/rpl_cabinet.usda
./isaaclab.sh -p scripts/validate_cabinet.py src/rpl_cabinet/assets/rpl_cabinet.usda   # 0 failures
./isaaclab.sh -p scripts/preview_cabinet.py src/rpl_cabinet/assets/rpl_cabinet.usda
```

See `src/rpl_cabinet/README.md` for the dimension table, the authored articulation (`/Cabinet` + four joints), and the "latch lock is kept but not wired" note (the real dataset shows no distinct lever-turn phase; the handle is treated as rigid geometry and the door opens on pull).

## Package layout

```
rpl_cabinet/
├── pyproject.toml
├── src/
│   ├── rpl_cabinet_autoregister.pth              ← .pth auto-import shim
│   └── rpl_cabinet/
│       ├── __init__.py                           ← gym.register
│       ├── README.md                             ← geometry + dataset notes
│       ├── teleop/                               ← dual-source (hand + controller) retargeters
│       ├── openarm/
│       │   ├── env_cfg.py                        ← base scene (floor + cabinet + robot stand)
│       │   ├── ik_rel_env_cfg.py                 ← Relative-IK teleop (camera-free)
│       │   ├── ik_abs_env_cfg.py                 ← Absolute-IK teleop (camera-free)
│       │   ├── visuomotor_env_cfg.py             ← + wrist/chest cameras + image obs (headless)
│       │   ├── ik_abs_mimic_env_cfg.py           ← low-dim Mimic env cfg
│       │   ├── visuomotor_mimic_env_cfg.py      ← image Mimic env cfg (generate_dataset)
│       │   └── mimic_env.py                      ← bimanual ManagerBasedRLMimicEnv impl
│       ├── mdp/                                  ← observations / events / terminations / subtasks
│       ├── agents/                               ← (empty, for robomimic JSONs later)
│       ├── assets/                               ← rpl_cabinet.usda
│       └── docs/                                 ← cabinet preview + comparison imagery
└── scripts/
    ├── prepare_assets.py                         ← verify + inspect the bundled USD
    ├── build_cabinet.py                          ← regenerate rpl_cabinet.usda from constants
    ├── validate_cabinet.py                       ← joint frames, interpenetration, swing
    ├── preview_cabinet.py                        ← render a PNG without Isaac
    ├── build_scene.py                            ← compose cabinet + OpenArm + cameras + lights
    ├── open_scene.py                             ← --exec target to open the scene in the GUI
    ├── openarm_cameras.py                        ← reusable three-camera rig on any OpenArm
    ├── isaac_latch.py                            ← handle-latch lock (NOT wired; see notes)
    ├── isaac_smoke_test.py                       ← headless check that the articulation behaves
    ├── inspect_openarm.py                        ← dump OpenArm asset DOFs / sizes
    ├── inspect_links.py                          ← dump an articulation's links / bodies
    └── diag_facing.py                            ← confirm the +X/-Y OpenArm facing convention
```
