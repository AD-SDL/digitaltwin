# 4) Collect episodes with Meta Quest 3 + centrifuge task

> **Input source (IsaacLab 3.0.0):** the VR tasks run through the IsaacTeleop
> pipeline and accept **both hand-tracking and the Touch controllers in the same
> session** — just hold the controllers, or track bare hands. Omit
> `--teleop_device` so the pipeline is selected (there is no legacy device path).
> VR-teleoperable task IDs: `Isaac-Centrifuge-Bimanual-OpenArm-IK-Abs-v0`,
> `Isaac-Centrifuge-Bimanual-OpenArm-IK-Rel-v0`, `Isaac-Centrifuge-SO101-IK-Abs-v0`,
> and `Isaac-Centrifuge-GR1T2-WaistEnabled-Abs-v0`. Set `CENTRIFUGE_TASK_ID` to
> whichever you are recording. (GR1T2 dexterous fingers need hand-tracking; the
> controllers drive its wrists with a trigger power grasp.)
>
> **Cameras / images:** the VR teleop tasks are **camera-free** (rendering a
> camera observation under the live XR pipeline crashes Kit). To get image
> datasets, record on a teleop/`-Mimic-v0` task here, then render images
> **headless** on the matching `-Visuomotor-Mimic-v0` task with
> `generate_dataset.py` — see the task package README
> ("Generating image datasets"). This mirrors the IsaacLab stack examples.

## (Optional) Train yourself for the manipulation
```
./isaaclab.sh -p ./scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Centrifuge-Bimanual-OpenArm-IK-Abs-v0  \
  --viz kit \
  --xr --kit_args="--enable omni.claude.bridge"
```

## (Optional) Visualize collision shapes (physics debug)

The teleop/record Kit app is minimal and does **not** load the PhysX debug
extensions by default, so there is no *Physics Debug* panel and no collision
wireframe overlay. To inspect colliders (e.g. to check the tube/bucket well fit
or diagnose insertion bounce), add `--enable omni.physx.ui` to `--kit_args` — it
also pulls in `omni.debugdraw`, which renders the overlay:

```
./isaaclab.sh -p ./scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Centrifuge-Bimanual-OpenArm-IK-Abs-v0 \
  --viz kit \
  --xr --kit_args="--enable omni.claude.bridge --enable omni.physx.ui"
```

Then turn on the overlay via **Window ▸ Physics ▸ Debug ▸ Colliders → All**.
The same flag works for `record_demos.py` — append `--enable omni.physx.ui` to
that command's `--kit_args` value.

Notes:
- Collision wireframes only refresh while physics is stepping (i.e. during
  active teleop/recording), and green lines trace the *actual* collider: SDF for
  the tube/bucket (follows the mesh), triangle mesh for the rack.
- This is a visualization aid only; leave it off for normal data collection.

## A. Prepare output path

```bash
export ISAACLAB_ROOT="<path-to-IsaacLab-root>"
cd "${ISAACLAB_ROOT}"
mkdir -p datasets/centrifuge
export DATASET_FILE=./datasets/centrifuge/quest3_centrifuge_dataset.hdf5
```

## B. Record demos (XR mode)

```bash
cd "${ISAACLAB_ROOT}"
./isaaclab.sh -p scripts/tools/record_demos.py \
  --task "${CENTRIFUGE_TASK_ID}" \
  --viz kit \
  --dataset_file "${DATASET_FILE}" \
  --num_demos 50 \
  --num_success_steps 10 \
  --reset_settle_seconds 5 \
  --xr
```

Notes:
- Set `--num_demos 0` for continuous recording until manual stop.
- Do not pass `--teleop_device` in XR mode unless your task explicitly requires legacy devices.
- `--reset_settle_seconds N` holds the robot at its home pose for `N` seconds after
  each episode reset (rendering only, no physics stepping) so you can move the XR
  controller back over the home gripper before the next episode records. Without it,
  the IK-Abs action snaps the arm to your hand on the first frame, ruining the demo.
  During the window a countdown shows in the instruction panel; set `0` to disable.

## C. Replay and sanity-check

```bash
cd "${ISAACLAB_ROOT}"
./isaaclab.sh -p scripts/tools/replay_demos.py \
  --task "${CENTRIFUGE_TASK_ID}" \
  --viz kit \
  --num_envs 1 \
  --dataset_file "${DATASET_FILE}"
```

## D. Quick dataset stats

```bash
cd "${ISAACLAB_ROOT}"
./isaaclab.sh -p - <<'PY'
import h5py
path = "./datasets/centrifuge/quest3_centrifuge_dataset.hdf5"
with h5py.File(path, "r") as f:
    demos = list(f.keys())
    print("episodes:", len(demos))
    print("first keys:", demos[:5])
PY
```
