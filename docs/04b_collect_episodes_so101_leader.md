# 4b) Collect episodes with a physical SO101 leader arm (LeRobot)

> **When to use this instead of [04](04_collect_episodes_meta_quest3.md):** you
> have a **physical SO101 leader arm** on USB and want to teleoperate the
> single-arm centrifuge task by hand, then produce a **LeRobot dataset**.
>
> Task ids: `Isaac-Centrifuge-SO101-LeRobot-Leader-v0` (and `-DR`). Unlike the VR
> tasks these run cameras **live** (plain gym env stepped from USB input, not the
> XR pipeline).
>
> **Single venv, single process.** The leader is read in-process via lerobot's
> ``so_leader`` module directly from the IsaacLab venv. We only touch the
> leader-teleop surface, which does not pull ``transformers`` into the import
> chain, so this works despite the ``lerobot`` ↔ ``transformers`` dep conflict
> that would otherwise break the GR00T eval path.

## A. One-time setup

Make sure the IsaacLab venv already has `lerobot` installed (it ships with
Centrifuge 3.0.0-beta2 — verify with
`"${ISAACLAB_ROOT}/.venv/bin/python" -c "import lerobot; print(lerobot.__version__)"`;
0.6.1 or newer is fine).

The feetech servo driver and `pyserial` come in via `lerobot`'s own deps, so no
extra install step is needed.

Serial access: `sudo usermod -aG dialout $USER` (re-login) or
`sudo chmod a+rw /dev/ttyACM0`.

## B. Calibrate the leader arm (once per arm id)

`connect()` needs a saved calibration for the arm `id`, written to
`~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/<id>.json`:

```bash
export ISAACLAB_ROOT="<path-to-IsaacLab-root>"
"${ISAACLAB_ROOT}/.venv/bin/lerobot-calibrate" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=leader_arm_1
```

Find the port with `ls /dev/ttyACM*`. Follow the on-screen prompts to drive each
joint through its full range.

## C. Record demos

Single terminal — the sim agent reads the leader directly:

```bash
export ISAACLAB_ROOT="<path-to-IsaacLab-root>"
export TASK_PACKAGE_DIR="<guide-repo-root>/tasks/centrifuge/tasks"
cd "${ISAACLAB_ROOT}"

./isaaclab.sh -p "${TASK_PACKAGE_DIR}/scripts/so101_lerobot_agent.py" \
  --task Isaac-Centrifuge-SO101-LeRobot-Leader-v0 \
  --port /dev/ttyACM0 --robot_id leader_arm_1 \
  --output_file ./datasets/centrifuge_so101_leader.hdf5
```

The agent opens the **Omniverse Kit GUI** by default (it requests `--viz kit`
for you, since IsaacLab 3.0.0 is otherwise headless). First launch takes a while
to compile shaders. Pass `--viz none` to run headless (no viewport, no keyboard).

**On launch** the agent force-switches the active Kit viewport to a scene camera
(``camera_table`` by default — in this Kit app the stock Persp viewport fails to
render mesh prims, so a scene camera is used instead). Pick a different one from
the viewport camera dropdown if you want (``camera_wrist``, ``camera_robot``).

In the Kit viewport:

- **S** — start / stop the current episode manually (first press starts; after
  that the agent re-arms recording automatically between episodes),
- **R** — reset the scene (also stops recording),
- **C** — cancel the current episode (discard buffered frames),
- **P** — dump a snapshot of each scene camera to ``/tmp/sim_cam_<name>.png`` (debug).

**Auto save/discard on env termination.** The env's termination manager has
three terms: ``success`` (tube settled in a bucket well for 3 s), ``tube_dropped``,
and ``time_out``. On ``success`` the agent **auto-saves** the current buffer as
the next demo; on failure/timeout it **discards**. The env auto-resets right
after, and recording stays armed, so a typical workflow is:

```
press S once -> perform demo -> env auto-resets on success -> demo saved -> next demo starts -> ...
```

Notes:

- Omit `--output_file` to **teleop only** (no recording). Recording is fully local.
- Until the first leader read lands the sim arm **holds its pose** (no snap).
- Transient Feetech `"Incorrect status packet"` errors are logged and the arm
  holds its last action (lerobot's `num_read_retries` already soaks most of them).
- `--task Isaac-Centrifuge-SO101-LeRobot-Leader-DR-v0` widens reset variety.
- `--fps` (default **20** = `1/(sim.dt*decimation)`) is stored in the HDF5.
- Recorded `actions` / `obs/joint_pos` are in **real-arm space** (deg-normalized),
  matching the physical arm and the LeRobot convention.

## D. Convert HDF5 → LeRobot dataset

```bash
cd "${ISAACLAB_ROOT}"
./isaaclab.sh -p "${TASK_PACKAGE_DIR}/scripts/convert_to_lerobot.py" \
  --input_file ./datasets/centrifuge_so101_leader.hdf5 \
  --repo_id local/centrifuge-so101-leader \
  --root ./datasets/centrifuge_so101_lerobot \
  --fps 20 --robot_type so101_follower
```

This is **local** — `--repo_id` is just a name; nothing is uploaded (add
`--push_to_hub` only if you want it on the Hub). Scene cameras ``table`` /
``wrist`` / ``robot`` become ``observation.images.table`` /
``observation.images.wrist`` / ``observation.images.robot``; state/action are the
6 SO101 joints in deg-normalized leader space.

## E. Calibration sanity

The leader→sim mapping normalizes leader percent into the USD joint ranges (see
`src/rpl_centrifuge/teleop/so101_mapping.py::SO101_USD_MAPPING`) → radians. If a
joint moves the wrong way or over/undershoots on first connect, tune that joint's
range there (or re-check calibration). Action/obs order is pinned
(`preserve_order=True`) to `[Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw]`
= leader `[shoulder_pan.pos … gripper.pos]`.

## F. Next

Publish per [05](05_upload_dataset_huggingface.md).

> **DR parity note:** `-DR` here only widens the rack/tube/bucket reset ranges.
> The Workshop's photoreal visual DR (HDRI sky, lightbox, robot recolor, mat
> jitter) needs assets not shipped in this repo and is not ported.
