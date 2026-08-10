# 4) Collect episodes with Meta Quest 3 + centrifuge task

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
  --xr
```

Notes:
- Set `--num_demos 0` for continuous recording until manual stop.
- Do not pass `--teleop_device` in XR mode unless your task explicitly requires legacy devices.

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
