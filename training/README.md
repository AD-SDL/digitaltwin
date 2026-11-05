# Training

VLA policy training for the OpenArm cabinet task.

Adapted from
[gemblerz/aiet2-il:scripts/train_vla_policy.sh](https://github.com/gemblerz/aiet2-il/blob/main/scripts/train_vla_policy.sh),
specialized for the OpenArm cabinet LeRobot v3.0 dataset.

## What this trains against

The lerobot CLI (`lerobot-train`) consumes a **LeRobot v3.0 dataset
directory**, not a raw HDF5 file. For the cabinet task that directory looks
like:

```
<isaaclab_root>/datasets/cabinet/openarm_vr_cabinet_lerobot/
  meta/info.json       # 100 episodes, 101,869 frames, 20 fps
  data/chunk-000/...   # parquet
  videos/.../*.mp4     # chest, left_wrist, right_wrist (h264)
```

The sibling file `openarm_vr_cabinet_images.hdf5` (~56 GB) is the pre-conversion
Isaac Lab dataset and is **not** directly trainable here; convert it with
[tasks/cabinet/tasks/scripts/convert_to_lerobot_parallel.py](../tasks/cabinet/tasks/scripts/convert_to_lerobot_parallel.py)
first if you need to (re)generate the LeRobot directory.

Dataset feature summary (from `meta/info.json`):

| key                             | dtype  | shape          |
|---------------------------------|--------|----------------|
| action                          | f32    | (16,)          |
| observation.state               | f32    | (22,)          |
| observation.images.chest        | video  | 480×848×3 @20  |
| observation.images.left_wrist   | video  | 480×640×3 @20  |
| observation.images.right_wrist  | video  | 480×640×3 @20  |

Action is the bimanual EE target (pos+quat+gripper for each arm, 8+8).

## Prerequisites

- LeRobot checkout at `<lerobot_root>` with `.venv/bin/lerobot-train`,
  `accelerate-launch`, and `hf` on `PATH`. The script auto-prepends
  `$LEROBOT_DIR/.venv/bin` when present (`LEROBOT_DIR` defaults to
  `$HOME/repo/lerobot`; override to match your checkout). If any binary is
  missing, install the training extras from the LeRobot checkout:
  ```bash
  INSTALL_EXTRAS=training,smolvla,pi,groot \
    <lerobot_root>/scripts/install_lerobot.sh
  ```
- `jq` and `curl` on `PATH`.
- At least one CUDA GPU (pi05 fine-tune needs ≥48 GB VRAM with the default
  bf16 + gradient-checkpointing + train-expert-only recipe; see the pi05
  comments in [scripts/train_vla_policy.sh](scripts/train_vla_policy.sh)).

## Quick start

```bash
cd <guide_repo_root>/training

# Pi0.5 fine-tune
./scripts/train_vla_policy.sh pi05

# Or SmolVLA / GR00T
./scripts/train_vla_policy.sh smolvla
EMBODIMENT_TAG=new_embodiment ./scripts/train_vla_policy.sh groot
```

Outputs land under `artifacts/train/<job_name>/` and logs under
`artifacts/logs/<job_name>.log`. If the output dir already exists and
`RESUME` is not `true`, a timestamp suffix is added so prior runs are
preserved.

## Common overrides

| var               | default                                                                 |
|-------------------|-------------------------------------------------------------------------|
| `LEROBOT_DIR`     | `$HOME/repo/lerobot`                                                    |
| `DATASET_ROOT`    | `$HOME/repo/IsaacLab/datasets/cabinet/openarm_vr_cabinet_lerobot`       |
| `DATASET_REPO_ID` | `openarm/vr_cabinet` (logical tag; local root is what's actually read)  |
| `EPOCHS`          | `20`                                                                    |
| `BATCH_SIZE`      | `8` per GPU                                                             |
| `NUM_PROCESSES`   | `1` -- see "Multi-GPU" note below                                       |
| `NUM_WORKERS`     | `4`                                                                     |
| `SAVE_FREQ`       | `2000` steps                                                            |
| `LOG_FREQ`        | `10` steps                                                              |
| `WANDB_ENABLE`    | `false` (set `true` + `WANDB_MODE=online` to stream)                    |
| `RESUME`          | `false` (set `true` to continue latest checkpoint in `OUTPUT_DIR`)      |
| `PRETRAINED_PATH` | `lerobot/pi05_base` for pi05, `lerobot/smolvla_base` for smolvla        |
| `BASE_MODEL_PATH` | `nvidia/GR00T-N1.7-3B` (groot only)                                     |
| `EMBODIMENT_TAG`  | prompted for groot                                                      |

The two path defaults assume the layout these docs recommend; they are
override-first, not hardcoded.

Step count is computed as `ceil(total_frames * EPOCHS / (BATCH_SIZE * NUM_PROCESSES))`
and passed as `--steps` to `lerobot-train`.

Any arguments after the policy name are forwarded verbatim to `lerobot-train`,
so you can pin-prick individual flags without editing the script:

```bash
./scripts/train_vla_policy.sh pi05 \
  --optimizer.lr=1e-4 \
  --policy.chunk_size=50
```

## Layout

```
training/
  README.md
  scripts/
    train_vla_policy.sh
  artifacts/
    train/   # checkpoints land here (git-ignored)
    logs/    # tee'd stdout+stderr per run (git-ignored)
```

## Multi-GPU note

Default is `NUM_PROCESSES=1`. To use multiple GPUs:

```bash
NUM_PROCESSES=2 ./scripts/train_vla_policy.sh pi05
```

The lerobot training loop runs `make_train_eval_datasets(cfg)` on rank 0 only
(see `lerobot/src/lerobot/scripts/lerobot_train.py:261-264`) and then calls
`accelerator.wait_for_everyone()`. On a first run against a new dataset the
rank-0-only work includes computing episode stats and warming the video
decoder — easily longer than NCCL's default 10-minute watchdog, which then
kills both ranks with `ALLREDUCE` timeout / SIGABRT.

The script bumps `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` and `NCCL_TIMEOUT` to one
hour as a workaround. If your dataset is huge, raise them further:

```bash
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 NCCL_TIMEOUT=7200 \
  NUM_PROCESSES=2 ./scripts/train_vla_policy.sh pi05
```

Either way, doing a short single-GPU run first (lets rank 0 cache dataset
stats + any downloaded weights) and *then* switching to multi-GPU is the
safest first-run recipe.

## Tested on

This script has been smoke-tested with:

- `lerobot 0.6.1`, `torch 2.11.0+cu128`, `accelerate 1.15.0`
- 2× NVIDIA RTX 6000 Ada (48 GB each)
- Python 3.12

Other versions / GPUs likely work, but the pi05 memory recipe above was tuned
for the 48 GB envelope.
