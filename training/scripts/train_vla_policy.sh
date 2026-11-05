#!/usr/bin/env bash
# Train a VLA policy (smolvla | pi05 | groot) against the OpenArm cabinet
# LeRobot v3.0 dataset using the lerobot training CLI.
#
# Adapted from:
#   https://github.com/gemblerz/aiet2-il/blob/main/scripts/train_vla_policy.sh
#
# Default assumptions (override via the env vars listed below):
#   - LeRobot checkout:    $HOME/repo/lerobot (venv at $HOME/repo/lerobot/.venv)
#   - Dataset (local):     $HOME/repo/IsaacLab/datasets/cabinet/openarm_vr_cabinet_lerobot
#   - GPUs:                NUM_PROCESSES=1 (see "NCCL timeout" note at end of file
#                          for why multi-GPU is opt-in)
#
# Usage:
#   scripts/train_vla_policy.sh <smolvla|pi05|groot> [extra lerobot-train args...]
#
# Common overrides (env vars):
#   LEROBOT_DIR, DATASET_REPO_ID, DATASET_ROOT, OUTPUT_DIR, EPOCHS,
#   BATCH_SIZE, NUM_PROCESSES, NUM_WORKERS, DEVICE, LOG_FREQ, SAVE_FREQ,
#   WANDB_ENABLE, WANDB_MODE, WANDB_PROJECT, RESUME, PRETRAINED_PATH,
#   BASE_MODEL_PATH, EMBODIMENT_TAG
#
# NOTE: The hdf5 file ~/.../openarm_vr_cabinet_images.hdf5 is NOT a LeRobot
# dataset; lerobot-train only consumes the LeRobot v3.0 layout (meta/, data/,
# videos/). The sibling openarm_vr_cabinet_lerobot/ directory IS that layout
# and is what this script trains on.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <smolvla|pi05|groot> [extra lerobot-train args...]" >&2
  exit 2
fi

POLICY="$1"; shift
case "$POLICY" in
  smolvla|pi05|groot) ;;
  *) echo "Unknown policy '$POLICY' (expected: smolvla | pi05 | groot)" >&2; exit 2 ;;
esac

# ---------------------------------------------------------------------------
# Config (override via env)
# ---------------------------------------------------------------------------
LEROBOT_DIR="${LEROBOT_DIR:-$HOME/repo/lerobot}"

# Use the LeRobot v3.0 directory, not the raw IsaacLab hdf5.
DATASET_ROOT="${DATASET_ROOT:-$HOME/repo/IsaacLab/datasets/cabinet/openarm_vr_cabinet_lerobot}"
# DATASET_REPO_ID is a *logical* id used to tag the run + forms the default
# output path. It does not need to exist on the HF Hub because DATASET_ROOT
# is set and info.json is present locally.
DATASET_REPO_ID="${DATASET_REPO_ID:-openarm/vr_cabinet}"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"      # Default to 1; see the NCCL-timeout
                                          # note at the end of this file for
                                          # why multi-GPU is opt-in.
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
LOG_FREQ="${LOG_FREQ:-10}"
SAVE_FREQ="${SAVE_FREQ:-2000}"

WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_PROJECT="${WANDB_PROJECT:-lerobot-vla-openarm-cabinet}"

JOB_NAME="${JOB_NAME:-${POLICY}_${DATASET_REPO_ID##*/}_e${EPOCHS}}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/artifacts/train/${JOB_NAME}}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/artifacts/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${JOB_NAME}.log}"
RESUME="${RESUME:-false}"

# If the output dir already exists and we are not resuming, suffix with a
# timestamp so we don't clobber a previous run.
if [[ -d "$OUTPUT_DIR" && "$RESUME" != "true" ]]; then
  RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
  OUTPUT_DIR="${OUTPUT_DIR}_${RUN_SUFFIX}"
  LOG_FILE="${LOG_DIR}/${JOB_NAME}_${RUN_SUFFIX}.log"
fi

mkdir -p "$LOG_DIR"  # NOTE: do NOT pre-create OUTPUT_DIR; lerobot-train refuses to reuse an existing dir

# ---------------------------------------------------------------------------
# Tool resolution
# ---------------------------------------------------------------------------
if [[ -x "$LEROBOT_DIR/.venv/bin/lerobot-train" ]]; then
  export PATH="$LEROBOT_DIR/.venv/bin:$PATH"
fi

for bin in lerobot-train accelerate-launch jq curl; do
  if ! command -v "$bin" >/dev/null 2>&1; then
    echo "Missing required command on PATH: $bin" >&2
    echo "Install LeRobot training extras first, e.g.:" >&2
    echo "  INSTALL_EXTRAS=training,smolvla,pi,groot $LEROBOT_DIR/scripts/install_lerobot.sh" >&2
    exit 1
  fi
done

# ---------------------------------------------------------------------------
# Dataset + step count
# ---------------------------------------------------------------------------
INFO_JSON="$DATASET_ROOT/meta/info.json"
if [[ ! -f "$INFO_JSON" ]]; then
  echo "LeRobot info.json not found at $INFO_JSON" >&2
  echo "Expected a LeRobot v3.0 dataset directory. The raw hdf5 file" >&2
  echo "(openarm_vr_cabinet_images.hdf5) is not directly trainable here --" >&2
  echo "convert it first (see tasks/cabinet/tasks/scripts/convert_to_lerobot.py)." >&2
  exit 1
fi

total_frames="$(jq -r '.total_frames' "$INFO_JSON")"
if ! [[ "$total_frames" =~ ^[0-9]+$ ]]; then
  echo "Could not read total_frames from $INFO_JSON" >&2
  exit 1
fi

effective_batch=$(( BATCH_SIZE * NUM_PROCESSES ))
# ceil(total_frames * EPOCHS / effective_batch)
steps=$(( (total_frames * EPOCHS + effective_batch - 1) / effective_batch ))

# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------
# NCCL timeout: the first rank runs `make_train_eval_datasets` alone while the
# other ranks sit at a `wait_for_everyone()` barrier (lerobot_train.py:261-264).
# For large datasets that first call can easily exceed NCCL's default 10 min
# watchdog and crash both ranks with ALLREDUCE timeout + SIGABRT. Bump the
# watchdog + heartbeat generously.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
# Reduce allocator fragmentation -- helps keep tight-fitting VLAs (pi05)
# inside the 48 GB envelope, as the OOM error message itself suggests.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Make sure all GPUs are visible unless the caller narrowed CUDA_VISIBLE_DEVICES.

cmd=(
  accelerate-launch
  --num_processes="$NUM_PROCESSES"
  "$(command -v lerobot-train)"
  --dataset.repo_id="$DATASET_REPO_ID"
  --dataset.root="$DATASET_ROOT"
  --output_dir="$OUTPUT_DIR"
  --job_name="$JOB_NAME"
  --batch_size="$BATCH_SIZE"
  --num_workers="$NUM_WORKERS"
  --steps="$steps"
  --policy.device="$DEVICE"
  --policy.push_to_hub=false
  --log_freq="$LOG_FREQ"
  --save_freq="$SAVE_FREQ"
)

if [[ "$RESUME" == "true" ]]; then
  cmd+=( --resume=true )
fi

if [[ "$WANDB_ENABLE" == "true" ]]; then
  cmd+=(
    --wandb.enable=true
    --wandb.mode="$WANDB_MODE"
    --wandb.project="$WANDB_PROJECT"
  )
else
  cmd+=( --wandb.enable=false )
fi

# -- policy-specific ---------------------------------------------------------
# The cabinet dataset exposes three image streams:
#   observation.images.chest
#   observation.images.left_wrist
#   observation.images.right_wrist
# No rename_map is applied; the policy will see all three.

case "$POLICY" in
  smolvla)
    PRETRAINED_PATH="${PRETRAINED_PATH:-lerobot/smolvla_base}"
    cmd+=(
      --policy.path="$PRETRAINED_PATH"
    )
    ;;
  pi05)
    PRETRAINED_PATH="${PRETRAINED_PATH:-lerobot/pi05_base}"
    # pi05 = 4.14B-param VLA (PaliGemma VLM + Gemma-300M action expert).
    # In fp32 full fine-tune at bs=8 needs ~45+ GB just for weights/grads and
    # OOMs the 48 GB RTX 6000 Ada before step 1 (AdamW fp32 moments alone are
    # ~33 GB). Even bf16 full fine-tune with AdamW stays >48 GB.
    #
    # Default recipe that actually fits:
    #   - bf16 weights/activations
    #   - gradient_checkpointing to shrink activation memory
    #   - train_expert_only=true: freeze VLM, train only the ~300M action
    #     expert + projection heads. This is the standard pi05 fine-tune
    #     recipe from the openpi reference.
    #   - batch_size left at $BATCH_SIZE (default 8); drop if OOM recurs.
    #
    # Switch to full fine-tune with PI05_TRAIN_EXPERT_ONLY=false (you will
    # also need a smaller BATCH_SIZE and/or an 8-bit optimizer).
    PI05_DTYPE="${PI05_DTYPE:-bfloat16}"
    PI05_GRAD_CKPT="${PI05_GRAD_CKPT:-true}"
    PI05_TRAIN_EXPERT_ONLY="${PI05_TRAIN_EXPERT_ONLY:-true}"
    PI05_FREEZE_VISION="${PI05_FREEZE_VISION:-false}"
    cmd+=(
      --policy.type=pi05
      --policy.pretrained_path="$PRETRAINED_PATH"
      --policy.dtype="$PI05_DTYPE"
      --policy.gradient_checkpointing="$PI05_GRAD_CKPT"
      --policy.train_expert_only="$PI05_TRAIN_EXPERT_ONLY"
      --policy.freeze_vision_encoder="$PI05_FREEZE_VISION"
      --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}'
    )
    ;;
  groot)
    BASE_MODEL_PATH="${BASE_MODEL_PATH:-nvidia/GR00T-N1.7-3B}"
    if [[ -z "${EMBODIMENT_TAG:-}" ]]; then
      if [[ -t 0 ]]; then
        read -r -p "Enter GR00T embodiment tag (e.g. new_embodiment, libero_sim): " EMBODIMENT_TAG
      else
        echo "EMBODIMENT_TAG must be set for groot (non-interactive shell)." >&2
        exit 1
      fi
    fi
    cmd+=(
      --policy.type=groot
      --policy.base_model_path="$BASE_MODEL_PATH"
      --policy.embodiment_tag="$EMBODIMENT_TAG"
      --policy.use_bf16=true
      --dataset.image_transforms.enable=true
    )
    ;;
esac

# Any extra args the caller passed get appended verbatim.
cmd+=( "$@" )

# ---------------------------------------------------------------------------
# Summary + run
# ---------------------------------------------------------------------------
cat <<EOF
================ train_vla_policy ================
policy         : $POLICY
dataset repo   : $DATASET_REPO_ID
dataset root   : $DATASET_ROOT
total frames   : $total_frames
epochs         : $EPOCHS
num processes  : $NUM_PROCESSES
batch per GPU  : $BATCH_SIZE  (effective $effective_batch)
steps          : $steps
output dir     : $OUTPUT_DIR
log file       : $LOG_FILE
command        :
  ${cmd[*]}
==================================================
EOF

set -o pipefail
"${cmd[@]}" 2>&1 | tee "$LOG_FILE"
