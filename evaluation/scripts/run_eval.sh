#!/usr/bin/env bash
# Orchestrate a two-venv rollout evaluation of a LeRobot VLA policy on the
# OpenArm cabinet door task.
#
#   1. start policy_server.py   (lerobot venv,  GPU $POLICY_GPU)
#   2. wait for it to load      (~2 min for pi05: 9.4 GB of weights)
#   3. run eval_rollouts.py     (IsaacLab venv, GPU $SIM_GPU, headless)
#   4. summarize                (stdlib python3)
#   5. tear the server down     (also on Ctrl-C / failure, via trap)
#
# The split is forced: the IsaacLab venv pins huggingface-hub 1.32, which
# makes `import transformers` raise, and pi05 needs transformers. See
# docs/07_evaluate_policy_rollouts.md.
#
# Usage:
#   evaluation/scripts/run_eval.sh [extra eval_rollouts.py args...]
#
# Common overrides (env vars):
#   CHECKPOINT, NUM_EPISODES, EPISODE_LENGTH_S, OUT_DIR, RUN_NAME,
#   POLICY_GPU, SIM_GPU, PORT, LEROBOT_DIR, ISAACLAB_DIR,
#   N_ACTION_STEPS, NUM_INFERENCE_STEPS, VIDEO_EPISODES, SEED

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CHECKPOINT="${CHECKPOINT:-/data1/yongho/models/242000/pretrained_model}"
LEROBOT_DIR="${LEROBOT_DIR:-$HOME/repo/lerobot}"
ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/repo/IsaacLab}"

NUM_EPISODES="${NUM_EPISODES:-50}"
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-80}"
SEED="${SEED:-10000}"
VIDEO_EPISODES="${VIDEO_EPISODES:-3}"

POLICY_GPU="${POLICY_GPU:-1}"
SIM_GPU="${SIM_GPU:-0}"
PORT="${PORT:-5557}"

RUN_NAME="${RUN_NAME:-$(basename "$(dirname "$CHECKPOINT")")_$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/evaluation/artifacts/$RUN_NAME}"

LEROBOT_PY="$LEROBOT_DIR/.venv/bin/python"
READY_FILE="$OUT_DIR/.server_ready"
SERVER_LOG="$OUT_DIR/policy_server.log"
ROLLOUT_LOG="$OUT_DIR/rollouts.log"

for f in "$LEROBOT_PY" "$ISAACLAB_DIR/isaaclab.sh"; do
  [[ -x "$f" ]] || { echo "Missing: $f" >&2; exit 1; }
done
[[ -d "$CHECKPOINT" ]] || { echo "Checkpoint dir not found: $CHECKPOINT" >&2; exit 1; }
if [[ ! -f "$CHECKPOINT/config.json" ]]; then
  echo "No config.json in $CHECKPOINT." >&2
  echo "Point CHECKPOINT at the checkpoint's pretrained_model/ subdirectory." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
rm -f "$READY_FILE"

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[run_eval] stopping policy server (pid $SERVER_PID)"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -f "$READY_FILE"
}
trap cleanup EXIT INT TERM

cat <<EOF
================ run_eval ================
checkpoint    : $CHECKPOINT
episodes      : $NUM_EPISODES
horizon       : ${EPISODE_LENGTH_S}s
policy GPU    : $POLICY_GPU
sim GPU       : $SIM_GPU
out dir       : $OUT_DIR
==========================================
EOF

# --- 1. policy server -------------------------------------------------------
server_args=(
  "$SCRIPT_DIR/policy_server.py"
  --checkpoint "$CHECKPOINT"
  --device cuda:0                 # cuda:0 *within* CUDA_VISIBLE_DEVICES=$POLICY_GPU
  --port "$PORT"
  --ready-file "$READY_FILE"
)
[[ -n "${N_ACTION_STEPS:-}" ]] && server_args+=( --n-action-steps "$N_ACTION_STEPS" )
[[ -n "${NUM_INFERENCE_STEPS:-}" ]] && server_args+=( --num-inference-steps "$NUM_INFERENCE_STEPS" )

echo "[run_eval] starting policy server -> $SERVER_LOG"
CUDA_VISIBLE_DEVICES="$POLICY_GPU" "$LEROBOT_PY" -I "${server_args[@]}" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo -n "[run_eval] waiting for model load"
for _ in $(seq 1 600); do          # up to 10 min
  if [[ -f "$READY_FILE" ]]; then echo " ready"; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo " FAILED"; echo "--- policy_server.log ---"; cat "$SERVER_LOG"; exit 1
  fi
  echo -n "."; sleep 1
done
[[ -f "$READY_FILE" ]] || { echo " timeout"; cat "$SERVER_LOG"; exit 1; }

# --- 2. rollouts ------------------------------------------------------------
echo "[run_eval] running rollouts -> $ROLLOUT_LOG"
cd "$ISAACLAB_DIR"
set +e
HEADLESS=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$SIM_GPU" \
  ./isaaclab.sh -p "$SCRIPT_DIR/eval_rollouts.py" \
    --num-episodes "$NUM_EPISODES" \
    --episode-length-s "$EPISODE_LENGTH_S" \
    --seed "$SEED" \
    --port "$PORT" \
    --video-episodes "$VIDEO_EPISODES" \
    --out-dir "$OUT_DIR" \
    "$@" 2>&1 | tee "$ROLLOUT_LOG"
rc=${PIPESTATUS[0]}
set -e
cd - >/dev/null

[[ $rc -eq 0 ]] || { echo "[run_eval] rollouts exited $rc" >&2; exit $rc; }

# --- 3. report --------------------------------------------------------------
echo
python3 "$SCRIPT_DIR/summarize_eval.py" "$OUT_DIR" | tee "$OUT_DIR/report.txt"
python3 "$SCRIPT_DIR/summarize_eval.py" "$OUT_DIR" --json > "$OUT_DIR/report.json"
echo "[run_eval] artifacts in $OUT_DIR"
