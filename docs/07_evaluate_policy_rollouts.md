# 7) Evaluate a trained policy with sim rollouts

Measure a fine-tuned VLA policy on the OpenArm cabinet door task: success
rate, partial progress, wall-clock cost, inference latency, and GPU load.

Scripts live in [`evaluation/scripts/`](../evaluation/scripts/).

## Two hard constraints

Both of these silently produce wrong results if you ignore them.

### 1. The policy and the simulator cannot share an interpreter

The IsaacLab venv pins `huggingface-hub==1.32.0`. `transformers` requires
`<1.0` and raises on import there:

```text
ImportError: huggingface-hub>=0.34.0,<1.0 is required ..., but found huggingface-hub==1.32.0
```

pi05 is PaliGemma + a Gemma-300M action expert, so it needs `transformers`.
The harness therefore runs the env in the IsaacLab venv, the policy in the
lerobot venv, and passes observations and actions over a loopback TCP socket.

### 2. The task cfg's episode horizon is shorter than the demonstrations

`OpenDoorEnvCfg.episode_length_s = 40.0` → 800 steps at 20 Hz. But in
`openarm_vr_cabinet_images.hdf5` the success condition first fires at:

| | steps | seconds |
| --- | --- | --- |
| p50 | 1018 | 50.9 |
| p90 | 1185 | 59.3 |
| max | 1384 | 69.2 |

Not one demo reaches success inside 800 steps. Evaluating at the cfg default
reports **0% success for any policy, however good**. `eval_rollouts.py`
defaults `--episode-length-s 80` for this reason; do not lower it below ~70.

**How much is enough?** Measured, not guessed — a 160 s run of checkpoint
242000 found its last success at step 1800 (90 s) and nothing at all in the
remaining 70 s:

| cap | successes |
| --- | --- |
| 40 s | 3/20 |
| 60 s | 4/20 |
| 80 s | 5/20 |
| **100 s** | **6/20** |
| 120-160 s | 6/20 |

So ~100 s is the knee. Past it you pay a full episode (~5 min wall) per
episode to watch an arm that stopped moving: 14/14 failures plateaued, with a
median 135 s of motionless time after their peak, and only 1/14 was still
improving in its final 10%. Going 80 s -> 160 s bought one success out of
twenty for 38 extra minutes. `--episode-length-s 100` is the recommended
setting; 80 is the default only because it was set before this was measured.

## A. What counts as success

The same term the dataset was scored with —
`rpl_cabinet.mdp.terminations.door_opened`: either `hinge_left` or
`hinge_right` past **1.40 rad (80.2°)**, held continuously for **1.0 s**.
98/100 recorded demos satisfy it, so the metric is calibrated against the
training distribution.

The margin is thin: demos top out at 1.56 rad (89°) median, only ~9° past
threshold. A binary pass/fail therefore throws away most of the signal, so
the harness also records per episode:

- `max_door_angle_deg` — how far the door actually got,
- `grasped_handle` — whether the `grasp_handle` subtask signal ever fired.
  Read this as **engaged**, not grasped: it is a conjunction of "EE within
  12 cm of the handle hub" AND "that door already past 10°", so it cannot
  fire before the door moves, and it stops counting if the gripper trails the
  hub as the door swings through its arc. Episodes with `grasp=n` and 30-50°
  of door travel do occur, and mean the door opened without a sustained hold
  near the hub,
- `success_step` — when the dwell completed.

A run with 0% success but 70° median door angle is a different problem from
one where the gripper never reaches the handle.

## B. Run the evaluation

```bash
cd <guide_repo_root>

CHECKPOINT=/data1/yongho/models/242000/pretrained_model \
NUM_EPISODES=50 \
evaluation/scripts/run_eval.sh
```

`CHECKPOINT` must point at the `pretrained_model/` subdirectory of a
`lerobot-train` checkpoint (the one holding `config.json` and
`model.safetensors`), not at the step directory above it.

The orchestrator starts the policy server on GPU 1, waits for the weights to
load (~105 s for pi05's 9.4 GB), runs the rollouts headless on GPU 0, writes
a report, and stops the server on exit — including on Ctrl-C.

### Useful overrides

| Variable | Default | Notes |
| --- | --- | --- |
| `NUM_EPISODES` | `50` | 50 gives roughly ±14 pp of 95% CI at mid-range rates. |
| `EPISODE_LENGTH_S` | `80` | Keep ≥ 70; see constraint 2. |
| `SEED` | `10000` | Episode *i* uses `SEED + i`, for both the scene and the policy's noise. Reuse to compare checkpoints on identical rollouts. |
| `POLICY_GPU` / `SIM_GPU` | `1` / `0` | Separate cards keep the latency and utilization numbers interpretable. |
| `N_ACTION_STEPS` | from cfg (50) | Replan horizon. Lower = more frequent re-planning, more inference cost. |
| `NUM_INFERENCE_STEPS` | from cfg (10) | Flow-matching denoise steps. Trades latency against action quality. |
| `VIDEO_EPISODES` | `3` | Rollout mp4s for the first N episodes (all three cameras). |

Arguments after the script name pass through to `eval_rollouts.py`:

```bash
# Wider scene randomization than the task default (±0.02 m, ±0.05 rad)
evaluation/scripts/run_eval.sh --cabinet-jitter 0.05,0.05,0.12
```

## C. Read the report

A real 20-episode run of checkpoint 242000 (seed 10000, 80 s horizon, policy
on GPU 1 / sim on GPU 0):

```text
SUCCESS RATE    35.0%   (7/20)   95% CI [18.1%, 56.7%]

PARTIAL PROGRESS (success needs door >= 80 deg held 1.0 s)
  engaged handle       90.0% of episodes   (EE within 12 cm of hub WHILE that door is past 10 deg)
  max door angle      p50  55.9 deg   p90  86.5 deg   best  87.4 deg

TIME
  total               43.8 min
  per episode         131.2 s wall, 1383 steps, realtime factor 0.52x
  time to success     p50 51.2 s   p90 58.5 s (sim time)

LATENCY (ms)                    n        mean      p50      p90      p99      max
  round trip (all steps)         27661     12.58     9.33    10.39   171.46   321.47
    replan steps                   557    172.43   171.54   175.82   179.78   321.47
    cached-chunk steps           27104      9.29     9.31    10.30    11.79    15.50
  server total                   27661      9.60     6.36     6.86   168.40   316.53
    policy forward               27661      6.15     3.06     3.11   164.98   291.02
    preprocess                   27661      3.33     3.26     3.62     5.21    24.40
    postprocess                  27661      0.12     0.12     0.13     0.31     1.10
  GPU device time                27661      9.51     6.27     6.77   168.31   316.31
  env.step (sim+render)          27661     80.52    79.41    92.39   102.29   341.06
  obs pack (sim side)            27661      0.32     0.32     0.33     0.36     1.49

GPU
  gpu0: util mean 50% / p90 60% / max 66% | mem peak 7.1 GiB | power mean 115 W | energy ~84.1 Wh
  gpu1: util mean  3% / p90  1% / max 91% | mem peak 9.7 GiB | power mean  92 W | energy ~66.7 Wh
```

The success interval is Wilson, not normal-approximation — rollout counts are
small and the normal interval runs outside [0, 1] near 0% or 100%. At 20
episodes the interval is ±19 pp wide, which is why 20 is a triage sample and
50 is the number to quote.

### Outcomes are bimodal, so read the angle distribution

Per-episode peak door angle across those 20 episodes, sorted:

```text
15 16 32 34 47 49 49 52 52 56 56 60 70 | 82 83 86 86 86 87 87
                                        ^ the 80.2 deg threshold sits here
```

There is a clean gap between 70° and 82°: an episode either carries the swing
through (82-87°) or stalls well short. Lowering the threshold to 70° would
convert exactly **one** failure, and to 60° only one more. So on this
checkpoint the result is not threshold-sensitive, and a near-miss reading of
the p50 would be wrong — the p50 of 55.9° is the centre of the *stalled*
mode, not a population hovering below the bar.

### Failures come in two kinds; do not pool them

Across a 160 s run the 14 failures split cleanly:

| mode | n | peak door angle |
| --- | --- | --- |
| never engaged | 8 | 0-12° |
| engaged, then stalled | 6 | 24-60° |

These are different problems — a reaching failure and a mid-pull failure —
and a single success rate hides both. `max_door_angle_deg` plus
`door_angle_trace_deg` separates them; the trace makes it obvious, since a
stall shows as a hard plateau while a success climbs monotonically.

Chunk noise alone decides which you get: the same checkpoint on the same
scene seeds produced 8 never-engaged failures in one run and 0 in another,
differing only in the flow-matching noise draw. Treat a success rate from 20
episodes as sensitive to that.

Check this distribution before concluding anything from a success rate. A
population pressed up against the threshold means something different from
two separated modes, and only the second says "the failures are a distinct
behaviour, go watch the video."

### Latency is bimodal, on purpose

pi05 with `n_action_steps=50` runs a full chunk inference once every 50
control steps and pops a cached action on the other 49. A single mean
describes neither population, so the report splits them:

| | p50 | what it is |
| --- | --- | --- |
| replan steps | 169 ms | PaliGemma + action expert, 10 denoise steps |
| cached-chunk steps | 9.3 ms | preprocess + tokenize + queue pop |

The control period is 50 ms at 20 Hz. Cached steps fit comfortably; **replan
steps do not** — a 169 ms inference stalls the loop for ~3.4 control periods.
In sim that only slows the rollout. On hardware it is a real-time violation,
and the fixes are async inference, a shorter chunk, fewer denoise steps
(`NUM_INFERENCE_STEPS`), or interpolating across the gap.

`GPU device time` is measured with CUDA events around the inference region,
so it excludes the socket and Python overhead that `round trip` includes.
The ~3 ms gap between `round trip` and `server total` on cached steps is the
3.06 MB/step of raw images crossing loopback.

### The simulator, not the policy, is the bottleneck

`env.step` costs **86 ms mean** against the policy's 9.4 ms on cached steps:
rendering three 480p cameras dominates, and the realtime factor is 0.50x.
Budget accordingly:

| horizon | worst-case steps | worst-case per episode | 50 episodes |
| --- | --- | --- | --- |
| 80 s | 1600 | ~2.7 min | **~2.2 h** |

Successful episodes end at the dwell and cost less (demos succeed around step
1020, so ~1.7 min). To speed a sweep up, lower `--num-episodes` first; the
camera rendering is not something the harness can avoid, since those images
are the policy's input.

### Reproducibility needs both seeds

pi05 is a flow-matching policy: every action chunk starts from a fresh
`torch.normal` draw, taken from the **global** RNG with no explicit generator
(`lerobot/policies/common/flow_matching.py::sample_noise`). Seeding only the
environment therefore does not pin a rollout — two runs over scene seed 10000
were measured reaching 42.6° and 51.6° on the same episode.

The harness sends the episode seed to both sides: the env seeds its reset
events, and the policy server seeds torch on each `reset`. Episode *i* is then
reproducible, while still drawing different noise from episode to episode.
Pass `--no-policy-seed` to the server to go back to free-running noise.

This matters most when comparing checkpoints. Without it, part of any
difference you measure is noise draw rather than model quality.

## D. Outputs

```text
evaluation/artifacts/<run>/
  manifest.json       run config: checkpoint, horizon, seed, server settings
  episodes.jsonl      one record per episode (success, timings, door angle)
  gpu_trace.csv       per-GPU utilization / memory / power, 2 Hz
  report.txt          the text report above
  report.json         the same numbers, machine-readable
  policy_server.log   model load + server lifecycle
  rollouts.log        per-episode progress lines
  videos/             rollout mp4s (see below)
```

### Videos

By default each recorded episode is one `episode_NNN.mp4` tiling all three
policy input streams into a 1280x960 frame:

```text
+----------------------------------+
|   pad |  chest 848x480  |  pad   |   1280 x 480
+----------------+-----------------+
|   left_wrist   |   right_wrist   |   1280 x 480
+----------------+-----------------+
```

Nothing is resampled — the chest view is letterboxed to the wrists' combined
width rather than stretched, so every pixel is a pixel the policy saw. Panes
are captioned when Pillow is importable; `--no-video-labels` turns that off.

For native-resolution files instead, one per camera
(`episode_000_chest.mp4`, `_left_wrist`, `_right_wrist`):

```bash
evaluation/scripts/run_eval.sh --video-layout separate
```

Frames stream straight to the encoder rather than accumulating in RAM — a
1600-step composite episode would otherwise hold ~5.9 GB.

Re-report without re-running:

```bash
python3 evaluation/scripts/summarize_eval.py evaluation/artifacts/<run>
```

## E. Comparing checkpoints

Hold `SEED`, `EPISODE_LENGTH_S` and `--cabinet-jitter` fixed so episodes are
scene-identical, then vary only `CHECKPOINT`:

```bash
for step in 100000 220000 242000 254000; do
  CHECKPOINT=/data1/yongho/models/$step/pretrained_model \
  RUN_NAME=pi05_$step NUM_EPISODES=50 SEED=10000 \
  evaluation/scripts/run_eval.sh
done
```

Training loss ranks 242000 best among the saved checkpoints, but that ranking
comes from a training loss on a dataset with `eval_split: 0.0` — there is no
held-out signal in the run. Rollout success rate is the first real
measurement, and it does not have to agree.

## Caveats

- **One episode at a time.** The policy holds a single action-chunk queue, so
  parallel envs would share and corrupt each other's plan. A batched variant
  needs per-env queues in the server.
- **Narrow eval distribution.** Only the cabinet pose is randomized (±0.02 m,
  ±0.05 rad by default) and the robot starts from the same joint state every
  episode. A high success rate here is not evidence of robustness; widen with
  `--cabinet-jitter`.
- **Quaternion labels are wrong in the dataset, harmlessly.** The LeRobot
  feature names say `quat_x..quat_w`, but the values are IsaacLab `wxyz`
  (demo frame 0 is `[0.9975, -0.009, -0.001, 0.0001]` — identity in wxyz).
  The harness passes the 16-D action through verbatim so it round-trips into
  the action space it was recorded in; only the two quaternion blocks are
  renormalized (`|q|` drifts to 0.991–1.010 even in the demos). Disable with
  `--no-normalize-quat`.
- **Sim success is not hardware success.** Camera mount offsets in
  `visuomotor_env_cfg.py` are fitted by eye and flagged upstream as the
  biggest open gap against the real robot.
