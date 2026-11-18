# pi05 rollout results

Rollout evaluations of pi05 checkpoints on
`Isaac-OpenDoor-Bimanual-OpenArm-IK-Abs-Visuomotor-v0`. Method and
reproduction: [`docs/07_evaluate_policy_rollouts.md`](../docs/07_evaluate_policy_rollouts.md).

| checkpoint | training data / sim chest camera | episodes | success @ 80 s | success @ 100 s |
| --- | --- | ---: | --- | --- |
| **254000** | fixed camera (ZED Mini, aimed at the doors) | 50 | 14/50 = 28% [17%, 42%] | **16/50 = 32% [21%, 46%]** |
| 242000 | old camera (aimed at the ceiling) | 40 (2 runs pooled) | 12/40 = 30% [18%, 45%] | 6/20 = 30% (h160 run only) |

Brackets are 95% Wilson intervals.

Handle-turn sub-task (254000 only, for comparison with real-robot policies
trained on handle-only data): **lever turned ≥ 45° in 41/50 = 82% [69%, 90%]
within 20 s, and 46/50 = 92% [81%, 97%] within 100 s.** See
[Handle turn](#handle-turn-sub-task) below.

## Checkpoint 254000 — trained on the camera-fixed dataset

The training data was re-recorded with the corrected chest camera (dataset
`argonne-rpl/openarm-cabinet-vr-dt`, re-uploaded 2026-11-12). The old
camera config put a wxyz quaternion into IsaacLab's xyzw `OffsetCfg.rot`,
so it pointed at the ceiling with the image rolled 90 degrees. The fix aims
the camera 10 degrees down at the doors and matches the ZED Mini field of
view (102 deg H). See
[`camera_check.png`](../tasks/cabinet/tasks/src/rpl_cabinet/docs/camera_check.png):
panel 1 is the old view, panel 3 is the new one.

Runs, both 50 episodes, 100 s horizon, `SEED=10000`, task-default cabinet
jitter:

- [`artifacts/pi05_254000_lever_e50_h100/`](artifacts/pi05_254000_lever_e50_h100/report.txt):
  the reference run. It logs lever angles and has **video for all 50
  episodes** (`videos/episode_000.mp4` … `episode_049.mp4`; chest + both
  wrist cameras tiled, 1280x960). 141 min wall clock.
- [`artifacts/pi05_254000_camfix_e50_h100/`](artifacts/pi05_254000_camfix_e50_h100/report.md):
  the first run. No lever logging, video for episodes 0–2 only.

The two runs gave **identical** door results episode by episode (same 16
successes, same horizon curve), because the env and the policy noise are both
seeded per episode. Every door number below holds for either run.

### Headline

**32% success (16/50, 95% CI [21%, 46%]) at 100 s.** Truncated to the 80 s
horizon used for 242000: **28% (14/50)**.

**The camera fix did not measurably change the success rate.** At 80 s,
28% vs 30% is a 2-point difference, far inside the overlapping intervals
(±12 pp each). This is the narrowest interval measured so far for this task,
so the result is a fairly firm "no change", not just too few episodes to tell.

What this does and does not compare: each checkpoint was evaluated with the
camera it was trained on. The 242000 runs (2026-11-09) predate the camera
fix commit (2026-11-10), so they most likely used the ceiling camera. So
this compares the whole pipeline (data plus sim) before and after the fix,
not two policies on the same scene. One reading consistent with the result:
the policy leans on the wrist cameras and joint state rather than the chest
view. That would explain why 242000 reached ~30% with a chest camera that
never saw the handles, and why fixing the view added nothing. This has not
been tested. An ablation that blanks the chest image would test it.

### Outcomes

The peak door angle splits the 50 episodes into the same three groups seen
for 242000, with empty bands between them:

| outcome | n/50 | peak door angle |
| --- | ---: | --- |
| never pulled | 13 | 0–17° |
| engaged, then stalled | 21 | 35–70° |
| success | 16 | 81–87° |

- **"Never pulled" is not "never reached the handle".** The lever logging
  shows 9 of these 13 episodes turned the lever ≥ 45° and then failed to pull
  the door. Only 4 (ep 25, 32, 46, 47) never really turned it. For 242000
  this group was called "never engaged", before lever angles were logged.
- **Stalling is now the main failure** (21/34 failures). The 242000 seeded
  run had the reverse split: 8 never pulled, 6 stalled.
- Stalled episodes still get the door to 35–70° and then stop. This is the
  same mid-pull plateau seen for 242000, and it looks like the bigger target
  for the next round of fixes.
- `grasp_handle` and peak angle disagree on 8/50 episodes (ep 1 and ep 37
  trip `grasp_handle` below 17°, while ep 6, 11, 19, 35 and 45 pull the door
  to 35–51° without it). Ep 34 succeeds (85.9°) without `grasp_handle` ever
  firing. As before, neither signal is enough on its own.

### Horizon

```
cap        20s   40s   60s   80s  100s
success   0/50  4/50 12/50 14/50 16/50
```

- Time to success: p50 50 s, p90 84 s, close to the demonstrations' 51 s
  median.
- Two episodes succeeded only after 80 s (ep 36 at 96.6 s, ep 34 at 99.2 s).
  The 100 s horizon earned those 2 successes.
- Failures are still plateaus: the median failure stopped improving with 68%
  of the episode left (68 s idle at the end). Only 2/34 were still improving
  in the final 10 s, so going past 100 s would add little.

### Handle turn (sub-task)

Purpose: compare pi05 against an ACT policy trained on real-robot episodes
(`openarm_open_lab_door`) that only turn the handle. ACT never learned to
open the door, so its full-task success rate can't be compared with pi05's.
This scores pi05 on the handle turn alone.

**Metric.** Success = either lever handle (`latch_left` / `latch_right`
joint) reaches |angle| ≥ threshold within the time cap. The run records the
first step each threshold is crossed, so every cap below is exact, from one
run.

```
lever >= | <=10s  <=20s  <=30s  <=40s  <=60s  <=100s | t_turn p50 / p90
   15°   |  28     48     48     48     48     48    |  8.6 / 13.1 s
   30°   |  25     45     46     46     46     47    |  9.7 / 14.5 s
   45°   |  14     41     45     45     45     46    | 12.1 / 19.6 s
   60°   |   5     39     44     45     45     46    | 13.8 / 22.4 s
                                       (successes out of 50)
```

Headline at 45° (half the lever's 90° throw):

| cap | handle turned |
| --- | --- |
| 10 s | 14/50 = 28% [17%, 42%] |
| 20 s | 41/50 = 82% [69%, 90%] |
| 30 s | 45/50 = 90% [79%, 96%] |
| 100 s | 46/50 = 92% [81%, 97%] |

- **Turning the handle is mostly solved. Pulling the door is the bottleneck.**
  92% turn the lever, but only 32% then open the door.
- **The turns are real, not the arm brushing the lever.** Turns are gradual
  (ep 3: 3° to 90° over ~6 s) and held (median 24–45 s above 45°, by outcome
  group). In all 42 episodes where the door moved, the lever reached 45°
  first, a median 8.9 s before the door reached 5°. Every one of the 16 door
  successes turned the lever first.
- **Only the right lever is ever turned** (50/50). The sim task is
  right-armed. The real `openarm_open_lab_door` episodes use the left arm on
  the left door, so the two policies don't even turn the same handle.
- **The cap decides the number.** The jump from 28% at 10 s to 82% at 20 s
  is the whole story. The real demos finish the handle motion within ~9 s
  (see the task README), and pi05's median 45° turn comes at 12 s. **Pick
  the column that matches the time limit of the real-robot ACT trials.**

**Caveats before comparing with ACT:**

1. **The sim lever has no latch and no spring** (stiffness 0, damping 0.05).
   Once pushed, it swings to the 90° stop on its own, which is why the
   threshold barely matters (median peak = 90° exactly). The real handle has
   to be held against a return spring. Read the sim number as "the policy
   went for the handle and started turning it, and how fast", not as proof
   it would beat a real latch. A torsion spring on `latch_*` would close this
   gap, but it changes the task the training data was collected on.
2. **Sim vs real.** pi05 is scored in sim and ACT will be scored on hardware,
   so a gap between the two numbers mixes policy quality with sim-to-real.
3. **Not the same arm or door** (see above).

Lever logging was added to `eval_rollouts.py` and reported by
`summarize_eval.py` (HANDLE TURN section of `report.txt`). `report.md` does
not include it yet.

### Cost and latency

Unchanged from 242000 (same architecture):

| | |
| --- | --- |
| realtime factor | 0.53x (160 s wall per episode) |
| `env.step` | 79 ms mean, the bottleneck |
| policy, cached-chunk step | 9.4 ms p50 |
| policy, replan step | 172 ms p50, once per 50 steps |
| policy VRAM | 9.7 GiB peak |
| energy, 50 episodes @ 100 s | ~460 Wh across both GPUs |

### Next steps

1. Chest-camera ablation: rerun 254000 with the chest image blanked. If the
   success rate holds near 30%, the chest view is not being used.
2. Look into the stalled group (21 episodes, door stuck at 35–70°). All 50
   episodes are on video in `artifacts/pi05_254000_lever_e50_h100/videos/`.
   Compare `episode_000.mp4` (stalled at 53°) with `episode_002.mp4`
   (success), and `episode_007.mp4` (lever turned to 90°, door never pulled:
   0.7°).
3. Add a return spring to the sim lever and rerun the handle-turn metric, if
   the ACT comparison needs the turn itself to be physically meaningful.
4. Checkpoint sweep on this run, which still has the same open question as
   242000 (checkpoint picked on training loss).

---

## Checkpoint 242000 — old camera

This checkpoint was selected from the first 20-epoch cabinet training run,
before the camera fix. The analysis below was written before 254000 existed.

Per-run detail with every episode: [`artifacts/pi05_242000_e20ep/report.md`](artifacts/pi05_242000_e20ep/report.md),
[`artifacts/pi05_242000_h160/report.md`](artifacts/pi05_242000_h160/report.md).

### Headline

**~30% success**, pooling both runs truncated at a common 1600-step horizon:
**12/40, 95% Wilson CI [18%, 45%]**.

| run | horizon | success | 95% CI | wall |
| --- | --- | --- | --- | --- |
| `pi05_242000_e20ep` | 80 s / 1600 steps | 7/20 = 35% | [18%, 57%] | 43.8 min |
| `pi05_242000_h160` | 160 s / 3200 steps | 6/20 = 30% | [15%, 52%] | 84.1 min |
| **pooled @ 1600 steps** | **80 s** | **12/40 = 30%** | **[18%, 45%]** | — |

The two runs are not paired: policy-noise seeding was added between them, so
episode *i* is a different rollout in each. They are two independent samples
of the same policy on the same scene distribution.

### The three outcomes

Across the 160 s run, peak door angle separates cleanly into three groups with
empty bands between them (success needs ≥ 80.2° held 1.0 s):

| outcome | n/20 | peak door angle |
| --- | ---: | --- |
| never engaged | 8 | 0–12° |
| engaged, then stalled | 6 | 24–60° |
| success | 6 | 82–87° |

Two distinct failures hide inside one success rate:

- **Never engaged** — the arm does not get the gripper onto the handle. A
  reaching/perception failure. The door barely moves at all.
- **Engaged, then stalled** — the gripper hooks the lever, pulls the door to
  24–60°, and stops. A mid-pull failure.

These likely need different fixes, so they are worth tracking separately
rather than as one number.

### More time does not help

The second run doubled the horizon specifically to test whether failures were
slow successes. They are not.

```
cap        40s   60s   80s  100s  120s  140s  160s
success   3/20  4/20  5/20  6/20  6/20  6/20  6/20
                             ^ knee; flat for the last 60 s
```

One episode (#17, step 1800 / 90 s) was genuinely rescued by the extra time.
It cost 38 minutes of extra wall clock to find, and nothing after step 2000
produced anything at all.

Failures are plateaus, not slow progress:

- 14/14 failures ran to the cap, but the median one stopped improving with
  **84% of the episode still to run**, then sat motionless for **135 s**.
- Only **1/14** was still improving in its final 10%.
- Episode 0 held exactly 48.4° for 123 consecutive seconds.

The one exception is worth naming: **episode 7** peaked at 59.9° on step 3043
of 3200 — still creeping upward when the 160 s cap hit, with only 8 s idle. It
is the single episode in 14 that a longer horizon might have converted, and at
~5 min of compute per episode it is not worth chasing.

**Recommended horizon: 100 s.** Below ~70 s the metric is invalid (the
demonstrations themselves need 51 s median, 69 s max). Above ~100 s you pay a
full episode of compute to watch a parked arm.

### Noise draw alone changes the failure mode

The two runs differ only in the flow-matching noise draw, and that alone moved
the failure distribution:

| | never-engaged failures |
| --- | --- |
| run 1 (free-running noise) | 0/13 |
| run 2 (seeded noise) | 8/14 |

Same checkpoint, same scene seeds, same horizon for the first 1600 steps. The
overall success rates are statistically indistinguishable (35% vs 30%, heavily
overlapping intervals), but *how* the policy fails is not stable across noise.

That is a fragility result in its own right: whether the arm ever finds the
handle can be decided by the sampled noise. It also means a 20-episode success
rate is sensitive to something you would not want it sensitive to — another
reason to quote the pooled 40-episode figure.

### The two progress signals disagree on edge cases

Peak door angle and the task's own `grasp_handle` term do not always agree:

- **ep 6**: door reached 48.2°, but `grasp_handle` never fired — the EE was
  never within 12 cm of the hub *while* the door was open past 10°. Consistent
  with shoving the door rather than pulling the lever.
- **ep 18**: `grasp_handle` fired, but the door only reached 12.3° — the
  signal tripped at its 10° gate and nothing followed.

Neither signal is sufficient alone. `grasp_handle` is a conjunction (near the
hub AND door past 10°), so it is better read as "engaged and pulling" than as
"grasped", and it can miss a genuine open or flag a stalled one.

### Cost and latency

| | |
| --- | --- |
| realtime factor | 0.50x |
| `env.step` (3× 480p cameras) | 76 ms mean — **the bottleneck** |
| policy, cached-chunk step | 9.4 ms p50 |
| policy, replan step | 172 ms p50, once per 50 steps |
| policy VRAM | 9.7 GiB peak |
| energy, 20 episodes @ 160 s | ~286 Wh across both GPUs |

The control period is 50 ms at 20 Hz. Cached steps fit; **replan steps do
not** — 172 ms stalls the loop for ~3.4 periods. Harmless in sim, a real-time
violation on hardware, and the reason to look at async inference, a shorter
chunk, or fewer denoise steps before any hardware deployment.

Inference is ~10% of wall clock. Episode count, not model speed, is the lever
on evaluation runtime.

### What these numbers do not cover

- **Narrow distribution.** Only cabinet pose is randomized (±2 cm, ±0.05 rad)
  and the robot starts from an identical joint state every episode. This is
  not a robustness measurement; `--cabinet-jitter` widens it.
- **One checkpoint.** 242000 was picked on smoothed *training* loss, from a
  run with `eval_split: 0.0`. Whether it is actually the best of the ~127
  saved checkpoints is untested — that needs the sweep in
  [`docs/07_evaluate_policy_rollouts.md`](../docs/07_evaluate_policy_rollouts.md#e-comparing-checkpoints).
- **Sim only.** Camera mount offsets in `visuomotor_env_cfg.py` are fitted by
  eye and flagged upstream as the largest open gap against the real robot.

### Suggested next steps

1. Watch `artifacts/pi05_242000_h160/videos/episode_004.mp4` (never engaged)
   against `episode_009.mp4` (clean success) — the two failure modes are
   visually distinct and should point at which to attack.
2. 50 episodes at 100 s for a quotable rate (±13 pp instead of ±19 pp).
3. Checkpoint sweep, fixed seeds, to test whether training loss ranked them
   correctly.
