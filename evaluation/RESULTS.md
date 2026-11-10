# pi05 checkpoint 242000 — rollout results

Evaluation of the pi05 checkpoint selected from the 20-epoch cabinet training
run, on `Isaac-OpenDoor-Bimanual-OpenArm-IK-Abs-Visuomotor-v0`.

Method and reproduction: [`docs/07_evaluate_policy_rollouts.md`](../docs/07_evaluate_policy_rollouts.md).
Per-run detail with every episode: [`artifacts/pi05_242000_e20ep/report.md`](artifacts/pi05_242000_e20ep/report.md),
[`artifacts/pi05_242000_h160/report.md`](artifacts/pi05_242000_h160/report.md).

## Headline

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

## The three outcomes

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

## More time does not help

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

## Noise draw alone changes the failure mode

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

## The two progress signals disagree on edge cases

Peak door angle and the task's own `grasp_handle` term do not always agree:

- **ep 6**: door reached 48.2°, but `grasp_handle` never fired — the EE was
  never within 12 cm of the hub *while* the door was open past 10°. Consistent
  with shoving the door rather than pulling the lever.
- **ep 18**: `grasp_handle` fired, but the door only reached 12.3° — the
  signal tripped at its 10° gate and nothing followed.

Neither signal is sufficient alone. `grasp_handle` is a conjunction (near the
hub AND door past 10°), so it is better read as "engaged and pulling" than as
"grasped", and it can miss a genuine open or flag a stalled one.

## Cost and latency

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

## What these numbers do not cover

- **Narrow distribution.** Only cabinet pose is randomized (±2 cm, ±0.05 rad)
  and the robot starts from an identical joint state every episode. This is
  not a robustness measurement; `--cabinet-jitter` widens it.
- **One checkpoint.** 242000 was picked on smoothed *training* loss, from a
  run with `eval_split: 0.0`. Whether it is actually the best of the ~127
  saved checkpoints is untested — that needs the sweep in
  [`docs/07_evaluate_policy_rollouts.md`](../docs/07_evaluate_policy_rollouts.md#e-comparing-checkpoints).
- **Sim only.** Camera mount offsets in `visuomotor_env_cfg.py` are fitted by
  eye and flagged upstream as the largest open gap against the real robot.

## Suggested next steps

1. Watch `artifacts/pi05_242000_h160/videos/episode_004.mp4` (never engaged)
   against `episode_009.mp4` (clean success) — the two failure modes are
   visually distinct and should point at which to attack.
2. 50 episodes at 100 s for a quotable rate (±13 pp instead of ±19 pp).
3. Checkpoint sweep, fixed seeds, to test whether training loss ranked them
   correctly.
