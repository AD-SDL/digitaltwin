# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render a run directory into a readable Markdown report with per-episode detail.

Complements ``summarize_eval.py`` (terminal/JSON) with something you can commit,
diff and hand to someone who was not watching the run.

Stdlib only.

Usage::

    python3 evaluation/scripts/make_report_md.py <out_dir> [-o report.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import summarize_eval as se

# Peak door angle below this means the arm never meaningfully engaged the door.
# Sits in the empty band between the two observed failure modes (0-12 deg vs
# 24-60 deg), so the split is not sensitive to the exact cut.
ENGAGE_FLOOR_DEG = 14.0
SPARK = "▁▂▃▄▅▆▇█"


def sparkline(trace: list[float], lo: float = 0.0, hi: float = 95.0) -> str:
    if not trace:
        return ""
    out = []
    for v in trace:
        f = (v - lo) / (hi - lo) if hi > lo else 0.0
        out.append(SPARK[max(0, min(len(SPARK) - 1, int(f * (len(SPARK) - 1) + 0.5)))])
    return "".join(out)


def classify(e: dict) -> str:
    if e["success"]:
        return "success"
    return "never engaged" if e["max_door_angle_deg"] < ENGAGE_FLOOR_DEG else "engaged, stalled"


def main() -> int:
    ap = argparse.ArgumentParser(description="Markdown report for a rollout eval run.")
    ap.add_argument("out_dir")
    ap.add_argument("-o", "--output", default=None, help="Default: <out_dir>/report.md")
    a = ap.parse_args()

    out_dir = Path(a.out_dir).expanduser()
    manifest = json.loads((out_dir / "manifest.json").read_text())
    eps = [json.loads(l) for l in (out_dir / "episodes.jsonl").read_text().splitlines() if l.strip()]
    n = len(eps)
    succ = [e for e in eps if e["success"]]
    lo, hi = se.wilson(len(succ), n)
    step_dt = 1.0 / manifest["control_hz"]

    L: list[str] = []
    w = L.append

    ckpt = (manifest.get("server") or {}).get("checkpoint", "unknown")
    w(f"# Rollout evaluation — {out_dir.name}")
    w("")
    w(f"- **Checkpoint**: `{ckpt}`")
    w(f"- **Task**: `{manifest['task']}`")
    w(f"- **Episodes**: {n} @ {manifest['episode_length_s']:.0f} s horizon "
      f"({manifest['max_steps']} steps, {manifest['control_hz']:.0f} Hz)")
    w(f"- **Base seed**: {manifest['base_seed']} (episode *i* uses seed + *i*)")
    w(f"- **Scene randomization**: {manifest['cabinet_jitter']}")
    w(f"- **Prompt**: \"{manifest['task_prompt']}\"")
    w(f"- **Wall clock**: {manifest.get('total_wall_s', 0) / 60:.1f} min")
    w("")
    w("## Headline")
    w("")
    w(f"**{100 * len(succ) / n:.1f}% success ({len(succ)}/{n})**, 95% Wilson CI "
      f"[{100 * lo:.1f}%, {100 * hi:.1f}%].")
    w("")

    # -- horizon sensitivity
    curve = se._horizon_curve(eps, manifest)
    if curve:
        w("## Would a different horizon change this?")
        w("")
        w("Exact, not a projection — `success_step` is recorded, so truncating at")
        w("any cap is just counting.")
        w("")
        w("| cap (steps) | cap (s) | successes | rate |")
        w("| ---: | ---: | ---: | ---: |")
        for row in curve:
            w(f"| {row['steps']} | {row['seconds']:.0f} | {row['successes']}/{n} | {row['rate'] * 100:.1f}% |")
        w("")

    # -- stall
    st = se._stall_summary(eps)
    if st:
        w("## Were the failures still making progress?")
        w("")
        w(f"- Failures: **{st['failures']}**")
        w(f"- Episode fraction elapsed *after* the door stopped improving: "
          f"p50 **{st['stalled_fraction_p50'] * 100:.0f}%**, p90 {st['stalled_fraction_p90'] * 100:.0f}%")
        w(f"- Motionless time at the end: p50 **{st['idle_s_p50']:.0f} s**, min {st['idle_s_min']:.0f} s")
        w(f"- Still improving in the final 10% of the episode: "
          f"**{st['still_improving_last_10pct']}/{st['failures']}** — only these could be rescued by more time")
        w("")

    # -- failure taxonomy
    fails = [e for e in eps if not e["success"]]
    never = [e for e in fails if e["max_door_angle_deg"] < ENGAGE_FLOOR_DEG]
    stalled = [e for e in fails if e["max_door_angle_deg"] >= ENGAGE_FLOOR_DEG]
    w("## Failure modes")
    w("")
    w("| mode | n | peak door angle |")
    w("| --- | ---: | --- |")
    if never:
        w(f"| never engaged | {len(never)} | "
          f"{min(e['max_door_angle_deg'] for e in never):.0f}–{max(e['max_door_angle_deg'] for e in never):.0f}° |")
    if stalled:
        w(f"| engaged, then stalled | {len(stalled)} | "
          f"{min(e['max_door_angle_deg'] for e in stalled):.0f}–{max(e['max_door_angle_deg'] for e in stalled):.0f}° |")
    if succ:
        w(f"| success | {len(succ)} | "
          f"{min(e['max_door_angle_deg'] for e in succ):.0f}–{max(e['max_door_angle_deg'] for e in succ):.0f}° |")
    w("")
    w(f"A reaching failure and a mid-pull failure are different problems; a single")
    w("success rate hides both. The split above is on peak door angle")
    w(f"(< {ENGAGE_FLOOR_DEG:.0f}° = never engaged), chosen to sit in the empty band between the")
    w("two observed modes.")
    w("")

    # The task's own grasp signal and the angle split can disagree; say so
    # rather than letting the tidy table imply they never do.
    disagree = [e for e in fails
                if (e["max_door_angle_deg"] < ENGAGE_FLOOR_DEG) != (not e["grasped_handle"])]
    if disagree:
        w(f"**{len(disagree)} episode(s) where the angle split and the `grasp_handle` signal "
          "disagree**, worth knowing before leaning on either alone:")
        w("")
        for e in disagree:
            w(f"- ep {e['episode']}: peak {e['max_door_angle_deg']:.1f}°, "
              f"engaged={'yes' if e['grasped_handle'] else 'no'} — "
              + ("door moved well past the floor but the EE was never within 12 cm of the hub "
                 "while it did (pushed rather than pulled?)" if e["grasped_handle"] is False
                 else "signal fired at the 10° gate but the door barely moved after"))
        w("")

    # Episodes that were still improving when the horizon hit.
    improving = [e for e in fails
                 if e.get("stalled_fraction") is not None and e["stalled_fraction"] < 0.10]
    if improving:
        w("**Still improving at the cap** (the only failures a longer horizon could plausibly rescue):")
        w("")
        for e in improving:
            idle = (e["steps"] - e["max_door_angle_step"]) * step_dt
            w(f"- ep {e['episode']}: peak {e['max_door_angle_deg']:.1f}° at step "
              f"{e['max_door_angle_step']} of {e['steps']}, only {idle:.0f} s idle")
        w("")

    # -- per episode
    w("## Per-episode")
    w("")
    w(f"`trace` is the peak-door-angle sparkline at "
      f"{eps[0].get('door_trace_hz') or 1:.0f} Hz, scaled 0–95°. A success climbs; a stall is flat.")
    w("")
    w("| ep | seed | outcome | peak door | peak @ step | idle after peak | engaged | steps |")
    w("| ---: | ---: | --- | ---: | ---: | ---: | :-: | ---: |")
    for e in eps:
        kind = classify(e)
        idle = (e["steps"] - e.get("max_door_angle_step", 0)) * step_dt
        outcome = (f"**success** @ {e['success_step']} ({e['success_sim_s']:.0f} s)"
                   if e["success"] else kind)
        w(f"| {e['episode']} | {e['seed']} | {outcome} | {e['max_door_angle_deg']:.1f}° | "
          f"{e.get('max_door_angle_step', 0)} | {idle:.0f} s | "
          f"{'yes' if e['grasped_handle'] else 'no'} | {e['steps']} |")
    w("")

    if any(e.get("door_angle_trace_deg") for e in eps):
        w("### Door-angle traces")
        w("")
        w("```text")
        for e in eps:
            tr = e.get("door_angle_trace_deg") or []
            tag = "OK  " if e["success"] else "    "
            w(f"{tag}ep{e['episode']:>2}  {e['max_door_angle_deg']:5.1f}°  {sparkline(tr)}")
        w("```")
        w("")

    # -- performance
    r_lat = {k: se._agg(eps, [v]) for k, v in
             (("rtt_all", "rtt_ms"), ("rtt_replan", "rtt_replan_ms"),
              ("rtt_cached", "rtt_cached_ms"), ("forward", "forward_ms"),
              ("gpu", "gpu_ms"), ("env_step", "env_step_ms"))}
    w("## Performance")
    w("")
    w(f"- Realtime factor: **{sum(e['realtime_factor'] for e in eps if e['realtime_factor']) / n:.2f}x**")
    w(f"- Wall per episode: {sum(e['wall_s'] for e in eps) / n:.0f} s")
    w("")
    w("| stage | n | mean (ms) | p50 | p90 | max |")
    w("| --- | ---: | ---: | ---: | ---: | ---: |")
    for label, key in (("round trip, all steps", "rtt_all"),
                       ("round trip, replan steps", "rtt_replan"),
                       ("round trip, cached steps", "rtt_cached"),
                       ("policy forward", "forward"),
                       ("GPU device time", "gpu"),
                       ("env.step (sim + render)", "env_step")):
        d = r_lat.get(key)
        if d:
            w(f"| {label} | {d['n']} | {d['mean']:.2f} | {d['p50_of_ep_p50']:.2f} | "
              f"{d['p90_of_ep_p90']:.2f} | {d['max']:.2f} |")
    w("")
    gpu = se._gpu_summary(out_dir / "gpu_trace.csv")
    if gpu:
        w("| GPU | util mean | mem peak | power mean | energy (est) |")
        w("| --- | ---: | ---: | ---: | ---: |")
        for name, g in gpu.items():
            w(f"| {name} | {g['util_mean_pct']:.0f}% | {g['mem_peak_mib'] / 1024:.1f} GiB | "
              f"{g['power_mean_w']:.0f} W | {g['energy_wh_est']:.0f} Wh |")
        w("")

    vids = sorted((out_dir / "videos").glob("*.mp4")) if (out_dir / "videos").exists() else []
    if vids:
        w(f"## Videos ({len(vids)})")
        w("")
        w("`videos/episode_NNN.mp4` — chest + both wrist cameras tiled, 1280x960.")
        w("")
        interesting = []
        if succ:
            fastest = min(succ, key=lambda e: e["success_step"])
            interesting.append((fastest["episode"], f"fastest success, step {fastest['success_step']}"))
            latest = max(succ, key=lambda e: e["success_step"])
            if latest["episode"] != fastest["episode"]:
                interesting.append((latest["episode"], f"slowest success, step {latest['success_step']}"))
        if stalled:
            worst = max(stalled, key=lambda e: e["max_door_angle_deg"])
            interesting.append((worst["episode"], f"stalled at {worst['max_door_angle_deg']:.0f}°, closest near-miss"))
        if never:
            interesting.append((never[0]["episode"], "never engaged the door"))
        for ep_ix, why in interesting:
            w(f"- `episode_{ep_ix:03d}.mp4` — {why}")
        w("")

    text = "\n".join(L)
    target = Path(a.output) if a.output else out_dir / "report.md"
    target.write_text(text)
    print(f"wrote {target} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
