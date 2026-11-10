# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Turn an ``eval_rollouts.py`` output directory into a report.

Stdlib only, so it runs under any of the three interpreters on this box
(system python3, the IsaacLab venv, or the lerobot venv).

Usage::

    python3 evaluation/scripts/summarize_eval.py <out_dir> [--json]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval.

    Preferred over the normal approximation because rollout counts are small
    (tens of episodes) and the interval stays inside [0, 1] even at 0% or
    100% success -- which is exactly where a first eval tends to land.
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _pct(values, q):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)


def _agg(episodes, path):
    """Pool a per-episode latency stat across episodes, weighted by sample count."""
    total_n = 0
    weighted = 0.0
    maxima = []
    p50s, p90s, p99s = [], [], []
    for ep in episodes:
        node = ep["latency"]
        for key in path:
            node = node.get(key, {}) if isinstance(node, dict) else {}
        if not isinstance(node, dict) or not node.get("n"):
            continue
        total_n += node["n"]
        weighted += node["mean"] * node["n"]
        maxima.append(node["max"])
        p50s.append(node["p50"])
        p90s.append(node["p90"])
        p99s.append(node["p99"])
    if total_n == 0:
        return None
    return {
        "n": total_n,
        "mean": weighted / total_n,
        "p50_of_ep_p50": _pct(p50s, 0.5),
        "p90_of_ep_p90": _pct(p90s, 0.5),
        "p99_of_ep_p99": _pct(p99s, 0.5),
        "max": max(maxima),
    }


def _horizon_curve(episodes, manifest):
    """Success rate the run would have reported at shorter horizons.

    Exact, not a projection: `success_step` is recorded, so truncating at H is
    just counting episodes whose dwell completed at or before H. Answers "would
    a longer horizon have helped?" without a second run.
    """
    n = len(episodes)
    max_steps = int(manifest.get("max_steps") or 0)
    if not max_steps:
        return None
    step_dt = 1.0 / manifest["control_hz"]

    marks = []
    h = 400
    while h < max_steps:
        marks.append(h)
        h += 400
    marks.append(max_steps)

    out = []
    for h in marks:
        k = sum(1 for e in episodes if e["success"] and 0 < e["success_step"] <= h)
        out.append({"steps": h, "seconds": h * step_dt, "successes": k, "rate": k / n})
    return out


def _stall_summary(episodes):
    """How long failures sat without improving the door angle before time ran out."""
    fails = [e for e in episodes if not e["success"] and e.get("stalled_fraction") is not None]
    if not fails:
        return None
    fr = sorted(e["stalled_fraction"] for e in fails)
    idle_s = sorted(
        (e["steps"] - e["max_door_angle_step"]) * (e["sim_s"] / e["steps"]) for e in fails
    )
    return {
        "failures": len(fails),
        "stalled_fraction_p50": _pct(fr, 0.5),
        "stalled_fraction_p90": _pct(fr, 0.9),
        "idle_s_p50": _pct(idle_s, 0.5),
        "idle_s_min": idle_s[0],
        "still_improving_last_10pct": sum(1 for x in fr if x < 0.10),
    }


def _gpu_summary(path: Path):
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return None
    by_gpu: dict[str, list[dict]] = {}
    for r in rows:
        by_gpu.setdefault(r["gpu_index"], []).append(r)

    out = {}
    for idx, samples in sorted(by_gpu.items()):
        def col(name):
            vals = []
            for s in samples:
                try:
                    vals.append(float(s[name]))
                except (ValueError, KeyError):
                    pass
            return vals

        util, mem, power = col("util_gpu_pct"), col("mem_used_mib"), col("power_w")
        times = col("wall_time")
        duration = (max(times) - min(times)) if len(times) > 1 else 0.0
        energy_wh = (sum(power) / len(power) * duration / 3600.0) if power and duration else None
        out[f"gpu{idx}"] = {
            "samples": len(samples),
            "duration_s": duration,
            "util_mean_pct": sum(util) / len(util) if util else None,
            "util_p90_pct": _pct(util, 0.9),
            "util_max_pct": max(util) if util else None,
            "mem_peak_mib": max(mem) if mem else None,
            "power_mean_w": sum(power) / len(power) if power else None,
            "power_max_w": max(power) if power else None,
            "energy_wh_est": energy_wh,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize a rollout eval directory.")
    ap.add_argument("out_dir")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    a = ap.parse_args()

    out_dir = Path(a.out_dir).expanduser()
    manifest = json.loads((out_dir / "manifest.json").read_text())
    episodes = [json.loads(line) for line in (out_dir / "episodes.jsonl").read_text().splitlines() if line.strip()]
    if not episodes:
        print("no episodes found")
        return 1

    n = len(episodes)
    succ = [e for e in episodes if e["success"]]
    lo, hi = wilson(len(succ), n)

    door = sorted(e["max_door_angle_deg"] for e in episodes)
    grasp_n = sum(1 for e in episodes if e.get("grasped_handle"))

    report = {
        "checkpoint": manifest.get("server", {}).get("checkpoint") or manifest.get("server"),
        "task": manifest["task"],
        "episodes": n,
        "horizon_s": manifest["episode_length_s"],
        "control_hz": manifest["control_hz"],
        "success": {
            "count": len(succ),
            "rate": len(succ) / n,
            "wilson95": [lo, hi],
        },
        "progress": {
            "grasped_handle_rate": grasp_n / n,
            "max_door_angle_deg_p50": _pct(door, 0.5),
            "max_door_angle_deg_p90": _pct(door, 0.9),
            "max_door_angle_deg_best": door[-1],
            "success_threshold_deg": 80.2,
        },
        "horizon_sensitivity": _horizon_curve(episodes, manifest),
        "stall": _stall_summary(episodes),
        "time": {
            "total_wall_min": manifest.get("total_wall_s", 0) / 60,
            "wall_s_per_episode_mean": sum(e["wall_s"] for e in episodes) / n,
            "steps_mean": sum(e["steps"] for e in episodes) / n,
            "realtime_factor_mean": (lambda v: sum(v) / len(v) if v else None)(
                [e["realtime_factor"] for e in episodes if e["realtime_factor"]]),
            "time_to_success_s_p50": _pct([e["success_sim_s"] for e in succ], 0.5) if succ else None,
            "time_to_success_s_p90": _pct([e["success_sim_s"] for e in succ], 0.9) if succ else None,
        },
        "latency_ms": {
            "rtt_all": _agg(episodes, ["rtt_ms"]),
            "rtt_replan": _agg(episodes, ["rtt_replan_ms"]),
            "rtt_cached": _agg(episodes, ["rtt_cached_ms"]),
            "server_total": _agg(episodes, ["server_ms"]),
            "policy_forward": _agg(episodes, ["forward_ms"]),
            "preprocess": _agg(episodes, ["pre_ms"]),
            "postprocess": _agg(episodes, ["post_ms"]),
            "gpu_device_time": _agg(episodes, ["gpu_ms"]),
            "env_step": _agg(episodes, ["env_step_ms"]),
            "obs_pack": _agg(episodes, ["obs_pack_ms"]),
        },
        "gpu": _gpu_summary(out_dir / "gpu_trace.csv"),
    }

    if a.json:
        print(json.dumps(report, indent=2))
        return 0

    r = report
    w = print
    w("=" * 68)
    w(f"  {r['task']}")
    w(f"  {r['episodes']} episodes @ {r['horizon_s']:.0f}s horizon, {r['control_hz']:.0f} Hz control")
    w("=" * 68)
    s = r["success"]
    w(f"\nSUCCESS RATE   {s['rate'] * 100:5.1f}%   ({s['count']}/{r['episodes']})"
      f"   95% CI [{s['wilson95'][0] * 100:.1f}%, {s['wilson95'][1] * 100:.1f}%]")

    p = r["progress"]
    w(f"\nPARTIAL PROGRESS (success needs door >= {p['success_threshold_deg']:.0f} deg held 1.0 s)")
    w(f"  engaged handle      {p['grasped_handle_rate'] * 100:5.1f}% of episodes"
      "   (EE within 12 cm of hub WHILE that door is past 10 deg)")
    w(f"  max door angle      p50 {p['max_door_angle_deg_p50']:5.1f} deg   "
      f"p90 {p['max_door_angle_deg_p90']:5.1f} deg   best {p['max_door_angle_deg_best']:5.1f} deg")

    hc = r.get("horizon_sensitivity")
    if hc:
        w("\nHORIZON SENSITIVITY (exact: would a shorter/longer cap change the result?)")
        for row in hc:
            bar = "#" * int(round(row["rate"] * 40))
            w(f"  <= {row['steps']:5d} steps ({row['seconds']:5.0f}s) : "
              f"{row['successes']:2d}/{r['episodes']:<2d} = {row['rate'] * 100:5.1f}%  {bar}")

    st = r.get("stall")
    if st:
        w(f"\nFAILURE STALL (n={st['failures']}; how much of the episode ran after the door stopped improving)")
        w(f"  stalled fraction    p50 {st['stalled_fraction_p50'] * 100:.0f}%   p90 {st['stalled_fraction_p90'] * 100:.0f}%")
        w(f"  idle time at end    p50 {st['idle_s_p50']:.0f} s   min {st['idle_s_min']:.0f} s")
        w(f"  still improving in the final 10% of the episode: "
          f"{st['still_improving_last_10pct']}/{st['failures']}"
          "   <- only these could plausibly be rescued by a longer horizon")

    t = r["time"]
    w("\nTIME")
    w(f"  total               {t['total_wall_min']:.1f} min")
    w(f"  per episode         {t['wall_s_per_episode_mean']:.1f} s wall, "
      f"{t['steps_mean']:.0f} steps, realtime factor {t['realtime_factor_mean']:.2f}x")
    if t["time_to_success_s_p50"] is not None:
        w(f"  time to success     p50 {t['time_to_success_s_p50']:.1f} s   p90 {t['time_to_success_s_p90']:.1f} s (sim time)")

    w("\nLATENCY (ms)                    n        mean      p50      p90      p99      max")
    for label, key in [
        ("round trip (all steps)", "rtt_all"),
        ("  replan steps", "rtt_replan"),
        ("  cached-chunk steps", "rtt_cached"),
        ("server total", "server_total"),
        ("  policy forward", "policy_forward"),
        ("  preprocess", "preprocess"),
        ("  postprocess", "postprocess"),
        ("GPU device time", "gpu_device_time"),
        ("env.step (sim+render)", "env_step"),
        ("obs pack (sim side)", "obs_pack"),
    ]:
        d = r["latency_ms"].get(key)
        if not d:
            continue
        w(f"  {label:<28} {d['n']:>7} {d['mean']:>9.2f} {d['p50_of_ep_p50']:>8.2f} "
          f"{d['p90_of_ep_p90']:>8.2f} {d['p99_of_ep_p99']:>8.2f} {d['max']:>8.2f}")

    if r["gpu"]:
        w("\nGPU")
        for name, g in r["gpu"].items():
            w(f"  {name}: util mean {g['util_mean_pct']:.0f}% / p90 {g['util_p90_pct']:.0f}% / max {g['util_max_pct']:.0f}%"
              f" | mem peak {g['mem_peak_mib'] / 1024:.1f} GiB"
              f" | power mean {g['power_mean_w']:.0f} W"
              + (f" | energy ~{g['energy_wh_est']:.1f} Wh" if g["energy_wh_est"] else ""))
    w("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
