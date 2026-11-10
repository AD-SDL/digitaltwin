# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Background ``nvidia-smi`` sampler.

``pynvml`` is not installed in either venv, so this shells out to
``nvidia-smi`` on a timer and appends CSV rows. Sampling is per-GPU, which is
what you want here: the sim renders on one card and the policy infers on
another, so a single aggregate number would hide which side is the
bottleneck.

Energy is integrated from the power samples (trapezoid over the sample
interval), so it is an estimate whose accuracy is bounded by ``hz`` -- at the
default 2 Hz it is good to a few percent over a multi-minute run, and it is
reported as an estimate rather than a measurement.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import threading
import time
from pathlib import Path

_QUERY = "index,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.sm"


class GpuSampler:
    """Polls ``nvidia-smi`` on a thread and writes one CSV row per GPU per tick."""

    def __init__(self, hz: float, out_path: Path):
        self.interval = 1.0 / hz if hz and hz > 0 else None
        self.out_path = Path(out_path)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._available = shutil.which("nvidia-smi") is not None

    def start(self) -> None:
        if self.interval is None:
            return
        if not self._available:
            print("[gpu] nvidia-smi not found; GPU trace disabled", flush=True)
            return
        self._thread = threading.Thread(target=self._run, name="gpu-sampler", daemon=True)
        self._thread.start()
        print(f"[gpu] sampling every {self.interval:.2f}s -> {self.out_path}", flush=True)

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "wall_time", "gpu_index", "util_gpu_pct", "util_mem_pct",
                "mem_used_mib", "power_w", "temp_c", "sm_clock_mhz",
            ])
            while not self._stop.is_set():
                t = time.time()
                try:
                    out = subprocess.run(
                        ["nvidia-smi", f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=5.0, check=True,
                    ).stdout
                except Exception:
                    self._stop.wait(self.interval)
                    continue

                for line in out.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 7:
                        writer.writerow([f"{t:.3f}", *parts])
                fh.flush()
                self._stop.wait(self.interval)
