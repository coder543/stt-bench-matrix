from __future__ import annotations

from dataclasses import dataclass
import statistics
import time
from typing import Callable
import sys

import pyperf

from .samples import SampleSpec


@dataclass(frozen=True)
class PerfConfig:
    warmups: int
    runs: int


@dataclass(frozen=True)
class PerfStats:
    rtfx_values: list[float]
    rtfx_mean: float
    rtfx_stdev: float
    wall_seconds: float


def measure_rtfx(
    *,
    name: str,
    sample: SampleSpec,
    run_once: Callable[[], None],
    config: PerfConfig,
) -> PerfStats:
    start_wall = time.perf_counter()
    if config.warmups:
        sys.stderr.write(f"warmup: {name} ({config.warmups} run(s))\n")
        sys.stderr.flush()
    for _ in range(config.warmups):
        run_once()

    elapsed_values: list[float] = []
    for _ in range(config.runs):
        start = time.perf_counter()
        run_once()
        elapsed = time.perf_counter() - start
        elapsed_values.append(elapsed)

    # Store elapsed seconds with a valid pyperf unit to satisfy metadata rules.
    _ = pyperf.Run(
        elapsed_values,
        metadata={"name": name, "unit": "second"},
        collect_metadata=False,
    )

    rtfx_values = [sample.duration_seconds / v for v in elapsed_values]
    rtfx_mean = statistics.fmean(rtfx_values)
    rtfx_stdev = statistics.stdev(rtfx_values) if len(rtfx_values) >= 2 else 0.0
    wall_seconds = time.perf_counter() - start_wall

    return PerfStats(
        rtfx_values=rtfx_values,
        rtfx_mean=rtfx_mean,
        rtfx_stdev=rtfx_stdev,
        wall_seconds=wall_seconds,
    )
