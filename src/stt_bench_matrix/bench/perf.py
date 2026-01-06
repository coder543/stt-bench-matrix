from __future__ import annotations

from dataclasses import dataclass
import statistics
import time
from typing import Callable
import sys

import pyperf

from .samples import SampleSpec
from .progress import RunProgress


@dataclass(frozen=True)
class PerfConfig:
    warmups: int
    runs: int
    auto: bool = True
    auto_min_runs: int = 5
    auto_max_runs: int = 30
    auto_target_cv: float = 0.05


@dataclass(frozen=True)
class PerfStats:
    rtfx_values: list[float]
    rtfx_mean: float
    rtfx_stdev: float
    wall_seconds: float
    elapsed_values: list[float]
    transcripts: list[str | None]


def measure_rtfx(
    *,
    name: str,
    sample: SampleSpec,
    run_once: Callable[[], str | None],
    warmup_run_once: Callable[[], str | None] | None = None,
    config: PerfConfig,
    progress_label: str | None = None,
) -> PerfStats:
    start_wall = time.perf_counter()
    if config.warmups:
        sys.stderr.write(f"warmup: {name} ({config.warmups} run(s))\n")
        sys.stderr.flush()
    for _ in range(config.warmups):
        if warmup_run_once is not None:
            warmup_run_once()
        else:
            run_once()

    run_label = progress_label or name
    expected_runs = config.auto_min_runs if config.auto else config.runs
    expected_runs = max(1, expected_runs)
    run_progress = None
    if config.auto or config.runs > 1:
        run_progress = RunProgress(label=run_label, total=expected_runs)

    elapsed_values: list[float] = []
    transcripts: list[str | None] = []
    if config.auto:
        target_cv = max(0.0, config.auto_target_cv)
        min_runs = max(1, config.auto_min_runs)
        max_runs = max(min_runs, config.auto_max_runs)
        while len(elapsed_values) < max_runs:
            start = time.perf_counter()
            transcript = run_once()
            elapsed = time.perf_counter() - start
            elapsed_values.append(elapsed)
            transcripts.append(transcript)
            current_runs = len(elapsed_values)
            display_target = min_runs
            if current_runs > display_target:
                display_target = current_runs
            if run_progress is not None:
                run_progress.update(current_runs, display_target)
            if len(elapsed_values) < min_runs:
                continue
            mean = statistics.fmean(elapsed_values)
            if mean <= 0:
                continue
            stdev = (
                statistics.stdev(elapsed_values)
                if len(elapsed_values) >= 2
                else 0.0
            )
            cv = stdev / mean if mean else 0.0
            if cv <= target_cv:
                break
    else:
        for _ in range(config.runs):
            start = time.perf_counter()
            transcript = run_once()
            elapsed = time.perf_counter() - start
            elapsed_values.append(elapsed)
            transcripts.append(transcript)
            current_runs = len(elapsed_values)
            display_target = max(1, config.runs)
            if run_progress is not None:
                run_progress.update(current_runs, display_target)
    if run_progress is not None:
        run_progress.finish()

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
        elapsed_values=elapsed_values,
        transcripts=transcripts,
    )
