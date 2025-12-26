from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..platforms.detect import HostInfo


@dataclass(frozen=True)
class ModelBenchmark:
    model_name: str
    model_size: str
    rtfx_mean: float | None
    rtfx_stdev: float | None
    bench_seconds: float | None
    notes: str | None


@dataclass(frozen=True)
class FrameworkBenchmark:
    framework: str
    supported: bool
    reason: str | None
    models: Sequence[ModelBenchmark]


@dataclass(frozen=True)
class BenchmarkResults:
    host: HostInfo
    frameworks: Sequence[FrameworkBenchmark]
    total_seconds: float
