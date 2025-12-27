from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..platforms.detect import HostInfo


@dataclass(frozen=True)
class RunResult:
    rtfx: float | None
    seconds: float | None
    wer: float | None
    transcript: str | None


@dataclass(frozen=True)
class ModelBenchmark:
    model_name: str
    model_size: str
    model_variant: str | None
    rtfx_mean: float | None
    rtfx_stdev: float | None
    bench_seconds: float | None
    device: str | None
    notes: str | None
    transcript: str | None
    wer: float | None
    wer_stdev: float | None
    runs: Sequence[RunResult]


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
    sample_name: str
    sample_path: str
    language: str
