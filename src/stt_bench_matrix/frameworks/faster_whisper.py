from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..bench.perf import PerfConfig, measure_rtfx
from ..bench.samples import SampleSpec
from ..bench.types import ModelBenchmark
from ..models.registry import ModelSpec
from ..platforms.detect import HostInfo
from ..platforms.cuda import cuda_is_usable
from .base import FrameworkInfo


@dataclass(frozen=True)
class FasterWhisperFramework:
    info: FrameworkInfo = FrameworkInfo(
        name="faster-whisper",
        description="faster-whisper (CTranslate2) Whisper",
        supports_whisper=True,
        supports_parakeet=False,
        supports_canary=False,
    )

    def is_supported(self, host: HostInfo) -> bool:
        return host.is_macos or host.is_linux


def _model_id(size: str) -> str:
    if size == "large-v3":
        return "large-v3"
    return size


def benchmark_whisper_models(
    sample: SampleSpec,
    models: list[ModelSpec],
    perf_config: PerfConfig,
    progress: Callable[[str], None] | None = None,
) -> list[ModelBenchmark]:
    try:
        import torch
        from faster_whisper import WhisperModel
    except Exception as exc:  # noqa: BLE001
        return [
            ModelBenchmark(
                model_name=model.name,
                model_size=model.size,
                rtfx_mean=None,
                rtfx_stdev=None,
                bench_seconds=None,
                notes=f"faster-whisper unavailable: {exc}",
            )
            for model in models
        ]

    cuda_ok, cuda_err = cuda_is_usable()
    device = "cpu"
    compute_type = "float32"
    device_note = "device: cpu"
    if cuda_ok:
        device = "cuda"
        compute_type = "float16"
        device_note = "device: cuda"
    elif torch.backends.mps.is_available():
        device = "cpu"
        compute_type = "float32"
        device_note = "device: cpu (mps unsupported)"
    elif cuda_err and torch.cuda.is_available():
        device_note = f"device: cpu (cuda unavailable: {cuda_err})"

    results: list[ModelBenchmark] = []

    for model in models:
        model_id = _model_id(model.size)
        try:
            try:
                whisper = WhisperModel(
                    model_id,
                    device=device,
                    compute_type=compute_type,
                )
            except Exception:  # noqa: BLE001
                whisper = WhisperModel(
                    model_id,
                    device="cpu",
                    compute_type="float32",
                )
                device_note = "device: cpu (cuda failed)"

            def run_once() -> None:
                segments, _ = whisper.transcribe(
                    str(sample.audio_path),
                    language="en",
                    task="transcribe",
                )
                for _ in segments:
                    pass

            stats = measure_rtfx(
                name=f"faster-whisper:{model.size}",
                sample=sample,
                run_once=run_once,
                config=perf_config,
            )
            results.append(
                ModelBenchmark(
                    model_name=model.name,
                    model_size=model.size,
                    rtfx_mean=stats.rtfx_mean,
                    rtfx_stdev=stats.rtfx_stdev,
                    bench_seconds=stats.wall_seconds,
                    notes=f"model: {model_id}; {device_note}",
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                ModelBenchmark(
                    model_name=model.name,
                    model_size=model.size,
                    rtfx_mean=None,
                    rtfx_stdev=None,
                    bench_seconds=None,
                    notes=f"faster-whisper failed: {exc}; {device_note}",
                )
            )
        if progress is not None:
            progress(f"faster-whisper {model.name} {model.size}")

    return results
