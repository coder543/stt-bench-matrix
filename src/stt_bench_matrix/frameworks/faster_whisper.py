from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..bench.perf import PerfConfig, measure_rtfx
from ..bench.samples import SampleSpec
from ..bench.types import ModelBenchmark, RunResult
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
        supports_moonshine=False,
        supports_granite=False,
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
    language: str,
    progress: Callable[[str], None] | None = None,
    on_result: Callable[[ModelBenchmark], None] | None = None,
) -> list[ModelBenchmark]:
    try:
        import torch
        from faster_whisper import WhisperModel
    except Exception as exc:  # noqa: BLE001
        return [
            ModelBenchmark(
                model_name=model.name,
                model_size=model.size,
                model_variant=model.variant,
                rtfx_mean=None,
                rtfx_stdev=None,
                bench_seconds=None,
                device=None,
                notes=f"faster-whisper unavailable: {exc}",
                transcript=None,
                wer=None,
                wer_stdev=None,
                runs=[],
            )
            for model in models
        ]

    cuda_ok, cuda_err = cuda_is_usable()
    device = "cpu"
    compute_type = "float32"
    device_note = "cpu"
    if cuda_ok:
        device = "cuda"
        compute_type = "float16"
        device_note = "cuda"
    elif torch.backends.mps.is_available():
        device = "cpu"
        compute_type = "float32"
        device_note = "cpu"
    elif cuda_err and torch.cuda.is_available():
        device_note = "cpu"

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
            last_transcript: str | None = None

            def run_once() -> str | None:
                segments, _ = whisper.transcribe(
                    str(sample.audio_path),
                    language=language,
                    task="transcribe",
                )
                texts: list[str] = []
                for segment in segments:
                    text = getattr(segment, "text", None)
                    if text:
                        texts.append(text.strip())
                transcript = " ".join(texts).strip() or None
                return transcript

            stats = measure_rtfx(
                name=f"faster-whisper:{model.size}",
                sample=sample,
                run_once=run_once,
                config=perf_config,
            )
            last_transcript = (
                stats.transcripts[-1] if stats.transcripts else None
            )
            results.append(
                ModelBenchmark(
                    model_name=model.name,
                    model_size=model.size,
                    model_variant=model.variant,
                    rtfx_mean=stats.rtfx_mean,
                    rtfx_stdev=stats.rtfx_stdev,
                    bench_seconds=stats.wall_seconds,
                    device=device_note,
                    notes=f"model: {model_id}",
                    transcript=last_transcript,
                    wer=None,
                    wer_stdev=None,
                    runs=[
                        RunResult(
                            rtfx=rtfx,
                            seconds=elapsed,
                            wer=None,
                            transcript=transcript,
                        )
                        for rtfx, elapsed, transcript in zip(
                            stats.rtfx_values,
                            stats.elapsed_values,
                            stats.transcripts,
                        )
                    ],
                )
            )
            if on_result is not None:
                on_result(results[-1])
        except Exception as exc:  # noqa: BLE001
            note = f"faster-whisper failed: {exc}"
            if cuda_err and torch.cuda.is_available():
                note = f"{note}; cuda unavailable: {cuda_err}"
            results.append(
                ModelBenchmark(
                    model_name=model.name,
                    model_size=model.size,
                    model_variant=model.variant,
                    rtfx_mean=None,
                    rtfx_stdev=None,
                    bench_seconds=None,
                    device=device_note,
                    notes=note,
                    transcript=None,
                    wer=None,
                    wer_stdev=None,
                    runs=[],
                )
            )
            if on_result is not None:
                on_result(results[-1])
        if progress is not None:
            progress(f"faster-whisper {model.name} {model.size}")

    return results
