from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from ..bench.transcripts import extract_transcript

from .base import FrameworkInfo
from .mlx_cleanup import cleanup_mlx
from ..platforms.detect import HostInfo
from ..bench.samples import SampleSpec
from ..bench.types import ModelBenchmark, RunResult
from ..bench.perf import PerfConfig, measure_rtfx
from ..models.registry import ModelSpec


@dataclass(frozen=True)
class LightningWhisperMlxFramework:
    info: FrameworkInfo = FrameworkInfo(
        name="lightning-whisper-mlx",
        description="Optimized Whisper on MLX",
        supports_whisper=True,
        supports_parakeet=False,
        supports_canary=False,
        supports_moonshine=False,
        supports_granite=False,
    )

    def is_supported(self, host: HostInfo) -> bool:
        return host.is_macos and host.is_apple_silicon


if TYPE_CHECKING:
    from lightning_whisper_mlx import LightningWhisperMLX


def _model_name_for_size(size: str) -> str:
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
        from lightning_whisper_mlx import LightningWhisperMLX  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        return [
            ModelBenchmark(
                model_name=model.name,
                model_size=model.size,
                model_variant=model.variant,
                model_id=None,
                rtfx_mean=None,
                rtfx_stdev=None,
                bench_seconds=None,
                device=None,
                notes=f"lightning-whisper-mlx unavailable: {exc}",
                transcript=None,
                wer=None,
                wer_stdev=None,
                runs=[],
            )
            for model in models
        ]

    results: list[ModelBenchmark] = []

    for model in models:
        model_name = _model_name_for_size(model.size)
        try:
            runner = LightningWhisperMLX(model=model_name)

            def run_once() -> str | None:
                result = runner.transcribe(str(sample.audio_path))
                return extract_transcript(result)

            stats = measure_rtfx(
                name=f"lightning-whisper-mlx:{model.size}",
                sample=sample,
                run_once=run_once,
                config=perf_config,
            )
            results.append(
                ModelBenchmark(
                    model_name=model.name,
                    model_size=model.size,
                    model_variant=model.variant,
                    model_id=model_name,
                    rtfx_mean=stats.rtfx_mean,
                    rtfx_stdev=stats.rtfx_stdev,
                    bench_seconds=stats.wall_seconds,
                    device="mps",
                    notes=None,
                    transcript=None,
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
            if progress is not None:
                progress(f"lightning-whisper-mlx {model.name} {model.size}")
        except Exception as exc:  # noqa: BLE001
            results.append(
                ModelBenchmark(
                    model_name=model.name,
                    model_size=model.size,
                    model_variant=model.variant,
                    model_id=model_name,
                    rtfx_mean=None,
                    rtfx_stdev=None,
                    bench_seconds=None,
                    device="mps",
                    notes=f"lightning-whisper-mlx failed: {exc}",
                    transcript=None,
                    wer=None,
                    wer_stdev=None,
                    runs=[],
                )
            )
            if on_result is not None:
                on_result(results[-1])
            if progress is not None:
                progress(f"lightning-whisper-mlx {model.name} {model.size}")
        finally:
            try:
                del runner
            except UnboundLocalError:
                pass
            cleanup_mlx()

    return results
