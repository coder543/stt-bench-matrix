from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ..bench.samples import SampleSpec
from ..bench.perf import PerfConfig, measure_rtfx
from ..bench.types import ModelBenchmark, RunResult
from ..models.registry import ModelSpec
from ..platforms.detect import HostInfo
from .base import FrameworkInfo
from .mlx_cleanup import cleanup_mlx
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError


@dataclass(frozen=True)
class WhisperMlxFramework:
    info: FrameworkInfo = FrameworkInfo(
        name="whisper-mlx",
        description="Apple MLX Whisper implementation",
        supports_whisper=True,
        supports_parakeet=False,
        supports_canary=False,
        supports_moonshine=False,
        supports_granite=False,
    )

    def is_supported(self, host: HostInfo) -> bool:
        return host.is_macos and host.is_apple_silicon


if TYPE_CHECKING:
    import mlx_whisper


def _mlx_whisper_repo_candidates(size: str) -> list[str]:
    if size == "large-v3":
        return ["mlx-community/whisper-large-v3-mlx"]
    return [
        f"mlx-community/whisper-{size}",
        f"mlx-community/whisper-{size}-mlx",
    ]


def benchmark_whisper_models(
    sample: SampleSpec,
    models: list[ModelSpec],
    use_cache: bool,
    perf_config: PerfConfig,
    language: str,
    progress: Callable[[str], None] | None = None,
    on_result: Callable[[ModelBenchmark], None] | None = None,
) -> list[ModelBenchmark]:
    try:
        import mlx_whisper  # type: ignore[import-not-found]
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
                notes=f"mlx-whisper unavailable: {exc}",
                transcript=None,
                wer=None,
                wer_stdev=None,
                runs=[],
            )
            for model in models
        ]

    results: list[ModelBenchmark] = []

    for model in models:
        last_error: Exception | None = None
        for repo in _mlx_whisper_repo_candidates(model.size):
            try:
                local_path: str | Path | None = None
                if use_cache:
                    try:
                        local_path = snapshot_download(
                            repo_id=repo,
                            local_files_only=True,
                        )
                    except LocalEntryNotFoundError:
                        local_path = snapshot_download(
                            repo_id=repo,
                            local_files_only=False,
                        )
                    def run_once() -> None:
                        _ = mlx_whisper.transcribe(
                            str(sample.audio_path),
                            path_or_hf_repo=local_path,
                            verbose=False,
                        )

                    stats = measure_rtfx(
                        name=f"whisper-mlx:{model.size}",
                        sample=sample,
                        run_once=run_once,
                        config=perf_config,
                    )
                    results.append(
                        ModelBenchmark(
                            model_name=model.name,
                            model_size=model.size,
                            model_variant=model.variant,
                            rtfx_mean=stats.rtfx_mean,
                            rtfx_stdev=stats.rtfx_stdev,
                            bench_seconds=stats.wall_seconds,
                            device="mps",
                            notes=f"repo: {repo}",
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
                        progress(f"{framework_name()} {model.name} {model.size}")
                    last_error = None
                    break

                from tempfile import TemporaryDirectory

                with TemporaryDirectory() as tmp_dir:
                    local_path = snapshot_download(
                        repo_id=repo,
                        local_dir=tmp_dir,
                    )
                    def run_once() -> None:
                        _ = mlx_whisper.transcribe(
                            str(sample.audio_path),
                            path_or_hf_repo=local_path,
                            verbose=False,
                        )

                    stats = measure_rtfx(
                        name=f"whisper-mlx:{model.size}",
                        sample=sample,
                        run_once=run_once,
                        config=perf_config,
                    )
                    results.append(
                        ModelBenchmark(
                            model_name=model.name,
                            model_size=model.size,
                            model_variant=model.variant,
                            rtfx_mean=stats.rtfx_mean,
                            rtfx_stdev=stats.rtfx_stdev,
                            bench_seconds=stats.wall_seconds,
                            device="mps",
                            notes=f"repo: {repo}",
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
                        progress(f"{framework_name()} {model.name} {model.size}")
                    last_error = None
                    break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        if last_error is not None:
            results.append(
                ModelBenchmark(
                    model_name=model.name,
                    model_size=model.size,
                    model_variant=model.variant,
                    rtfx_mean=None,
                    rtfx_stdev=None,
                    bench_seconds=None,
                    device="mps",
                    notes=f"mlx-whisper failed: {last_error}",
                    transcript=None,
                    wer=None,
                    wer_stdev=None,
                    runs=[],
                )
            )
            if on_result is not None:
                on_result(results[-1])
            if progress is not None:
                progress(f"{framework_name()} {model.name} {model.size}")
        cleanup_mlx()

    return results


def framework_name() -> str:
    return "whisper-mlx"
