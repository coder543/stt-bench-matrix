from __future__ import annotations

from dataclasses import dataclass
import shutil
import subprocess
from typing import Callable

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError

from ..bench.perf import PerfConfig, measure_rtfx
from ..bench.samples import SampleSpec
from ..bench.types import ModelBenchmark
from ..models.registry import ModelSpec
from ..platforms.detect import HostInfo
from .base import FrameworkInfo


@dataclass(frozen=True)
class WhisperCppFramework:
    info: FrameworkInfo = FrameworkInfo(
        name="whisper.cpp",
        description="C/C++ whisper.cpp CLI",
        supports_whisper=True,
        supports_parakeet=False,
        supports_canary=False,
    )

    def is_supported(self, host: HostInfo) -> bool:
        return host.is_macos or host.is_linux


def _whisper_cli() -> str | None:
    return shutil.which("whisper-cli")


def has_whisper_cli() -> bool:
    return _whisper_cli() is not None


def _ggml_model_filename(size: str) -> str:
    if size == "large-v3":
        return "ggml-large-v3.bin"
    return f"ggml-{size}.bin"


def _download_model(filename: str) -> str:
    repo_id = "ggerganov/whisper.cpp"
    try:
        return hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True)
    except LocalEntryNotFoundError:
        return hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=False)


def benchmark_whisper_models(
    sample: SampleSpec,
    models: list[ModelSpec],
    perf_config: PerfConfig,
    progress: Callable[[str], None] | None = None,
) -> list[ModelBenchmark]:
    whisper_cli = _whisper_cli()
    if whisper_cli is None:
        return [
            ModelBenchmark(
                model_name=model.name,
                model_size=model.size,
                rtfx_mean=None,
                rtfx_stdev=None,
                bench_seconds=None,
                notes="whisper-cli not found in PATH",
            )
            for model in models
        ]

    results: list[ModelBenchmark] = []

    for model in models:
        filename = _ggml_model_filename(model.size)
        try:
            model_path = _download_model(filename)

            def run_once() -> None:
                subprocess.run(
                    [whisper_cli, "-m", model_path, "-f", str(sample.audio_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                )

            stats = measure_rtfx(
                name=f"whisper.cpp:{model.size}",
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
                    notes=f"model: {filename}",
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
                    notes=f"whisper.cpp failed: {exc}",
                )
            )
        if progress is not None:
            progress(f"whisper.cpp {model.name} {model.size}")

    return results
