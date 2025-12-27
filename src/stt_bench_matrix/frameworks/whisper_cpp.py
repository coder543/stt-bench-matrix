from __future__ import annotations

from dataclasses import dataclass
import os
import re
import shutil
import statistics
import subprocess
import time
import tempfile
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
    language: str,
    progress: Callable[[str], None] | None = None,
) -> list[ModelBenchmark]:
    whisper_cli = _whisper_cli()
    if whisper_cli is None:
        return [
            ModelBenchmark(
                model_name=model.name,
                model_size=model.size,
                model_variant=model.variant,
                rtfx_mean=None,
                rtfx_stdev=None,
                bench_seconds=None,
                device=None,
                notes="whisper-cli not found in PATH",
                transcript=None,
                wer=None,
            )
            for model in models
        ]

    results: list[ModelBenchmark] = []

    timing_re = re.compile(r"whisper_print_timings:.*total time\s*=\s*([0-9.]+)\s*ms")
    gpu_re = re.compile(r"use gpu\s*=\s*(\d)")

    for model in models:
        filename = _ggml_model_filename(model.size)
        try:
            model_path = _download_model(filename)
            device_note = "cpu"
            note_suffix = ""
            timings_missing = False
            last_transcript: str | None = None
            force_cpu = os.environ.get("STT_BENCH_WHISPER_CPP_FORCE_CPU", "") in {
                "1",
                "true",
                "yes",
                "y",
            }

            def _run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                )

            def run_once(out_base: str) -> float:
                nonlocal device_note, note_suffix, timings_missing, last_transcript
                base_cmd = [
                    whisper_cli,
                    "-m",
                    model_path,
                    "-f",
                    str(sample.audio_path),
                    "-l",
                    language,
                    "-of",
                    out_base,
                    "-otxt",
                ]
                cmd = base_cmd + (["-ng"] if force_cpu else [])
                wall_start = time.perf_counter()
                result = _run_cmd(cmd)
                wall_elapsed = time.perf_counter() - wall_start
                output = (result.stdout or "") + (result.stderr or "")
                if "unknown argument" in output or "usage:" in output:
                    result = subprocess.CompletedProcess(result.args, 2, output, "")
                if result.returncode != 0:
                    wall_start = time.perf_counter()
                    result_cpu = _run_cmd(base_cmd + ["-ng"])
                    wall_elapsed = time.perf_counter() - wall_start
                    output = (result_cpu.stdout or "") + (result_cpu.stderr or "")
                    if "unknown argument" in output or "usage:" in output:
                        result_cpu = subprocess.CompletedProcess(
                            result_cpu.args, 2, output, ""
                        )
                    if result_cpu.returncode != 0:
                        raise subprocess.CalledProcessError(
                            result_cpu.returncode, result_cpu.args, output, None
                        )
                    device_note = "cpu"
                    note_suffix = f"; gpu failed: exit {result.returncode}"
                    match = timing_re.search(output)
                    if match:
                        return float(match.group(1)) / 1000.0
                    if not timings_missing:
                        note_suffix = f"{note_suffix}; timings missing"
                        timings_missing = True
                    return wall_elapsed
                gpu_match = gpu_re.search(output)
                if gpu_match:
                    device_note = "cuda" if gpu_match.group(1) == "1" else "cpu"
                txt_path = f"{out_base}.txt"
                if os.path.exists(txt_path):
                    try:
                        last_transcript = (
                            open(txt_path, encoding="utf-8").read().strip() or None
                        )
                    except Exception:
                        last_transcript = last_transcript
                match = timing_re.search(output)
                if match:
                    return float(match.group(1)) / 1000.0
                if not timings_missing:
                    note_suffix = f"{note_suffix}; timings missing"
                    timings_missing = True
                return wall_elapsed

            elapsed_values: list[float] = []
            with tempfile.TemporaryDirectory() as temp_dir:
                out_base = os.path.join(temp_dir, "whisper_output")
                for _ in range(perf_config.warmups):
                    _ = run_once(out_base)
                for _ in range(perf_config.runs):
                    elapsed_values.append(run_once(out_base))

            rtfx_values = [sample.duration_seconds / v for v in elapsed_values]
            rtfx_mean = statistics.fmean(rtfx_values)
            rtfx_stdev = (
                statistics.stdev(rtfx_values) if len(rtfx_values) >= 2 else 0.0
            )
            wall_seconds = sum(elapsed_values)
            results.append(
                ModelBenchmark(
                    model_name=model.name,
                    model_size=model.size,
                    model_variant=model.variant,
                    rtfx_mean=rtfx_mean,
                    rtfx_stdev=rtfx_stdev,
                    bench_seconds=wall_seconds,
                    device=device_note,
                    notes=f"model: {filename}{note_suffix}",
                    transcript=last_transcript,
                    wer=None,
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                ModelBenchmark(
                    model_name=model.name,
                    model_size=model.size,
                    model_variant=model.variant,
                    rtfx_mean=None,
                    rtfx_stdev=None,
                    bench_seconds=None,
                    device="cpu",
                    notes=f"whisper.cpp failed: {exc}",
                    transcript=None,
                    wer=None,
                )
            )
        if progress is not None:
            progress(f"whisper.cpp {model.name} {model.size}")

    return results
