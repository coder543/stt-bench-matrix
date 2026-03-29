from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
from typing import Callable

from ..bench.perf import PerfConfig
from ..bench.samples import SampleSpec
from ..bench.types import ModelBenchmark, RunResult
from ..models.registry import ModelSpec
from ..platforms.detect import HostInfo
from .base import FrameworkInfo


@dataclass(frozen=True)
class AppleSpeechFramework:
    info: FrameworkInfo = FrameworkInfo(
        name="speech-analyzer",
        description="Apple SpeechAnalyzer with SpeechTranscriber",
        supports_whisper=False,
        supports_parakeet=False,
        supports_canary=False,
        supports_moonshine=False,
        supports_granite=False,
        supports_apple_speech=True,
    )

    def is_supported(self, host: HostInfo) -> bool:
        return host.is_macos


@dataclass(frozen=True)
class AppleSpeechRunResult:
    rtfx_mean: float | None
    rtfx_stdev: float | None
    wall_seconds: float | None
    device: str | None
    transcript: str | None
    elapsed_values: list[float]
    transcripts: list[str | None]
    error: str | None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _runner_dir() -> Path:
    return _project_root() / "tools" / "apple_speech_runner"


def _runner_source() -> Path:
    return _runner_dir() / "run.swift"


def _runner_binary() -> Path:
    return _runner_dir() / ".build" / "apple_speech_runner"


def _runner_error(prefix: str, raw: str | None) -> AppleSpeechRunResult:
    error = " ".join((raw or "").splitlines()).strip()
    if len(error) > 400:
        error = f"{error[:400]}..."
    if not error:
        error = prefix
    return AppleSpeechRunResult(
        rtfx_mean=None,
        rtfx_stdev=None,
        wall_seconds=None,
        device=None,
        transcript=None,
        elapsed_values=[],
        transcripts=[],
        error=error,
    )


def _ensure_runner_binary() -> str | None:
    source = _runner_source()
    binary = _runner_binary()
    if not source.exists():
        return f"apple speech runner missing at {source}"
    binary.parent.mkdir(parents=True, exist_ok=True)
    if binary.exists() and binary.stat().st_mtime >= source.stat().st_mtime:
        return None
    proc = subprocess.run(
        [
            "xcrun",
            "swiftc",
            "-parse-as-library",
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return proc.stderr.strip() or proc.stdout.strip() or "swiftc failed"
    return None


def _model_id(spec: ModelSpec) -> str | None:
    if spec.name == "speech-transcriber":
        return "apple/speech-transcriber"
    return None


def run_apple_speech_benchmark(
    *,
    model_id: str,
    sample: SampleSpec,
    perf_config: PerfConfig,
    warmup_sample: SampleSpec | None,
    language: str,
) -> AppleSpeechRunResult:
    compile_error = _ensure_runner_binary()
    if compile_error is not None:
        return _runner_error("apple speech runner compile failed", compile_error)
    binary = _runner_binary()
    env = dict(os.environ)
    cmd = [
        str(binary),
        "--model-id",
        model_id,
        "--audio-path",
        str(sample.audio_path),
        "--sample-seconds",
        str(sample.duration_seconds),
        "--locale",
        language,
        "--warmups",
        str(perf_config.warmups),
        "--runs",
        str(perf_config.runs),
        "--auto-min-runs",
        str(perf_config.auto_min_runs),
        "--auto-max-runs",
        str(perf_config.auto_max_runs),
        "--auto-target-cv",
        str(perf_config.auto_target_cv),
    ]
    if perf_config.auto:
        cmd.append("--auto")
    if warmup_sample is not None:
        cmd.extend(["--warmup-audio-path", str(warmup_sample.audio_path)])
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        detail = proc.stderr or proc.stdout
        if not detail:
            detail = f"exit code {proc.returncode}"
        return _runner_error("apple speech runner failed", detail)
    stdout = proc.stdout.strip()
    if not stdout:
        return _runner_error(
            "apple speech runner empty output", proc.stderr or proc.stdout
        )
    payload_line = stdout.splitlines()[-1]
    try:
        payload = json.loads(payload_line)
    except json.JSONDecodeError:
        return _runner_error("apple speech runner invalid JSON", payload_line)
    return AppleSpeechRunResult(
        rtfx_mean=payload.get("rtfx_mean"),
        rtfx_stdev=payload.get("rtfx_stdev"),
        wall_seconds=payload.get("wall_seconds"),
        device=payload.get("device"),
        transcript=payload.get("transcript"),
        elapsed_values=payload.get("elapsed_values") or [],
        transcripts=payload.get("transcripts") or [],
        error=None,
    )


def benchmark_apple_speech_models(
    sample: SampleSpec,
    models: list[ModelSpec],
    perf_config: PerfConfig,
    language: str,
    warmup_sample: SampleSpec | None = None,
    progress: Callable[[str], None] | None = None,
    on_result: Callable[[ModelBenchmark], None] | None = None,
) -> list[ModelBenchmark]:
    results: list[ModelBenchmark] = []
    for model in models:
        model_id = _model_id(model)
        if model_id is None:
            result = ModelBenchmark(
                model_name=model.name,
                model_size=model.size,
                model_variant=model.variant,
                model_id=None,
                rtfx_mean=None,
                rtfx_stdev=None,
                bench_seconds=None,
                device=None,
                notes="speech-analyzer only supports speech-transcriber",
                transcript=None,
                wer=None,
                wer_stdev=None,
                runs=[],
            )
            results.append(result)
            if on_result is not None:
                on_result(result)
            if progress is not None:
                progress(f"{framework_name()} {model.name} {model.size}")
            continue
        run_result = run_apple_speech_benchmark(
            model_id=model_id,
            sample=sample,
            perf_config=perf_config,
            warmup_sample=warmup_sample,
            language=language,
        )
        if run_result.error is not None:
            result = ModelBenchmark(
                model_name=model.name,
                model_size=model.size,
                model_variant=model.variant,
                model_id=model_id,
                rtfx_mean=None,
                rtfx_stdev=None,
                bench_seconds=None,
                device=run_result.device or "system",
                notes=f"speech-analyzer failed: {run_result.error}",
                transcript=None,
                wer=None,
                wer_stdev=None,
                runs=[],
            )
        else:
            rtfx_values = [
                sample.duration_seconds / elapsed if elapsed > 0 else 0.0
                for elapsed in run_result.elapsed_values
            ]
            result = ModelBenchmark(
                model_name=model.name,
                model_size=model.size,
                model_variant=model.variant,
                model_id=model_id,
                rtfx_mean=run_result.rtfx_mean,
                rtfx_stdev=run_result.rtfx_stdev,
                bench_seconds=run_result.wall_seconds,
                device=run_result.device or "system",
                notes=f"locale: {language}",
                transcript=run_result.transcript,
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
                        rtfx_values,
                        run_result.elapsed_values,
                        run_result.transcripts,
                    )
                ],
            )
        results.append(result)
        if on_result is not None:
            on_result(result)
        if progress is not None:
            progress(f"{framework_name()} {model.name} {model.size}")
    return results


def framework_name() -> str:
    return "speech-analyzer"
