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


DEFAULT_MODEL_ID = "mistralai/Voxtral-Mini-4B-Realtime-2602"
_RUNNER_SYNCED = False


@dataclass(frozen=True)
class VoxtralTransformersFramework:
    info: FrameworkInfo = FrameworkInfo(
        name="voxtral-transformers",
        description="Mistral Voxtral Realtime via Transformers",
        supports_whisper=False,
        supports_parakeet=False,
        supports_canary=False,
        supports_moonshine=False,
        supports_granite=False,
        supports_voxtral=True,
    )

    def is_supported(self, host: HostInfo) -> bool:
        return host.is_linux or host.is_macos


@dataclass(frozen=True)
class VoxtralRunResult:
    rtfx_mean: float | None
    rtfx_stdev: float | None
    wall_seconds: float | None
    device: str | None
    transcript: str | None
    elapsed_values: list[float]
    transcripts: list[str | None]
    notes: str | None
    error: str | None


def _model_id(model: ModelSpec) -> str:
    if model.name != "voxtral":
        raise ValueError("voxtral-transformers only supports voxtral models")
    return os.environ.get("STT_BENCH_VOXTRAL_MODEL_ID", DEFAULT_MODEL_ID)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _runner_dir() -> Path:
    return _project_root() / "tools" / "voxtral_runner"


def _runner_script() -> Path:
    return _runner_dir() / "run.py"


def _int_env(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _runner_error(prefix: str, raw: str | None) -> VoxtralRunResult:
    error = " ".join((raw or "").splitlines()).strip()
    if len(error) > 400:
        error = f"{error[:400]}…"
    if not error:
        error = prefix
    return VoxtralRunResult(
        rtfx_mean=None,
        rtfx_stdev=None,
        wall_seconds=None,
        device=None,
        transcript=None,
        elapsed_values=[],
        transcripts=[],
        notes=None,
        error=error,
    )


def _ensure_runner_synced(env: dict[str, str]) -> str | None:
    global _RUNNER_SYNCED
    if _RUNNER_SYNCED:
        return None

    sync_cmd = [
        "uv",
        "sync",
        "--project",
        str(_runner_dir()),
    ]
    sync_proc = subprocess.run(
        sync_cmd,
        capture_output=True,
        text=True,
        env=env,
    )
    if sync_proc.returncode != 0:
        return sync_proc.stderr or sync_proc.stdout

    _RUNNER_SYNCED = True
    return None


def run_voxtral_benchmark(
    *,
    model_id: str,
    sample: SampleSpec,
    perf_config: PerfConfig,
    warmup_sample: SampleSpec | None,
    max_new_tokens: int | None,
) -> VoxtralRunResult:
    runner_script = _runner_script()
    if not runner_script.exists():
        return _runner_error("voxtral runner missing", f"{runner_script} not found")
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    env.pop("UV_NO_SYNC", None)
    sync_error = _ensure_runner_synced(env)
    if sync_error is not None:
        return _runner_error("voxtral runner sync failed", sync_error)
    cmd = [
        "uv",
        "run",
        "--project",
        str(_runner_dir()),
        "python",
        str(runner_script),
        "--model-id",
        model_id,
        "--audio-path",
        str(sample.audio_path),
        "--sample-seconds",
        str(sample.duration_seconds),
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
    if warmup_sample is not None:
        cmd.extend(["--warmup-audio-path", str(warmup_sample.audio_path)])
    if perf_config.auto:
        cmd.append("--auto")
    if max_new_tokens is not None:
        cmd.extend(["--max-new-tokens", str(max_new_tokens)])
    run_env = dict(env)
    run_env["UV_NO_SYNC"] = "1"
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=run_env,
    )
    if proc.returncode != 0:
        return _runner_error("voxtral runner failed", proc.stderr or proc.stdout)
    stdout = proc.stdout.strip()
    if not stdout:
        return _runner_error("voxtral runner empty output", proc.stderr or proc.stdout)
    payload_line = stdout.splitlines()[-1]
    try:
        payload = json.loads(payload_line)
    except json.JSONDecodeError:
        return _runner_error("voxtral runner invalid JSON", payload_line)
    return VoxtralRunResult(
        rtfx_mean=payload.get("rtfx_mean"),
        rtfx_stdev=payload.get("rtfx_stdev"),
        wall_seconds=payload.get("wall_seconds"),
        device=payload.get("device"),
        transcript=payload.get("transcript"),
        elapsed_values=payload.get("elapsed_values") or [],
        transcripts=payload.get("transcripts") or [],
        notes=payload.get("notes"),
        error=None,
    )


def benchmark_voxtral_models(
    sample: SampleSpec,
    models: list[ModelSpec],
    perf_config: PerfConfig,
    warmup_sample: SampleSpec | None = None,
    progress: Callable[[str], None] | None = None,
    on_result: Callable[[ModelBenchmark], None] | None = None,
) -> list[ModelBenchmark]:
    if not models:
        return []
    max_new_tokens = _int_env("STT_BENCH_VOXTRAL_MAX_NEW_TOKENS")

    results: list[ModelBenchmark] = []

    for model in models:
        model_id = _model_id(model)
        run_result = run_voxtral_benchmark(
            model_id=model_id,
            sample=sample,
            perf_config=perf_config,
            warmup_sample=warmup_sample,
            max_new_tokens=max_new_tokens,
        )
        if run_result.error is not None:
            results.append(
                ModelBenchmark(
                    model_name=model.name,
                    model_size=model.size,
                    model_variant=model.variant,
                    model_id=model_id,
                    rtfx_mean=None,
                    rtfx_stdev=None,
                    bench_seconds=None,
                    device=run_result.device,
                    notes=f"voxtral failed: {run_result.error}",
                    transcript=None,
                    wer=None,
                    wer_stdev=None,
                    runs=[],
                )
            )
        else:
            results.append(
                ModelBenchmark(
                    model_name=model.name,
                    model_size=model.size,
                    model_variant=model.variant,
                    model_id=model_id,
                    rtfx_mean=run_result.rtfx_mean,
                    rtfx_stdev=run_result.rtfx_stdev,
                    bench_seconds=run_result.wall_seconds,
                    device=run_result.device,
                    notes=run_result.notes,
                    transcript=run_result.transcript,
                    wer=None,
                    wer_stdev=None,
                    runs=[
                        RunResult(
                            rtfx=(sample.duration_seconds / elapsed) if elapsed else None,
                            seconds=elapsed,
                            wer=None,
                            transcript=transcript,
                        )
                        for elapsed, transcript in zip(
                            run_result.elapsed_values,
                            run_result.transcripts,
                        )
                    ],
                )
            )
        if on_result is not None:
            on_result(results[-1])
        if progress is not None:
            progress(f"voxtral {model.name} {model.size}")

    return results
