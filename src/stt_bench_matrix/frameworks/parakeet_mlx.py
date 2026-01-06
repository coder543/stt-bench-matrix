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
class ParakeetMlxFramework:
    info: FrameworkInfo = FrameworkInfo(
        name="parakeet-mlx",
        description="NVIDIA Parakeet via parakeet-mlx (MLX)",
        supports_whisper=False,
        supports_parakeet=True,
        supports_canary=False,
        supports_moonshine=False,
        supports_granite=False,
    )

    def is_supported(self, host: HostInfo) -> bool:
        return host.is_macos and host.is_apple_silicon


@dataclass(frozen=True)
class ParakeetMlxRunResult:
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
    return _project_root() / "tools" / "parakeet_runner"


def _runner_script() -> Path:
    return _runner_dir() / "run.py"


def _model_repo_candidates(spec: ModelSpec) -> list[str]:
    if spec.name == "parakeet-ctc":
        base = f"parakeet-ctc-{spec.size}"
        return [f"mlx-community/{base}"]
    if spec.name == "parakeet-rnnt":
        base = f"parakeet-rnnt-{spec.size}"
        return [f"mlx-community/{base}"]
    if spec.name == "parakeet-tdt":
        base = f"parakeet-tdt-{spec.size}"
        return [f"mlx-community/{base}"]
    if spec.name == "parakeet-tdt-ctc":
        base = f"parakeet-tdt-ctc-{spec.size}"
        underscored = f"parakeet-tdt_ctc-{spec.size}"
        return [f"mlx-community/{base}", f"mlx-community/{underscored}"]
    if spec.name == "parakeet-realtime-eou":
        base = f"parakeet-realtime-eou-{spec.size}"
        return [f"mlx-community/{base}"]
    return []


def _is_metal_malloc_error(error: str | None) -> bool:
    if not error:
        return False
    return "metal::malloc" in error or "maximum allowed buffer size" in error


def _runner_error(prefix: str, raw: str | None) -> ParakeetMlxRunResult:
    error = " ".join((raw or "").splitlines()).strip()
    if len(error) > 400:
        error = f"{error[:400]}…"
    if not error:
        error = prefix
    return ParakeetMlxRunResult(
        rtfx_mean=None,
        rtfx_stdev=None,
        wall_seconds=None,
        device=None,
        transcript=None,
        elapsed_values=[],
        transcripts=[],
        error=error,
    )


def run_parakeet_mlx_benchmark(
    *,
    model_id: str,
    sample: SampleSpec,
    perf_config: PerfConfig,
    chunk_seconds: float | None,
    overlap_seconds: float | None,
    local_attn: bool,
) -> ParakeetMlxRunResult:
    runner_script = _runner_script()
    if not runner_script.exists():
        return _runner_error("parakeet runner missing", f"{runner_script} not found")
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    env.pop("UV_NO_SYNC", None)
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
        return _runner_error(
            "parakeet runner sync failed", sync_proc.stderr or sync_proc.stdout
        )
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
    if perf_config.auto:
        cmd.append("--auto")
    if chunk_seconds is not None:
        cmd.extend(["--chunk-seconds", str(chunk_seconds)])
    if overlap_seconds is not None:
        cmd.extend(["--overlap-seconds", str(overlap_seconds)])
    if local_attn:
        cmd.append("--local-attn")
    run_env = dict(env)
    run_env["UV_NO_SYNC"] = "1"
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=run_env,
    )
    if proc.returncode != 0:
        return _runner_error("parakeet runner failed", proc.stderr or proc.stdout)
    stdout = proc.stdout.strip()
    if not stdout:
        return _runner_error("parakeet runner empty output", proc.stderr or proc.stdout)
    payload_line = stdout.splitlines()[-1]
    try:
        payload = json.loads(payload_line)
    except json.JSONDecodeError:
        return _runner_error("parakeet runner invalid JSON", payload_line)
    return ParakeetMlxRunResult(
        rtfx_mean=payload.get("rtfx_mean"),
        rtfx_stdev=payload.get("rtfx_stdev"),
        wall_seconds=payload.get("wall_seconds"),
        device=payload.get("device"),
        transcript=payload.get("transcript"),
        elapsed_values=payload.get("elapsed_values") or [],
        transcripts=payload.get("transcripts") or [],
        error=None,
    )


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _env_bool(name: str) -> bool:
    raw = os.environ.get(name, "")
    return raw.lower() in {"1", "true", "yes", "y"}


def benchmark_parakeet_models(
    sample: SampleSpec,
    models: list[ModelSpec],
    perf_config: PerfConfig,
    progress: Callable[[str], None] | None = None,
    on_result: Callable[[ModelBenchmark], None] | None = None,
) -> list[ModelBenchmark]:
    results: list[ModelBenchmark] = []
    env_chunk = _env_float("STT_BENCH_PARAKEET_MLX_CHUNK_SECONDS")
    env_overlap = _env_float("STT_BENCH_PARAKEET_MLX_OVERLAP_SECONDS")
    env_local_attn = _env_bool("STT_BENCH_PARAKEET_MLX_LOCAL_ATTN")
    default_chunk = 40.0
    default_overlap = 0.0

    for model in models:
        candidates = _model_repo_candidates(model)
        if not candidates:
            results.append(
                ModelBenchmark(
                    model_name=model.name,
                    model_size=model.size,
                    model_variant=model.variant,
                    model_id=None,
                    rtfx_mean=None,
                    rtfx_stdev=None,
                    bench_seconds=None,
                    device="mps",
                    notes="parakeet-mlx unsupported model",
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
            continue
        last_error: str | None = None
        for repo in candidates:
            note_suffix = ""
            chunk_seconds = env_chunk if env_chunk is not None else default_chunk
            overlap_seconds = env_overlap if env_overlap is not None else default_overlap
            local_attn = env_local_attn
            run_result = run_parakeet_mlx_benchmark(
                model_id=repo,
                sample=sample,
                perf_config=perf_config,
                chunk_seconds=chunk_seconds,
                overlap_seconds=overlap_seconds,
                local_attn=local_attn,
            )
            if run_result.error and _is_metal_malloc_error(run_result.error):
                fallback_chunk = 30.0 if chunk_seconds is None else chunk_seconds
                fallback_overlap = 5.0 if overlap_seconds is None else overlap_seconds
                fallback_attn = True if not local_attn else local_attn
                run_result = run_parakeet_mlx_benchmark(
                    model_id=repo,
                    sample=sample,
                    perf_config=perf_config,
                    chunk_seconds=fallback_chunk,
                    overlap_seconds=fallback_overlap,
                    local_attn=fallback_attn,
                )
                if run_result.error is None:
                    note_suffix = "; fallback: local-attn, chunked"
            if run_result.error:
                last_error = run_result.error
                continue
            results.append(
                ModelBenchmark(
                    model_name=model.name,
                    model_size=model.size,
                    model_variant=model.variant,
                    model_id=repo,
                    rtfx_mean=run_result.rtfx_mean,
                    rtfx_stdev=run_result.rtfx_stdev,
                    bench_seconds=run_result.wall_seconds,
                    device=run_result.device or "mps",
                    notes=note_suffix.lstrip("; ") if note_suffix else None,
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
                            _rtfx_values(sample.duration_seconds, run_result.elapsed_values),
                            run_result.elapsed_values,
                            run_result.transcripts,
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
        if last_error is not None:
            results.append(
                ModelBenchmark(
                    model_name=model.name,
                    model_size=model.size,
                    model_variant=model.variant,
                    model_id=None,
                    rtfx_mean=None,
                    rtfx_stdev=None,
                    bench_seconds=None,
                    device="mps",
                    notes=f"parakeet-mlx failed: {last_error}",
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

    return results


def _rtfx_values(duration_seconds: float, elapsed_values: list[float]) -> list[float]:
    return [duration_seconds / v if v > 0 else 0.0 for v in elapsed_values]


def framework_name() -> str:
    return "parakeet-mlx"
