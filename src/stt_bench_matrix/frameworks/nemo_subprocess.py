from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import os
from typing import Sequence

from ..bench.perf import PerfConfig
from ..bench.types import ModelBenchmark, RunResult
from ..bench.samples import SampleSpec
from ..models.registry import ModelSpec


@dataclass(frozen=True)
class NemoRunResult:
    rtfx_mean: float | None
    rtfx_stdev: float | None
    wall_seconds: float | None
    device: str | None
    decode: str | None
    transcript: str | None
    elapsed_values: list[float]
    transcripts: list[str | None]
    error: str | None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _runner_dir() -> Path:
    return _project_root() / "tools" / "nemo_runner"


def _runner_script() -> Path:
    return _runner_dir() / "run.py"


def run_nemo_benchmark(
    *,
    task: str,
    model_id: str,
    model_type: str | None,
    decode_mode: str | None,
    sample: SampleSpec,
    perf_config: PerfConfig,
    chunk_seconds: float = 40.0,
) -> NemoRunResult:
    runner_script = _runner_script()
    if not runner_script.exists():
        return NemoRunResult(
            rtfx_mean=None,
            rtfx_stdev=None,
            wall_seconds=None,
            device=None,
            decode=None,
            transcript=None,
            elapsed_values=[],
            transcripts=[],
            error=f"nemo runner missing at {runner_script}",
        )
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    chunk_override = env.get("STT_BENCH_NEMO_CHUNK_SECONDS")
    chunk_seconds = float(chunk_override) if chunk_override else chunk_seconds
    cmd = [
        "uv",
        "run",
        "--project",
        str(_runner_dir()),
        "python",
        str(runner_script),
        "--task",
        task,
        "--model-id",
        model_id,
        "--audio-path",
        str(sample.audio_path),
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
        "--chunk-seconds",
        str(chunk_seconds),
    ]
    if perf_config.auto:
        cmd.append("--auto")
    if model_type is not None:
        cmd.extend(["--model-type", model_type])
    if decode_mode is not None:
        cmd.extend(["--decode-mode", decode_mode])
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        raw_error = proc.stderr.strip() or proc.stdout.strip() or "nemo runner failed"
        error = " ".join(raw_error.splitlines())
        if len(error) > 400:
            error = f"{error[:400]}…"
        return NemoRunResult(
            rtfx_mean=None,
            rtfx_stdev=None,
            wall_seconds=None,
            device=None,
            decode=None,
            transcript=None,
            elapsed_values=[],
            transcripts=[],
            error=error,
        )
    stdout = proc.stdout.strip()
    if not stdout:
        return NemoRunResult(
            rtfx_mean=None,
            rtfx_stdev=None,
            wall_seconds=None,
            device=None,
            decode=None,
            transcript=None,
            elapsed_values=[],
            transcripts=[],
            error="nemo runner empty output",
        )
    payload_line = stdout.splitlines()[-1]
    try:
        payload = json.loads(payload_line)
    except json.JSONDecodeError:
        return NemoRunResult(
            rtfx_mean=None,
            rtfx_stdev=None,
            wall_seconds=None,
            device=None,
            decode=None,
            transcript=None,
            elapsed_values=[],
            transcripts=[],
            error=f"nemo runner invalid JSON: {payload_line}",
        )
    return NemoRunResult(
        rtfx_mean=payload.get("rtfx_mean"),
        rtfx_stdev=payload.get("rtfx_stdev"),
        wall_seconds=payload.get("wall_seconds"),
        device=payload.get("device"),
        decode=payload.get("decode"),
        transcript=payload.get("transcript"),
        elapsed_values=payload.get("elapsed_values") or [],
        transcripts=payload.get("transcripts") or [],
        error=None,
    )


def benchmark_nemo_models(
    *,
    task: str,
    sample: SampleSpec,
    models: Sequence[ModelSpec],
    perf_config: PerfConfig,
    model_id_fn,
    model_type_fn=None,
    decode_mode_fn=None,
    chunk_seconds: float | None = None,
    progress=None,
    on_result=None,
) -> list[ModelBenchmark]:
    results: list[ModelBenchmark] = []
    for model in models:
        model_id = model_id_fn(model)
        model_type = model_type_fn(model) if model_type_fn is not None else None
        decode_mode = decode_mode_fn(model) if decode_mode_fn is not None else None
        run_result = run_nemo_benchmark(
            task=task,
            model_id=model_id,
            model_type=model_type,
            decode_mode=decode_mode,
            sample=sample,
            perf_config=perf_config,
            chunk_seconds=chunk_seconds or 40.0,
        )
        if run_result.error:
            results.append(
                ModelBenchmark(
                    model_name=model.name,
                    model_size=model.size,
                    model_variant=model.variant,
                    rtfx_mean=None,
                    rtfx_stdev=None,
                    bench_seconds=None,
                    device=None,
                    notes=f"nemo failed: {run_result.error}",
                    transcript=None,
                    wer=None,
                    wer_stdev=None,
                    runs=[],
                )
            )
            if on_result is not None:
                on_result(results[-1])
        else:
            notes = f"model: {model_id}"
            if run_result.decode:
                notes = f"{notes}; decode: {run_result.decode}"
            run_rtfx_values = [
                sample.duration_seconds / v
                for v in run_result.elapsed_values
                if v > 0
            ]
            runs = [
                RunResult(
                    rtfx=rtfx,
                    seconds=elapsed,
                    wer=None,
                    transcript=transcript,
                )
                for rtfx, elapsed, transcript in zip(
                    run_rtfx_values,
                    run_result.elapsed_values,
                    run_result.transcripts,
                )
            ]
            results.append(
                ModelBenchmark(
                    model_name=model.name,
                    model_size=model.size,
                    model_variant=model.variant,
                    rtfx_mean=run_result.rtfx_mean,
                    rtfx_stdev=run_result.rtfx_stdev,
                    bench_seconds=run_result.wall_seconds,
                    device=run_result.device,
                    notes=notes,
                    transcript=run_result.transcript,
                    wer=None,
                    wer_stdev=None,
                    runs=runs,
                )
            )
            if on_result is not None:
                on_result(results[-1])
        if progress is not None:
            progress(f"{task} {model.name} {model.size}")
    return results
