from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import os
from typing import Sequence

from ..bench.perf import PerfConfig
from ..bench.types import ModelBenchmark
from ..bench.samples import SampleSpec
from ..models.registry import ModelSpec


@dataclass(frozen=True)
class NemoRunResult:
    rtfx_mean: float | None
    rtfx_stdev: float | None
    wall_seconds: float | None
    device: str | None
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
            error=f"nemo runner missing at {runner_script}",
        )
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
        "--chunk-seconds",
        str(chunk_seconds),
    ]
    if model_type is not None:
        cmd.extend(["--model-type", model_type])
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
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
            rtfx_mean=None, rtfx_stdev=None, wall_seconds=None, device=None, error=error
        )
    stdout = proc.stdout.strip()
    if not stdout:
        return NemoRunResult(
            rtfx_mean=None,
            rtfx_stdev=None,
            wall_seconds=None,
            device=None,
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
            error=f"nemo runner invalid JSON: {payload_line}",
        )
    return NemoRunResult(
        rtfx_mean=payload.get("rtfx_mean"),
        rtfx_stdev=payload.get("rtfx_stdev"),
        wall_seconds=payload.get("wall_seconds"),
        device=payload.get("device"),
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
    progress=None,
) -> list[ModelBenchmark]:
    results: list[ModelBenchmark] = []
    for model in models:
        model_id = model_id_fn(model)
        model_type = model_type_fn(model) if model_type_fn is not None else None
        run_result = run_nemo_benchmark(
            task=task,
            model_id=model_id,
            model_type=model_type,
            sample=sample,
            perf_config=perf_config,
        )
        if run_result.error:
            results.append(
                ModelBenchmark(
                    model_name=model.name,
                    model_size=model.size,
                    rtfx_mean=None,
                    rtfx_stdev=None,
                    bench_seconds=None,
                    notes=f"nemo failed: {run_result.error}",
                )
            )
        else:
            device_note = f"device: {run_result.device}" if run_result.device else None
            notes = f"model: {model_id}"
            if device_note:
                notes = f"{notes}; {device_note}"
            results.append(
                ModelBenchmark(
                    model_name=model.name,
                    model_size=model.size,
                    rtfx_mean=run_result.rtfx_mean,
                    rtfx_stdev=run_result.rtfx_stdev,
                    bench_seconds=run_result.wall_seconds,
                    notes=notes,
                )
            )
        if progress is not None:
            progress(f"{task} {model.name} {model.size}")
    return results
