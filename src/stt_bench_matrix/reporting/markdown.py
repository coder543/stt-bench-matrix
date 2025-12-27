from __future__ import annotations

import os
import subprocess

from ..bench.types import BenchmarkResults


def _format_bytes(value: int | None) -> str | None:
    if value is None:
        return None
    gib = value / (1024**3)
    return f"{gib:.1f} GiB"


def _git_revision() -> str | None:
    env_rev = os.environ.get("STT_BENCH_GIT_REV")
    if env_rev and env_rev.lower() != "unknown":
        return env_rev
    env_sha = os.environ.get("STT_BENCH_GIT_SHA")
    if env_sha and env_sha.lower() != "unknown":
        dirty = os.environ.get("STT_BENCH_GIT_DIRTY", "").lower()
        if dirty in {"1", "true", "yes", "y"}:
            return f"{env_sha} (dirty)"
        return env_sha
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True)
        if dirty.strip():
            return f"{rev} (dirty)"
        return rev
    except Exception:
        return None


def render_markdown(results: BenchmarkResults) -> str:
    lines: list[str] = []
    lines.append("## stt-bench-matrix")
    lines.append("")
    lines.append("**System**")
    accelerator_mem = _format_bytes(results.host.accelerator_memory_bytes)
    ram_mem = _format_bytes(results.host.ram_bytes)
    system_lines = [f"- OS: {results.host.os}", f"- Accelerator: {results.host.accelerator}"]
    if accelerator_mem:
        system_lines.append(f"- Accelerator memory: {accelerator_mem}")
    if results.host.cpu:
        system_lines.append(f"- CPU: {results.host.cpu}")
    if ram_mem:
        system_lines.append(f"- System RAM: {ram_mem}")
    system_lines.append(f"- CUDA available: {str(results.host.cuda_available).lower()}")
    system_lines.append(f"- cuDNN available: {str(results.host.cudnn_available).lower()}")
    system_lines.append(f"- MPS available: {str(results.host.mps_available).lower()}")
    if results.host.cuda_error:
        system_lines.append(f"- CUDA error: {results.host.cuda_error}")
    git_rev = _git_revision()
    if git_rev:
        system_lines.append(f"- Git: {git_rev}")
    system_lines.append(f"- Total time: {results.total_seconds:.2f}s")
    system_lines.append(f"- Sample: {results.sample_name} ({results.sample_path})")
    system_lines.append(f"- Language: {results.language}")
    lines.append("\n".join(system_lines))
    lines.append("")
    lines.append("**Benchmarks**")
    lines.append("")
    lines.append("| Framework | Model | RTFx (mean ± stdev) | Time (s) | Device | Notes |")
    lines.append("| --- | --- | --- | --- | --- | --- |")

    for framework in results.frameworks:
        if not framework.models:
            lines.append(f"| {framework.framework} | - | n/a | - |")
            continue

        for model in framework.models:
            rtf_x = "n/a"
            if model.rtfx_mean and model.rtfx_mean > 0:
                if model.rtfx_stdev is not None:
                    rtf_x = f"{model.rtfx_mean:.2f}x ± {model.rtfx_stdev:.2f}"
                else:
                    rtf_x = f"{model.rtfx_mean:.2f}x"
            bench_seconds = "n/a"
            if model.bench_seconds is not None:
                bench_seconds = f"{model.bench_seconds:.2f}"
            device = model.device or "n/a"
            model_name = f"{model.model_name} {model.model_size}"
            notes = model.notes or ""
            lines.append(
                f"| {framework.framework} | {model_name} | {rtf_x} | {bench_seconds} | {device} | {notes} |"
            )

    return "\n".join(lines)
