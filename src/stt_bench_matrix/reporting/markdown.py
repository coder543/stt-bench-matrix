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
    env_dirty = os.environ.get("STT_BENCH_GIT_DIRTY", "").lower()
    if env_rev and env_rev.lower() != "unknown":
        if env_dirty in {"1", "true", "yes", "y"}:
            return f"{env_rev} (dirty)"
        return _append_runtime_dirty(env_rev)
    env_sha = os.environ.get("STT_BENCH_GIT_SHA")
    if env_sha and env_sha.lower() != "unknown":
        if env_dirty in {"1", "true", "yes", "y"}:
            return f"{env_sha} (dirty)"
        return _append_runtime_dirty(env_sha)
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


def _append_runtime_dirty(rev: str) -> str:
    if "dirty" in rev:
        return rev
    try:
        inside = subprocess.check_output(
            ["git", "rev-parse", "--is-inside-work-tree"],
            text=True,
        ).strip()
        if inside != "true":
            return rev
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True)
        if dirty.strip():
            return f"{rev} (dirty)"
    except Exception:
        return rev
    return rev


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
    lines.append(
        "| Framework | Model | RTFx (mean ± stdev) | Runs | Time (s) | Device | WER (mean ± stdev) | Notes |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")

    for framework in results.frameworks:
        if not framework.models:
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
            if model.model_variant:
                model_name = f"{model_name} ({model.model_variant})"
            wer = "n/a"
            if model.wer is not None:
                if model.wer_stdev is not None:
                    wer = f"{model.wer:.3f} ± {model.wer_stdev:.3f}"
                else:
                    wer = f"{model.wer:.3f}"
            notes = model.notes or ""
            runs_count = len(model.runs) if model.runs is not None else 0
            lines.append(
                f"| {framework.framework} | {model_name} | {rtf_x} | {runs_count} | {bench_seconds} | {device} | {wer} | {notes} |"
            )

    return "\n".join(lines)
