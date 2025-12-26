from __future__ import annotations

import subprocess

from ..bench.types import BenchmarkResults


def _format_bytes(value: int | None) -> str | None:
    if value is None:
        return None
    gib = value / (1024**3)
    return f"{gib:.1f} GiB"


def _git_revision() -> str | None:
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
    git_rev = _git_revision()
    if git_rev:
        system_lines.append(f"- Git: {git_rev}")
    system_lines.append(f"- Total time: {results.total_seconds:.2f}s")
    lines.append("\n".join(system_lines))
    lines.append("")
    lines.append("**Benchmarks**")
    lines.append("")
    lines.append("| Framework | Model | RTFx (mean ± stdev) | Time (s) | Notes |")
    lines.append("| --- | --- | --- | --- | --- |")

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
            model_name = f"{model.model_name} {model.model_size}"
            notes = model.notes or ""
            lines.append(
                f"| {framework.framework} | {model_name} | {rtf_x} | {bench_seconds} | {notes} |"
            )

    return "\n".join(lines)
