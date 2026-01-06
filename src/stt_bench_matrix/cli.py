from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .bench.runner import run_benchmarks
from .platforms.detect import detect_host
from .bench.samples import default_sample, default_warmup_sample, sample_from_path
from .reporting.markdown import render_markdown
from .frameworks.whisper_cpp import has_whisper_cli
from .frameworks.registry import all_frameworks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stt-bench-matrix",
        description="Cross-platform STT benchmarking tool",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable model caching for this run",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
        help="Number of warmup runs per model (default: 1)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Number of measured runs per model (disables auto mode)",
    )
    parser.add_argument(
        "--auto-min-runs",
        type=int,
        default=5,
        help="Minimum runs in auto mode (default: 5)",
    )
    parser.add_argument(
        "--auto-max-runs",
        type=int,
        default=30,
        help="Maximum runs in auto mode (default: 30)",
    )
    parser.add_argument(
        "--auto-target-cv",
        type=float,
        default=0.05,
        help="Target coefficient of variation in auto mode (default: 0.05)",
    )
    parser.add_argument(
        "--parakeet",
        action="store_true",
        help="Run only Parakeet benchmarks (respects --quick/--quick-2)",
    )
    parser.add_argument(
        "--heavy",
        action="store_true",
        help="Include heavy models (Granite 2B; Canary 2.5B; 8B is opt-in via --models)",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="Language code to use when supported (default: en)",
    )
    parser.add_argument(
        "--sample",
        help="Path to a 16kHz mono WAV sample to benchmark",
    )
    parser.add_argument(
        "--warmup-sample",
        help="Path to a 16kHz mono WAV sample to use for warmups (default: 20s sample)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit raw JSON instead of Markdown",
    )
    parser.add_argument(
        "--frameworks",
        help="Comma-separated list of frameworks to run (by name)",
    )
    parser.add_argument(
        "--models",
        help=(
            "Comma-separated list of model sizes or names to run "
            "(e.g., tiny,base,parakeet-ctc)"
        ),
    )
    parser.add_argument(
        "--include-per-run-transcripts",
        action="store_true",
        help="Include per-run transcripts in JSON output (large)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print available frameworks and models, then exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    host = detect_host()
    if args.list:
        from .models.registry import (
            whisper_models,
            whisper_optional_models,
            parakeet_models,
            canary_models,
            moonshine_models,
            nemotron_models,
            granite_models,
            granite_optional_models,
        )

        print("Frameworks:")
        for framework in all_frameworks():
            print(f"- {framework.info.name}: {framework.info.description}")
        print("")
        print("Models:")
        def _print_models(label: str, models, optional: bool = False) -> None:
            suffix = " (optional)" if optional else ""
            print(f"- {label}{suffix}:")
            for model in models:
                variant = f" ({model.variant})" if model.variant else ""
                print(f"  - {model.name} {model.size}{variant}")
        _print_models("whisper", whisper_models() + whisper_optional_models())
        _print_models("parakeet", parakeet_models())
        _print_models("canary", canary_models())
        _print_models("moonshine", moonshine_models())
        _print_models("nemotron", nemotron_models())
        _print_models("granite", granite_models() + granite_optional_models())
        return 0
    selected_frameworks = None
    if args.frameworks:
        selected_frameworks = {name.strip() for name in args.frameworks.split(",") if name.strip()}
        available = {framework.info.name for framework in all_frameworks()}
        unknown = sorted(selected_frameworks - available)
        if unknown:
            parser.error(
                f"Unknown framework(s): {', '.join(unknown)}. "
                f"Available: {', '.join(sorted(available))}"
            )
    model_filters = None
    if args.models:
        model_filters = {name.strip().lower() for name in args.models.split(",") if name.strip()}
    warn_missing_whisper_cli = host.is_macos or host.is_linux
    if selected_frameworks is not None and "whisper.cpp" not in selected_frameworks:
        warn_missing_whisper_cli = False
    if warn_missing_whisper_cli and not has_whisper_cli():
        warning = (
            "warning: whisper.cpp not found (whisper-cli missing from PATH); "
            "skipping whisper.cpp benchmarks"
        )
        print(warning)

    sample = default_sample()
    if args.sample:
        sample = sample_from_path(Path(args.sample))
    warmup_sample = default_warmup_sample()
    if args.warmup_sample:
        warmup_sample = sample_from_path(Path(args.warmup_sample))
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = output_dir / f"{timestamp}.md"
    output_json_path = output_dir / f"{timestamp}.json"

    def _atomic_write_text(path: Path, contents: str) -> None:
        import os
        import tempfile

        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as handle:
            handle.write(contents)
            temp_name = handle.name
        os.replace(temp_name, path)

    def _write_outputs(results) -> None:
        import json

        markdown = render_markdown(results)
        json_payload = json.dumps(asdict(results), indent=2)
        _atomic_write_text(output_path, markdown)
        _atomic_write_text(output_json_path, json_payload)

    results = run_benchmarks(
        host=host,
        use_cache=not args.no_cache,
        sample=sample,
        warmup_sample=warmup_sample,
        language=args.lang,
        warmups=args.warmups,
        runs=args.runs if args.runs is not None else 3,
        auto=args.runs is None,
        auto_min_runs=args.auto_min_runs,
        auto_max_runs=args.auto_max_runs,
        auto_target_cv=args.auto_target_cv,
        parakeet_only=args.parakeet,
        heavy=args.heavy,
        frameworks=selected_frameworks,
        model_filters=model_filters,
        include_run_transcripts=args.include_per_run_transcripts,
        on_update=_write_outputs,
    )

    import json
    markdown = render_markdown(results)
    json_payload = json.dumps(asdict(results), indent=2)
    if args.json:
        print(json_payload)
    else:
        print(markdown)

    if warn_missing_whisper_cli and not has_whisper_cli():
        warning = (
            "warning: whisper.cpp not found (whisper-cli missing from PATH); "
            "skipping whisper.cpp benchmarks"
        )
        print(warning)
    _write_outputs(results)
    return 0
