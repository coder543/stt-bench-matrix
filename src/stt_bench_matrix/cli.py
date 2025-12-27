from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .bench.runner import run_benchmarks
from .platforms.detect import detect_host
from .bench.samples import default_sample, sample_from_path
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
        "--quick",
        action="store_true",
        help="Run a fast dev-only benchmark (Whisper tiny only)",
    )
    parser.add_argument(
        "--quick-2",
        action="store_true",
        help="Run a fast dev-only benchmark (Whisper tiny + base)",
    )
    parser.add_argument(
        "--parakeet",
        action="store_true",
        help="Run only Parakeet benchmarks (respects --quick/--quick-2)",
    )
    parser.add_argument(
        "--heavy",
        action="store_true",
        help="Include heavy models (e.g. Granite Speech 3.3 8B)",
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
        "--json",
        action="store_true",
        help="Emit raw JSON instead of Markdown",
    )
    parser.add_argument(
        "--frameworks",
        help="Comma-separated list of frameworks to run (by name)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    host = detect_host()
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
    results = run_benchmarks(
        host=host,
        use_cache=not args.no_cache,
        sample=sample,
        language=args.lang,
        quick=args.quick,
        quick_2=args.quick_2,
        parakeet_only=args.parakeet,
        heavy=args.heavy,
        frameworks=selected_frameworks,
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
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = output_dir / f"{timestamp}.md"
    output_path.write_text(markdown, encoding="utf-8")
    output_json_path = output_dir / f"{timestamp}.json"
    output_json_path.write_text(json_payload, encoding="utf-8")
    return 0
