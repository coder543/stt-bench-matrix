from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .bench.runner import run_benchmarks
from .platforms.detect import detect_host
from .reporting.markdown import render_markdown
from .frameworks.whisper_cpp import has_whisper_cli


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
        "--json",
        action="store_true",
        help="Emit raw JSON instead of Markdown",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    host = detect_host()
    warn_missing_whisper_cli = host.is_macos or host.is_linux
    if warn_missing_whisper_cli and not has_whisper_cli():
        warning = (
            "warning: whisper.cpp not found (whisper-cli missing from PATH); "
            "skipping whisper.cpp benchmarks"
        )
        print(warning)

    results = run_benchmarks(
        host=host,
        use_cache=not args.no_cache,
        quick=args.quick,
        quick_2=args.quick_2,
        parakeet_only=args.parakeet,
    )

    if args.json:
        import json

        print(json.dumps(asdict(results), indent=2))
        return 0

    if warn_missing_whisper_cli and not has_whisper_cli():
        warning = (
            "warning: whisper.cpp not found (whisper-cli missing from PATH); "
            "skipping whisper.cpp benchmarks"
        )
        print(warning)
    markdown = render_markdown(results)
    print(markdown)
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = output_dir / f"{timestamp}.md"
    output_path.write_text(markdown, encoding="utf-8")
    return 0
