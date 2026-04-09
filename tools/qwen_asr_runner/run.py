from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any


_LANGUAGE_MAP = {
    "ar": "Arabic",
    "cantonese": "Cantonese",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fi": "Finnish",
    "fil": "Filipino",
    "fr": "French",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "yue": "Cantonese",
    "zh": "Chinese",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Qwen3-ASR benchmarks.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--warmup-audio-path")
    parser.add_argument("--sample-seconds", type=float, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--auto-min-runs", type=int, default=5)
    parser.add_argument("--auto-max-runs", type=int, default=30)
    parser.add_argument("--auto-target-cv", type=float, default=0.05)
    parser.add_argument("--language", default="")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    return parser


def _device_config(torch_module: Any) -> tuple[Any, str, str]:
    if torch_module.cuda.is_available():
        return torch_module.bfloat16, "cuda:0", "cuda:0"
    if torch_module.backends.mps.is_available():
        return torch_module.float16, "mps", "mps"
    return torch_module.float32, "cpu", "cpu"


def _normalize_language(value: str) -> str | None:
    normalized = value.strip()
    if not normalized:
        return None
    return _LANGUAGE_MAP.get(normalized.lower(), normalized)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    import torch
    from qwen_asr import Qwen3ASRModel

    dtype, device_map, device_note = _device_config(torch)
    load_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_map": device_map,
    }
    if args.max_new_tokens is not None:
        load_kwargs["max_new_tokens"] = args.max_new_tokens
    model = Qwen3ASRModel.from_pretrained(args.model_id, **load_kwargs)

    def run_once(audio_path: str = args.audio_path) -> str | None:
        transcribe_kwargs: dict[str, Any] = {
            "audio": audio_path,
        }
        language = _normalize_language(args.language)
        if language:
            transcribe_kwargs["language"] = language
        results = model.transcribe(**transcribe_kwargs)
        if not results:
            return None
        text = getattr(results[0], "text", None)
        if text is None:
            return None
        normalized = str(text).strip()
        return normalized or None

    start_wall = time.perf_counter()
    for _ in range(args.warmups):
        if args.warmup_audio_path:
            run_once(args.warmup_audio_path)
        else:
            run_once()

    elapsed_values: list[float] = []
    transcripts: list[str | None] = []
    if args.auto:
        target_cv = max(0.0, args.auto_target_cv)
        min_runs = max(1, args.auto_min_runs)
        max_runs = max(min_runs, args.auto_max_runs)
        while len(elapsed_values) < max_runs:
            start = time.perf_counter()
            transcript = run_once()
            elapsed = time.perf_counter() - start
            elapsed_values.append(elapsed)
            transcripts.append(transcript)
            if len(elapsed_values) < min_runs:
                continue
            mean = statistics.fmean(elapsed_values)
            if mean <= 0:
                continue
            stdev = statistics.stdev(elapsed_values) if len(elapsed_values) >= 2 else 0.0
            cv = stdev / mean if mean else 0.0
            if cv <= target_cv:
                break
    else:
        for _ in range(args.runs):
            start = time.perf_counter()
            transcript = run_once()
            elapsed = time.perf_counter() - start
            elapsed_values.append(elapsed)
            transcripts.append(transcript)

    rtfx_values = [
        (args.sample_seconds / value) if value > 0 else 0.0 for value in elapsed_values
    ]
    note_parts = [
        f"dtype:{dtype}",
        f"device_map:{device_map}",
    ]
    if args.max_new_tokens is not None:
        note_parts.append(f"max_new_tokens:{args.max_new_tokens}")
    payload = {
        "rtfx_mean": statistics.fmean(rtfx_values) if rtfx_values else None,
        "rtfx_stdev": statistics.stdev(rtfx_values) if len(rtfx_values) >= 2 else 0.0,
        "wall_seconds": time.perf_counter() - start_wall,
        "device": device_note,
        "transcript": transcripts[-1] if transcripts else None,
        "elapsed_values": elapsed_values,
        "transcripts": transcripts,
        "notes": "; ".join(note_parts),
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
