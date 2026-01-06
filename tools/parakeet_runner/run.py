from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import cast
from collections.abc import Mapping


def _extract_transcript(result: object) -> str | None:
    if result is None:
        return None
    if isinstance(result, str):
        text = result
    elif isinstance(result, Mapping):
        result_dict = cast(Mapping[str, object], result)
        text = result_dict.get("text") or result_dict.get("transcript")
        if text is None:
            sentences = result_dict.get("sentences")
            if isinstance(sentences, list):
                parts = []
                for sentence in sentences:
                    if isinstance(sentence, Mapping):
                        sent_dict = cast(Mapping[str, object], sentence)
                        piece = sent_dict.get("text")
                    else:
                        piece = getattr(sentence, "text", None)
                    if piece:
                        parts.append(str(piece))
                text = " ".join(parts) if parts else None
    else:
        text = getattr(result, "text", None) or getattr(result, "transcript", None)
        if text is None:
            sentences = getattr(result, "sentences", None)
            if isinstance(sentences, list):
                parts = []
                for sentence in sentences:
                    piece = None
                    if isinstance(sentence, dict):
                        piece = sentence.get("text")
                    else:
                        piece = getattr(sentence, "text", None)
                    if piece:
                        parts.append(str(piece))
                text = " ".join(parts) if parts else None
    if text is None:
        return None
    text = str(text).strip()
    return text or None


def _configure_local_attention(model: object) -> bool:
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return False
    set_attention_model = getattr(encoder, "set_attention_model", None)
    if not callable(set_attention_model):
        return False
    try:
        set_attention_model("rel_pos_local_attn", (256, 256))
        return True
    except Exception:
        return False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run parakeet-mlx benchmarks.")
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
    parser.add_argument("--chunk-seconds", type=float, default=None)
    parser.add_argument("--overlap-seconds", type=float, default=None)
    parser.add_argument("--local-attn", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    from parakeet_mlx import from_pretrained  # type: ignore[import-not-found]

    model = from_pretrained(args.model_id)
    if args.local_attn:
        _configure_local_attention(model)

    chunk_seconds = args.chunk_seconds
    overlap_seconds = args.overlap_seconds

    def run_once(audio_path: str = args.audio_path) -> str | None:
        kwargs = {}
        if chunk_seconds is not None:
            kwargs["chunk_duration"] = chunk_seconds
        if overlap_seconds is not None:
            kwargs["overlap_duration"] = overlap_seconds
        result = model.transcribe(audio_path, **kwargs)
        return _extract_transcript(result)

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
            stdev = (
                statistics.stdev(elapsed_values)
                if len(elapsed_values) >= 2
                else 0.0
            )
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
        (args.sample_seconds / v) if v > 0 else 0.0 for v in elapsed_values
    ]
    rtfx_mean = statistics.fmean(rtfx_values) if rtfx_values else None
    rtfx_stdev = (
        statistics.stdev(rtfx_values) if len(rtfx_values) >= 2 else 0.0
    )
    wall_seconds = time.perf_counter() - start_wall
    transcript = transcripts[-1] if transcripts else None

    payload = {
        "rtfx_mean": rtfx_mean,
        "rtfx_stdev": rtfx_stdev,
        "wall_seconds": wall_seconds,
        "device": "mps",
        "transcript": transcript,
        "elapsed_values": elapsed_values,
        "transcripts": transcripts,
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
