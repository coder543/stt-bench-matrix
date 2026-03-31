from __future__ import annotations

import argparse
import importlib
import json
import statistics
import time
from typing import Any, cast


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Voxtral benchmarks.")
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
    parser.add_argument("--max-new-tokens", type=int, default=None)
    return parser

def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration

    Audio = getattr(
        importlib.import_module("mistral_common.tokens.tokenizers.audio"),
        "Audio",
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
        args.model_id,
        device_map="auto",
        dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    model_device = cast(Any, model).device
    device_note = str(model_device)
    first_param = next(cast(Any, model).parameters())
    model_dtype_note = str(model.dtype)
    first_param_dtype_note = str(first_param.dtype)
    first_param_device_note = str(first_param.device)

    def run_once(audio_path: str = args.audio_path) -> str | None:
        audio = Audio.from_file(audio_path, strict=False)
        audio.resample(processor.feature_extractor.sampling_rate)
        inputs = processor(audio.audio_array, return_tensors="pt")
        inputs = inputs.to(model_device, dtype=model.dtype)
        generate_kwargs: dict[str, Any] = {}
        if args.max_new_tokens is not None:
            generate_kwargs["max_new_tokens"] = args.max_new_tokens
        generated_ids = cast(Any, model).generate(**inputs, **generate_kwargs)
        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
        if decoded and decoded[0].strip():
            return decoded[0].strip()
        return None

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
        "dtype:auto",
        "device_map:auto",
        f"model_dtype:{model_dtype_note}",
        f"param_dtype:{first_param_dtype_note}",
        f"param_device:{first_param_device_note}",
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
