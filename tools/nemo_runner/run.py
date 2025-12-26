from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import tempfile
import time
import wave


def _wav_duration_seconds(path: str) -> float:
    with wave.open(path, "rb") as wav:
        if wav.getnchannels() != 1:
            raise ValueError("Expected mono WAV")
        if wav.getframerate() != 16000:
            raise ValueError("Expected 16kHz WAV")
        frames = wav.getnframes()
        rate = wav.getframerate()
    return frames / float(rate)


def _write_wav_chunks(path: str, chunk_seconds: float, out_dir: str) -> list[str]:
    chunk_paths: list[str] = []
    with wave.open(path, "rb") as wav:
        if wav.getnchannels() != 1:
            raise ValueError("Expected mono WAV")
        if wav.getframerate() != 16000:
            raise ValueError("Expected 16kHz WAV")
        rate = wav.getframerate()
        sampwidth = wav.getsampwidth()
        frames_per_chunk = int(rate * chunk_seconds)
        idx = 0
        while True:
            frames = wav.readframes(frames_per_chunk)
            if not frames:
                break
            chunk_path = os.path.join(out_dir, f"chunk_{idx}.wav")
            with wave.open(chunk_path, "wb") as out_wav:
                out_wav.setnchannels(1)
                out_wav.setsampwidth(sampwidth)
                out_wav.setframerate(rate)
                out_wav.writeframes(frames)
            chunk_paths.append(chunk_path)
            idx += 1
    return chunk_paths


def _load_model(task: str, model_id: str, model_type: str | None):
    if task == "canary":
        from nemo.collections.asr.models import EncDecMultiTaskModel  # type: ignore[unresolved-import]

        return EncDecMultiTaskModel.from_pretrained(model_id)
    if task == "parakeet":
        from nemo.collections.asr.models import (  # type: ignore[unresolved-import]
            ASRModel,
            EncDecCTCModelBPE,
            EncDecRNNTBPEModel,
        )

        if model_type == "ctc":
            return EncDecCTCModelBPE.from_pretrained(model_id)
        if model_type in {"rnnt", "tdt"}:
            return EncDecRNNTBPEModel.from_pretrained(model_id)
        if model_type == "tdt-ctc":
            return ASRModel.from_pretrained(model_id)
        raise ValueError(f"Unknown parakeet model_type: {model_type}")
    raise ValueError(f"Unknown task: {task}")


def _configure_logging() -> None:
    logging.basicConfig(level=logging.ERROR)
    os.environ.setdefault("NEMO_LOG_LEVEL", "ERROR")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("canary", "parakeet"))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-type")
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--chunk-seconds", type=float, default=40.0)
    args = parser.parse_args()

    _configure_logging()

    import torch

    model = _load_model(args.task, args.model_id, args.model_type)
    model.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    if args.task == "canary":
        decode_cfg = model.cfg.decoding
        decode_cfg.beam.beam_size = 1
        model.change_decoding_strategy(decode_cfg)

    sample_seconds = _wav_duration_seconds(args.audio_path)

    def run_once() -> None:
        if sample_seconds <= args.chunk_seconds:
            _ = model.transcribe(
                audio=[args.audio_path],
                batch_size=1,
                num_workers=0,
                verbose=False,
            )
            return
        with tempfile.TemporaryDirectory() as temp_dir:
            chunk_paths = _write_wav_chunks(args.audio_path, args.chunk_seconds, temp_dir)
            _ = model.transcribe(
                audio=chunk_paths,
                batch_size=1,
                num_workers=0,
                verbose=False,
            )

    wall_start = time.perf_counter()
    for _ in range(args.warmups):
        run_once()

    elapsed_values: list[float] = []
    for _ in range(args.runs):
        start = time.perf_counter()
        run_once()
        elapsed_values.append(time.perf_counter() - start)
    wall_seconds = time.perf_counter() - wall_start

    rtfx_values = [sample_seconds / v for v in elapsed_values]
    rtfx_mean = statistics.fmean(rtfx_values)
    rtfx_stdev = statistics.stdev(rtfx_values) if len(rtfx_values) >= 2 else 0.0
    payload = {
        "rtfx_mean": rtfx_mean,
        "rtfx_stdev": rtfx_stdev,
        "wall_seconds": wall_seconds,
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
