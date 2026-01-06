from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable
import wave

import numpy as np

from ..bench.perf import PerfConfig, measure_rtfx
from ..bench.samples import SampleSpec
from ..bench.types import ModelBenchmark, RunResult
from ..models.registry import ModelSpec
from ..platforms.detect import HostInfo
from ..platforms.cuda import cuda_is_usable
from .base import FrameworkInfo


@dataclass(frozen=True)
class GraniteTransformersFramework:
    info: FrameworkInfo = FrameworkInfo(
        name="granite-transformers",
        description="IBM Granite Speech via Transformers",
        supports_whisper=False,
        supports_parakeet=False,
        supports_canary=False,
        supports_moonshine=False,
        supports_granite=True,
    )

    def is_supported(self, host: HostInfo) -> bool:
        return host.is_macos or host.is_linux


def _model_id(size: str) -> str:
    if size == "2b":
        return "ibm-granite/granite-speech-3.3-2b"
    if size == "8b":
        return "ibm-granite/granite-speech-3.3-8b"
    return size


def _load_wav_16k_mono(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wav:
        if wav.getnchannels() != 1:
            raise ValueError("Expected mono WAV")
        if wav.getframerate() != 16000:
            raise ValueError("Expected 16kHz WAV")
        frames = wav.readframes(wav.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def _chunk_audio(
    audio: np.ndarray, sample_rate: int, chunk_seconds: float, overlap_seconds: float
) -> list[np.ndarray]:
    if chunk_seconds <= 0:
        return [audio]
    chunk_size = int(chunk_seconds * sample_rate)
    overlap_size = int(max(0.0, overlap_seconds) * sample_rate)
    if chunk_size <= 0:
        return [audio]
    if overlap_size >= chunk_size:
        overlap_size = 0
    chunks: list[np.ndarray] = []
    start = 0
    step = chunk_size - overlap_size
    while start < len(audio):
        end = min(len(audio), start + chunk_size)
        chunks.append(audio[start:end])
        if end == len(audio):
            break
        start += step
    return chunks


def benchmark_granite_models(
    sample: SampleSpec,
    models: list[ModelSpec],
    perf_config: PerfConfig,
    language: str,
    progress: Callable[[str], None] | None = None,
    on_result: Callable[[ModelBenchmark], None] | None = None,
) -> list[ModelBenchmark]:
    del language
    if not models:
        return []
    try:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    except Exception as exc:  # noqa: BLE001
        return [
            ModelBenchmark(
                model_name=model.name,
                model_size=model.size,
                model_variant=model.variant,
                model_id=None,
                rtfx_mean=None,
                rtfx_stdev=None,
                bench_seconds=None,
                device=None,
                notes=f"granite unavailable: {exc}",
                transcript=None,
                wer=None,
                wer_stdev=None,
                runs=[],
            )
            for model in models
        ]

    cuda_ok, cuda_err = cuda_is_usable()
    prefer_cuda = cuda_ok
    prefer_mps = torch.backends.mps.is_available()
    device = torch.device("cpu")
    dtype = torch.float32
    device_note = "cpu"
    if prefer_cuda:
        device = torch.device("cuda:0")
        dtype = torch.float16
        device_note = "cuda"
    elif prefer_mps:
        device = torch.device("mps")
        dtype = torch.float32
        device_note = "mps"
    elif cuda_err and torch.cuda.is_available():
        device_note = "cpu"

    audio = _load_wav_16k_mono(str(sample.audio_path))
    chunk_seconds = 60.0 if sample.duration_seconds > 60 else 0.0
    overlap_seconds = float(os.getenv("STT_BENCH_GRANITE_CHUNK_OVERLAP", "0") or 0.0)
    audio_chunks = _chunk_audio(audio, 16000, chunk_seconds, overlap_seconds)

    results: list[ModelBenchmark] = []

    for model in models:
        model_id = _model_id(model.size)
        try:
            processor = AutoProcessor.from_pretrained(model_id)
            tokenizer = processor.tokenizer
            try:
                asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id, dtype=dtype
                ).to(device)
            except Exception:  # noqa: BLE001
                device = torch.device("cpu")
                dtype = torch.float32
                device_note = "cpu"
                asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id, dtype=dtype
                ).to(device)
            asr_model.eval()
            last_transcript: str | None = None

            def run_once() -> str | None:
                nonlocal last_transcript
                chunks_text: list[str] = []
                for chunk in audio_chunks:
                    chunk_tensor = torch.from_numpy(chunk).unsqueeze(0)
                    chat = [
                        {
                            "role": "system",
                            "content": (
                                "Knowledge Cutoff Date: April 2024.\n"
                                "Today's Date: December 27, 2025.\n"
                                "You are Granite, developed by IBM. "
                                "You are a helpful AI assistant."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "<|audio|>can you transcribe the speech into a written format?"
                            ),
                        },
                    ]
                    prompt = tokenizer.apply_chat_template(
                        chat, tokenize=False, add_generation_prompt=True
                    )
                    inputs = processor(
                        prompt,
                        chunk_tensor,
                        device=str(device),
                        return_tensors="pt",
                    ).to(device=device, dtype=dtype)
                    max_new_tokens = 200
                    with torch.inference_mode():
                        generated = asr_model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            num_beams=4,
                            do_sample=False,
                            min_length=1,
                            top_p=1.0,
                            repetition_penalty=3.0,
                            length_penalty=1.0,
                            temperature=1.0,
                            bos_token_id=tokenizer.bos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    num_input_tokens = inputs["input_ids"].shape[-1]
                    new_tokens = generated[:, num_input_tokens:]
                    decoded = tokenizer.batch_decode(
                        new_tokens,
                        skip_special_tokens=True,
                        add_special_tokens=False,
                        clean_up_tokenization_spaces=True,
                    )
                    if decoded:
                        text = decoded[0].strip()
                        if text:
                            chunks_text.append(text)
                if chunks_text:
                    return " ".join(chunks_text)
                return None

            stats = measure_rtfx(
                name=f"granite:{model.size}",
                sample=sample,
                run_once=run_once,
                config=perf_config,
            )
            last_transcript = (
                stats.transcripts[-1] if stats.transcripts else None
            )
            note_suffix = ""
            if chunk_seconds > 0:
                note_suffix = f"; chunked:{int(chunk_seconds)}s"
            note_suffix = ""
            if chunk_seconds > 0:
                note_suffix = f"; chunked:{int(chunk_seconds)}s"
            note_suffix = f"{note_suffix}; beams:4; max_new_tokens:200"
            results.append(
                ModelBenchmark(
                    model_name=model.name,
                    model_size=model.size,
                    model_variant=model.variant,
                    model_id=model_id,
                    rtfx_mean=stats.rtfx_mean,
                    rtfx_stdev=stats.rtfx_stdev,
                    bench_seconds=stats.wall_seconds,
                    device=device_note,
                    notes=note_suffix.lstrip("; ") if note_suffix else None,
                    transcript=last_transcript,
                    wer=None,
                    wer_stdev=None,
                    runs=[
                        RunResult(
                            rtfx=rtfx,
                            seconds=elapsed,
                            wer=None,
                            transcript=transcript,
                        )
                        for rtfx, elapsed, transcript in zip(
                            stats.rtfx_values,
                            stats.elapsed_values,
                            stats.transcripts,
                        )
                    ],
                )
            )
            if on_result is not None:
                on_result(results[-1])
        except Exception as exc:  # noqa: BLE001
            note = f"granite failed: {exc}"
            if cuda_err and torch.cuda.is_available():
                note = f"{note}; cuda unavailable: {cuda_err}"
            results.append(
                ModelBenchmark(
                    model_name=model.name,
                    model_size=model.size,
                    model_variant=model.variant,
                    model_id=model_id,
                    rtfx_mean=None,
                    rtfx_stdev=None,
                    bench_seconds=None,
                    device=device_note,
                    notes=note,
                    transcript=None,
                    wer=None,
                    wer_stdev=None,
                    runs=[],
                )
            )
            if on_result is not None:
                on_result(results[-1])
        if progress is not None:
            progress(f"granite {model.name} {model.size}")

    return results
