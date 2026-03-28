from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable
import wave

import numpy as np

from ..bench.perf import PerfConfig, measure_rtfx
from ..bench.samples import SampleSpec
from ..bench.types import ModelBenchmark, RunResult
from ..models.registry import ModelSpec
from ..platforms.detect import HostInfo
from ..platforms.cuda import cuda_is_usable
from .base import FrameworkInfo


DEFAULT_MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
DEFAULT_LANGUAGE = os.environ.get("COHERE_LANGUAGE", "en")
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("STT_BENCH_COHERE_MAX_NEW_TOKENS", "256"))
FORCE_MATH_SDP = os.environ.get("STT_BENCH_COHERE_FORCE_MATH_SDP", "0").lower() not in {
    "0",
    "false",
    "no",
    "n",
}


@dataclass(frozen=True)
class CohereTransformersFramework:
    info: FrameworkInfo = FrameworkInfo(
        name="cohere-transformers",
        description="Cohere Transcribe via Transformers",
        supports_whisper=False,
        supports_parakeet=False,
        supports_canary=False,
        supports_moonshine=False,
        supports_granite=False,
        supports_cohere=True,
    )

    def is_supported(self, host: HostInfo) -> bool:
        return host.is_macos or host.is_linux


def _model_id(model: ModelSpec) -> str:
    if model.name != "cohere-transcribe":
        raise ValueError("cohere framework only supports cohere-transcribe models")
    return os.environ.get("COHERE_MODEL_ID", DEFAULT_MODEL_ID)


def _load_wav_16k_mono(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wav:
        if wav.getnchannels() != 1:
            raise ValueError("Expected mono WAV")
        if wav.getframerate() != 16000:
            raise ValueError("Expected 16kHz WAV")
        frames = wav.readframes(wav.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def _normalize_text(decoded: object) -> str:
    if isinstance(decoded, str):
        return decoded.strip()
    if isinstance(decoded, list):
        cleaned = [str(item).strip() for item in decoded if str(item).strip()]
        if len(cleaned) == 1:
            return cleaned[0]
        return "\n\n".join(cleaned).strip()
    return str(decoded or "").strip()


def benchmark_cohere_models(
    sample: SampleSpec,
    models: list[ModelSpec],
    perf_config: PerfConfig,
    language: str,
    warmup_sample: SampleSpec | None = None,
    progress: Callable[[str], None] | None = None,
    on_result: Callable[[ModelBenchmark], None] | None = None,
) -> list[ModelBenchmark]:
    if not models:
        return []
    try:
        import torch
        from transformers import AutoProcessor
        from transformers.models.cohere_asr import CohereAsrForConditionalGeneration
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
                notes=f"cohere unavailable: {exc}",
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
    warmup_audio = None
    if warmup_sample is not None:
        warmup_audio = _load_wav_16k_mono(str(warmup_sample.audio_path))

    results: list[ModelBenchmark] = []

    for model in models:
        model_id = _model_id(model)
        try:
            processor = AutoProcessor.from_pretrained(model_id)
            try:
                asr_model = CohereAsrForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                ).to(device)
            except Exception:  # noqa: BLE001
                device = torch.device("cpu")
                dtype = torch.float32
                device_note = "cpu"
                asr_model = CohereAsrForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                ).to(device)
            asr_model.eval()
            sdp_note = None
            if prefer_cuda and FORCE_MATH_SDP:
                try:
                    torch.backends.cuda.enable_flash_sdp(False)
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
                    torch.backends.cuda.enable_math_sdp(True)
                    sdp_note = "sdp: math"
                except Exception:
                    sdp_note = None

            def run_once(audio_input=audio) -> str | None:
                inputs = processor(
                    audio=audio_input,
                    sampling_rate=16000,
                    return_tensors="pt",
                    language=(language or DEFAULT_LANGUAGE).strip() or DEFAULT_LANGUAGE,
                    punctuation=True,
                )
                audio_chunk_index = inputs.get("audio_chunk_index")
                for key, value in list(inputs.items()):
                    if not torch.is_tensor(value):
                        continue
                    if torch.is_floating_point(value):
                        inputs[key] = value.to(device=device, dtype=dtype)
                    else:
                        inputs[key] = value.to(device=device)
                with torch.inference_mode():
                    outputs = asr_model.generate(
                        **inputs,
                        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                    )
                decoded = processor.decode(
                    outputs,
                    skip_special_tokens=True,
                    audio_chunk_index=audio_chunk_index,
                    language=(language or DEFAULT_LANGUAGE).strip() or DEFAULT_LANGUAGE,
                )
                text = _normalize_text(decoded)
                return text or None

            warmup_run_once = None
            if warmup_audio is not None:
                warmup_run_once = lambda: run_once(warmup_audio)
            stats = measure_rtfx(
                name=f"cohere:{model.size}",
                sample=sample,
                run_once=run_once,
                warmup_run_once=warmup_run_once,
                config=perf_config,
                progress_label=f"cohere {model.name} {model.size}",
            )
            note_parts = [f"max_new_tokens:{DEFAULT_MAX_NEW_TOKENS}"]
            if sdp_note is not None:
                note_parts.append(sdp_note)
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
                    notes="; ".join(note_parts),
                    transcript=stats.transcripts[-1] if stats.transcripts else None,
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
            note = f"cohere failed: {type(exc).__name__}: {exc}"
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
            progress(f"cohere {model.name} {model.size}")

    return results
