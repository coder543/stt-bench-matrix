from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import wave

import numpy as np

from ..bench.perf import PerfConfig, measure_rtfx
from ..bench.samples import SampleSpec
from ..bench.types import ModelBenchmark
from ..models.registry import ModelSpec
from ..platforms.detect import HostInfo
from .base import FrameworkInfo


@dataclass(frozen=True)
class TransformersWhisperFramework:
    info: FrameworkInfo = FrameworkInfo(
        name="transformers",
        description="Hugging Face Transformers Whisper",
        supports_whisper=True,
        supports_parakeet=False,
        supports_canary=False,
    )

    def is_supported(self, host: HostInfo) -> bool:
        return host.is_macos or host.is_linux


def _model_id(size: str) -> str:
    if size == "large-v3":
        return "openai/whisper-large-v3"
    return f"openai/whisper-{size}"


def _load_wav_16k_mono(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wav:
        if wav.getnchannels() != 1:
            raise ValueError("Expected mono WAV")
        if wav.getframerate() != 16000:
            raise ValueError("Expected 16kHz WAV")
        frames = wav.readframes(wav.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def benchmark_whisper_models(
    sample: SampleSpec,
    models: list[ModelSpec],
    perf_config: PerfConfig,
    progress: Callable[[str], None] | None = None,
) -> list[ModelBenchmark]:
    try:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
    except Exception as exc:  # noqa: BLE001
        return [
            ModelBenchmark(
                model_name=model.name,
                model_size=model.size,
                rtfx_mean=None,
                rtfx_stdev=None,
                bench_seconds=None,
                notes=f"transformers unavailable: {exc}",
            )
            for model in models
        ]

    device = torch.device("cpu")
    dtype = torch.float32
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float32

    audio = _load_wav_16k_mono(str(sample.audio_path))

    results: list[ModelBenchmark] = []

    for model in models:
        model_id = _model_id(model.size)
        try:
            processor = WhisperProcessor.from_pretrained(model_id)
            whisper = WhisperForConditionalGeneration.from_pretrained(
                model_id, dtype=dtype
            ).to(device)

            def run_once() -> None:
                inputs = processor(
                    audio,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding="longest",
                    truncation=False,
                    return_attention_mask=True,
                )
                input_features = inputs.input_features.to(device=device, dtype=dtype)
                attention_mask = inputs.attention_mask.to(device)
                with torch.no_grad():
                    _ = whisper.generate(
                        input_features,
                        attention_mask=attention_mask,
                        task="transcribe",
                        language="en",
                        return_timestamps=True,
                    )

            stats = measure_rtfx(
                name=f"transformers:{model.size}",
                sample=sample,
                run_once=run_once,
                config=perf_config,
            )
            results.append(
                ModelBenchmark(
                    model_name=model.name,
                    model_size=model.size,
                    rtfx_mean=stats.rtfx_mean,
                    rtfx_stdev=stats.rtfx_stdev,
                    bench_seconds=stats.wall_seconds,
                    notes=f"model: {model_id}",
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                ModelBenchmark(
                    model_name=model.name,
                    model_size=model.size,
                    rtfx_mean=None,
                    rtfx_stdev=None,
                    bench_seconds=None,
                    notes=f"transformers failed: {exc}",
                )
            )
        if progress is not None:
            progress(f"transformers {model.name} {model.size}")

    return results
