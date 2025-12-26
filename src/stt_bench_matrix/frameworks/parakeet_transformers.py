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
from ..platforms.cuda import cuda_is_usable
from .base import FrameworkInfo


@dataclass(frozen=True)
class ParakeetTransformersFramework:
    info: FrameworkInfo = FrameworkInfo(
        name="parakeet-transformers",
        description="NVIDIA Parakeet via Transformers AutoModelForCTC",
        supports_whisper=False,
        supports_parakeet=True,
        supports_canary=False,
    )

    def is_supported(self, host: HostInfo) -> bool:
        return host.is_macos or host.is_linux


def _model_id(spec: ModelSpec) -> str:
    if spec.name != "parakeet-ctc":
        raise ValueError("parakeet transformers only supports CTC models")
    if spec.size == "1.1b":
        return "nvidia/parakeet-ctc-1.1b"
    if spec.size == "0.6b":
        return "nvidia/parakeet-ctc-0.6b"
    return f"nvidia/parakeet-ctc-{spec.size}"


def _load_wav_16k_mono(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wav:
        if wav.getnchannels() != 1:
            raise ValueError("Expected mono WAV")
        if wav.getframerate() != 16000:
            raise ValueError("Expected 16kHz WAV")
        frames = wav.readframes(wav.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def benchmark_parakeet_models(
    sample: SampleSpec,
    models: list[ModelSpec],
    perf_config: PerfConfig,
    progress: Callable[[str], None] | None = None,
) -> list[ModelBenchmark]:
    supported_models = [model for model in models if model.name == "parakeet-ctc"]
    if not supported_models:
        return []
    try:
        import torch
        from transformers import AutoModelForCTC, AutoProcessor
    except Exception as exc:  # noqa: BLE001
        return [
            ModelBenchmark(
                model_name=model.name,
                model_size=model.size,
                rtfx_mean=None,
                rtfx_stdev=None,
                bench_seconds=None,
                notes=f"parakeet transformers unavailable: {exc}",
            )
            for model in models
        ]

    cuda_ok, cuda_err = cuda_is_usable()
    prefer_cuda = cuda_ok
    prefer_mps = torch.backends.mps.is_available()
    device = torch.device("cpu")
    dtype = torch.float32
    device_note = "device: cpu"
    if prefer_cuda:
        device = torch.device("cuda:0")
        dtype = torch.float16
        device_note = "device: cuda"
    elif prefer_mps:
        device = torch.device("mps")
        dtype = torch.float32
        device_note = "device: mps"
    elif cuda_err and torch.cuda.is_available():
        device_note = f"device: cpu (cuda unavailable: {cuda_err})"

    audio = _load_wav_16k_mono(str(sample.audio_path))

    results: list[ModelBenchmark] = []

    for model in supported_models:
        model_id = _model_id(model)
        try:
            processor = AutoProcessor.from_pretrained(model_id)
            try:
                asr_model = AutoModelForCTC.from_pretrained(
                    model_id, dtype=dtype
                ).to(device)
            except Exception:  # noqa: BLE001
                device = torch.device("cpu")
                dtype = torch.float32
                device_note = "device: cpu (cuda failed)"
                asr_model = AutoModelForCTC.from_pretrained(
                    model_id, dtype=dtype
                ).to(device)
            asr_model.eval()
            sampling_rate = processor.feature_extractor.sampling_rate
            hop_length = processor.feature_extractor.hop_length
            max_len = None
            encoder = getattr(asr_model, "encoder", None)
            encoder_config = getattr(encoder, "config", None)
            if encoder_config is not None:
                max_len = getattr(encoder_config, "max_position_embeddings", None)
            if max_len is None:
                for attr in (
                    "max_position_embeddings",
                    "max_source_positions",
                    "max_input_length",
                    "max_sequence_length",
                ):
                    max_len = getattr(asr_model.config, attr, None)
                    if max_len is not None:
                        break
            max_samples = None
            if max_len is not None and hop_length:
                max_samples = int(max_len * hop_length)

            def run_segment(segment: np.ndarray) -> list[str]:
                inputs = processor(
                    segment,
                    sampling_rate=sampling_rate,
                    return_tensors="pt",
                    padding="longest",
                )
                inputs = inputs.to(device=device, dtype=dtype)
                with torch.inference_mode():
                    predicted_ids = asr_model.generate(**inputs)
                return processor.batch_decode(predicted_ids, skip_special_tokens=True)

            def run_once() -> None:
                if max_samples is not None and audio.shape[0] > max_samples:
                    decoded = []
                    for start in range(0, audio.shape[0], max_samples):
                        end = start + max_samples
                        segment = audio[start:end]
                        decoded.extend(run_segment(segment))
                    _ = " ".join(text.strip() for text in decoded if text)
                else:
                    _ = run_segment(audio)

            stats = measure_rtfx(
                name=f"parakeet:{model.size}",
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
                    notes=f"model: {model_id}; {device_note}",
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
                    notes=f"parakeet failed: {type(exc).__name__}: {exc}; {device_note}",
                )
            )
        if progress is not None:
            progress(f"parakeet {model.name} {model.size}")

    return results
