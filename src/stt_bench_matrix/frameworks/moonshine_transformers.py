from __future__ import annotations

from dataclasses import dataclass
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
class MoonshineTransformersFramework:
    info: FrameworkInfo = FrameworkInfo(
        name="moonshine-transformers",
        description="Moonshine via Transformers MoonshineForConditionalGeneration",
        supports_whisper=False,
        supports_parakeet=False,
        supports_canary=False,
        supports_moonshine=True,
        supports_granite=False,
    )

    def is_supported(self, host: HostInfo) -> bool:
        return host.is_macos or host.is_linux


def _model_id(size: str) -> str:
    if size == "tiny":
        return "UsefulSensors/moonshine-tiny"
    if size == "base":
        return "UsefulSensors/moonshine-base"
    return f"UsefulSensors/moonshine-{size}"


def _load_wav_16k_mono(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wav:
        if wav.getnchannels() != 1:
            raise ValueError("Expected mono WAV")
        if wav.getframerate() != 16000:
            raise ValueError("Expected 16kHz WAV")
        frames = wav.readframes(wav.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def benchmark_moonshine_models(
    sample: SampleSpec,
    models: list[ModelSpec],
    perf_config: PerfConfig,
    language: str,
    warmup_sample: SampleSpec | None = None,
    progress: Callable[[str], None] | None = None,
    on_result: Callable[[ModelBenchmark], None] | None = None,
) -> list[ModelBenchmark]:
    del language
    try:
        import torch
        from transformers import AutoProcessor, MoonshineForConditionalGeneration
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
                notes=f"moonshine unavailable: {exc}",
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
        model_id = _model_id(model.size)
        try:
            processor = AutoProcessor.from_pretrained(model_id)
            try:
                asr_model = MoonshineForConditionalGeneration.from_pretrained(
                    model_id, dtype=dtype
                ).to(device)
            except Exception:  # noqa: BLE001
                device = torch.device("cpu")
                dtype = torch.float32
                device_note = "cpu"
                asr_model = MoonshineForConditionalGeneration.from_pretrained(
                    model_id, dtype=dtype
                ).to(device)
            asr_model.eval()
            last_transcript: str | None = None

            max_output_tokens = (
                asr_model.generation_config.max_length
                if asr_model.generation_config.max_length
                else asr_model.config.max_position_embeddings
            )
            if max_output_tokens is None:
                max_output_tokens = 194
            max_chunk_seconds = max_output_tokens / 6.5
            chunk_seconds = max(10.0, min(30.0, max_chunk_seconds))
            chunk_samples = int(processor.feature_extractor.sampling_rate * chunk_seconds)

            def run_once(audio_input=audio) -> str | None:
                nonlocal last_transcript
                chunks: list[np.ndarray]
                if audio_input.shape[0] > chunk_samples:
                    chunks = [
                        audio_input[start : start + chunk_samples]
                        for start in range(0, audio_input.shape[0], chunk_samples)
                    ]
                else:
                    chunks = [audio_input]
                outputs: list[str] = []
                for chunk in chunks:
                    inputs = processor(
                        chunk,
                        sampling_rate=processor.feature_extractor.sampling_rate,
                        return_tensors="pt",
                    )
                    inputs = inputs.to(device=device, dtype=dtype)
                    token_limit_factor = (
                        6.5 / processor.feature_extractor.sampling_rate
                    )
                    seq_lens = inputs.attention_mask.sum(dim=-1)
                    max_length = int((seq_lens * token_limit_factor).max().item())
                    max_length = max(1, min(max_output_tokens, max_length))
                    with torch.inference_mode():
                        generated_ids = asr_model.generate(
                            **inputs, max_length=max_length, do_sample=False
                        )
                    decoded = processor.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )
                    if decoded and decoded[0].strip():
                        outputs.append(decoded[0].strip())
                return " ".join(outputs).strip() or None

            warmup_run_once = None
            if warmup_audio is not None:
                warmup_run_once = lambda: run_once(warmup_audio)
            stats = measure_rtfx(
                name=f"moonshine:{model.size}",
                sample=sample,
                run_once=run_once,
                warmup_run_once=warmup_run_once,
                config=perf_config,
                progress_label=f"moonshine {model.name} {model.size}",
            )
            last_transcript = (
                stats.transcripts[-1] if stats.transcripts else None
            )
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
                    notes=None,
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
            note = f"moonshine failed: {exc}"
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
            progress(f"moonshine {model.name} {model.size}")

    return results
