from __future__ import annotations

from contextlib import nullcontext
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
class ParakeetTransformersFramework:
    info: FrameworkInfo = FrameworkInfo(
        name="parakeet-transformers",
        description="NVIDIA Parakeet via Transformers AutoModelForCTC",
        supports_whisper=False,
        supports_parakeet=True,
        supports_canary=False,
        supports_moonshine=False,
        supports_granite=False,
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
    warmup_sample: SampleSpec | None = None,
    progress: Callable[[str], None] | None = None,
    on_result: Callable[[ModelBenchmark], None] | None = None,
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
                model_variant=model.variant,
                model_id=None,
                rtfx_mean=None,
                rtfx_stdev=None,
                bench_seconds=None,
                device=None,
                notes=f"parakeet transformers unavailable: {exc}",
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
    use_math_sdp_only = False
    if prefer_cuda:
        try:
            capability = torch.cuda.get_device_capability(0)
        except Exception:
            capability = None
        if capability and (capability[0] >= 12):
            # Blackwell can trip flash-attn kernels built for sm80-sm100.
            use_math_sdp_only = True
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

    def _disable_sdp_flash() -> bool:
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            return True
        except Exception:
            return False

    def _is_flash_attention_error(exc: Exception) -> bool:
        msg = str(exc)
        if not msg:
            return False
        needles = (
            "fmha_cutlass",
            "sm80-sm100",
            "flash attention",
            "flash_attn",
            "flash_sdp",
        )
        lower = msg.lower()
        return any(needle in lower or needle in msg for needle in needles)

    for model in supported_models:
        model_id = _model_id(model)
        try:
            processor = AutoProcessor.from_pretrained(model_id)
            sdp_fallback = False
            try:
                asr_model = AutoModelForCTC.from_pretrained(
                    model_id, dtype=dtype
                ).to(device)
            except Exception:  # noqa: BLE001
                device = torch.device("cpu")
                dtype = torch.float32
                device_note = "cpu"
                asr_model = AutoModelForCTC.from_pretrained(
                    model_id, dtype=dtype
                ).to(device)
            asr_model.eval()
            dtype_note = None
            if prefer_cuda and model.size == "0.6b":
                if use_math_sdp_only:
                    dtype = torch.float32
                    dtype_note = "precision: fp32"
                    asr_model = AutoModelForCTC.from_pretrained(
                        model_id, dtype=dtype
                    ).to(device)
                    asr_model.eval()
            if use_math_sdp_only:
                try:
                    torch.backends.cuda.enable_flash_sdp(False)
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
                    torch.backends.cuda.enable_math_sdp(True)
                except Exception:
                    pass
            last_transcript: str | None = None
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
                nonlocal sdp_fallback
                inputs = processor(
                    segment,
                    sampling_rate=sampling_rate,
                    return_tensors="pt",
                    padding="longest",
                )
                inputs = inputs.to(device=device, dtype=dtype)
                sdp_cm = nullcontext()
                if use_math_sdp_only:
                    attn = getattr(torch.nn, "attention", None)
                    sdpa_kernel = getattr(attn, "sdpa_kernel", None) if attn else None
                    sdp_backend = getattr(attn, "SDPBackend", None) if attn else None
                    if sdpa_kernel is not None and sdp_backend is not None:
                        sdp_cm = sdpa_kernel([sdp_backend.MATH])
                    elif hasattr(torch.backends.cuda, "sdp_kernel"):
                        sdp_cm = torch.backends.cuda.sdp_kernel(
                            enable_flash=False,
                            enable_mem_efficient=False,
                            enable_math=True,
                        )
                with torch.inference_mode(), sdp_cm:
                    try:
                        logits = asr_model(**inputs).logits
                    except RuntimeError as exc:
                        if (
                            prefer_cuda
                            and not sdp_fallback
                            and _is_flash_attention_error(exc)
                            and _disable_sdp_flash()
                        ):
                            sdp_fallback = True
                            logits = asr_model(**inputs).logits
                        else:
                            raise
                    predicted_ids = torch.argmax(logits, dim=-1)
                return processor.batch_decode(predicted_ids, skip_special_tokens=True)

            def run_once(audio_input=audio) -> str | None:
                if max_samples is not None and audio_input.shape[0] > max_samples:
                    decoded = []
                    for start in range(0, audio_input.shape[0], max_samples):
                        end = start + max_samples
                        segment = audio_input[start:end]
                        decoded.extend(run_segment(segment))
                    return " ".join(text.strip() for text in decoded if text).strip() or None
                else:
                    decoded = run_segment(audio_input)
                    return " ".join(text.strip() for text in decoded if text).strip() or None

            warmup_run_once = None
            if warmup_audio is not None:
                warmup_run_once = lambda: run_once(warmup_audio)
            stats = measure_rtfx(
                name=f"parakeet:{model.size}",
                sample=sample,
                run_once=run_once,
                warmup_run_once=warmup_run_once,
                config=perf_config,
                progress_label=f"parakeet {model.name} {model.size}",
            )
            note_parts: list[str] = []
            if sdp_fallback:
                note_parts.append("sdp: math (flash fallback)")
            if dtype_note:
                note_parts.append(dtype_note)
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
                    notes="; ".join(note_parts) if note_parts else None,
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
            note = f"parakeet failed: {type(exc).__name__}: {exc}"
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
            progress(f"parakeet {model.name} {model.size}")

    return results
