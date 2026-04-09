from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable

from ..bench.perf import PerfConfig, measure_rtfx
from ..bench.samples import SampleSpec
from ..bench.types import ModelBenchmark, RunResult
from ..models.registry import ModelSpec
from ..platforms.detect import HostInfo
from ..platforms.cuda import cuda_is_usable
from .base import FrameworkInfo


DEFAULT_MODEL_ID = "microsoft/VibeVoice-ASR-HF"
DEFAULT_PROMPT = os.getenv("STT_BENCH_VIBEVOICE_PROMPT")


@dataclass(frozen=True)
class VibeVoiceTransformersFramework:
    info: FrameworkInfo = FrameworkInfo(
        name="vibevoice-transformers",
        description="Microsoft VibeVoice ASR via Transformers",
        supports_whisper=False,
        supports_parakeet=False,
        supports_canary=False,
        supports_moonshine=False,
        supports_granite=False,
        supports_vibevoice=True,
    )

    def is_supported(self, host: HostInfo) -> bool:
        return host.is_macos or host.is_linux


def _model_id(model: ModelSpec) -> str:
    if model.name != "vibevoice-asr":
        raise ValueError("vibevoice-transformers only supports vibevoice-asr models")
    return DEFAULT_MODEL_ID


def _int_env(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _normalize_text(decoded: object) -> str:
    if isinstance(decoded, str):
        return decoded.strip()
    if isinstance(decoded, list):
        cleaned = [str(item).strip() for item in decoded if str(item).strip()]
        return " ".join(cleaned).strip()
    return str(decoded or "").strip()


def benchmark_vibevoice_models(
    sample: SampleSpec,
    models: list[ModelSpec],
    perf_config: PerfConfig,
    language: str,
    warmup_sample: SampleSpec | None = None,
    progress: Callable[[str], None] | None = None,
    on_result: Callable[[ModelBenchmark], None] | None = None,
) -> list[ModelBenchmark]:
    del language
    if not models:
        return []
    _ = progress
    try:
        import torch
        from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration
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
                notes=f"vibevoice unavailable: {exc}",
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
        dtype = torch.bfloat16
        device_note = "cuda"
    elif prefer_mps:
        device = torch.device("mps")
        dtype = torch.float16
        device_note = "mps"
    elif cuda_err and torch.cuda.is_available():
        device_note = "cpu"

    prompt = DEFAULT_PROMPT.strip() if DEFAULT_PROMPT else None
    max_new_tokens = _int_env("STT_BENCH_VIBEVOICE_MAX_NEW_TOKENS")
    tokenizer_chunk_size = _int_env("STT_BENCH_VIBEVOICE_TOKENIZER_CHUNK_SIZE")

    results: list[ModelBenchmark] = []

    for model in models:
        model_id = _model_id(model)
        try:
            processor = AutoProcessor.from_pretrained(model_id)
            try:
                asr_model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                ).to(device)
            except Exception:  # noqa: BLE001
                device = torch.device("cpu")
                dtype = torch.float32
                device_note = "cpu"
                asr_model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                ).to(device)
            asr_model.eval()

            def run_once(audio_path: str = str(sample.audio_path)) -> str | None:
                inputs = processor.apply_transcription_request(
                    audio=audio_path,
                    prompt=prompt,
                ).to(device=device, dtype=dtype)
                generate_kwargs: dict[str, int] = {}
                if max_new_tokens is not None:
                    generate_kwargs["max_new_tokens"] = max_new_tokens
                if tokenizer_chunk_size is not None:
                    generate_kwargs["tokenizer_chunk_size"] = tokenizer_chunk_size
                with torch.inference_mode():
                    output_ids = asr_model.generate(**inputs, **generate_kwargs)
                generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
                decoded = processor.decode(
                    generated_ids,
                    return_format="transcription_only",
                )
                text = _normalize_text(decoded[0] if decoded else None)
                return text or None

            warmup_run_once = None
            if warmup_sample is not None:
                warmup_run_once = lambda: run_once(str(warmup_sample.audio_path))
            stats = measure_rtfx(
                name=f"vibevoice:{model.size}",
                sample=sample,
                run_once=run_once,
                warmup_run_once=warmup_run_once,
                config=perf_config,
                progress_label=f"vibevoice {model.name} {model.size}",
            )
            note_parts = ["return_format:transcription_only"]
            if prompt:
                note_parts.append(f"prompt:{prompt}")
            if max_new_tokens is not None:
                note_parts.append(f"max_new_tokens:{max_new_tokens}")
            if tokenizer_chunk_size is not None:
                note_parts.append(f"tokenizer_chunk_size:{tokenizer_chunk_size}")
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
            note = f"vibevoice failed: {type(exc).__name__}: {exc}"
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

    return results
