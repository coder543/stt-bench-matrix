from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable

from ..bench.perf import PerfConfig, measure_rtfx
from ..bench.samples import SampleSpec
from ..bench.types import ModelBenchmark, RunResult
from ..models.registry import ModelSpec
from ..platforms.cuda import cuda_is_usable
from ..platforms.detect import HostInfo
from .base import FrameworkInfo


_MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B"


@dataclass(frozen=True)
class LfmAudioFramework:
    info: FrameworkInfo = FrameworkInfo(
        name="lfm-audio",
        description="LiquidAI LFM2.5 Audio via liquid-audio",
        supports_whisper=False,
        supports_parakeet=False,
        supports_canary=False,
        supports_moonshine=False,
        supports_granite=False,
        supports_nemotron=False,
        supports_gemma=False,
        supports_lfm=True,
    )

    def is_supported(self, host: HostInfo) -> bool:
        return host.is_linux or host.is_macos


def _model_id(size: str) -> str | None:
    if size == "1.5b":
        return _MODEL_ID
    return None


def _load_audio(torchaudio, path: str):
    audio, sr = torchaudio.load(path)
    if audio.dim() > 1 and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
        sr = 16000
    return audio, sr


def _decode_tokens(processor, tokens) -> str:
    if hasattr(tokens, "detach"):
        tokens = tokens.detach().cpu()
    text = processor.text.decode(tokens)
    return text


def benchmark_lfm_models(
    sample: SampleSpec,
    models: list[ModelSpec],
    perf_config: PerfConfig,
    warmup_sample: SampleSpec | None = None,
    progress: Callable[[str], None] | None = None,
    on_result: Callable[[ModelBenchmark], None] | None = None,
) -> list[ModelBenchmark]:
    try:
        import importlib
        import torch
        import torchaudio
        liquid_audio = importlib.import_module("liquid_audio")
        LFM2AudioModel = getattr(liquid_audio, "LFM2AudioModel")
        LFM2AudioProcessor = getattr(liquid_audio, "LFM2AudioProcessor")
        ChatState = getattr(liquid_audio, "ChatState")
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
                notes=f"lfm-audio unavailable: {exc}",
                transcript=None,
                wer=None,
                wer_stdev=None,
                runs=[],
            )
            for model in models
        ]

    cuda_ok, cuda_err = cuda_is_usable()
    device = torch.device("cuda") if cuda_ok else torch.device("cpu")
    device_note = "cuda" if cuda_ok else "cpu"
    if cuda_err and torch.cuda.is_available():
        device_note = f"cpu (cuda unavailable: {cuda_err})"

    max_new_tokens_env = os.getenv("STT_BENCH_LFM_MAX_NEW_TOKENS", "4096")
    try:
        max_new_tokens = int(max_new_tokens_env)
    except ValueError:
        max_new_tokens = 4096
    system_prompt = os.getenv("STT_BENCH_LFM_SYSTEM_PROMPT", "Perform ASR.")
    chunk_seconds_env = os.getenv("STT_BENCH_LFM_CHUNK_SECS", "20")
    overlap_seconds_env = os.getenv("STT_BENCH_LFM_CHUNK_OVERLAP_SECS", "0")
    try:
        chunk_seconds = float(chunk_seconds_env)
    except ValueError:
        chunk_seconds = 0.0
    try:
        overlap_seconds = float(overlap_seconds_env)
    except ValueError:
        overlap_seconds = 0.0

    results: list[ModelBenchmark] = []

    for model in models:
        model_id = _model_id(model.size)
        if model_id is None:
            results.append(
                ModelBenchmark(
                    model_name=model.name,
                    model_size=model.size,
                    model_variant=model.variant,
                    model_id=None,
                    rtfx_mean=None,
                    rtfx_stdev=None,
                    bench_seconds=None,
                    device=None,
                    notes="lfm-audio only supports 1.5b for now",
                    transcript=None,
                    wer=None,
                    wer_stdev=None,
                    runs=[],
                )
            )
            if on_result is not None:
                on_result(results[-1])
            if progress is not None:
                progress(f"lfm-audio {model.name} {model.size}")
            continue

        try:
            processor = LFM2AudioProcessor.from_pretrained(model_id)
            processor = processor.to(device)
            processor.eval()
            lfm_model = LFM2AudioModel.from_pretrained(model_id)
            lfm_model = lfm_model.to(device)
            lfm_model.eval()
        except Exception as exc:  # noqa: BLE001
            results.append(
                ModelBenchmark(
                    model_name=model.name,
                    model_size=model.size,
                    model_variant=model.variant,
                    model_id=model_id,
                    rtfx_mean=None,
                    rtfx_stdev=None,
                    bench_seconds=None,
                    device=None,
                    notes=f"lfm-audio failed to load: {exc}",
                    transcript=None,
                    wer=None,
                    wer_stdev=None,
                    runs=[],
                )
            )
            if on_result is not None:
                on_result(results[-1])
            if progress is not None:
                progress(f"lfm-audio {model.name} {model.size}")
            continue

        def _run_chunk(audio, sr: int) -> str | None:
            chat_state = ChatState(processor)
            chat_state.new_turn("system")
            chat_state.add_text(system_prompt)
            chat_state.end_turn()
            chat_state.new_turn("user")
            chat_state.add_audio(audio, sampling_rate=sr)
            chat_state.end_turn()
            chat_state.new_turn("assistant")
            pieces: list[str] = []
            with torch.inference_mode():
                for token in lfm_model.generate_sequential(
                    **chat_state,
                    max_new_tokens=max_new_tokens,
                ):
                    if getattr(token, "numel", lambda: 0)() == 1:
                        pieces.append(_decode_tokens(processor, token))
            transcript = "".join(pieces).strip()
            return transcript or None

        def _run_audio(path: str) -> str | None:
            audio, sr = _load_audio(torchaudio, path)
            if chunk_seconds <= 0:
                return _run_chunk(audio, sr)
            chunk_len = int(chunk_seconds * sr)
            if chunk_len <= 0 or audio.shape[-1] <= chunk_len:
                return _run_chunk(audio, sr)
            overlap_len = max(0, int(overlap_seconds * sr))
            if overlap_len >= chunk_len:
                overlap_len = 0
            step = max(1, chunk_len - overlap_len)
            total = audio.shape[-1]
            transcripts: list[str] = []
            start = 0
            while start < total:
                end = min(start + chunk_len, total)
                chunk = audio[..., start:end]
                chunk_text = _run_chunk(chunk, sr)
                if chunk_text:
                    transcripts.append(chunk_text)
                start += step
            transcript = " ".join(transcripts).strip()
            return transcript or None

        def run_once() -> str | None:
            return _run_audio(str(sample.audio_path))

        warmup_run = None
        if warmup_sample is not None:
            warmup_run = lambda: _run_audio(str(warmup_sample.audio_path))

        stats = measure_rtfx(
            name=f"lfm-audio {model.size}",
            sample=sample,
            run_once=run_once,
            warmup_run_once=warmup_run,
            config=perf_config,
            progress_label=f"lfm-audio {model.name} {model.size}",
        )
        runs = [
            RunResult(rtfx=rtfx, seconds=elapsed, wer=None, transcript=transcript)
            for rtfx, elapsed, transcript in zip(
                stats.rtfx_values,
                stats.elapsed_values,
                stats.transcripts,
            )
        ]
        result = ModelBenchmark(
            model_name=model.name,
            model_size=model.size,
            model_variant=model.variant,
            model_id=model_id,
            rtfx_mean=stats.rtfx_mean,
            rtfx_stdev=stats.rtfx_stdev,
            bench_seconds=stats.wall_seconds,
            device=device_note,
            notes=None,
            transcript=stats.transcripts[-1] if stats.transcripts else None,
            wer=None,
            wer_stdev=None,
            runs=runs,
        )
        results.append(result)
        if on_result is not None:
            on_result(result)
        if progress is not None:
            progress(f"lfm-audio {model.name} {model.size}")

    return results
