from __future__ import annotations

from dataclasses import dataclass
import os
import re
from pathlib import Path
from typing import Callable

from huggingface_hub import snapshot_download

from ..bench.perf import PerfConfig, measure_rtfx
from ..bench.samples import SampleSpec
from ..bench.types import ModelBenchmark, RunResult
from ..models.registry import ModelSpec
from ..platforms.detect import HostInfo
from ..platforms.cuda import cuda_is_usable
from .base import FrameworkInfo


@dataclass(frozen=True)
class Gemma3nOnnxFramework:
    info: FrameworkInfo = FrameworkInfo(
        name="gemma-3n-onnx",
        description="Google Gemma 3n via ONNX Runtime",
        supports_whisper=False,
        supports_parakeet=False,
        supports_canary=False,
        supports_moonshine=False,
        supports_granite=False,
        supports_nemotron=False,
        supports_gemma=True,
    )

    def is_supported(self, host: HostInfo) -> bool:
        return host.is_linux


_MODEL_IDS = {
    "e2b": "onnx-community/gemma-3n-E2B-it-ONNX",
}

_PROCESSOR_IDS = {
    "e2b": "google/gemma-3n-E2B-it",
    "e4b": "google/gemma-3n-E4B-it",
}


def _model_id(size: str) -> str | None:
    return _MODEL_IDS.get(size)


def _processor_id(size: str) -> str:
    return _PROCESSOR_IDS.get(size, size)


def _extract_text(decoded: str) -> str | None:
    marker = "<start_of_turn>model"
    if marker in decoded:
        decoded = decoded.split(marker, 1)[1]
    decoded = decoded.replace("<end_of_turn>", "")
    decoded = decoded.replace("<pad>", "").replace("<eos>", "")
    decoded = re.sub(r"\s+", " ", decoded).strip()
    return decoded or None


def _resolve_model_dir(model_id: str) -> Path:
    override = os.getenv("STT_BENCH_GEMMA_ONNX_DIR")
    if override:
        return Path(override)
    snapshot = snapshot_download(
        model_id,
        allow_patterns=["onnx/*", "*.json", "*.txt", "*.model", "tokenizer*"],
    )
    return Path(snapshot)


def _select_providers(ort, cuda_ok: bool) -> list[str]:
    available = ort.get_available_providers()
    env = os.getenv("STT_BENCH_GEMMA_ONNX_PROVIDERS")
    if env:
        requested = [p.strip() for p in env.split(",") if p.strip()]
    else:
        if cuda_ok and "CUDAExecutionProvider" in available:
            requested = ["CUDAExecutionProvider"]
        else:
            requested = ["CPUExecutionProvider"]
    providers = [p for p in requested if p in available]
    return providers or available


def benchmark_gemma_onnx_models(
    sample: SampleSpec,
    models: list[ModelSpec],
    perf_config: PerfConfig,
    language: str,
    warmup_sample: SampleSpec | None = None,
    progress: Callable[[str], None] | None = None,
    on_result: Callable[[ModelBenchmark], None] | None = None,
) -> list[ModelBenchmark]:
    try:
        import numpy as np
        import onnxruntime as ort
        import soundfile as sf
        from transformers import AutoConfig, AutoProcessor
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
                notes=f"gemma-3n onnx unavailable: {exc}",
                transcript=None,
                wer=None,
                wer_stdev=None,
                runs=[],
            )
            for model in models
        ]

    cuda_ok, _ = cuda_is_usable()
    default_providers = _select_providers(ort, cuda_ok)
    default_device = "cuda" if "CUDAExecutionProvider" in default_providers else "cpu"

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
                    notes="gemma-3n ONNX only supports e2b for now",
                    transcript=None,
                    wer=None,
                    wer_stdev=None,
                    runs=[],
                )
            )
            continue

        try:
            model_dir = _resolve_model_dir(model_id)
            processor_id = _processor_id(model.size)
            processor = AutoProcessor.from_pretrained(processor_id)
            config = AutoConfig.from_pretrained(processor_id)

            embed_path = model_dir / "onnx" / "embed_tokens_quantized.onnx"
            audio_path = model_dir / "onnx" / "audio_encoder.onnx"
            decoder_path = model_dir / "onnx" / "decoder_model_merged_q4.onnx"
            if not embed_path.exists() or not audio_path.exists() or not decoder_path.exists():
                raise FileNotFoundError("missing ONNX files under onnx/")

            model_providers = list(default_providers)
            device_note = default_device
            try:
                embed_session = ort.InferenceSession(str(embed_path), providers=model_providers)
                audio_session = ort.InferenceSession(str(audio_path), providers=model_providers)
                decoder_session = ort.InferenceSession(str(decoder_path), providers=model_providers)
            except Exception:
                if model_providers == ["CUDAExecutionProvider"] and "CPUExecutionProvider" in ort.get_available_providers():
                    model_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    device_note = "cuda"
                    embed_session = ort.InferenceSession(str(embed_path), providers=model_providers)
                    audio_session = ort.InferenceSession(str(audio_path), providers=model_providers)
                    decoder_session = ort.InferenceSession(str(decoder_path), providers=model_providers)
                else:
                    raise

            num_key_value_heads = config.text_config.num_key_value_heads
            head_dim = config.text_config.head_dim
            num_hidden_layers = config.text_config.num_hidden_layers
            eos_token_id = 106
            audio_token_id = config.audio_token_id

            def _load_audio(sample_spec: SampleSpec):
                audio, sr = sf.read(sample_spec.audio_path)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != 16000:
                    import librosa

                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                    sr = 16000
                return audio.astype("float32"), sr

            def _build_inputs(sample_spec: SampleSpec):
                audio, sr = _load_audio(sample_spec)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio"},
                            {"type": "text", "text": "Transcribe this audio into English."},
                        ],
                    }
                ]
                prompt = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                inputs = processor(
                    text=prompt,
                    audio=audio,
                    sampling_rate=sr,
                    add_special_tokens=False,
                    return_tensors="np",
                )
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                position_ids = np.cumsum(attention_mask, axis=-1) - 1
                input_features = inputs.get("input_features")
                input_features_mask = inputs.get("input_features_mask")
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "input_features": input_features,
                    "input_features_mask": input_features_mask,
                }

            def run_once(sample_spec: SampleSpec = sample) -> str | None:
                payload = _build_inputs(sample_spec)
                input_ids = payload["input_ids"]
                attention_mask = payload["attention_mask"]
                position_ids = payload["position_ids"]
                input_features = payload["input_features"]
                input_features_mask = payload["input_features_mask"]

                batch_size = input_ids.shape[0]
                past_key_values = {
                    f"past_key_values.{layer}.{kv}": np.zeros(
                        [batch_size, num_key_value_heads, 0, head_dim], dtype=np.float32
                    )
                    for layer in range(num_hidden_layers)
                    for kv in ("key", "value")
                }

                embed_input_name = embed_session.get_inputs()[0].name
                embed_outputs = embed_session.run(None, {embed_input_name: input_ids})
                inputs_embeds = embed_outputs[0]
                per_layer_inputs = embed_outputs[1] if len(embed_outputs) > 1 else None

                if input_features is not None and input_features_mask is not None:
                    audio_inputs = {}
                    for inp in audio_session.get_inputs():
                        if "mask" in inp.name:
                            audio_inputs[inp.name] = input_features_mask
                        else:
                            audio_inputs[inp.name] = input_features
                    audio_outputs = audio_session.run(None, audio_inputs)
                    audio_features = audio_outputs[0]
                    mask = (input_ids == audio_token_id).reshape(-1)
                    flat_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
                    flat_embeds[mask] = audio_features.reshape(-1, audio_features.shape[-1])
                    inputs_embeds = flat_embeds.reshape(inputs_embeds.shape)

                decoder_input_names = {inp.name for inp in decoder_session.get_inputs()}
                generated_tokens = []
                max_new_tokens = 256
                for _ in range(max_new_tokens):
                    decoder_inputs = {
                        "inputs_embeds": inputs_embeds,
                        "position_ids": position_ids,
                        **past_key_values,
                    }
                    if per_layer_inputs is not None and "per_layer_inputs" in decoder_input_names:
                        decoder_inputs["per_layer_inputs"] = per_layer_inputs
                    outputs = decoder_session.run(None, decoder_inputs)
                    logits = outputs[0]
                    present_key_values = outputs[1:]

                    input_ids = logits[:, -1].argmax(-1, keepdims=True)
                    attention_mask = np.ones_like(input_ids)
                    position_ids = position_ids[:, -1:] + 1
                    for idx, key in enumerate(past_key_values):
                        past_key_values[key] = present_key_values[idx]

                    generated_tokens.append(input_ids)
                    if (input_ids == eos_token_id).all():
                        break
                    inputs_embeds, per_layer_inputs = embed_session.run(
                        None, {embed_input_name: input_ids}
                    )

                if not generated_tokens:
                    return None
                generated = np.concatenate(generated_tokens, axis=-1)
                decoded = processor.batch_decode(generated, skip_special_tokens=True)[0]
                return _extract_text(decoded)

            warmup_run_once = None
            if warmup_sample is not None:
                warmup_run_once = lambda: run_once(warmup_sample)

            stats = measure_rtfx(
                name=f"gemma-3n:{model.size}",
                sample=sample,
                run_once=run_once,
                warmup_run_once=warmup_run_once,
                config=perf_config,
                progress_label=f"gemma-3n {model.name} {model.size} (onnx)",
            )
            last_transcript = stats.transcripts[-1] if stats.transcripts else None
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
            note = f"gemma-3n onnx failed: {exc}"
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
            progress(f"gemma-3n {model.name} {model.size} (onnx)")

    return results
