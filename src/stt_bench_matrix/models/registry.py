from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    size: str
    family: str
    variant: str | None = None


def whisper_models() -> list[ModelSpec]:
    return [
        ModelSpec(name="whisper", size="tiny", family="whisper"),
        ModelSpec(name="whisper", size="base", family="whisper"),
        ModelSpec(name="whisper", size="small", family="whisper"),
        ModelSpec(name="whisper", size="medium", family="whisper"),
        ModelSpec(name="whisper", size="large-v3", family="whisper"),
    ]


def whisper_optional_models() -> list[ModelSpec]:
    return [
        ModelSpec(name="whisper", size="tiny.en", family="whisper"),
        ModelSpec(name="whisper", size="base.en", family="whisper"),
        ModelSpec(name="whisper", size="small.en", family="whisper"),
        ModelSpec(name="whisper", size="medium.en", family="whisper"),
        ModelSpec(name="whisper", size="large-v1", family="whisper"),
        ModelSpec(name="whisper", size="large-v2", family="whisper"),
    ]


def canary_models() -> list[ModelSpec]:
    return [
        ModelSpec(name="canary", size="180m-flash", family="canary"),
        ModelSpec(name="canary", size="1b-flash", family="canary"),
        ModelSpec(name="canary", size="1b-v2", family="canary"),
    ]


def canary_optional_models() -> list[ModelSpec]:
    return [
        ModelSpec(name="canary", size="qwen-2.5b", family="canary"),
    ]


def moonshine_models() -> list[ModelSpec]:
    return [
        ModelSpec(name="moonshine", size="tiny", family="moonshine"),
        ModelSpec(name="moonshine", size="base", family="moonshine"),
    ]


def cohere_models() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="cohere-transcribe",
            size="03-2026",
            family="cohere",
        ),
    ]


def qwen3_asr_models() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="qwen3-asr",
            size="0.6b",
            family="qwen3-asr",
        ),
    ]


def qwen3_asr_optional_models() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="qwen3-asr",
            size="1.7b",
            family="qwen3-asr",
        ),
    ]


def vibevoice_models() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="vibevoice-asr",
            size="8b",
            family="vibevoice",
        ),
    ]


def voxtral_models() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="voxtral",
            size="mini-4b-realtime-2602",
            family="voxtral",
        ),
    ]


def apple_speech_models() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="speech-transcriber",
            size="default",
            family="apple-speech",
        ),
    ]


def nemotron_models() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="nemotron-speech-streaming",
            size="0.6b",
            family="nemotron",
        ),
    ]


def granite_models() -> list[ModelSpec]:
    return [
        ModelSpec(name="granite-speech-3.3", size="2b", family="granite"),
    ]


def granite_optional_models() -> list[ModelSpec]:
    return [
        ModelSpec(name="granite-speech-3.3", size="8b", family="granite"),
    ]


def gemma_models() -> list[ModelSpec]:
    return [
        ModelSpec(name="gemma-3n", size="e2b", family="gemma"),
        ModelSpec(name="gemma-3n", size="e4b", family="gemma"),
    ]


def lfm_models() -> list[ModelSpec]:
    return [
        ModelSpec(name="lfm2.5-audio", size="1.5b", family="lfm"),
    ]


def parakeet_models() -> list[ModelSpec]:
    return [
        ModelSpec(name="parakeet-ctc", size="0.6b", family="parakeet"),
        ModelSpec(name="parakeet-ctc", size="1.1b", family="parakeet"),
        ModelSpec(name="parakeet-rnnt", size="0.6b", family="parakeet"),
        ModelSpec(name="parakeet-rnnt", size="1.1b", family="parakeet"),
        ModelSpec(name="parakeet-tdt", size="0.6b-v3", family="parakeet"),
        ModelSpec(name="parakeet-tdt", size="1.1b", family="parakeet"),
        ModelSpec(name="parakeet-tdt-ctc", size="110m", family="parakeet", variant="tdt"),
        ModelSpec(name="parakeet-tdt-ctc", size="110m", family="parakeet", variant="ctc"),
        ModelSpec(name="parakeet-realtime-eou", size="120m-v1", family="parakeet"),
    ]
