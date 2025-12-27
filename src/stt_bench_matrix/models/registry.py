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


def canary_models() -> list[ModelSpec]:
    return [
        ModelSpec(name="canary", size="180m-flash", family="canary"),
        ModelSpec(name="canary", size="1b-flash", family="canary"),
        ModelSpec(name="canary", size="1b-v2", family="canary"),
    ]


def moonshine_models() -> list[ModelSpec]:
    return [
        ModelSpec(name="moonshine", size="tiny", family="moonshine"),
        ModelSpec(name="moonshine", size="base", family="moonshine"),
    ]


def granite_models() -> list[ModelSpec]:
    return [
        ModelSpec(name="granite-speech-3.3", size="2b", family="granite"),
        ModelSpec(name="granite-speech-3.3", size="8b", family="granite"),
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
