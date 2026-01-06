from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..bench.perf import PerfConfig
from ..bench.samples import SampleSpec
from ..bench.types import ModelBenchmark
from ..models.registry import ModelSpec
from ..platforms.detect import HostInfo
from .base import FrameworkInfo
from .nemo_subprocess import benchmark_nemo_models


@dataclass(frozen=True)
class NemotronNemoFramework:
    info: FrameworkInfo = FrameworkInfo(
        name="nemotron-nemo",
        description="NVIDIA Nemotron Speech via NeMo ASRModel",
        supports_whisper=False,
        supports_parakeet=False,
        supports_canary=False,
        supports_moonshine=False,
        supports_granite=False,
        supports_nemotron=True,
    )

    def is_supported(self, host: HostInfo) -> bool:
        return host.is_linux or host.is_macos


def _model_id(spec: ModelSpec) -> str:
    if spec.size == "0.6b" and (spec.variant == "streaming-en" or spec.variant == "en"):
        return "nvidia/nemotron-speech-streaming-en-0.6b"
    return spec.size


def _model_type(spec: ModelSpec) -> str | None:
    if spec.size == "0.6b":
        return "cache-aware-rnnt"
    return None


def benchmark_nemotron_models(
    sample: SampleSpec,
    models: list[ModelSpec],
    perf_config: PerfConfig,
    warmup_sample: SampleSpec | None = None,
    progress: Callable[[str], None] | None = None,
    on_result: Callable[[ModelBenchmark], None] | None = None,
) -> list[ModelBenchmark]:
    return benchmark_nemo_models(
        task="nemotron",
        sample=sample,
        warmup_sample=warmup_sample,
        models=models,
        perf_config=perf_config,
        model_id_fn=_model_id,
        model_type_fn=_model_type,
        chunk_seconds=20.0,
        progress=progress,
        on_result=on_result,
    )
