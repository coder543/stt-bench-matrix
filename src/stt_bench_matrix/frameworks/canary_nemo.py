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
class CanaryNemoFramework:
    info: FrameworkInfo = FrameworkInfo(
        name="canary-nemo",
        description="NVIDIA Canary via NeMo EncDecMultiTaskModel",
        supports_whisper=False,
        supports_parakeet=False,
        supports_canary=True,
        supports_moonshine=False,
        supports_granite=False,
    )

    def is_supported(self, host: HostInfo) -> bool:
        return host.is_linux or host.is_macos


def _model_id(spec: ModelSpec) -> str:
    if spec.size == "1b-flash":
        return "nvidia/canary-1b-flash"
    if spec.size == "180m-flash":
        return "nvidia/canary-180m-flash"
    if spec.size == "1b-v2":
        return "nvidia/canary-1b-v2"
    if spec.size == "qwen-2.5b":
        return "nvidia/canary-qwen-2.5b"
    return spec.size


def _model_type(spec: ModelSpec) -> str | None:
    if spec.size == "qwen-2.5b":
        return "salm"
    return None


def benchmark_canary_models(
    sample: SampleSpec,
    models: list[ModelSpec],
    perf_config: PerfConfig,
    progress: Callable[[str], None] | None = None,
    on_result: Callable[[ModelBenchmark], None] | None = None,
) -> list[ModelBenchmark]:
    return benchmark_nemo_models(
        task="canary",
        sample=sample,
        models=models,
        perf_config=perf_config,
        model_id_fn=_model_id,
        model_type_fn=_model_type,
        chunk_seconds=20.0,
        progress=progress,
        on_result=on_result,
    )
