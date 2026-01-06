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
class ParakeetNemoFramework:
    info: FrameworkInfo = FrameworkInfo(
        name="parakeet-nemo",
        description="NVIDIA Parakeet via NeMo EncDecCTCModelBPE",
        supports_whisper=False,
        supports_parakeet=True,
        supports_canary=False,
        supports_moonshine=False,
        supports_granite=False,
    )

    def is_supported(self, host: HostInfo) -> bool:
        return host.is_linux or host.is_macos


def _model_id(spec: ModelSpec) -> str:
    if spec.name == "parakeet-ctc":
        if spec.size == "1.1b":
            return "nvidia/parakeet-ctc-1.1b"
        if spec.size == "0.6b":
            return "nvidia/parakeet-ctc-0.6b"
        return f"nvidia/parakeet-ctc-{spec.size}"
    if spec.name == "parakeet-rnnt":
        if spec.size == "1.1b":
            return "nvidia/parakeet-rnnt-1.1b"
        if spec.size == "0.6b":
            return "nvidia/parakeet-rnnt-0.6b"
        return f"nvidia/parakeet-rnnt-{spec.size}"
    if spec.name == "parakeet-tdt":
        if spec.size == "1.1b":
            return "nvidia/parakeet-tdt-1.1b"
        if spec.size == "0.6b-v3":
            return "nvidia/parakeet-tdt-0.6b-v3"
        return f"nvidia/parakeet-tdt-{spec.size}"
    if spec.name == "parakeet-tdt-ctc":
        if spec.size == "110m":
            return "nvidia/parakeet-tdt_ctc-110m"
        if spec.size == "1.1b":
            return "nvidia/parakeet-tdt_ctc-1.1b"
        return f"nvidia/parakeet-tdt_ctc-{spec.size}"
    if spec.name == "parakeet-realtime-eou":
        return "nvidia/parakeet_realtime_eou_120m-v1"
    return spec.size


def _model_type(spec: ModelSpec) -> str:
    if spec.name == "parakeet-ctc":
        return "ctc"
    if spec.name == "parakeet-rnnt":
        return "rnnt"
    if spec.name == "parakeet-tdt":
        return "tdt"
    if spec.name == "parakeet-tdt-ctc":
        return "tdt-ctc"
    if spec.name == "parakeet-realtime-eou":
        return "rnnt"
    raise ValueError(f"unknown parakeet flavor: {spec.name}")


def _decode_mode(spec: ModelSpec) -> str | None:
    if spec.name == "parakeet-tdt-ctc":
        return spec.variant or "tdt"
    return None


def benchmark_parakeet_models(
    sample: SampleSpec,
    models: list[ModelSpec],
    perf_config: PerfConfig,
    warmup_sample: SampleSpec | None = None,
    progress: Callable[[str], None] | None = None,
    on_result: Callable[[ModelBenchmark], None] | None = None,
) -> list[ModelBenchmark]:
    return benchmark_nemo_models(
        task="parakeet",
        sample=sample,
        warmup_sample=warmup_sample,
        models=models,
        perf_config=perf_config,
        model_id_fn=_model_id,
        model_type_fn=_model_type,
        decode_mode_fn=_decode_mode,
        progress=progress,
        on_result=on_result,
    )
