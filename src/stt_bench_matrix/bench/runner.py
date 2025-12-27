from __future__ import annotations

import time
from dataclasses import replace

from .types import BenchmarkResults, FrameworkBenchmark, ModelBenchmark
from .perf import PerfConfig
from .progress import ProgressTracker
from .samples import SampleSpec
from .wer import word_error_rate
from ..frameworks.base import Framework
from ..frameworks.registry import all_frameworks
from ..frameworks.whisper_mlx import benchmark_whisper_models, WhisperMlxFramework
from ..frameworks.lightning_whisper_mlx import (
    benchmark_whisper_models as benchmark_lightning_whisper_models,
    LightningWhisperMlxFramework,
)
from ..frameworks.whisper_cpp import (
    benchmark_whisper_models as benchmark_whisper_cpp_models,
    has_whisper_cli,
    WhisperCppFramework,
)
from ..frameworks.transformers_whisper import (
    benchmark_whisper_models as benchmark_transformers_models,
    TransformersWhisperFramework,
)
from ..frameworks.faster_whisper import (
    benchmark_whisper_models as benchmark_faster_whisper_models,
    FasterWhisperFramework,
)
from ..frameworks.whisperx import (
    benchmark_whisper_models as benchmark_whisperx_models,
    WhisperXFramework,
)
from ..frameworks.parakeet_transformers import (
    benchmark_parakeet_models,
    ParakeetTransformersFramework,
)
from ..frameworks.canary_nemo import (
    benchmark_canary_models,
    CanaryNemoFramework,
)
from ..frameworks.parakeet_nemo import (
    benchmark_parakeet_models as benchmark_parakeet_nemo_models,
    ParakeetNemoFramework,
)
from ..frameworks.moonshine_transformers import (
    benchmark_moonshine_models,
    MoonshineTransformersFramework,
)
from ..frameworks.granite_transformers import (
    benchmark_granite_models,
    GraniteTransformersFramework,
)
from ..models.registry import (
    whisper_models,
    parakeet_models,
    canary_models,
    moonshine_models,
    granite_models,
    ModelSpec,
)
from ..platforms.detect import HostInfo


def _benchmark_framework(
    framework: Framework,
    host: HostInfo,
    whisper_model_list: list[ModelSpec],
    canary_model_list: list[ModelSpec],
    parakeet_model_list: list[ModelSpec],
    moonshine_model_list: list[ModelSpec],
    granite_model_list: list[ModelSpec],
    use_cache: bool,
    perf_config: PerfConfig,
    sample: SampleSpec,
    language: str,
    progress: ProgressTracker | None,
) -> FrameworkBenchmark | None:
    if not framework.is_supported(host):
        return None

    progress_cb = progress.step if progress is not None else None

    if isinstance(framework, WhisperMlxFramework):
        models = benchmark_whisper_models(
            sample,
            whisper_model_list,
            use_cache=use_cache,
            perf_config=perf_config,
            language=language,
            progress=progress_cb,
        )
    elif isinstance(framework, LightningWhisperMlxFramework):
        models = benchmark_lightning_whisper_models(
            sample,
            whisper_model_list,
            perf_config=perf_config,
            language=language,
            progress=progress_cb,
        )
    elif isinstance(framework, WhisperCppFramework):
        models = benchmark_whisper_cpp_models(
            sample,
            whisper_model_list,
            perf_config=perf_config,
            language=language,
            progress=progress_cb,
        )
    elif isinstance(framework, TransformersWhisperFramework):
        models = benchmark_transformers_models(
            sample,
            whisper_model_list,
            perf_config=perf_config,
            language=language,
            progress=progress_cb,
        )
    elif isinstance(framework, FasterWhisperFramework):
        models = benchmark_faster_whisper_models(
            sample,
            whisper_model_list,
            perf_config=perf_config,
            language=language,
            progress=progress_cb,
        )
    elif isinstance(framework, WhisperXFramework):
        models = benchmark_whisperx_models(
            sample,
            whisper_model_list,
            perf_config=perf_config,
            language=language,
            progress=progress_cb,
        )
    elif isinstance(framework, ParakeetTransformersFramework):
        parakeet_list = parakeet_model_list
        models = benchmark_parakeet_models(
            sample,
            parakeet_list,
            perf_config=perf_config,
            progress=progress_cb,
        )
    elif isinstance(framework, ParakeetNemoFramework):
        parakeet_list = parakeet_model_list
        models = benchmark_parakeet_nemo_models(
            sample,
            parakeet_list,
            perf_config=perf_config,
            progress=progress_cb,
        )
    elif isinstance(framework, CanaryNemoFramework):
        canary_list = canary_model_list
        models = benchmark_canary_models(
            sample,
            canary_list,
            perf_config=perf_config,
            progress=progress_cb,
        )
    elif isinstance(framework, MoonshineTransformersFramework):
        moonshine_list = moonshine_model_list
        models = benchmark_moonshine_models(
            sample,
            moonshine_list,
            perf_config=perf_config,
            language=language,
            progress=progress_cb,
        )
    elif isinstance(framework, GraniteTransformersFramework):
        granite_list = granite_model_list
        models = benchmark_granite_models(
            sample,
            granite_list,
            perf_config=perf_config,
            language=language,
            progress=progress_cb,
        )
    else:
        # Placeholder: real benchmarking will be added once framework runners exist.
        models = [
            ModelBenchmark(
                model_name=model.name,
                model_size=model.size,
                model_variant=model.variant,
                rtfx_mean=None,
                rtfx_stdev=None,
                bench_seconds=None,
                device=None,
                notes="Benchmark not yet implemented",
                transcript=None,
                wer=None,
            )
            for model in whisper_model_list
        ]

    return FrameworkBenchmark(
        framework=framework.info.name,
        supported=True,
        reason=None,
        models=models,
    )


def run_benchmarks(
    host: HostInfo,
    use_cache: bool,
    sample: SampleSpec,
    language: str,
    quick: bool = False,
    quick_2: bool = False,
    parakeet_only: bool = False,
    heavy: bool = False,
    frameworks: set[str] | None = None,
) -> BenchmarkResults:
    start = time.perf_counter()
    _ = use_cache
    whisper_model_list = whisper_models()
    canary_model_list = canary_models()
    parakeet_model_list = parakeet_models()
    moonshine_model_list = moonshine_models()
    granite_model_list = granite_models() if heavy else []
    if quick_2:
        whisper_model_list = [
            ModelSpec(name="whisper", size="tiny", family="whisper"),
            ModelSpec(name="whisper", size="base", family="whisper"),
        ]
        canary_model_list = [canary_model_list[0]]
        moonshine_model_list = [
            ModelSpec(name="moonshine", size="tiny", family="moonshine"),
            ModelSpec(name="moonshine", size="base", family="moonshine"),
        ]
        parakeet_model_list = [
            ModelSpec(name="parakeet-ctc", size="0.6b", family="parakeet"),
            ModelSpec(name="parakeet-rnnt", size="0.6b", family="parakeet"),
            ModelSpec(name="parakeet-tdt", size="0.6b-v3", family="parakeet"),
            ModelSpec(name="parakeet-tdt-ctc", size="110m", family="parakeet", variant="tdt"),
            ModelSpec(name="parakeet-tdt-ctc", size="110m", family="parakeet", variant="ctc"),
            ModelSpec(name="parakeet-realtime-eou", size="120m-v1", family="parakeet"),
        ]
        granite_model_list = []
        perf_config = PerfConfig(warmups=0, runs=2)
    elif quick:
        whisper_model_list = [ModelSpec(name="whisper", size="tiny", family="whisper")]
        canary_model_list = [canary_model_list[0]]
        moonshine_model_list = [
            ModelSpec(name="moonshine", size="tiny", family="moonshine"),
            ModelSpec(name="moonshine", size="base", family="moonshine"),
        ]
        parakeet_model_list = [
            ModelSpec(name="parakeet-ctc", size="0.6b", family="parakeet"),
            ModelSpec(name="parakeet-rnnt", size="0.6b", family="parakeet"),
            ModelSpec(name="parakeet-tdt", size="0.6b-v3", family="parakeet"),
            ModelSpec(name="parakeet-tdt-ctc", size="110m", family="parakeet", variant="tdt"),
            ModelSpec(name="parakeet-tdt-ctc", size="110m", family="parakeet", variant="ctc"),
            ModelSpec(name="parakeet-realtime-eou", size="120m-v1", family="parakeet"),
        ]
        granite_model_list = []
        perf_config = PerfConfig(warmups=0, runs=2)
    else:
        perf_config = PerfConfig(warmups=1, runs=3)
    frameworks_to_run: list[Framework] = []
    for framework in all_frameworks():
        if parakeet_only and not framework.info.supports_parakeet:
            continue
        if frameworks is not None and framework.info.name not in frameworks:
            continue
        if not framework.is_supported(host):
            continue
        if framework.info.supports_granite and not granite_model_list:
            continue
        if framework.info.supports_moonshine and not moonshine_model_list:
            continue
        if isinstance(framework, WhisperCppFramework) and not has_whisper_cli():
            continue
        frameworks_to_run.append(framework)
    total_steps = 0
    for framework in frameworks_to_run:
        if framework.info.supports_whisper:
            total_steps += len(whisper_model_list)
        if framework.info.supports_parakeet:
            total_steps += len(parakeet_model_list)
        if framework.info.supports_canary:
            total_steps += len(canary_model_list)
        if framework.info.supports_moonshine:
            total_steps += len(moonshine_model_list)
        if framework.info.supports_granite:
            total_steps += len(granite_model_list)
    progress = ProgressTracker(total_steps=total_steps)
    progress.start()
    framework_results: list[FrameworkBenchmark] = []
    for framework in frameworks_to_run:
        result = _benchmark_framework(
            framework,
            host,
            whisper_model_list,
            canary_model_list,
            parakeet_model_list,
            moonshine_model_list,
            granite_model_list,
            use_cache=use_cache,
            perf_config=perf_config,
            sample=sample,
            language=language,
            progress=progress,
        )
        if result is not None:
            framework_results.append(result)
    if sample.transcript_path and sample.transcript_path.exists():
        reference_text = sample.transcript_path.read_text(encoding="utf-8")
        updated_frameworks: list[FrameworkBenchmark] = []
        for framework in framework_results:
            updated_models: list[ModelBenchmark] = []
            for model in framework.models:
                wer = None
                if model.transcript:
                    wer = word_error_rate(reference_text, model.transcript)
                updated_models.append(replace(model, wer=wer))
            updated_frameworks.append(replace(framework, models=updated_models))
        framework_results = updated_frameworks
    total_seconds = time.perf_counter() - start
    return BenchmarkResults(
        host=host,
        frameworks=framework_results,
        total_seconds=total_seconds,
        sample_name=sample.name,
        sample_path=str(sample.audio_path),
        language=language,
    )
