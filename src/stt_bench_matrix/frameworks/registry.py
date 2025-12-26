from __future__ import annotations

from .base import Framework
from .lightning_whisper_mlx import LightningWhisperMlxFramework
from .whisper_mlx import WhisperMlxFramework
from .whisper_cpp import WhisperCppFramework
from .transformers_whisper import TransformersWhisperFramework
from .parakeet_transformers import ParakeetTransformersFramework
from .canary_nemo import CanaryNemoFramework
from .parakeet_nemo import ParakeetNemoFramework
from .faster_whisper import FasterWhisperFramework
from .whisperx import WhisperXFramework


def all_frameworks() -> list[Framework]:
    return [
        LightningWhisperMlxFramework(),
        WhisperMlxFramework(),
        WhisperCppFramework(),
        TransformersWhisperFramework(),
        FasterWhisperFramework(),
        WhisperXFramework(),
        ParakeetTransformersFramework(),
        ParakeetNemoFramework(),
        CanaryNemoFramework(),
    ]
