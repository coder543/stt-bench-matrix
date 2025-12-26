from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..platforms.detect import HostInfo


@dataclass(frozen=True)
class FrameworkInfo:
    name: str
    description: str
    supports_whisper: bool
    supports_parakeet: bool
    supports_canary: bool


class Framework(Protocol):
    info: FrameworkInfo

    def is_supported(self, host: HostInfo) -> bool: ...


@dataclass(frozen=True)
class UnsupportedReason:
    framework: str
    reason: str
