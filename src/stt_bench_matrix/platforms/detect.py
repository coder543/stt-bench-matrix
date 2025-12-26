from __future__ import annotations

from dataclasses import dataclass
import platform
import subprocess
import sys
from typing import Tuple


@dataclass(frozen=True)
class HostInfo:
    os: str
    arch: str
    machine: str
    cpu: str | None
    ram_bytes: int | None
    accelerator: str
    accelerator_memory_bytes: int | None
    is_macos: bool
    is_linux: bool
    is_apple_silicon: bool


def detect_host() -> HostInfo:
    os_name = sys.platform
    machine = platform.machine().lower()
    arch = platform.processor() or machine

    is_macos = os_name == "darwin"
    is_linux = os_name.startswith("linux")
    is_apple_silicon = is_macos and machine in {"arm64", "aarch64"}
    cpu = _cpu_name(os_name)
    ram_bytes = _ram_bytes(os_name)
    accelerator, accelerator_memory_bytes = _accelerator_info()

    return HostInfo(
        os=os_name,
        arch=arch,
        machine=machine,
        cpu=cpu,
        ram_bytes=ram_bytes,
        accelerator=accelerator,
        accelerator_memory_bytes=accelerator_memory_bytes,
        is_macos=is_macos,
        is_linux=is_linux,
        is_apple_silicon=is_apple_silicon,
    )


def _cpu_name(os_name: str) -> str | None:
    if os_name.startswith("linux"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            return None
    if os_name == "darwin":
        try:
            output = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            )
            return output.strip()
        except Exception:
            return None
    return platform.processor() or None


def _ram_bytes(os_name: str) -> int | None:
    if os_name.startswith("linux"):
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemTotal"):
                        parts = line.split()
                        if len(parts) >= 2:
                            return int(parts[1]) * 1024
        except OSError:
            return None
    if os_name == "darwin":
        try:
            output = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return int(output.strip())
        except Exception:
            return None
    return None


def _accelerator_info() -> Tuple[str, int | None]:
    try:
        import torch
    except Exception:
        return ("CPU", None)
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        return (name, int(props.total_memory))
    if torch.backends.mps.is_available():
        return ("Apple GPU (MPS)", None)
    return ("CPU", None)
