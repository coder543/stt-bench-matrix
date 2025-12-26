from __future__ import annotations

import ctypes
from typing import Tuple


def cuda_is_usable() -> Tuple[bool, str | None]:
    try:
        import torch
    except Exception:
        return (False, "torch not available")
    if not torch.cuda.is_available():
        return (False, "cuda not available")
    for lib in (
        "libcudnn_ops.so.9.1.0",
        "libcudnn_ops.so.9.1",
        "libcudnn_ops.so.9",
        "libcudnn_ops.so",
    ):
        try:
            ctypes.CDLL(lib)
            break
        except OSError:
            continue
    else:
        return (False, "cudnn not found")
    try:
        _ = torch.cuda.get_device_name(0)
    except Exception as exc:
        return (False, f"cuda init failed: {exc}")
    return (True, None)
