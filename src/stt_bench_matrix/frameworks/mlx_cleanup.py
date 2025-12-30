from __future__ import annotations

import gc


def cleanup_mlx() -> None:
    """Best-effort release of MLX caches and Python refs."""
    try:
        import mlx.core as mx  # type: ignore[import-not-found]
    except Exception:
        gc.collect()
        return

    try:
        clear_cache = getattr(mx, "clear_cache", None)
        if callable(clear_cache):
            clear_cache()
        reset_peak = getattr(mx, "reset_peak_memory", None)
        if callable(reset_peak):
            reset_peak()
        if not callable(clear_cache) or not callable(reset_peak):
            metal = getattr(mx, "metal", None)
            if metal is not None:
                if not callable(clear_cache):
                    metal_clear = getattr(metal, "clear_cache", None)
                    if callable(metal_clear):
                        metal_clear()
                if not callable(reset_peak):
                    metal_reset = getattr(metal, "reset_peak_memory", None)
                    if callable(metal_reset):
                        metal_reset()
    except Exception:
        pass
    finally:
        gc.collect()
