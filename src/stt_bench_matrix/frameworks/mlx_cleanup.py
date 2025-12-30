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
        metal = getattr(mx, "metal", None)
        if metal is not None:
            clear_cache = getattr(metal, "clear_cache", None)
            if callable(clear_cache):
                clear_cache()
            reset_peak = getattr(metal, "reset_peak_memory", None)
            if callable(reset_peak):
                reset_peak()
    except Exception:
        pass
    finally:
        gc.collect()
