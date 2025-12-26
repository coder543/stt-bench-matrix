from __future__ import annotations

from dataclasses import dataclass
import sys
import time


@dataclass
class ProgressTracker:
    total_steps: int
    completed: int = 0
    start_time: float = 0.0

    def start(self) -> None:
        self.start_time = time.perf_counter()

    def step(self, label: str) -> None:
        self.completed += 1
        elapsed = time.perf_counter() - self.start_time
        avg = elapsed / self.completed if self.completed else 0.0
        remaining = max(self.total_steps - self.completed, 0)
        eta = remaining * avg
        eta_str = _format_seconds(eta)
        sys.stderr.write(
            f"[{self.completed}/{self.total_steps}] {label} | ETA {eta_str}\n"
        )
        sys.stderr.flush()


def _format_seconds(seconds: float) -> str:
    total = int(round(seconds))
    mins, secs = divmod(total, 60)
    if mins:
        return f"{mins}m{secs:02d}s"
    return f"{secs}s"
