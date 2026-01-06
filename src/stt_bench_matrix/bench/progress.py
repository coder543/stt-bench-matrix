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
        line = (
            f"Overall {_bar(self.completed, self.total_steps)} "
            f"{self.completed}/{self.total_steps} | {label} | ETA {eta_str}"
        )
        _write_progress_line(line)


def _format_seconds(seconds: float) -> str:
    total = int(round(seconds))
    mins, secs = divmod(total, 60)
    if mins:
        return f"{mins}m{secs:02d}s"
    return f"{secs}s"


def _bar(current: int, total: int, width: int = 24) -> str:
    total = max(total, 1)
    current = min(max(current, 0), total)
    filled = int(width * current / total)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _write_progress_line(line: str) -> None:
    if sys.stderr.isatty():
        sys.stderr.write("\r\x1b[2K" + line)
        sys.stderr.flush()
        sys.stderr.write("\n")
    else:
        sys.stderr.write(line + "\n")
        sys.stderr.flush()


@dataclass
class RunProgress:
    label: str
    total: int
    current: int = 0
    _last_line: str | None = None

    def update(self, current: int, total: int) -> None:
        self.current = current
        self.total = max(total, 1)
        line = (
            f"Runs {_bar(self.current, self.total)} "
            f"{self.current}/{self.total} | {self.label}"
        )
        if sys.stderr.isatty():
            sys.stderr.write("\r\x1b[2K" + line)
            sys.stderr.flush()
        else:
            if line != self._last_line:
                sys.stderr.write(line + "\n")
                sys.stderr.flush()
        self._last_line = line

    def finish(self) -> None:
        if sys.stderr.isatty() and self._last_line:
            sys.stderr.write("\n")
            sys.stderr.flush()
