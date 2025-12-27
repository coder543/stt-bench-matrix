from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

from .audio import audio_duration_seconds


@dataclass(frozen=True)
class SampleSpec:
    name: str
    audio_path: Path
    transcript_path: Path | None
    duration_seconds: float


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _samples_dir() -> Path:
    return _repo_root() / "samples"


def _ensure_audio_from_video(video_path: Path, audio_path: Path) -> None:
    if audio_path.exists():
        return
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(audio_path),
        ],
        check=True,
    )


def default_sample() -> SampleSpec:
    samples_dir = _samples_dir()
    video = samples_dir / "President Kennedy's Speech at Rice University [WZyRbnpGyzQ].webm"
    audio = samples_dir / "jfk_rice_16k.wav"
    transcript = samples_dir / "jfk_rice_16k.txt"

    if video.exists():
        _ensure_audio_from_video(video, audio)

    if not audio.exists():
        raise FileNotFoundError(
            "Sample audio not found. Place a sample in samples/ or provide a custom audio file."
        )

    duration_seconds = audio_duration_seconds(audio)
    transcript_path = transcript if transcript.exists() else None

    return SampleSpec(
        name="jfk_rice",
        audio_path=audio,
        transcript_path=transcript_path,
        duration_seconds=duration_seconds,
    )


def sample_from_path(path: Path) -> SampleSpec:
    if not path.exists():
        raise FileNotFoundError(f"Sample audio not found: {path}")
    duration_seconds = audio_duration_seconds(path)
    return SampleSpec(
        name=path.stem,
        audio_path=path,
        transcript_path=None,
        duration_seconds=duration_seconds,
    )
