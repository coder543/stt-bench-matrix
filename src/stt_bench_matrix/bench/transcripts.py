from __future__ import annotations

from collections.abc import Mapping
from typing import cast


def extract_transcript(result: object) -> str | None:
    if result is None:
        return None
    if isinstance(result, str):
        text = result
    elif isinstance(result, Mapping):
        result_dict = cast(Mapping[str, object], result)
        text = (
            result_dict.get("text")
            or result_dict.get("transcript")
            or result_dict.get("transcription")
        )
        if text is None:
            segments = result_dict.get("segments")
            if isinstance(segments, list):
                parts = []
                for segment in segments:
                    if isinstance(segment, Mapping):
                        seg_map = cast(Mapping[str, object], segment)
                        piece = seg_map.get("text")
                    else:
                        piece = getattr(segment, "text", None)
                    if piece:
                        parts.append(str(piece))
                text = " ".join(parts) if parts else None
    else:
        text = getattr(result, "text", None) or getattr(result, "transcript", None)
        if text is None:
            segments = getattr(result, "segments", None)
            if isinstance(segments, list):
                parts = []
                for segment in segments:
                    piece = None
                    if isinstance(segment, Mapping):
                        seg_map = cast(Mapping[str, object], segment)
                        piece = seg_map.get("text")
                    else:
                        piece = getattr(segment, "text", None)
                    if piece:
                        parts.append(str(piece))
                text = " ".join(parts) if parts else None
    if text is None:
        return None
    text = str(text).strip()
    return text or None
