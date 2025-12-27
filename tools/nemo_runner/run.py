from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import statistics
import tempfile
import time
import wave
from contextlib import nullcontext
import re


def _wav_duration_seconds(path: str) -> float:
    with wave.open(path, "rb") as wav:
        if wav.getnchannels() != 1:
            raise ValueError("Expected mono WAV")
        if wav.getframerate() != 16000:
            raise ValueError("Expected 16kHz WAV")
        frames = wav.getnframes()
        rate = wav.getframerate()
    return frames / float(rate)


def _write_wav_chunks(path: str, chunk_seconds: float, out_dir: str) -> list[str]:
    chunk_paths: list[str] = []
    with wave.open(path, "rb") as wav:
        if wav.getnchannels() != 1:
            raise ValueError("Expected mono WAV")
        if wav.getframerate() != 16000:
            raise ValueError("Expected 16kHz WAV")
        rate = wav.getframerate()
        sampwidth = wav.getsampwidth()
        frames_per_chunk = int(rate * chunk_seconds)
        idx = 0
        while True:
            frames = wav.readframes(frames_per_chunk)
            if not frames:
                break
            chunk_path = os.path.join(out_dir, f"chunk_{idx}.wav")
            with wave.open(chunk_path, "wb") as out_wav:
                out_wav.setnchannels(1)
                out_wav.setsampwidth(sampwidth)
                out_wav.setframerate(rate)
                out_wav.writeframes(frames)
            chunk_paths.append(chunk_path)
            idx += 1
    return chunk_paths


def _load_model(task: str, model_id: str, model_type: str | None):
    if task == "canary":
        from nemo.collections.asr.models import EncDecMultiTaskModel  # type: ignore[unresolved-import]

        return EncDecMultiTaskModel.from_pretrained(model_id)
    if task == "parakeet":
        from nemo.collections.asr.models import (  # type: ignore[unresolved-import]
            ASRModel,
            EncDecCTCModelBPE,
            EncDecRNNTBPEModel,
        )

        if model_type == "ctc":
            return EncDecCTCModelBPE.from_pretrained(model_id)
        if model_type in {"rnnt", "tdt", "realtime-eou"}:
            return EncDecRNNTBPEModel.from_pretrained(model_id)
        if model_type == "tdt-ctc":
            return ASRModel.from_pretrained(model_id)
        raise ValueError(f"Unknown parakeet model_type: {model_type}")
    raise ValueError(f"Unknown task: {task}")


def _set_attr_path(obj, path: list[str], value) -> bool:
    target = obj
    for key in path[:-1]:
        if not hasattr(target, key):
            return False
        target = getattr(target, key)
    last = path[-1]
    if not hasattr(target, last):
        return False
    setattr(target, last, value)
    return True


def _configure_decoding(model, task: str, model_type: str | None) -> None:
    if not hasattr(model, "change_decoding_strategy"):
        return
    cfg = getattr(model, "cfg", None)
    decoding = getattr(cfg, "decoding", None) if cfg is not None else None
    if decoding is None:
        return
    if task == "canary" or model_type in {"rnnt", "tdt", "tdt-ctc", "realtime-eou"}:
        # Prefer fast greedy decoding when available.
        _set_attr_path(decoding, ["strategy"], "greedy_batch")
        _set_attr_path(decoding, ["rnnt_decoding", "strategy"], "greedy_batch")
        # Force beam size to 1 when possible.
        for path in (
            ["beam", "beam_size"],
            ["rnnt_decoding", "beam_size"],
            ["rnnt_decoding", "beam", "beam_size"],
        ):
            _set_attr_path(decoding, path, 1)
    try:
        model.change_decoding_strategy(decoding)
    except Exception:
        return


def _set_decoding_type(model, decoding, decoding_type: str) -> bool:
    if hasattr(model, "set_decoding_type"):
        try:
            model.set_decoding_type(decoding_type)
            return True
        except Exception:
            return False
    if decoding is not None:
        changed = False
        changed |= _set_attr_path(decoding, ["decoding_type"], decoding_type)
        changed |= _set_attr_path(decoding, ["model_type"], decoding_type)
        if decoding_type == "ctc":
            changed |= _set_attr_path(decoding, ["strategy"], "greedy")
        if changed:
            try:
                model.change_decoding_strategy(decoding)
                return True
            except Exception:
                return False
    return False


def _configure_logging() -> None:
    logging.basicConfig(level=logging.ERROR)
    os.environ.setdefault("NEMO_LOG_LEVEL", "ERROR")


def _cuda_is_usable() -> tuple[bool, str | None]:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("canary", "parakeet"))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-type")
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--chunk-seconds", type=float, default=40.0)
    parser.add_argument(
        "--decode-mode",
        choices=("tdt", "ctc"),
        help="Optional decode mode override for TDT-CTC models",
    )
    args = parser.parse_args()

    _configure_logging()

    import torch

    cuda_ok, cuda_err = _cuda_is_usable()
    device = torch.device("cuda") if cuda_ok else torch.device("cpu")
    device_note = "cuda" if cuda_ok else "cpu"
    if cuda_err and torch.cuda.is_available():
        device_note = f"cpu (cuda unavailable: {cuda_err})"

    model = _load_model(args.task, args.model_id, args.model_type)
    model.eval()
    model = model.to(device)
    if os.environ.get("STT_BENCH_NEMO_ALLOW_TF32", "").lower() in {"1", "true", "yes", "y"}:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    matmul_precision = os.environ.get("STT_BENCH_NEMO_MATMUL_PRECISION")
    if matmul_precision:
        try:
            torch.set_float32_matmul_precision(matmul_precision)
        except Exception:
            pass
    if os.environ.get("STT_BENCH_NEMO_CUDNN_BENCHMARK", "").lower() in {"1", "true", "yes", "y"}:
        torch.backends.cudnn.benchmark = True

    autocast_env = os.environ.get("STT_BENCH_NEMO_AUTOCAST", "auto").lower()
    if autocast_env in {"0", "false", "no", "n"}:
        use_autocast = False
    elif autocast_env in {"1", "true", "yes", "y"}:
        use_autocast = cuda_ok
    else:
        # Auto: enable for CTC/CTC-like models, disable for RNNT/TDT unless overridden.
        use_autocast = cuda_ok and args.model_type in {"ctc", "tdt-ctc"}
    _configure_decoding(model, args.task, args.model_type)
    decode_note = None
    if args.model_type == "tdt-ctc" and args.decode_mode:
        cfg = getattr(model, "cfg", None)
        decoding = getattr(cfg, "decoding", None) if cfg is not None else None
        if args.decode_mode == "ctc":
            if _set_decoding_type(model, decoding, "ctc"):
                decode_note = "ctc"
        else:
            decode_note = "tdt"
    elif args.decode_mode:
        decode_note = args.decode_mode

    sample_seconds = _wav_duration_seconds(args.audio_path)

    last_transcript: str | None = None

    _TAG_RE = re.compile(r"<[^>]+>")

    def _strip_tags(text: str) -> str:
        return _TAG_RE.sub(" ", text).strip()

    def _text_from_obj(obj) -> str | None:
        if obj is None:
            return None
        if isinstance(obj, str):
            cleaned = _strip_tags(obj)
            return cleaned or None
        if isinstance(obj, dict):
            for key in ("text", "transcript", "pred_text", "sentence"):
                value = obj.get(key)
                if isinstance(value, str) and value.strip():
                    return _strip_tags(value)
            return None
        for attr in ("text", "transcript", "pred_text", "sentence"):
            value = getattr(obj, attr, None)
            if isinstance(value, str) and value.strip():
                return _strip_tags(value)
        return None

    def _extract_transcript(output) -> str | None:
        if output is None:
            return None
        if isinstance(output, (list, tuple)):
            texts: list[str] = []
            for item in output:
                text = _text_from_obj(item)
                if text:
                    texts.append(text)
                    continue
                hyps = getattr(item, "hypotheses", None)
                if isinstance(hyps, (list, tuple)):
                    for hyp in hyps:
                        hyp_text = _text_from_obj(hyp)
                        if hyp_text:
                            texts.append(hyp_text)
                            break
            joined = " ".join(text for text in texts if text).strip()
            return joined or None
        return _text_from_obj(output)

    def run_once() -> str | None:
        autocast_dtype = os.environ.get("STT_BENCH_NEMO_AUTOCAST_DTYPE", "fp16").lower()
        autocast_dtype_t = torch.float16 if autocast_dtype in {"fp16", "float16"} else torch.bfloat16
        autocast_cm = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype_t)
            if use_autocast
            else nullcontext()
        )
        with torch.inference_mode(), autocast_cm:
            if sample_seconds <= args.chunk_seconds:
                outputs = model.transcribe(
                    audio=[args.audio_path],
                    batch_size=1,
                    num_workers=0,
                    verbose=False,
                )
                return _extract_transcript(outputs)
            with tempfile.TemporaryDirectory() as temp_dir:
                chunk_paths = _write_wav_chunks(args.audio_path, args.chunk_seconds, temp_dir)
                outputs = model.transcribe(
                    audio=chunk_paths,
                    batch_size=1,
                    num_workers=0,
                    verbose=False,
                )
                return _extract_transcript(outputs)

    wall_start = time.perf_counter()
    for _ in range(args.warmups):
        run_once()

    elapsed_values: list[float] = []
    transcripts: list[str | None] = []
    for _ in range(args.runs):
        start = time.perf_counter()
        transcript = run_once()
        elapsed_values.append(time.perf_counter() - start)
        transcripts.append(transcript)
    wall_seconds = time.perf_counter() - wall_start

    rtfx_values = [sample_seconds / v for v in elapsed_values]
    rtfx_mean = statistics.fmean(rtfx_values)
    rtfx_stdev = statistics.stdev(rtfx_values) if len(rtfx_values) >= 2 else 0.0
    last_transcript = transcripts[-1] if transcripts else None
    payload = {
        "rtfx_mean": rtfx_mean,
        "rtfx_stdev": rtfx_stdev,
        "wall_seconds": wall_seconds,
        "device": device_note,
        "decode": decode_note,
        "transcript": last_transcript,
        "elapsed_values": elapsed_values,
        "transcripts": transcripts,
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
