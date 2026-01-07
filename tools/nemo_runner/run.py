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
import sys


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
        if model_type == "salm":
            from nemo.collections.speechlm2.models import SALM  # type: ignore[unresolved-import]

            return SALM.from_pretrained(model_id)
        from nemo.collections.asr.models import EncDecMultiTaskModel  # type: ignore[unresolved-import]

        return EncDecMultiTaskModel.from_pretrained(model_id)
    if task == "nemotron":
        from nemo.collections.asr.models import ASRModel  # type: ignore[unresolved-import]

        return ASRModel.from_pretrained(model_name=model_id)
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


def _configure_decoding(model, task: str, model_type: str | None) -> list[str]:
    if not hasattr(model, "change_decoding_strategy"):
        return []
    cfg = getattr(model, "cfg", None)
    decoding = getattr(cfg, "decoding", None) if cfg is not None else None
    if decoding is None:
        return []
    if task == "canary" or model_type in {
        "rnnt",
        "tdt",
        "tdt-ctc",
        "realtime-eou",
        "cache-aware-rnnt",
    }:
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
    decode_notes: list[str] = []
    decoding_strategy = os.environ.get("STT_BENCH_NEMO_DECODING_STRATEGY")
    if decoding_strategy:
        if _set_attr_path(decoding, ["strategy"], decoding_strategy):
            decode_notes.append(f"strategy={decoding_strategy}")
    rnnt_strategy = os.environ.get("STT_BENCH_NEMO_RNNT_DECODING_STRATEGY")
    if rnnt_strategy:
        if _set_attr_path(decoding, ["rnnt_decoding", "strategy"], rnnt_strategy):
            decode_notes.append(f"rnnt_strategy={rnnt_strategy}")
    beam_size_env = os.environ.get("STT_BENCH_NEMO_BEAM_SIZE")
    if beam_size_env:
        try:
            beam_size = int(beam_size_env)
        except ValueError:
            beam_size = None
        if beam_size is not None:
            for path in (
                ["beam", "beam_size"],
                ["rnnt_decoding", "beam_size"],
                ["rnnt_decoding", "beam", "beam_size"],
            ):
                if _set_attr_path(decoding, path, beam_size):
                    decode_notes.append(f"beam={beam_size}")
                    break
    try:
        model.change_decoding_strategy(decoding)
    except Exception:
        return decode_notes
    return decode_notes


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


def _render_run_progress(label: str, current: int, total: int) -> None:
    total = max(total, 1)
    current = min(max(current, 0), total)
    width = 24
    filled = int(width * current / total)
    bar = "[" + ("#" * filled) + ("-" * (width - filled)) + "]"
    line = f"Runs {bar} {current}/{total} | {label}"
    if sys.stderr.isatty():
        sys.stderr.write("\r\x1b[2K" + line)
        sys.stderr.flush()
    else:
        sys.stderr.write(line + "\n")
        sys.stderr.flush()


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
    parser.add_argument("--task", required=True, choices=("canary", "parakeet", "nemotron"))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-type")
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--warmup-audio-path")
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--auto-min-runs", type=int, default=5)
    parser.add_argument("--auto-max-runs", type=int, default=30)
    parser.add_argument("--auto-target-cv", type=float, default=0.05)
    parser.add_argument("--chunk-seconds", type=float, default=40.0)
    parser.add_argument(
        "--decode-mode",
        choices=("tdt", "ctc"),
        help="Optional decode mode override for TDT-CTC models",
    )
    args = parser.parse_args()

    _configure_logging()

    import torch
    try:
        from torch.utils.data import Sampler
        sampler_type = Sampler
    except Exception:
        sampler_type = None
    if sampler_type is not None and sampler_type.__init__ is object.__init__:
        def _sampler_init(self, data_source=None) -> None:
            return None
        sampler_type.__init__ = _sampler_init

    cuda_ok, cuda_err = _cuda_is_usable()
    device = torch.device("cuda") if cuda_ok else torch.device("cpu")
    device_note = "cuda" if cuda_ok else "cpu"
    if cuda_err and torch.cuda.is_available():
        device_note = f"cpu (cuda unavailable: {cuda_err})"

    model = _load_model(args.task, args.model_id, args.model_type)
    model.eval()
    model = model.to(device)
    if args.model_type in {"salm", "cache-aware-rnnt"}:
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass
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
    att_context_env = os.environ.get("STT_BENCH_NEMO_ATT_CONTEXT_SIZE")
    if att_context_env is None and args.model_type == "cache-aware-rnnt":
        att_context_env = "70,13"
    att_context_note = None
    if att_context_env:
        cleaned = att_context_env.strip().strip("[]()")
        parts = [p.strip() for p in cleaned.split(",") if p.strip()]
        if len(parts) == 2:
            try:
                att_context = [int(parts[0]), int(parts[1])]
            except ValueError:
                att_context = None
            if att_context is not None:
                applied = False
                encoder = getattr(model, "encoder", None)
                if encoder is not None and hasattr(encoder, "set_default_att_context_size"):
                    try:
                        encoder.set_default_att_context_size(att_context)
                        applied = True
                    except Exception:
                        applied = False
                if not applied and hasattr(model, "set_default_att_context_size"):
                    try:
                        model.set_default_att_context_size(att_context)
                        applied = True
                    except Exception:
                        applied = False
                if applied:
                    att_context_note = f"att_context={att_context}"

    autocast_env = os.environ.get("STT_BENCH_NEMO_AUTOCAST", "auto").lower()
    if autocast_env in {"0", "false", "no", "n"}:
        use_autocast = False
    elif autocast_env in {"1", "true", "yes", "y"}:
        use_autocast = cuda_ok
    else:
        # Auto: enable for CTC/CTC-like models, disable for RNNT/TDT unless overridden.
        use_autocast = cuda_ok and args.model_type in {"ctc", "tdt-ctc"}
    precision_note = None
    if (
        autocast_env == "auto"
        and args.task == "parakeet"
        and args.model_type == "ctc"
        and args.model_id.endswith("parakeet-ctc-0.6b")
    ):
        # Avoid garbage transcripts on this model when autocast is enabled.
        use_autocast = False
        precision_note = "fp32"
    decode_notes = _configure_decoding(model, args.task, args.model_type)
    decode_note = "; ".join(decode_notes) if decode_notes else None
    if args.model_type == "tdt-ctc" and args.decode_mode:
        cfg = getattr(model, "cfg", None)
        decoding = getattr(cfg, "decoding", None) if cfg is not None else None
        if args.decode_mode == "ctc":
            if _set_decoding_type(model, decoding, "ctc"):
                decode_note = "ctc" if decode_note is None else f"{decode_note}; type=ctc"
        else:
            decode_note = "tdt" if decode_note is None else f"{decode_note}; type=tdt"
    elif args.decode_mode:
        decode_note = args.decode_mode if decode_note is None else f"{decode_note}; type={args.decode_mode}"
    decode_type_env = os.environ.get("STT_BENCH_NEMO_DECODING_TYPE")
    if decode_type_env:
        cfg = getattr(model, "cfg", None)
        decoding = getattr(cfg, "decoding", None) if cfg is not None else None
        if _set_decoding_type(model, decoding, decode_type_env):
            decode_note = (
                f"type={decode_type_env}" if decode_note is None else f"{decode_note}; type={decode_type_env}"
            )
    if att_context_note:
        decode_note = att_context_note if decode_note is None else f"{decode_note}; {att_context_note}"

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

    def run_once(audio_path: str = args.audio_path) -> str | None:
        autocast_dtype = os.environ.get("STT_BENCH_NEMO_AUTOCAST_DTYPE", "fp16").lower()
        autocast_dtype_t = torch.float16 if autocast_dtype in {"fp16", "float16"} else torch.bfloat16
        autocast_cm = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype_t)
            if use_autocast
            else nullcontext()
        )
        salm_prompt = os.environ.get("STT_BENCH_SALM_PROMPT", "Transcribe the following:")
        salm_max_tokens_env = os.environ.get("STT_BENCH_SALM_MAX_NEW_TOKENS")
        try:
            salm_max_tokens = int(salm_max_tokens_env) if salm_max_tokens_env else 128
        except ValueError:
            salm_max_tokens = 128
        with torch.inference_mode(), autocast_cm:
            if args.task == "canary" and args.model_type == "salm":
                audio_tag = getattr(model, "audio_locator_tag", "<|audioplaceholder|>")
                content = salm_prompt if audio_tag in salm_prompt else f"{salm_prompt} {audio_tag}"

                def _decode_salm(ids) -> str | None:
                    if ids is None:
                        return None
                    if hasattr(ids, "detach"):
                        ids = ids.detach().cpu()
                    if isinstance(ids, (list, tuple)):
                        token_ids = ids
                    else:
                        token_ids = ids.tolist() if hasattr(ids, "tolist") else ids
                    if isinstance(token_ids, (list, tuple)) and token_ids:
                        first = token_ids[0]
                        if isinstance(first, (list, tuple)):
                            token_ids = first
                    tokenizer = getattr(model, "tokenizer", None)
                    if tokenizer is not None:
                        if hasattr(tokenizer, "ids_to_text"):
                            text = tokenizer.ids_to_text(token_ids)
                        elif hasattr(tokenizer, "decode"):
                            text = tokenizer.decode(token_ids)
                        else:
                            text = str(token_ids)
                    else:
                        text = str(token_ids)
                    cleaned = _strip_tags(text)
                    return cleaned or None

                def _salm_transcribe(audio_path: str) -> str | None:
                    prompts = [
                        [
                            {
                                "role": "user",
                                "content": content,
                                "audio": [audio_path],
                            }
                        ]
                    ]
                    answer_ids = model.generate(
                        prompts=prompts,
                        max_new_tokens=salm_max_tokens,
                    )
                    if isinstance(answer_ids, (list, tuple)) and answer_ids:
                        return _decode_salm(answer_ids[0])
                    return _decode_salm(answer_ids)

                if sample_seconds <= args.chunk_seconds:
                    return _salm_transcribe(audio_path)
                with tempfile.TemporaryDirectory() as temp_dir:
                    chunk_paths = _write_wav_chunks(audio_path, args.chunk_seconds, temp_dir)
                    texts: list[str] = []
                    for chunk in chunk_paths:
                        text = _salm_transcribe(chunk)
                        if text:
                            texts.append(text)
                    joined = " ".join(texts).strip()
                    return joined or None
            if sample_seconds <= args.chunk_seconds:
                transcribe_kwargs = {}
                if args.task == "canary":
                    task = os.environ.get("STT_BENCH_CANARY_TASK", "asr")
                    source_lang = os.environ.get("STT_BENCH_CANARY_SOURCE_LANG", "en")
                    target_lang = os.environ.get("STT_BENCH_CANARY_TARGET_LANG", "en")
                    pnc_env = os.environ.get("STT_BENCH_CANARY_PNC", "pnc").lower()
                    if pnc_env in {"1", "true", "yes", "y"}:
                        pnc = "yes"
                    elif pnc_env in {"0", "false", "no", "n"}:
                        pnc = "no"
                    else:
                        pnc = pnc_env
                    transcribe_kwargs.update(
                        task=task,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        pnc=pnc,
                    )
                outputs = model.transcribe(
                    audio=[audio_path],
                    batch_size=1,
                    num_workers=0,
                    verbose=False,
                    **transcribe_kwargs,
                )
                return _extract_transcript(outputs)
            with tempfile.TemporaryDirectory() as temp_dir:
                chunk_paths = _write_wav_chunks(audio_path, args.chunk_seconds, temp_dir)
                transcribe_kwargs = {}
                if args.task == "canary":
                    task = os.environ.get("STT_BENCH_CANARY_TASK", "asr")
                    source_lang = os.environ.get("STT_BENCH_CANARY_SOURCE_LANG", "en")
                    target_lang = os.environ.get("STT_BENCH_CANARY_TARGET_LANG", "en")
                    pnc_env = os.environ.get("STT_BENCH_CANARY_PNC", "pnc").lower()
                    if pnc_env in {"1", "true", "yes", "y"}:
                        pnc = "yes"
                    elif pnc_env in {"0", "false", "no", "n"}:
                        pnc = "no"
                    else:
                        pnc = pnc_env
                    transcribe_kwargs.update(
                        task=task,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        pnc=pnc,
                    )
                outputs = model.transcribe(
                    audio=chunk_paths,
                    batch_size=1,
                    num_workers=0,
                    verbose=False,
                    **transcribe_kwargs,
                )
                return _extract_transcript(outputs)

    wall_start = time.perf_counter()
    for _ in range(args.warmups):
        if args.warmup_audio_path:
            run_once(args.warmup_audio_path)
        else:
            run_once()

    elapsed_values: list[float] = []
    transcripts: list[str | None] = []
    def _should_stop_auto(values: list[float]) -> bool:
        if not values:
            return False
        min_runs = max(1, args.auto_min_runs)
        if len(values) < min_runs:
            return False
        max_runs = max(min_runs, args.auto_max_runs)
        if len(values) >= max_runs:
            return True
        mean = statistics.fmean(values)
        if mean <= 0:
            return False
        stdev = statistics.stdev(values) if len(values) >= 2 else 0.0
        cv = stdev / mean if mean > 0 else 0.0
        return cv <= max(0.0, args.auto_target_cv)

    run_target = args.runs if not args.auto else None
    run_label = f"{args.task} {args.model_id}"
    show_run_progress = not (run_target == 1)
    while True:
        start = time.perf_counter()
        transcript = run_once()
        elapsed_values.append(time.perf_counter() - start)
        transcripts.append(transcript)
        current_runs = len(elapsed_values)
        if run_target is not None:
            display_target = max(1, run_target)
        else:
            display_target = max(1, args.auto_min_runs)
            if current_runs > display_target:
                display_target = current_runs
        if show_run_progress:
            _render_run_progress(run_label, current_runs, display_target)
        if run_target is not None:
            if len(elapsed_values) >= run_target:
                break
        elif _should_stop_auto(elapsed_values):
            break
    if show_run_progress and sys.stderr.isatty():
        sys.stderr.write("\n")
        sys.stderr.flush()
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
        "precision": precision_note,
        "transcript": last_transcript,
        "elapsed_values": elapsed_values,
        "transcripts": transcripts,
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
