# stt-bench-matrix

Cross-platform, single-command STT benchmarking that outputs a **ready-to-paste Markdown blob** for GitHub issues.

The goal is to run on **macOS or Linux**, across **Apple Silicon / NVIDIA / AMD**, and automatically select compatible frameworks and models for the host machine.

---

## Goals

- **One command** to benchmark popular STT models and implementations
- **Auto-detect platform** and select compatible frameworks
- **Auto-download + cache** all models on first run
- **Emit a single Markdown report** that can be pasted into a GitHub issue comment
- **Well-typed Python** (Astral `ty`) and **Astral `uv`** for dependency management

---

## Models

- **OpenAI Whisper** — tiny → large-v3 (optional: `.en`, large-v1/v2)
- **NVIDIA Parakeet** — CTC, RNNT, TDT, TDT-CTC (110M), realtime-EOU
- **NVIDIA Canary** — 180m-flash, 1b-flash, 1b-v2 (optional: qwen-2.5b SALM)
- **Moonshine** — tiny, base
- **Cohere Transcribe** — 03-2026
- **Qwen3-ASR** — 0.6b (optional heavy: 1.7b)
- **VibeVoice ASR** — 8b (heavy)
- **Mistral Voxtral Realtime** — Mini 4B Realtime 2602 (heavy)
- **Nemotron Speech Streaming** — 0.6b
- **Granite Speech 3.3** — 2b (optional heavy: 8b)
- **Gemma 3n** — e2b, e4b (heavy, ONNX runtime)
- **LiquidAI LFM2.5 Audio** — 1.5b (heavy)

---

## Architecture sketch (modular by platform + framework)

The system is structured to support multiple OS/accelerators without rewriting core logic.

- **`platforms/`**
  - Detect OS + hardware and declare supported frameworks
- **`frameworks/`**
  - Framework adapters (whisper-mlx, lightning-whisper-mlx, etc.)
- **`models/`**
  - Model inventory + sizes + cache locations
- **`bench/`**
  - Benchmark runner, timing, metrics, and dataset handling
- **`reporting/`**
  - Markdown report generator

---

## Output

The CLI should emit a **single Markdown blob** with:

- system info (OS, CPU/GPU)
- frameworks tested
- models/sizes tested
- timing + throughput
- accuracy metrics (WER or CER if available)

---

## CLI

```bash
uv run stt-bench-matrix
uv run stt-bench-matrix --frameworks whisper.cpp --models tiny,base
uv run stt-bench-matrix --runs 3
uv run stt-bench-matrix --auto-min-runs 3 --auto-target-cv 0.03
uv run stt-bench-matrix --list
```

---

## Docker (CUDA)

Build:

```bash
docker build -f Dockerfile.cuda -t stt-bench-matrix:cuda13-bw .
```

Build (optional source CTranslate2 / other native deps):

```bash
docker build -f Dockerfile.cuda -t stt-bench-matrix:cuda13-bw \
  --build-arg CTRANSLATE2_SOURCE=1 \
  .
```

Run (persists caches and writes output):

```bash
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -e HF_HOME=/workspace/.cache/huggingface \
  -v "$HF_HOME:/workspace/.cache/huggingface" \
  -v "$HOME/.cache/uv:/workspace/.cache/uv" \
  -v "$(pwd)/output:/workspace/output" \
  -w /workspace \
  stt-bench-matrix:cuda13-bw
```

---

## Notes

- Model caching should be **transparent** and **stable**, with a single cache directory per framework.
- The tool should degrade gracefully: if a framework isn’t supported on a machine, it should be skipped with a clear reason.
- This repo targets Python **3.12+** (some optional frameworks, like LiquidAI LFM2.5 Audio, require it).
- Official PyTorch 2.11 CUDA wheels are now sufficient for the CUDA transformer paths here; custom Torch/Torchaudio wheel overrides are no longer required for standard runs.
- NeMo on bare-metal Linux still depends on the host CUDA/cuDNN runtime matching the official Torch 2.11 cu13 stack. If you hit `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`, install system cuDNN 13 runtime libraries (for example `libcudnn9-cuda-13`) or use `Dockerfile.cuda`, which already includes the matching runtime.
- Cohere Transcribe runs via `cohere-transformers` against `CohereLabs/cohere-transcribe-03-2026` by default. Override with `COHERE_MODEL_ID`, adjust decode length with `STT_BENCH_COHERE_MAX_NEW_TOKENS`, and force math SDPA only if needed with `STT_BENCH_COHERE_FORCE_MATH_SDP=1`.
- Qwen3-ASR runs via `qwen3-asr-transformers` in an isolated `tools/qwen_asr_runner` `uv` project because the upstream `qwen-asr` package currently pins its own Transformers stack. The benchmark uses `Qwen/Qwen3-ASR-0.6B` by default and adds `Qwen/Qwen3-ASR-1.7B` under `--heavy` or explicit `--models`.
- VibeVoice ASR runs via `vibevoice-transformers` against `microsoft/VibeVoice-ASR-HF`. The benchmark decodes with `return_format="transcription_only"` so WER uses plain text, and you can pass context/hotwords with `STT_BENCH_VIBEVOICE_PROMPT` or lower memory usage with `STT_BENCH_VIBEVOICE_TOKENIZER_CHUNK_SIZE`.
- Voxtral Realtime runs via `voxtral-transformers` against `mistralai/Voxtral-Mini-4B-Realtime-2602` in an isolated `tools/voxtral_runner` `uv` project because `mistral-common[audio]` conflicts with some of the main-environment model deps. It is gated behind `--heavy` or explicit `--models voxtral`.
- Parakeet realtime EOU is a **streaming/EOU model**; offline WER on the full sample can look very poor even when GPU is working. Use it for latency/RTFx comparisons or stream-style evaluation rather than comparing WER directly.
- Canary Qwen 2.5B is a SALM model; it requires NeMo with SpeechLM2 support and uses a prompt + audio input path instead of `transcribe()`.
- Nemotron Speech Streaming 0.6B uses NeMo ASRModel with cache-aware streaming; the benchmark uses offline `transcribe()` by default, but you can set `STT_BENCH_NEMO_ATT_CONTEXT_SIZE="70,13"` to emulate a streaming chunk size when supported.
- The Nemotron model card recommends NeMo main / runtime engine 25.11; if it fails under the pinned NeMo, you may need to upgrade the NeMo runner environment.
- Gemma 3n runs via ONNX Runtime (`gemma-3n-onnx`) and uses the `onnx-community/gemma-3n-E2B-it-ONNX` checkpoint with the `google/gemma-3n-E2B-it` processor. CUDA requires `onnxruntime-gpu`; on arm64 (DGX Spark) this is currently a source build. You can override the model snapshot location with `STT_BENCH_GEMMA_ONNX_DIR=/path/to/snapshot`. Long audio is chunked by default (20s); set `STT_BENCH_GEMMA_ONNX_CHUNK_SECONDS=30` or similar to tune.
- LiquidAI LFM2.5 Audio uses the `liquid-audio` package and the `LiquidAI/LFM2.5-Audio-1.5B` checkpoint. The benchmark issues an ASR-style prompt ("Perform ASR.") and uses sequential generation; you can adjust output length with `STT_BENCH_LFM_MAX_NEW_TOKENS`. Long-form audio is chunked by default at 20s (`STT_BENCH_LFM_CHUNK_SECS`, optional overlap via `STT_BENCH_LFM_CHUNK_OVERLAP_SECS`). On arm64 you may need a local `torchcodec` build (FFmpeg dev libs + `I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1`) because wheels are not published.
