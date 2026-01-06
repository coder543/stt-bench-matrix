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

## Models (target)

- **OpenAI Whisper** — all sizes (tiny → large)
- **NVIDIA Parakeet**
- **NVIDIA Canary** — 180M and 1B

---

## macOS focus (phase 1)

We’re starting with macOS. For Whisper, the initial candidates are:

- `lightning-whisper-mlx`
- `whisper-mlx`

We still need to identify the **macOS-compatible** pathways for **Parakeet** and **Canary** (frameworks, model formats, and inference runtimes).

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

## Status

This repository is scaffolding the macOS implementation first. See `TODO.md` for concrete next steps.

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

Build (DGX Spark / Blackwell, source torch + torchaudio + CTranslate2):

```bash
docker build -f Dockerfile.cuda -t stt-bench-matrix:cuda13-bw \
  --build-arg TORCH_SOURCE=1 \
  --build-arg TORCHAUDIO_SOURCE=1 \
  --build-arg CTRANSLATE2_SOURCE=1 \
  --build-arg TORCH_CUDA_ARCH_LIST=12.1+PTX \
  --build-arg TORCH_MAX_JOBS=8 .
```

Run (persists caches and writes output):

```bash
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v "$HOME/.cache/huggingface:/workspace/.cache/huggingface" \
  -v "$HOME/.cache/uv:/workspace/.cache/uv" \
  -v "$(pwd)/output:/workspace/output" \
  -w /workspace \
  stt-bench-matrix:cuda13-bw
```

---

## Notes

- Model caching should be **transparent** and **stable**, with a single cache directory per framework.
- The tool should degrade gracefully: if a framework isn’t supported on a machine, it should be skipped with a clear reason.
- DGX Spark (arm64 Blackwell) currently requires **source-built CUDA torch/torchaudio**; use `Dockerfile.cuda` with `--build-arg TORCH_SOURCE=1 --build-arg TORCHAUDIO_SOURCE=1` to enable GPU for transformers-based runs.
- Parakeet realtime EOU is a **streaming/EOU model**; offline WER on the full sample can look very poor even when GPU is working. Use it for latency/RTFx comparisons or stream-style evaluation rather than comparing WER directly.
- Canary Qwen 2.5B is a SALM model; it requires NeMo with SpeechLM2 support and uses a prompt + audio input path instead of `transcribe()`.
- Nemotron Speech Streaming 0.6B uses NeMo ASRModel with cache-aware streaming; the benchmark uses offline `transcribe()` by default, but you can set `STT_BENCH_NEMO_ATT_CONTEXT_SIZE="70,13"` to emulate a streaming chunk size when supported.
- The Nemotron model card recommends NeMo main / runtime engine 25.11; if it fails under the pinned NeMo, you may need to upgrade the NeMo runner environment.
- Gemma 3n runs via ONNX Runtime (`gemma-3n-onnx`) and uses the `onnx-community/gemma-3n-E2B-it-ONNX` checkpoint with the `google/gemma-3n-E2B-it` processor. CUDA requires `onnxruntime-gpu`; on arm64 (DGX Spark) this is currently a source build. You can override the model snapshot location with `STT_BENCH_GEMMA_ONNX_DIR=/path/to/snapshot`.
