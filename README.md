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

## Notes

- Model caching should be **transparent** and **stable**, with a single cache directory per framework.
- The tool should degrade gracefully: if a framework isn’t supported on a machine, it should be skipped with a clear reason.
- DGX Spark (arm64 Blackwell) currently requires **source-built CUDA torch/torchaudio**; use `Dockerfile.cuda` with `--build-arg TORCH_SOURCE=1 --build-arg TORCHAUDIO_SOURCE=1` to enable GPU for transformers-based runs.
- Parakeet realtime EOU is a **streaming/EOU model**; offline WER on the full sample can look very poor even when GPU is working. Use it for latency/RTFx comparisons or stream-style evaluation rather than comparing WER directly.
- Canary Qwen 2.5B is a SALM model; it requires NeMo with SpeechLM2 support and uses a prompt + audio input path instead of `transcribe()`.
