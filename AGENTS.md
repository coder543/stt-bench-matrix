# Repository Guidelines

## Project Structure & Module Organization
- `src/stt_bench_matrix/`: core package.
  - `cli.py`: CLI entrypoint and output handling.
  - `bench/`: run orchestration, perf stats, samples, WER.
  - `frameworks/`: per‑model runners (whisper, parakeet, canary, moonshine, granite).
  - `models/registry.py`: model lists for full/quick runs.
  - `platforms/`: host/accelerator detection.
  - `reporting/`: Markdown report formatting.
- `tools/nemo_runner/`: separate uv project for NeMo subprocess runner.
- `samples/`: default audio + transcript.
- `Dockerfile.cuda` and `docker/entrypoint.sh`: CUDA container build/run.
- `output/`: auto‑saved benchmark markdown + json (git‑ignored).

## Build, Test, and Development Commands
- `uv run stt-bench-matrix`: run full benchmark suite with auto-run stabilization enabled.
- `uv run stt-bench-matrix --frameworks whisper.cpp --models tiny,base`: run a targeted subset.
- `uv run stt-bench-matrix --runs 3`: fixed number of runs (disables auto mode).
- `uv run stt-bench-matrix --heavy`: include heavy models (Granite 8B).
- `uv run stt-bench-matrix --list`: print known frameworks/models.
- `uv run ty check`: typecheck (primary validation step).
- `docker build -f Dockerfile.cuda -t stt-bench-matrix:cuda .`: build CUDA image.
- `docker run --rm --gpus all -v "$(pwd)/output:/workspace/output" -v "$HOME/.cache/huggingface:/root/.cache/huggingface" stt-bench-matrix:cuda --quick`: container quick run with cache/output mounts.

## Coding Style & Naming Conventions
- Python 3.11, type hints throughout; follow existing conventions.
- Keep filenames snake_case and framework classes `XFramework` with `benchmark_*` helpers.
- Prefer dataclasses for result structures; avoid heavy inline comments.

## Testing Guidelines
- No unit test suite yet. Validate by running `uv run ty check` and a targeted CLI run.
- Sample data: `samples/jfk_rice_16k.wav` with transcript `samples/jfk_rice_16k.txt`.
- For faster iteration, use the committed 20s sample (skips the first 17s of applause) with `--sample samples/jfk_rice_16k_20s.wav`.

## Commit & Pull Request Guidelines
- No formal commit convention enforced. Use clear, imperative summaries (e.g., “Add moonshine runner”).
- PRs should describe benchmark changes, include command output paths from `output/`, and note runtime/accelerator.

## Notes for Agents
- WER is computed only when a transcript is available; keep transcripts clean (strip tags, brackets).
- NeMo runs via `tools/nemo_runner` with its own dependencies; do not mix imports into main package.
- Quick runs include the smallest models; heavy models are gated by `--heavy`.
