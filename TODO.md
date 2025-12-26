# TODO

## Immediate (Mac implementation)

- [ ] Confirm macOS-compatible inference paths for:
  - [ ] NVIDIA Parakeet
  - [ ] NVIDIA Canary (180M + 1B)
- [ ] Decide benchmark dataset(s) (short/long audio, languages)
- [ ] Define metrics (RTF, WER/CER) and output schema
- [ ] Sketch CLI entrypoint and flags
- [ ] Define cache layout per framework

## Architecture / scaffolding

- [ ] Establish Python project scaffolding with `uv`
- [ ] Add `ty` type-checking config
- [ ] Create module layout:
  - [ ] `platforms/` (OS + hardware detection)
  - [ ] `frameworks/` (per-framework adapters)
  - [ ] `models/` (model registry + variants)
  - [ ] `bench/` (runner + metrics)
  - [ ] `reporting/` (Markdown output)

## macOS framework adapters

- [ ] Implement `whisper-mlx` adapter
- [ ] Implement `lightning-whisper-mlx` adapter
- [ ] Ensure both run all Whisper sizes

## Benchmark flow

- [ ] Auto-download + cache models on first run
- [ ] Run warmup vs timed runs
- [ ] Capture throughput + latency
- [ ] Optional accuracy metrics (if reference transcripts available)

## Reporting

- [ ] Generate one Markdown blob per run
- [ ] Include system summary and skipped frameworks

## Cross-platform (later)

- [ ] NVIDIA (CUDA) platform support
- [ ] AMD (ROCm) platform support
- [ ] Linux framework coverage

