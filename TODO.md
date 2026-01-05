# TODO

## Current priorities

- [ ] Review Moonshine chunking/max length settings vs official guidance
- [ ] Validate Granite Speech models (2B/8B) with correct processor usage
- [ ] Evaluate CPU-only full run feasibility or add a CPU preset

## Nice-to-have

- [ ] Add ROCm detection and AMD support
- [ ] Add lightweight test coverage for key utilities

## Notes (host Blackwell bring-up)

- Host venv GPU runs require `UV_NO_SYNC=1` to avoid uv reinstalling CPU-only torch.
- DGX Spark (arm64 Blackwell) lacks official CUDA torch/torchaudio wheels, so GPU runs require locally built CUDA+cuDNN wheels.
- Keep locally built CUDA+cuDNN torch/torchaudio wheels in a shared `python/wheels/` directory for reuse across projects.
- Repo venv uses the CUDA wheels; `uv run` + `UV_NO_SYNC=1` shows GPU acceleration for transformers/parakeet-transformers.
- NeMo subprocess uses its own venv; to force GPU, copy the CUDA torch wheel to `/opt/pytorch/dist` so `nemo_subprocess.py` installs it automatically.
- parakeet-nemo 0.6b runs on CUDA after the `/opt/pytorch/dist` wheel override; autocast disabled for this model yields sane transcripts (WER ~0.16 on the 20s sample).
- parakeet-realtime-eou 120m-v1 shows poor offline WER even on CUDA; likely needs stream-style evaluation to be meaningful.
