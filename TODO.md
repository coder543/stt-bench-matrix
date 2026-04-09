# TODO

## Current priorities

- [ ] Review Moonshine chunking/max length settings vs official guidance
- [ ] Validate Granite Speech models (2B/8B) with correct processor usage
- [ ] Evaluate CPU-only full run feasibility or add a CPU preset

## Nice-to-have

- [ ] Add ROCm detection and AMD support
- [ ] Add lightweight test coverage for key utilities

## Notes (host Blackwell bring-up)

- Official PyTorch 2.11 CUDA wheels are available, so custom torch/torchaudio wheel overrides should no longer be needed for standard GPU runs.
- Re-validate transformer-path performance on Blackwell now that CUTLASS issues are fixed in the official release.
- Confirm NeMo runner stays on GPU without the old local wheel override path.
- parakeet-realtime-eou 120m-v1 shows poor offline WER even on CUDA; likely needs stream-style evaluation to be meaningful.
