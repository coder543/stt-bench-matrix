#!/usr/bin/env bash
set -euo pipefail

if [[ -f /etc/stt-bench-git.env ]]; then
  # shellcheck disable=SC1091
  source /etc/stt-bench-git.env
fi

if [[ -x /workspace/.venv/bin/python ]]; then
  export UV_PYTHON=/workspace/.venv/bin/python
fi

exec uv run stt-bench-matrix "$@"
