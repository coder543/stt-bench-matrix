#!/usr/bin/env bash
set -euo pipefail

if [[ -f /etc/stt-bench-git.env ]]; then
  # shellcheck disable=SC1091
  source /etc/stt-bench-git.env
fi

exec uv run stt-bench-matrix "$@"
