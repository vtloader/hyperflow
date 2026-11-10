#!/usr/bin/env bash
# Launch the HyperFlow H3 WebUI with the Ascend virtualenv.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_PYTHON="$ROOT/.venv312/bin/python"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="$PYTHON_BIN"
elif [[ -x "$VENV_PYTHON" ]]; then
  PYTHON="$VENV_PYTHON"
else
  PYTHON="$(command -v python3)"
  echo "warning: .venv312 not found; using $PYTHON" >&2
fi

export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-3000}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "HyperFlow H3 WebUI: http://$HOST:$PORT"
exec "$PYTHON" -u -m webui.server --host "$HOST" --port "$PORT"
