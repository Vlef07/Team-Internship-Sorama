#!/usr/bin/env bash
# this script sets up paths and activates the python environment on snellius
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="${CODE_DIR:-$PROJECT_DIR/code}"
EVAL_PROJECT_DIR="${EVAL_PROJECT_DIR:-$PROJECT_DIR}"

if [ ! -d "$CODE_DIR" ]; then
  echo "CODE_DIR ontbreekt: $CODE_DIR"
  exit 1
fi

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

VENV="${VENV_OVERRIDE:-${VENV:-$HOME/venvs/eat_lora}}"
if [ ! -f "$VENV/bin/activate" ]; then
  echo "Missing venv at $VENV."
  echo "On a Snellius login node, from this project directory run:"
  echo "  bash setup_snellius.sh"
  exit 1
fi

# shellcheck source=/dev/null
source "$VENV/bin/activate"

if ! python -c "import sys" >/dev/null 2>&1; then
  echo "Broken venv at $VENV (python cannot start)."
  echo "Recreate it on a login node: bash setup_snellius.sh"
  exit 1
fi

PYTHON="$(python -c "import sys; print(sys.executable)")"
if [ -z "$PYTHON" ] || ! "$PYTHON" -c "import sys" >/dev/null 2>&1; then
  echo "Could not resolve a working python after activating $VENV."
  exit 1
fi

export VENV PYTHON VIRTUAL_ENV PROJECT_DIR CODE_DIR EVAL_PROJECT_DIR
export PROJECT_ROOT="${PROJECT_ROOT:-$PROJECT_DIR}"
export PATH="$VENV/bin:$PATH"
hash -r 2>/dev/null || true

"$PYTHON" - <<'PY'
import pandas
import torch
print("Omgeving klaar:", "python", __import__("sys").executable)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY
