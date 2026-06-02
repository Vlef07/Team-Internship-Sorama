#!/bin/bash
# one time venv setup on snellius login node:
#   cd ~/projects/EAT-LoRA_TSE_10pct_Synth_Snellius
#   bash setup_snellius.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ="${REQ:-$SCRIPT_DIR/requirements_snellius.txt}"

if [ ! -f "$REQ" ]; then
  echo "Missing requirements file: $REQ"
  exit 1
fi

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

VENV="${VENV:-$HOME/venvs/wang2025}"
mkdir -p "$(dirname "$VENV")"

venv_python_ok() {
  [ -f "$VENV/bin/activate" ] && "$VENV/bin/python" -c "import sys" >/dev/null 2>&1
}

if venv_python_ok; then
  echo "Using existing venv at $VENV"
else
  if [ -d "$VENV" ]; then
    echo "Recreating broken venv at $VENV"
    rm -rf "$VENV"
  else
    echo "Creating venv at $VENV"
  fi
  python -m venv "$VENV"
fi

# shellcheck source=/dev/null
source "$VENV/bin/activate"

if ! python -c "import sys; print(sys.executable)" >/dev/null 2>&1; then
  echo "FATAL: venv python still broken after create/recreate at $VENV"
  exit 1
fi

python -m pip install --upgrade pip

python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.1.2+cu121 torchaudio==2.1.2+cu121 torchvision==0.16.2+cu121

python -m pip install --no-cache-dir -r "$REQ"
python -m pip install --no-cache-dir timm==0.9.16 --no-deps

python - <<'PY'
import sys
import torch, torchaudio, torchvision, timm, numpy as np
print("python", sys.executable)
print("torch", torch.__version__)
print("torchaudio", torchaudio.__version__)
print("torchvision", torchvision.__version__)
print("timm", timm.__version__)
print("numpy", np.__version__)
print("cuda", torch.version.cuda)
PY

echo "Venv ready at $VENV"
