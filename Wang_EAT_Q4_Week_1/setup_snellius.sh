#!/bin/bash
# Eenmalige venv op Snellius. Pijplijn zelf: run_wang2025_snellius.slurm (zelfde stappen als het notebook).
set -euo pipefail

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

VENV="$HOME/venvs/wang2025"
python -m venv "$VENV"
source "$VENV/bin/activate"

python -m pip install --upgrade pip

python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.1.2+cu121 torchaudio==2.1.2+cu121 torchvision==0.16.2+cu121

REQ="$HOME/Sorama_Internship/EAT_TSE_same_pipeline_train_eval/eat_tse_knn_both/requirements_snellius.txt"
python -m pip install --no-cache-dir -r "$REQ"
python -m pip install --no-cache-dir timm==0.9.16 --no-deps

python - <<'PY'
import torch, torchaudio, torchvision, timm, numpy as np
print("torch", torch.__version__)
print("torchaudio", torchaudio.__version__)
print("torchvision", torchvision.__version__)
print("timm", timm.__version__)
print("numpy", np.__version__)
print("cuda", torch.version.cuda)
PY
