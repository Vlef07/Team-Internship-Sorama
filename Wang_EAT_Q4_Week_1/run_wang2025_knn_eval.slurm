#!/bin/bash
# Losse KNN eval job; zelfde eval script als STEP 3 in run_wang2025_snellius.slurm (notebook stappen 21 en 22).
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=wang2025-knn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
##SBATCH --reservation=terv92681

set -euo pipefail

module purge
module load 2023

VENV="$HOME/venvs/wang2025"
if [ ! -d "$VENV" ]; then
  echo "Missing venv at $VENV. Create it on a login node first."
  exit 1
fi
source "$VENV/bin/activate"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

PROJECT_DIR="$HOME/Sorama_Internship/EAT_TSE_same_pipeline_train_eval/eat_tse_knn_both"
HOME_DATA_ROOT="$PROJECT_DIR/data/dcase2025t2/dev_data/raw"
SCRATCH_BASE="${SCRATCH:-$HOME/scratch}"
SCRATCH_DATA_ROOT="$SCRATCH_BASE/Sorama_Internship/EAT_TSE_same_pipeline_train_eval/eat_tse_knn_both/data/dcase2025t2/dev_data/raw"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_DIR/checkpoints/eat_lora_system1}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-auto}"
RESULTS_CSV="${RESULTS_CSV:-$PROJECT_DIR/results/knn_eval_auto.csv}"
EMBED_BS="${EMBED_BS:-32}"
MACHINES="${MACHINES:-}"

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/results"
mkdir -p "$SCRATCH_DATA_ROOT"

if [ ! -d "$SCRATCH_DATA_ROOT" ] || [ -z "$(ls -A "$SCRATCH_DATA_ROOT")" ]; then
  echo "Staging dataset to scratch..."
  rsync -a "$HOME_DATA_ROOT/" "$SCRATCH_DATA_ROOT/"
fi

if [ "$CHECKPOINT_PATH" = "auto" ]; then
  latest_step_ckpt="$(ls -1 "$CHECKPOINT_DIR"/checkpoint_step_*.pt 2>/dev/null | sort -V | tail -n 1 || true)"
  if [ -n "$latest_step_ckpt" ]; then
    CHECKPOINT_PATH="$latest_step_ckpt"
  elif [ -f "$CHECKPOINT_DIR/checkpoint_last.pt" ]; then
    CHECKPOINT_PATH="$CHECKPOINT_DIR/checkpoint_last.pt"
  else
    echo "No checkpoint found in $CHECKPOINT_DIR"
    exit 1
  fi
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "Checkpoint not found: $CHECKPOINT_PATH"
  exit 1
fi

if [[ "$RESULTS_CSV" == *"knn_eval_auto.csv"* ]]; then
  ckpt_tag="$(basename "$CHECKPOINT_PATH" .pt | sed 's/checkpoint_//')"
  RESULTS_CSV="$PROJECT_DIR/results/knn_eval_${ckpt_tag}.csv"
fi

echo "Using checkpoint: $CHECKPOINT_PATH"
echo "Results csv: $RESULTS_CSV"

cd "$PROJECT_DIR"

if [ -n "$MACHINES" ]; then
  python -u eval_wang2025_knn_snellius.py \
    --data-root "$SCRATCH_DATA_ROOT" \
    --checkpoint "$CHECKPOINT_PATH" \
    --embed-batch-size "$EMBED_BS" \
    --output-csv "$RESULTS_CSV" \
    --machines $MACHINES
else
  python -u eval_wang2025_knn_snellius.py \
    --data-root "$SCRATCH_DATA_ROOT" \
    --checkpoint "$CHECKPOINT_PATH" \
    --embed-batch-size "$EMBED_BS" \
    --output-csv "$RESULTS_CSV"
fi

# Optioneel TSE-KNN (zelfde flags als eval_wang2025_knn_snellius.py):
#   --tse-checkpoint-dir "$PROJECT_DIR/checkpoints/tse" --tse-mode both
