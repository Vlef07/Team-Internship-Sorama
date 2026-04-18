#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=wang2025-eat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=12:00:00
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

PROJECT_DIR="$HOME/Sorama_Internship/Victor_week_9"
HOME_DATA_ROOT="$PROJECT_DIR/data/dcase2025t2/dev_data/raw"
SCRATCH_BASE="${SCRATCH:-$HOME/scratch}"
SCRATCH_DATA_ROOT="$SCRATCH_BASE/Sorama_Internship/Victor_week_9/data/dcase2025t2/dev_data/raw"
SAVE_DIR="$PROJECT_DIR/checkpoints/eat_lora_system1"

RESULTS_DIR="$PROJECT_DIR/results"
LOSS_CSV="$RESULTS_DIR/training_loss.csv"
TB_DIR="$RESULTS_DIR/tensorboard/${SLURM_JOB_ID:-local}"
EMBED_BS="${EMBED_BS:-32}"

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$RESULTS_DIR"
mkdir -p "$SCRATCH_DATA_ROOT"

if [ ! -d "$SCRATCH_DATA_ROOT" ] || [ -z "$(ls -A "$SCRATCH_DATA_ROOT")" ]; then
  echo "Staging dataset to scratch..."
  rsync -a "$HOME_DATA_ROOT/" "$SCRATCH_DATA_ROOT/"
fi

cd "$PROJECT_DIR"

# ---------------------------------------------------------------------------
# 1) Training
# ---------------------------------------------------------------------------
echo "=========================================="
echo "STEP 1/3 - Training"
echo "=========================================="
python -u train_wang2025_snellius.py \
  --data-root "$SCRATCH_DATA_ROOT" \
  --save-dir "$SAVE_DIR" \
  --batch-size 32 \
  --num-steps 20000 \
  --log-interval 100 \
  --save-interval 2000 \
  --num-workers 4 \
  --auto-resume \
  --loss-csv "$LOSS_CSV" \
  --tensorboard-dir "$TB_DIR"

# ---------------------------------------------------------------------------
# 2) KNN evaluation on every saved checkpoint
# ---------------------------------------------------------------------------
echo "=========================================="
echo "STEP 2/3 - KNN evaluation on all checkpoints"
echo "=========================================="

shopt -s nullglob
CKPTS=( "$SAVE_DIR"/checkpoint_step_*.pt "$SAVE_DIR"/checkpoint_last.pt )
shopt -u nullglob

if [ ${#CKPTS[@]} -eq 0 ]; then
  echo "No checkpoints found in $SAVE_DIR, skipping eval."
else
  for ckpt in "${CKPTS[@]}"; do
    [ -f "$ckpt" ] || continue
    tag="$(basename "$ckpt" .pt | sed 's/checkpoint_//')"
    out_csv="$RESULTS_DIR/knn_eval_${tag}.csv"

    if [ -f "$out_csv" ]; then
      echo "Skipping $ckpt (results already at $out_csv)"
      continue
    fi

    echo "----- Evaluating $ckpt -----"
    python -u eval_wang2025_knn_snellius.py \
      --data-root "$SCRATCH_DATA_ROOT" \
      --checkpoint "$ckpt" \
      --embed-batch-size "$EMBED_BS" \
      --output-csv "$out_csv"
  done
fi

# ---------------------------------------------------------------------------
# 3) Plots
# ---------------------------------------------------------------------------
echo "=========================================="
echo "STEP 3/3 - Plots"
echo "=========================================="

if [ -f "$LOSS_CSV" ]; then
  python -u plot_training_loss.py \
    --loss-csv "$LOSS_CSV" \
    --out-png "$RESULTS_DIR/training_loss.png" || echo "training loss plot failed"
else
  echo "No loss csv at $LOSS_CSV, skipping training loss plot."
fi

python -u plot_knn_per_machine.py \
  --pattern "$RESULTS_DIR/knn_eval_*.csv" \
  --out-png "$RESULTS_DIR/knn_per_machine.png" || echo "per-machine plot failed"

python -u plot_knn_performance.py \
  --pattern "$RESULTS_DIR/knn_eval_*.csv" \
  --out-csv "$RESULTS_DIR/knn_performance_over_steps.csv" \
  --out-png "$RESULTS_DIR/knn_performance_over_steps.png" || echo "perf-over-steps plot failed"

echo "Done. Results in $RESULTS_DIR"
