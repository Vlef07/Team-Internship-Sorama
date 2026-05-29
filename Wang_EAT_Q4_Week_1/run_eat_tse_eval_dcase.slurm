#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=eat-tse-eval-dcase
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
##SBATCH --reservation=terv92681
#
# Hertrain EAT-LoRA + TSE op de ADDITIONAL dataset en scoor de EVALUATION dataset,
# zodat de officiele dcase2025_task2_evaluator.py een ranking-score kan berekenen.
#
#   STEP 1/5  TSE training per machine        (Additional/<machine>/train)   -- uit met RUN_TSE=0
#   STEP 2/5  EAT-LoRA training               (Additional/<machine>/train)
#   STEP 3/5  Inference op Evaluation         -> teams/<team>/<system>/*.csv  (anomaly + decision)
#   STEP 4/5  Plots                            (training loss, TSE loss, score-histogrammen)
#   STEP 5/5  Officiele DCASE evaluator        -> teams_result/ + teams_additional_result/ (plots, csv)
#
# Toggles (zet vóór sbatch):
#   export RUN_TSE=0                 # sla TSE over (baseline EAT-LoRA only)
#   export TSE_MODE=both             # KNN/inference TSE: off | test-only | both (default both)
#   export EAT_TSE_IN_TRAINING=1     # Stap A: TSE ook in de EAT-train-loop (default 0 = Stap B)
#   export TEAM_NAME=... SYSTEM_NAME=...

set -euo pipefail

module purge
module load 2023

VENV="$HOME/venvs/wang2025"
if [ ! -d "$VENV" ]; then
  echo "Missing venv at $VENV. Create it on a login node first (zie README)."
  exit 1
fi
source "$VENV/bin/activate"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# ---- Project- en datalocaties (override met env vars indien nodig) ----------
# Default = de map waarvandaan je sbatch draait (SLURM_SUBMIT_DIR), zodat het
# script zelf-lokaliserend is en niet afhangt van een geexporteerde PROJECT_DIR.
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

# Train (Additional) en eval (Evaluation) data. Maak deze als symlinks naar je
# echte datamappen (zie README), of zet de env vars naar de juiste paden.
ADDITIONAL_DATA_ROOT="${ADDITIONAL_DATA_ROOT:-$PROJECT_DIR/data/additional}"
EVALUATION_DATA_ROOT="${EVALUATION_DATA_ROOT:-$PROJECT_DIR/data/evaluation}"

# Officiele evaluator (clone van nttcslab/dcase2025_task2_evaluator).
EVALUATOR_DIR="${EVALUATOR_DIR:-$PROJECT_DIR/dcase2025_task2_evaluator}"

TEAM_NAME="${TEAM_NAME:-Sorama_EAT_TSE}"
SYSTEM_NAME="${SYSTEM_NAME:-eat_lora_tse_additional}"
TEAM_OUT_DIR="$EVALUATOR_DIR/teams/$TEAM_NAME/$SYSTEM_NAME"

SAVE_DIR="$PROJECT_DIR/checkpoints/eat_lora_system1"
TSE_DIR="$PROJECT_DIR/checkpoints/tse"
RESULTS_DIR="$PROJECT_DIR/results"
LOSS_CSV="$RESULTS_DIR/training_loss.csv"
TSE_LOSS_CSV="$RESULTS_DIR/tse_loss.csv"
TB_DIR="$RESULTS_DIR/tensorboard/${SLURM_JOB_ID:-local}"
PLOTS_DIR="$RESULTS_DIR/eval_inference"

EMBED_BS="${EMBED_BS:-32}"
RUN_TSE="${RUN_TSE:-1}"
TSE_STEPS="${TSE_STEPS:-3000}"
TSE_BATCH="${TSE_BATCH:-8}"
TSE_MODE="${TSE_MODE:-both}"
EAT_TSE_IN_TRAINING="${EAT_TSE_IN_TRAINING:-0}"
EAT_STEPS="${EAT_STEPS:-20000}"

mkdir -p "$PROJECT_DIR/logs" "$RESULTS_DIR" "$TSE_DIR" "$SAVE_DIR" "$TEAM_OUT_DIR" "$PLOTS_DIR"

# ---- Stage data naar scratch (sneller I/O) ----------------------------------
SCRATCH_BASE="${SCRATCH:-$HOME/scratch}"
SCRATCH_ADD="$SCRATCH_BASE/eat_tse_eval_dcase/additional"
SCRATCH_EVAL="$SCRATCH_BASE/eat_tse_eval_dcase/evaluation"
mkdir -p "$SCRATCH_ADD" "$SCRATCH_EVAL"

if [ -z "$(ls -A "$SCRATCH_ADD" 2>/dev/null)" ]; then
  echo "Staging Additional dataset to scratch..."
  rsync -a "$ADDITIONAL_DATA_ROOT/" "$SCRATCH_ADD/"
fi
if [ -z "$(ls -A "$SCRATCH_EVAL" 2>/dev/null)" ]; then
  echo "Staging Evaluation dataset to scratch..."
  rsync -a "$EVALUATION_DATA_ROOT/" "$SCRATCH_EVAL/"
fi

cd "$PROJECT_DIR"



# 1) TSE training, een model per machine (Fujimura et al., 2025) op Additional data

echo "STEP 1/5 - TSE training (Additional dataset)"
if [ "$RUN_TSE" = "1" ]; then
  python -u train_tse_snellius.py \
    --data-root "$SCRATCH_ADD" \
    --save-dir "$TSE_DIR" \
    --num-steps "$TSE_STEPS" \
    --batch-size "$TSE_BATCH" \
    --log-interval 50 \
    --loss-csv "$TSE_LOSS_CSV" \
    --skip-existing
else
  echo "RUN_TSE=0, TSE training overgeslagen"
fi



# 2) EAT-LoRA training op Additional data

echo "STEP 2/5 - EAT-LoRA training (Additional dataset)"

eat_train_tse_args=()
if [ "$RUN_TSE" = "1" ] && [ "$EAT_TSE_IN_TRAINING" = "1" ]; then
  eat_train_tse_args=( --train-tse-checkpoint-dir "$TSE_DIR" )
  echo "EAT met TSE op train-waveforms (Stap A, EAT_TSE_IN_TRAINING=1)"
else
  echo "EAT op ruwe waveforms (Stap B); TSE alleen in inference indien RUN_TSE=1 en TSE_MODE!=off"
fi

python -u train_wang2025_snellius.py \
  --data-root "$SCRATCH_ADD" \
  --save-dir "$SAVE_DIR" \
  --batch-size 32 \
  --num-steps "$EAT_STEPS" \
  --log-interval 100 \
  --save-interval 2000 \
  --num-workers 4 \
  --auto-resume \
  --loss-csv "$LOSS_CSV" \
  --tensorboard-dir "$TB_DIR" \
  "${eat_train_tse_args[@]}"



# 3) Inference op de Evaluation dataset -> DCASE submission CSV's

echo "STEP 3/5 - Inference op Evaluation dataset -> $TEAM_OUT_DIR"

CKPT="$SAVE_DIR/checkpoint_last.pt"
if [ ! -f "$CKPT" ]; then
  echo "checkpoint_last.pt ontbreekt in $SAVE_DIR; kan geen inference doen."
  exit 1
fi

infer_tse_args=()
INFER_TSE_MODE="off"
if [ "$RUN_TSE" = "1" ] && [ "$TSE_MODE" != "off" ]; then
  infer_tse_args=( --tse-checkpoint-dir "$TSE_DIR" )
  INFER_TSE_MODE="$TSE_MODE"
fi

python -u infer_eval_dcase_snellius.py \
  --train-data-root "$SCRATCH_ADD" \
  --eval-data-root "$SCRATCH_EVAL" \
  --checkpoint "$CKPT" \
  --embed-batch-size "$EMBED_BS" \
  --out-dir "$TEAM_OUT_DIR" \
  --plots-dir "$PLOTS_DIR" \
  --tse-mode "$INFER_TSE_MODE" \
  "${infer_tse_args[@]}"



# 4) Plots (loss-curves; score-histogrammen komen uit STEP 3)

echo "STEP 4/5 - Plots"

if [ -f "$LOSS_CSV" ]; then
  python -u plot_training_loss.py \
    --loss-csv "$LOSS_CSV" \
    --out-png "$RESULTS_DIR/training_loss.png" || echo "training loss plot failed"
else
  echo "No loss csv at $LOSS_CSV, skipping training loss plot."
fi

if [ "$RUN_TSE" = "1" ] && [ -f "$TSE_LOSS_CSV" ]; then
  python -u plot_training_loss.py \
    --loss-csv "$TSE_LOSS_CSV" \
    --out-png "$RESULTS_DIR/tse_loss.png" || echo "tse loss plot failed (verwacht: ander csv-formaat)"
fi



# 5) Officiele DCASE evaluator (AUC, pAUC, official score, precision/recall/F1 + plots)

echo "STEP 5/5 - Officiele DCASE evaluator"

if [ ! -d "$EVALUATOR_DIR" ]; then
  echo "Evaluator-map ontbreekt: $EVALUATOR_DIR"
  echo "Clone: git clone https://github.com/nttcslab/dcase2025_task2_evaluator.git \"$EVALUATOR_DIR\""
  exit 1
fi

cd "$EVALUATOR_DIR"
python -u dcase2025_task2_evaluator.py \
  --teams_root_dir ./teams \
  --result_dir ./teams_result \
  --additional_result_dir ./teams_additional_result \
  --dir_depth 2 \
  --out_all True

echo "Done."
echo "  Submission CSV's : $TEAM_OUT_DIR"
echo "  Officiele scores : $EVALUATOR_DIR/teams_result/${SYSTEM_NAME}_result.csv"
echo "  Extra + plots    : $EVALUATOR_DIR/teams_additional_result/"
echo "  Score-histogram  : $PLOTS_DIR"
