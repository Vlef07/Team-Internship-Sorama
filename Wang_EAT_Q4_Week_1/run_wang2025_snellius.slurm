#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=wang2025-eat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
##SBATCH --reservation=terv92681
#
# Zelfde pijplijn en volgorde als in test_Wang2025_Luc.ipynb (uitvoer op cluster):
#   Stap 1 TSE train per machine  ->  notebook stappen 17 t/m 19  (uit met: RUN_TSE=0)
#   Stap 2 EAT LoRA               ->  notebook stap 20
#   Stap 3  KNN eval (baseline + evt. TSE)  ->  notebook stappen 21 t/m 22
#   Stap 4  Plots (loss, KNN)       ->  zelfde plot scripts als in het notebook na training
# Toggle TSE, zet vóór sbatch -> export RUN_TSE=0 om stap 1 en de TSE KNN-tak over te slaan.
# TSE_MODE hoort bij KNN-eval (off/test-only/both); default both = bank + test.
#
# Stap B (default): EAT trainen op ruwe mels, TSE alleen in KNN (both). Zet vóór sbatch:
#   export EAT_TSE_IN_TRAINING=1
# om Stap A te draaien (TSE ook in de EAT-training-loop via --train-tse-checkpoint-dir).

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

PROJECT_DIR="$HOME/Sorama_Internship/EAT_TSE_same_pipeline_train_eval/eat_tse_stap_b"
HOME_DATA_ROOT="$PROJECT_DIR/data/dcase2025t2/dev_data/raw"
SCRATCH_BASE="${SCRATCH:-$HOME/scratch}"
SCRATCH_DATA_ROOT="$SCRATCH_BASE/Sorama_Internship/EAT_TSE_same_pipeline_train_eval/eat_tse_stap_b/data/dcase2025t2/dev_data/raw"
SAVE_DIR="$PROJECT_DIR/checkpoints/eat_lora_system1"
TSE_DIR="$PROJECT_DIR/checkpoints/tse"

RESULTS_DIR="$PROJECT_DIR/results"
LOSS_CSV="$RESULTS_DIR/training_loss.csv"
TSE_LOSS_CSV="$RESULTS_DIR/tse_loss.csv"
TB_DIR="$RESULTS_DIR/tensorboard/${SLURM_JOB_ID:-local}"
EMBED_BS="${EMBED_BS:-32}"

# Toggle TSE for this run (1 = train TSE and evaluate with TSE, 0 = skip TSE).
RUN_TSE="${RUN_TSE:-1}"
TSE_STEPS="${TSE_STEPS:-3000}"
TSE_BATCH="${TSE_BATCH:-8}"
TSE_MODE="${TSE_MODE:-both}"
# Stap B default 0: geen TSE in EAT-train; Stap A: export EAT_TSE_IN_TRAINING=1
EAT_TSE_IN_TRAINING="${EAT_TSE_IN_TRAINING:-0}"

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$RESULTS_DIR"
mkdir -p "$SCRATCH_DATA_ROOT"
mkdir -p "$TSE_DIR"

if [ ! -d "$SCRATCH_DATA_ROOT" ] || [ -z "$(ls -A "$SCRATCH_DATA_ROOT")" ]; then
  echo "Staging dataset to scratch..."
  rsync -a "$HOME_DATA_ROOT/" "$SCRATCH_DATA_ROOT/"
fi

cd "$PROJECT_DIR"



# 1) TSE training, one model per machine type, based on Fujimura et al. (2025)


echo "STEP 1/4 - TSE training (Fujimura et al., 2025)"
if [ "$RUN_TSE" = "1" ]; then
  python -u train_tse_snellius.py \
    --data-root "$SCRATCH_DATA_ROOT" \
    --save-dir "$TSE_DIR" \
    --num-steps "$TSE_STEPS" \
    --batch-size "$TSE_BATCH" \
    --log-interval 50 \
    --loss-csv "$TSE_LOSS_CSV" \
    --skip-existing
else
  echo "RUN_TSE=0, overslaan TSE training"
fi



# 2) Training of the Wang 2025 EAT-LoRA System-1 classifier



echo "STEP 2/4 - EAT-LoRA training"

eat_train_tse_args=()
if [ "$RUN_TSE" = "1" ] && [ "$EAT_TSE_IN_TRAINING" = "1" ]; then
  eat_train_tse_args=( --train-tse-checkpoint-dir "$TSE_DIR" )
  echo "EAT met TSE op train-waveforms (Stap A, EAT_TSE_IN_TRAINING=1)"
else
  echo "EAT op ruwe waveforms (Stap B: geen --train-tse-checkpoint-dir); KNN gebruikt nog TSE indien RUN_TSE=1 en TSE_MODE"
fi

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
  --tensorboard-dir "$TB_DIR" \
  "${eat_train_tse_args[@]}"



# 3) KNN evaluation on every saved checkpoint, baseline and (optionally) TSE


echo "STEP 3/4 - KNN evaluation on all checkpoints"

shopt -s nullglob
CKPTS=( "$SAVE_DIR"/checkpoint_step_*.pt "$SAVE_DIR"/checkpoint_last.pt )
shopt -u nullglob

run_eval () {
  local ckpt="$1"
  local tag="$2"
  local tse_mode="$3"
  local tse_dir_arg=()

  local out_csv="$RESULTS_DIR/knn_eval_${tag}.csv"
  if [ -f "$out_csv" ]; then
    echo "Skipping $ckpt ($tag) - results already at $out_csv"
    return
  fi

  if [ "$tse_mode" != "off" ]; then
    tse_dir_arg=( --tse-checkpoint-dir "$TSE_DIR" )
  fi

  echo "Evaluating $ckpt (tse=$tse_mode)"
  python -u eval_wang2025_knn_snellius.py \
    --data-root "$SCRATCH_DATA_ROOT" \
    --checkpoint "$ckpt" \
    --embed-batch-size "$EMBED_BS" \
    --tse-mode "$tse_mode" \
    "${tse_dir_arg[@]}" \
    --output-csv "$out_csv"
}

if [ ${#CKPTS[@]} -eq 0 ]; then
  echo "No checkpoints found in $SAVE_DIR, skipping eval."
else
  for ckpt in "${CKPTS[@]}"; do
    [ -f "$ckpt" ] || continue
    base_tag="$(basename "$ckpt" .pt | sed 's/checkpoint_//')"

    run_eval "$ckpt" "$base_tag" "off"

    if [ "$RUN_TSE" = "1" ]; then
      run_eval "$ckpt" "${base_tag}_tse_${TSE_MODE}" "$TSE_MODE"
    fi
  done
fi




# 4) Plots


echo "STEP 4/4 - Plots"

if [ -f "$LOSS_CSV" ]; then
  python -u plot_training_loss.py \
    --loss-csv "$LOSS_CSV" \
    --out-png "$RESULTS_DIR/training_loss.png" || echo "training loss plot failed"
else
  echo "No loss csv at $LOSS_CSV, skipping training loss plot."
fi

python -u plot_knn_per_machine.py \
  --pattern "$RESULTS_DIR/knn_eval_*.csv" \
  --variant baseline \
  --out-png "$RESULTS_DIR/knn_per_machine_baseline.png" || echo "per-machine baseline plot failed"

if [ "$RUN_TSE" = "1" ]; then
  python -u plot_knn_per_machine.py \
    --pattern "$RESULTS_DIR/knn_eval_*.csv" \
    --variant tse \
    --out-png "$RESULTS_DIR/knn_per_machine_tse.png" || echo "per-machine TSE plot failed"
fi

python -u plot_knn_performance.py \
  --pattern "$RESULTS_DIR/knn_eval_*.csv" \
  --out-csv "$RESULTS_DIR/knn_performance_over_steps.csv" \
  --out-png "$RESULTS_DIR/knn_performance_over_steps.png" || echo "perf-over-steps plot failed"

echo "Done. Results in $RESULTS_DIR"
