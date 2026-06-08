#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=dcase-full-retrain
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=48:00:00
#SBATCH --output=/home/%u/%x_%j.out
#SBATCH --error=/home/%u/%x_%j.err
#
# full pipeline: synth + TSE + EAT-LoRA (10% synth mix) + KNN eval + plots
# cd to this folder on snellius then: sbatch run_full_pipeline.slurm

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  EVAL_PROJECT_DIR="${EVAL_PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
else
  EVAL_PROJECT_DIR="${EVAL_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
fi
CODE_DIR="${CODE_DIR:-$EVAL_PROJECT_DIR/code}"

export EVAL_PROJECT_DIR CODE_DIR

source "${EVAL_PROJECT_DIR}/scripts/activate_snellius_env.sh"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-9}"

SCRATCH_BASE="${SCRATCH:-$HOME/scratch}"
SCRATCH_STEM="${SCRATCH_PROJECT_STEM:-$(basename "$EVAL_PROJECT_DIR")}"

ADDITIONAL_TRAIN_HOME="${ADDITIONAL_TRAIN_HOME:-}"
EVAL_TEST_HOME="${EVAL_TEST_HOME:-}"
EVALUATOR_HOME="${EVALUATOR_HOME:-$EVAL_PROJECT_DIR/dcase2025_task2_evaluator}"

SAVE_DIR="${SAVE_DIR:-$EVAL_PROJECT_DIR/checkpoints/eat_lora_additional_10pct_synth}"
TSE_DIR="${TSE_DIR:-$EVAL_PROJECT_DIR/checkpoints/tse_additional_10pct_synth}"
RESULTS_DIR="${RESULTS_DIR:-$EVAL_PROJECT_DIR/results}"
EAT_CHECKPOINT="${EAT_CHECKPOINT:-$SAVE_DIR/checkpoint_step_6000.pt}"

TSE_MODE="${TSE_MODE:-both}"
EMBED_BS="${EMBED_BS:-32}"
TEAM_NAME="${TEAM_NAME:-our_team}"
SYSTEM_NAME="${SYSTEM_NAME:-eat_lora_tse_retrained_additional_10pct_synth_dcase_eval}"

mkdir -p "$SAVE_DIR" "$TSE_DIR" "$RESULTS_DIR"

echo "preflight and staging"
export SCRATCH_BASE SCRATCH_STEM
source "${EVAL_PROJECT_DIR}/scripts/stage_additional_eval_scratch.sh"

export SAVE_DIR TSE_DIR RESULTS_DIR SCRATCH_ADDITIONAL_RAW SCRATCH_ADDITIONAL_SYNTH
bash "${EVAL_PROJECT_DIR}/scripts/train_tse_eat_additional_10pct_synth.sh"

if [ ! -f "$EAT_CHECKPOINT" ]; then
  latest="$(ls -1 "$SAVE_DIR"/checkpoint_step_*.pt 2>/dev/null | sort -V | tail -n 1 || true)"
  if [ -n "$latest" ]; then
    EAT_CHECKPOINT="$latest"
  elif [ -f "$SAVE_DIR/checkpoint_last.pt" ]; then
    EAT_CHECKPOINT="$SAVE_DIR/checkpoint_last.pt"
  else
    echo "Geen EAT checkpoint in $SAVE_DIR"
    exit 1
  fi
fi
export EAT_CHECKPOINT

export RUN_TSE=1
export SCRATCH_EVAL_TEST
export EVALUATOR_HOME TEAM_NAME SYSTEM_NAME EMBED_BS TSE_DIR
bash "${EVAL_PROJECT_DIR}/scripts/run_dcase_knn_eval.sh"

export KNN_PATTERN="$RESULTS_DIR/knn_eval_*_dcase_eval.csv"
export TRAINING_LOSS_CSV="${TRAINING_LOSS_CSV:-$RESULTS_DIR/training_loss_additional_10pct_synth.csv}"
bash "${EVAL_PROJECT_DIR}/scripts/dcase_eval_snellius_plot.sh"

echo "Klaar."
echo "  EAT: $EAT_CHECKPOINT"
echo "  TSE: $TSE_DIR"
echo "  CSV: $RESULTS_DIR"
