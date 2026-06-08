#!/usr/bin/env bash
# this script turns knn result csv files into png plots and a latex table
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_DIR="${CODE_DIR:-$PROJECT_DIR/code}"

# shellcheck source=activate_snellius_env.sh
source "${SCRIPT_DIR}/activate_snellius_env.sh"

RESULTS_DIR="${RESULTS_DIR:-$PROJECT_DIR/results}"
KNN_PATTERN="${KNN_PATTERN:-$RESULTS_DIR/knn_eval_*_dcase_eval.csv}"
KNN_CSV_FOR_LATEX="${KNN_CSV_FOR_LATEX:-}"

mkdir -p "$RESULTS_DIR"

pick_latest_csv() {
  local pattern="$1"
  shopt -s nullglob
  local files=( $pattern )
  shopt -u nullglob
  if [ ${#files[@]} -eq 0 ]; then
    return 1
  fi
  ls -1 "${files[@]}" | sort -V | tail -n 1
}

if [ -z "$KNN_CSV_FOR_LATEX" ]; then
  KNN_CSV_FOR_LATEX="$(pick_latest_csv "$KNN_PATTERN" || true)"
fi

echo "Stap 1: KNN per machine (baseline)"
"${PYTHON}" -u "${CODE_DIR}/plot_knn_per_machine.py" \
  --pattern "$KNN_PATTERN" \
  --variant baseline \
  --prefer highest_step \
  --out-png "$RESULTS_DIR/knn_per_machine_baseline_dcase_eval.png" \
  || echo "baseline per-machine plot mislukt"

shopt -s nullglob
tse_csvs=( "$RESULTS_DIR"/knn_eval_*_tse_*_dcase_eval.csv )
shopt -u nullglob
if [ ${#tse_csvs[@]} -gt 0 ]; then
  echo "Stap 2: KNN per machine (TSE)"
  "${PYTHON}" -u "${CODE_DIR}/plot_knn_per_machine.py" \
    --pattern "$KNN_PATTERN" \
    --variant tse \
    --prefer highest_step \
    --out-png "$RESULTS_DIR/knn_per_machine_tse_dcase_eval.png" \
    || echo "TSE per-machine plot mislukt"
fi

echo "Stap 3: KNN performance over checkpoints"
"${PYTHON}" -u "${CODE_DIR}/plot_knn_performance.py" \
  --pattern "$KNN_PATTERN" \
  --out-csv "$RESULTS_DIR/knn_performance_over_steps_dcase_eval.csv" \
  --out-png "$RESULTS_DIR/knn_performance_over_steps_dcase_eval.png" \
  || echo "knn performance plot mislukt"

TRAINING_LOSS_CSV="${TRAINING_LOSS_CSV:-}"
if [ -n "$TRAINING_LOSS_CSV" ] && [ -f "$TRAINING_LOSS_CSV" ]; then
  echo "Stap 4: training loss"
  "${PYTHON}" -u "${CODE_DIR}/plot_training_loss.py" \
    --loss-csv "$TRAINING_LOSS_CSV" \
    --out-png "$RESULTS_DIR/training_loss_dcase_eval_context.png" \
    || echo "training loss plot mislukt"
else
  echo "Stap 4: training loss overgeslagen"
fi

if [ -n "$KNN_CSV_FOR_LATEX" ] && [ -f "$KNN_CSV_FOR_LATEX" ]; then
  echo "Stap 5: LaTeX tabel"
  "${PYTHON}" -u "${CODE_DIR}/export_knn_latex_table.py" \
    --input-csv "$KNN_CSV_FOR_LATEX" \
    --out-tex "$RESULTS_DIR/knn_dcase_eval_table.tex" \
    --caption "kNN evaluation on the DCASE 2025 Task 2 evaluation dataset (EAT-LoRA step 6000, 10\\% synthetic training mix, 15\\,dB synth SNR)." \
    --label "tab:knn_dcase_eval_step6000_snr15" \
    --machine-order AutoTrash BandSealer CoffeeGrinder HomeCamera Polisher ScrewFeeder ToyPet ToyRCCar
fi

echo "Klaar. Resultaten in $RESULTS_DIR"
