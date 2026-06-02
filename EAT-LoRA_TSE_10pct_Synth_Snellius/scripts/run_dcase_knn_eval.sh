#!/usr/bin/env bash
# this script runs knn anomaly detection on the official evaluation dataset
set -euo pipefail

: "${CODE_DIR:?CODE_DIR niet gezet}"
: "${SCRATCH_ADDITIONAL_RAW:?SCRATCH_ADDITIONAL_RAW niet gezet (KNN bank = raw additional train)}"
: "${PYTHON:?PYTHON niet gezet}"
: "${SCRATCH_EVAL_TEST:?SCRATCH_EVAL_TEST niet gezet}"
: "${EAT_CHECKPOINT:?EAT_CHECKPOINT niet gezet}"
: "${RESULTS_DIR:?RESULTS_DIR niet gezet}"

RUN_TSE="${RUN_TSE:-1}"
TSE_MODE="${TSE_MODE:-both}"
TSE_DIR="${TSE_DIR:-}"
EMBED_BS="${EMBED_BS:-32}"
EVALUATOR_ROOT="${EVALUATOR_HOME:-}"
TEAM_NAME="${TEAM_NAME:-our_team}"
SYSTEM_NAME="${SYSTEM_NAME:-eat_lora_tse_retrained_additional_10pct_synth_dcase_eval}"

ckpt_tag="$(basename "$EAT_CHECKPOINT" .pt | sed 's/checkpoint_//')"

run_one () {
  local tse_mode="$1"
  local tag_suffix="$2"
  local tse_args=()
  local out_csv="$RESULTS_DIR/knn_eval_${ckpt_tag}${tag_suffix}_dcase_eval.csv"

  if [ -f "$out_csv" ]; then
    echo "Overslaan (bestaat al): $out_csv"
    return
  fi

  if [ "$tse_mode" != "off" ]; then
    if [ -z "$TSE_DIR" ] || [ ! -d "$TSE_DIR" ]; then
      echo "TSE_DIR ontbreekt voor mode=$tse_mode, sla TSE-run over"
      return
    fi
    tse_args=( --tse-checkpoint-dir "$TSE_DIR" --tse-mode "$tse_mode" )
  else
    tse_args=( --tse-mode off )
  fi

  eval_args=(
    --additional-train-root "$SCRATCH_ADDITIONAL_RAW"
    --eval-test-root "$SCRATCH_EVAL_TEST"
    --checkpoint "$EAT_CHECKPOINT"
    --embed-batch-size "$EMBED_BS"
    --output-csv "$out_csv"
    --team-name "$TEAM_NAME"
    --system-name "$SYSTEM_NAME"
  )
  if [ -n "$EVALUATOR_ROOT" ] && [ -d "$EVALUATOR_ROOT" ]; then
    eval_args+=( --evaluator-root "$EVALUATOR_ROOT" )
  fi

  echo "dcase eval knn tse=$tse_mode -> $out_csv"
  "${PYTHON}" -u "${CODE_DIR}/eval_wang2025_dcase_eval_snellius.py" \
    "${eval_args[@]}" \
    "${tse_args[@]}"

  if [ -f "$out_csv" ]; then
    "${PYTHON}" -u "${CODE_DIR}/export_knn_latex_table.py" \
      --input-csv "$out_csv" \
      --out-tex "${out_csv%.csv}.tex" \
      --caption "kNN evaluation (${tse_mode}) on the DCASE 2025 Task 2 evaluation dataset (EAT-LoRA ${ckpt_tag}, 10\\% synthetic training mix, 15\\,dB synth SNR)." \
      --label "tab:knn_dcase_eval_${ckpt_tag}_${tse_mode}" \
      --machine-order AutoTrash BandSealer CoffeeGrinder HomeCamera Polisher ScrewFeeder ToyPet ToyRCCar \
      || true
  fi
}

run_one "off" ""

if [ "$RUN_TSE" = "1" ]; then
  run_one "$TSE_MODE" "_tse_${TSE_MODE}"
fi
