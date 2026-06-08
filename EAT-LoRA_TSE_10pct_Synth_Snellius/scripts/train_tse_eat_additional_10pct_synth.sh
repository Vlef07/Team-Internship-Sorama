#!/usr/bin/env bash
# this script runs the training part of the pipeline on the additional dataset
# step one optionally build synthetic copies of raw train wavs with 15 db noise
# step two train one tse model per machine using about ten percent synth batches
# step three train eat lora for 6000 steps using the same ten ninety mix
set -euo pipefail

: "${PYTHON:?}"
: "${CODE_DIR:?}"
: "${SCRATCH_ADDITIONAL_RAW:?}"
: "${SCRATCH_ADDITIONAL_SYNTH:?}"
: "${SAVE_DIR:?}"
: "${TSE_DIR:?}"
: "${RESULTS_DIR:?}"

TSE_STEPS="${TSE_STEPS:-3000}"
TSE_BATCH="${TSE_BATCH:-8}"
EAT_NUM_STEPS="${EAT_NUM_STEPS:-6000}"
EAT_SAVE_INTERVAL="${EAT_SAVE_INTERVAL:-2000}"
TRAIN_MIX_FRACTION="${TRAIN_MIX_SYNTH_FRACTION:-0.1}"
SYNTH_SN_DB="${SYNTH_SN_DB:-15}"
SYNTH_REGENERATE="${SYNTH_REGENERATE:-0}"

# tell python to mix ten percent from synth folder and ninety percent from raw folder
train_mix_args=(
  --mix-second-data-root "$SCRATCH_ADDITIONAL_RAW"
  --mix-primary-fraction "$TRAIN_MIX_FRACTION"
)

mkdir -p "$SAVE_DIR" "$TSE_DIR" "$RESULTS_DIR" "$SCRATCH_ADDITIONAL_SYNTH"

# step one create synthetic train data if the synth folder is empty or if forced
if [ "$SYNTH_REGENERATE" = "1" ] || [ -z "$(ls -A "$SCRATCH_ADDITIONAL_SYNTH" 2>/dev/null || true)" ]; then
  echo "Synthese additional train raw -> synth op scratch"
  synth_extra=()
  if [ -n "$SYNTH_SN_DB" ]; then
    synth_extra+=( --synth-noise-snr-db "$SYNTH_SN_DB" )
  fi
  "${PYTHON}" -u "${CODE_DIR}/synthesize_dcase_data.py" \
    --input-root "$SCRATCH_ADDITIONAL_RAW" \
    --output-root "$SCRATCH_ADDITIONAL_SYNTH" \
    --copies-per-source "${SYNTH_COPIES:-2}" \
    --sample-rate 16000 \
    --duration-sec 10.0 \
    --seed "${SYNTH_SEED:-42}" \
    "${synth_extra[@]}"
else
  echo "Hergebruik synth op scratch: $SCRATCH_ADDITIONAL_SYNTH"
fi

echo ""
echo "TSE training with synth primary and raw mix fraction ${TRAIN_MIX_FRACTION}"
# step two train tse per machine and save one checkpoint file per machine name
"${PYTHON}" -u "${CODE_DIR}/train_tse_snellius.py" \
  --data-root "$SCRATCH_ADDITIONAL_SYNTH" \
  --save-dir "$TSE_DIR" \
  --num-steps "$TSE_STEPS" \
  --batch-size "$TSE_BATCH" \
  --log-interval 50 \
  --loss-csv "$RESULTS_DIR/tse_loss_additional_10pct_synth.csv" \
  "${train_mix_args[@]}"

echo ""
echo "EAT LoRA training with the same mix"
# step three remove old eat checkpoints if synth was rebuilt so labels stay consistent
if [ "${SYNTH_REGENERATE:-0}" = "1" ] && ls "$SAVE_DIR"/checkpoint_*.pt >/dev/null 2>&1; then
  echo "SYNTH_REGENERATE=1: verwijder oude EAT checkpoints (labels kunnen gewijzigd zijn)"
  rm -f "$SAVE_DIR"/checkpoint_*.pt
fi
# step four train eat lora and write checkpoints every 2000 steps
"${PYTHON}" -u "${CODE_DIR}/train_eat_lora_snellius.py" \
  --data-root "$SCRATCH_ADDITIONAL_SYNTH" \
  --save-dir "$SAVE_DIR" \
  --batch-size 32 \
  --log-interval 100 \
  --save-interval "$EAT_SAVE_INTERVAL" \
  --num-workers 4 \
  --auto-resume \
  --loss-csv "$RESULTS_DIR/training_loss_additional_10pct_synth.csv" \
  --tensorboard-dir "$RESULTS_DIR/tensorboard" \
  "${train_mix_args[@]}"

echo "TSE in: $TSE_DIR"
echo "EAT in: $SAVE_DIR"
