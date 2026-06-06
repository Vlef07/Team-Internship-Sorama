#!/usr/bin/env bash
# this script runs before staging to find where the datasets live on disk
# it looks in common folder names inside the eval project directory
# you can also set ADDITIONAL_TRAIN_HOME and EVAL_TEST_HOME yourself
set -euo pipefail

: "${EVAL_PROJECT_DIR:?}"

# try a list of possible paths and return the first folder that exists
resolve_first_dir () {
  local label="$1"
  shift
  local p
  for p in "$@"; do
    if [ -n "$p" ] && [ -d "$p" ]; then
      echo "$p"
      return 0
    fi
  done
  echo "[$label] Geen map gevonden. Geprobeerd:" >&2
  for p in "$@"; do
    [ -n "$p" ] && echo "  - $p" >&2
  done
  return 1
}

echo "Preflight data paths (EVAL_PROJECT_DIR=$EVAL_PROJECT_DIR)"
echo "Inhoud projectmap:"
ls -la "$EVAL_PROJECT_DIR" 2>/dev/null | head -30 || true
echo ""

ADDITIONAL_TRAIN_HOME="${ADDITIONAL_TRAIN_HOME:-}"
if [ -z "$ADDITIONAL_TRAIN_HOME" ]; then
  ADDITIONAL_TRAIN_HOME="$(resolve_first_dir "Additional train" \
    "$EVAL_PROJECT_DIR/Additional dataset DCASE" \
    "$EVAL_PROJECT_DIR/additional dataset DCASE" \
    "$EVAL_PROJECT_DIR/data/dcase2025t2/eval_data/additional_train/raw" \
  )" || exit 1
fi

EVAL_TEST_HOME="${EVAL_TEST_HOME:-}"
if [ -z "$EVAL_TEST_HOME" ]; then
  EVAL_TEST_HOME="$(resolve_first_dir "Evaluation test" \
    "$EVAL_PROJECT_DIR/Evaluation dataset DCASE" \
    "$EVAL_PROJECT_DIR/evaluation dataset DCASE" \
    "$EVAL_PROJECT_DIR/data/dcase2025t2/eval_data/evaluation/raw" \
    "$HOME/scratch/EAT-LoRA_TSE_10pct_Synth_Snellius/evaluation_test" \
    "$HOME/scratch/Evaluation_dataset_Both_EAT-LoRA_and_TSE_retrained_on_additional_dataset_10pct_synth_train_100pct_DCASE_eval/evaluation_test" \
    "$HOME/scratch/10pct_Synth_Train_EAT-LoRA_TSE/dcase_eval/evaluation_test" \
  )" || {
    echo "" >&2
    echo "FIX: upload Evaluation dataset DCASE naar het project, bv.:" >&2
    echo "  scp -r \"Evaluation dataset DCASE\" ltheunissen@snellius:.../$EVAL_PROJECT_DIR/" >&2
    echo "Of zet pad handmatig: export EVAL_TEST_HOME=/pad/naar/Evaluation\\ dataset\\ DCASE" >&2
    exit 1
  }
fi

export ADDITIONAL_TRAIN_HOME EVAL_TEST_HOME
echo "OK Additional train: $ADDITIONAL_TRAIN_HOME"
echo "OK Evaluation test:  $EVAL_TEST_HOME"
