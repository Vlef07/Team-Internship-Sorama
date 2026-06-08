#!/usr/bin/env bash
# this script copies large dataset folders from the project disk to fast scratch storage
# scratch is reused if the folder already has files so reruns start faster
# it copies the additional train raw data and the evaluation test data
set -euo pipefail

: "${SCRATCH_BASE:?}"
: "${SCRATCH_STEM:?}"

# first check that the source dataset paths exist on the project disk
# shellcheck source=preflight_data_paths.sh
source "${EVAL_PROJECT_DIR}/scripts/preflight_data_paths.sh"

SCRATCH_ADDITIONAL_RAW="${SCRATCH_ADDITIONAL_RAW:-$SCRATCH_BASE/$SCRATCH_STEM/additional_train_raw}"
SCRATCH_ADDITIONAL_SYNTH="${SCRATCH_ADDITIONAL_SYNTH:-$SCRATCH_BASE/$SCRATCH_STEM/additional_train_synth}"
SCRATCH_EVAL_TEST="${SCRATCH_EVAL_TEST:-$SCRATCH_BASE/$SCRATCH_STEM/evaluation_test}"

# copy one source folder to scratch if the destination is still empty
stage_dir () {
  local src="$1"
  local dst="$2"
  local label="$3"

  mkdir -p "$dst"
  if [ -n "$(ls -A "$dst" 2>/dev/null || true)" ]; then
    echo "[$label] Hergebruik scratch: $dst"
    return 0
  fi

  if [ ! -d "$src" ]; then
    echo "[$label] Bronmap ontbreekt: $src"
    return 1
  fi

  echo "[$label] Staging $src -> $dst"
  rsync -a "$src/" "$dst/"
}

fail=0
stage_dir "$ADDITIONAL_TRAIN_HOME" "$SCRATCH_ADDITIONAL_RAW" "additional train" || fail=1
stage_dir "$EVAL_TEST_HOME" "$SCRATCH_EVAL_TEST" "evaluation test" || fail=1

if [ "$fail" -ne 0 ]; then
  echo ""
  echo "Staging mislukt. Zet beide datasets in EVAL_PROJECT_DIR of exporteer paden:"
  echo "  export ADDITIONAL_TRAIN_HOME=..."
  echo "  export EVAL_TEST_HOME=..."
  exit 1
fi

export SCRATCH_ADDITIONAL_RAW SCRATCH_ADDITIONAL_SYNTH SCRATCH_EVAL_TEST
