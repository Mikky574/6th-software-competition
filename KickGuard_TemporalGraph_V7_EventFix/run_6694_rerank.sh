#!/usr/bin/env bash
set -euo pipefail

BASE116=${BASE116:-./V5_feedback_dropbad_116.zip}
BASE124=${BASE124:-./V5_meta_plus4_124.zip}
WORK=${WORK:-./work_v7}
OUT=${OUT:-./rerank_6694}

ARGS=(--work-dir "$WORK" --output-dir "$OUT")
[[ -f "$BASE116" ]] && ARGS+=(--base116 "$BASE116")
[[ -f "$BASE124" ]] && ARGS+=(--base124 "$BASE124")

if [[ ! -f "$BASE116" && ! -f "$BASE124" ]]; then
  echo "Neither 66.9444 anchor was found."
  echo "Set BASE116=/path/to/V5_feedback_dropbad_116.zip and/or BASE124=/path/to/V5_meta_plus4_124.zip"
  exit 2
fi

python kickguard_6694_rerank.py "${ARGS[@]}"
