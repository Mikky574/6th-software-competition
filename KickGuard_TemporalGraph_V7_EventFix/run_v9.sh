#!/usr/bin/env bash
set -euo pipefail

python kickguard_v9_feedback.py \
  --anchor116 ./V5_feedback_dropbad_116.zip \
  --anchor124 ./V5_meta_plus4_124.zip \
  --v7-zip ./result_v7.zip \
  --v8-zero ./v8_submit/result_v8_nograph_top116.zip \
  --work-dir ./work_v7 \
  --output-dir ./v9_submit \
  2>&1 | tee v9_feedback.log
