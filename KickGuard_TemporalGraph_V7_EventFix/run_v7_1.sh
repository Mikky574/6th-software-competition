#!/usr/bin/env bash
set -euo pipefail

python kickguard_v7_1.py \
  --train-dir ../train \
  --test-dir ../test \
  --work-dir ./work_v7 \
  --output-dir ./v7_1_submit \
  "$@" \
  2>&1 | tee v7_1_train.log
