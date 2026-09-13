#!/usr/bin/env bash
set -euo pipefail

python kickguard_v8_retrain.py \
  --train-dir ../train \
  --test-dir ../test \
  --work-dir ./work_v7 \
  --output-dir ./v8_submit \
  --main-k 120 \
  --graph-weight 0.03 \
  2>&1 | tee v8_train.log
