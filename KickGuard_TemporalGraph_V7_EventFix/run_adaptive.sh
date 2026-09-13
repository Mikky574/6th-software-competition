#!/usr/bin/env bash
set -euo pipefail

python kickguard_v7_adaptive.py \
  --train-dir ../train \
  --test-dir ../test \
  --work-dir ./work_v7 \
  --output ./result_v7.zip \
  2>&1 | tee v7_train_adaptive.log
