#!/usr/bin/env bash
set -euo pipefail
python kickguard_v10_rebuild_strict.py \
  --train-dir ../train \
  --test-dir ../test \
  --work-dir ./work_v7 \
  --output-dir ./v10_submit \
  2>&1 | tee v10_strict_train.log
