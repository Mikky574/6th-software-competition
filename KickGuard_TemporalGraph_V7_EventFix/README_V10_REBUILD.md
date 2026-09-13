# KickGuard V10 Baseline-Rebuild

This version is used when historical high-score submission ZIPs are no longer available.

It rebuilds a submission from the original train/test data using the already validated EventFix reduced caches.

## Core design

- strict event-aware parsing inherited from `kickguard_v7.py`
- no post-event negatives
- balanced positive / hard-negative / far-negative sampling per well
- global HGB backbone
- source-well-biased HGB experts
- strict LOSO router/expert validation
- router based on stable well-domain features
- no Temporal Graph in the default score
- fixed Top-K outputs: 112, 116, 120, 124, 128

## Run

```bash
python kickguard_v10_rebuild.py \
  --train-dir ../train \
  --test-dir ../test \
  --work-dir ./work_v7 \
  --output-dir ./v10_submit
```

or:

```bash
bash run_v10.sh
```

Do not delete the validated `work_v7/WELL_*_reduced_v7_eventfix_1.pkl` caches.

## Outputs

- `v10_submit/result_v10.zip`
- `v10_submit/result_v10_top112.zip`
- `v10_submit/result_v10_top116.zip`
- `v10_submit/result_v10_top120.zip`
- `v10_submit/result_v10_top124.zip`
- `v10_submit/result_v10_top128.zip`
- global-only and MoE-only ablations for K=116/120/124
- `work_v7/V10_oof.csv`
- `work_v7/V10_test_debug.csv`
- `work_v7/V10_summary.json`

Always inspect strict LOSO metrics before spending leaderboard submissions.
