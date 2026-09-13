# KickGuard V8 ReTrain

This version performs a real retraining pass while reusing the event-time caches already validated by V7 EventFix.

## What changed

- No post-event leakage.
- Positive windows: event-1min to event-30min.
- Hard negatives: event-31min to event-60min.
- Far negatives: event-65min to event-6h, capped per well.
- Per-well/class sample weighting so WELL_000005 cannot be overwhelmed by dense wells.
- Two feature views:
  - signed log features;
  - absolute log features for direction-invariant event changes.
- Six model branches:
  - ExtraTrees signed;
  - HistGradientBoosting signed;
  - LogisticRegression signed;
  - ExtraTrees absolute;
  - HistGradientBoosting absolute;
  - ExtraTrees regression on continuous event proximity.
- Strict leave-one-well-out OOF evaluation.
- Final branch weights come only from OOF AUC above 0.5; weak branches get zero weight.
- Temporal Graph default contribution is only 3% and no-graph submissions are also emitted.
- Fixed Top-K outputs are produced around the historically useful 116-124 range.

## Important

Do **not** delete `work_v7` if the EventFix scan has already passed. V8 reuses:

```text
work_v7/WELL_000001_reduced_v7_eventfix_1.pkl
...
work_v7/WELL_000010_reduced_v7_eventfix_1.pkl
work_v7/test_features_v7_eventfix_1.npz
```

## Run

```bash
python kickguard_v8_retrain.py \
  --train-dir ../train \
  --test-dir ../test \
  --work-dir ./work_v7 \
  --output-dir ./v8_submit
```

or:

```bash
bash run_v8.sh
```

Use `--force-rebuild` only if you intentionally want to rebuild the V8 training feature cache. It does not need to be used for a normal first V8 run.

## Outputs

```text
v8_submit/result_v8.zip
v8_submit/result_v8_top112.zip
v8_submit/result_v8_top116.zip
v8_submit/result_v8_top120.zip
v8_submit/result_v8_top124.zip
v8_submit/result_v8_top128.zip
v8_submit/result_v8_nograph_top116.zip
v8_submit/result_v8_nograph_top120.zip
v8_submit/result_v8_nograph_top124.zip

work_v7/V8_oof.csv
work_v7/V8_test_debug.csv
work_v7/V8_graph_edges.csv
work_v7/V8_summary.json
```

## What to inspect before submitting

The key log block is:

```text
[V8 OOF] branch AUC:
...
[V8 OOF] weighted blend AUC=...
```

If weighted OOF AUC is still below 0.5, do not submit the retrained outputs. If it reaches a clearly useful level, compare graph and no-graph candidates before using leaderboard attempts.
