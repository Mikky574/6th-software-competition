# KickGuard Temporal-Graph V7.1

V7.1 is the follow-up to EventFix/Adaptive after V7 showed poor LOSO generalization.

## What changed

- Reuses validated `work_v7/WELL_*_reduced_v7_eventfix_1.pkl` caches.
- Does **not** reparse multi-GB raw data unless a reduced cache is missing.
- Learns continuous event-proximity risk instead of relying only on binary labels.
- Adds direction-invariant anomaly magnitude features to reduce cross-well sign reversal.
- Temporal Graph is a weak correction, not the main predictor.
- OOF positive rate is **not** used to choose final submission size.
- Always emits fixed Top-K candidates: `112/116/120/124/128/132`.
- Can optionally blend a previous TimeAware debug CSV. This is recommended when available because the earlier TimeAware branch validated much better than V7.

## Directory layout

Run from this project directory:

```text
competition/
├── train/
├── test/
└── base/  (or this downloaded project directory)
    ├── kickguard_v7.py
    ├── kickguard_v7_adaptive.py
    ├── kickguard_v7_1.py
    └── work_v7/
```

The train/test paths are expected to be:

```text
../train
../test
```

The EventFix code already handles:

- `WELL_0000010` as `WELL_000010`
- incorrect nested directory names such as `WELL_000006/WELL_000009时序数据`
- file-name based event-aware time reconstruction

## Install

```bash
pip install -r requirements.txt
```

## Recommended: reuse your existing validated cache

Do **not** delete `work_v7` if it already contains the successful EventFix reduced caches.

Expected files include:

```text
work_v7/WELL_000001_reduced_v7_eventfix_1.pkl
...
work_v7/WELL_000010_reduced_v7_eventfix_1.pkl
```

## Run standalone V7.1

```bash
python kickguard_v7_1.py \
  --train-dir ../train \
  --test-dir ../test \
  --work-dir ./work_v7 \
  --output-dir ./v7_1_submit
```

or:

```bash
bash run_v7_1.sh
```

## Recommended: blend with the old TimeAware branch

If you have the earlier `timeaware_blend_debug.csv` containing `切片ID` and `blend_prob`, run:

```bash
python kickguard_v7_1.py \
  --train-dir ../train \
  --test-dir ../test \
  --work-dir ./work_v7 \
  --output-dir ./v7_1_submit \
  --timeaware-debug ./timeaware_blend_debug.csv
```

Default rank-blend weights are:

```text
old TimeAware/old blend : 0.75
V7.1 invariant risk     : 0.20
Temporal Graph          : 0.05
```

You can change these with:

```text
--baseline-weight
--risk-weight
--graph-weight
```

When an external CSV is supplied, weights must sum to 1.

## Outputs

```text
v7_1_submit/result_v7_1.zip          # default K=120
v7_1_submit/result_v7_1_top112.zip
v7_1_submit/result_v7_1_top116.zip
v7_1_submit/result_v7_1_top120.zip
v7_1_submit/result_v7_1_top124.zip
v7_1_submit/result_v7_1_top128.zip
v7_1_submit/result_v7_1_top132.zip
```

Diagnostics:

```text
work_v7/V7_1_oof.csv
work_v7/V7_1_test_debug.csv
work_v7/V7_1_graph_edges.csv
work_v7/V7_1_summary.json
```

## What to send back after running

Please send the lines beginning with:

```text
[V7.1 LOSO]
[V7.1 OOF]
[domains]
[graph]
Temporal-Graph V7.1 completed
```

and the content of `work_v7/V7_1_summary.json` if convenient.

Do not submit all six candidates blindly. We use the diagnostics first to decide which 1-2 Top-K variants are worth leaderboard attempts.
