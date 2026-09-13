#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KickGuard Temporal-Graph V7 EventFix + adaptive negative sampling.

This entry point reuses the validated event-aware parser from kickguard_v7.py
and replaces only the sample builder / LOSO handling that caused WELL_000005
to fail when far-pre-event data were sparse.

Run this file for training.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import kickguard_v7 as base

ADAPTIVE_TRAIN_CACHE = "train_features_v7_eventfix_2_adaptive.npz"
OLD_REDUCED_TAG = "v7_eventfix_1"
TARGET_NEG_PER_WELL = 107


def _make_feature(base_df, event, delta, label):
    end_time = event - pd.Timedelta(seconds=int(delta))
    tmin = base_df["__time"].min()
    tmax = base_df["__time"].max()
    if end_time <= tmin or end_time > tmax:
        return None
    seg = base.slice_by_end_time(base_df, end_time, history=1800)
    if seg is None:
        return None
    try:
        f, _, _, _, regime, _ = base.extract_features(seg)
    except Exception:
        return None
    return f, int(label), int(regime), int(delta)


def build_well_samples_adaptive(df, well):
    event = pd.Timestamp(base.EVENT_TIMES[well])

    # Positive: slice end lies within the 30-minute pre-kick horizon.
    positive_deltas = np.arange(
        30,
        base.POSITIVE_HORIZON + 1,
        base.POSITIVE_STRIDE,
        dtype=int,
    )

    # Negative stage A: hard negatives immediately outside the 30-minute
    # horizon. Starting at 31 min guarantees the slice end is outside the
    # positive horizon while keeping the operating regime close to positives.
    near_negative_deltas = np.arange(
        base.POSITIVE_HORIZON + 60,
        3600 + 1,
        60,
        dtype=int,
    )

    # Negative stage B: farther negatives provide diversity whenever the
    # well has sufficiently continuous earlier data.
    far_negative_deltas = np.arange(
        3600 + 180,
        base.NEGATIVE_LOOKBACK + 1,
        base.NEGATIVE_STRIDE,
        dtype=int,
    )

    positives = []
    near_negatives = []
    far_negatives = []

    for delta in positive_deltas:
        item = _make_feature(df, event, delta, 1)
        if item is not None:
            positives.append(item)

    for delta in near_negative_deltas:
        item = _make_feature(df, event, delta, 0)
        if item is not None:
            near_negatives.append(item)

    for delta in far_negative_deltas:
        item = _make_feature(df, event, delta, 0)
        if item is not None:
            far_negatives.append(item)

    if len(positives) < 5:
        raise RuntimeError(
            f"Too few positive samples for {well}: {len(positives)}. "
            "Event-time parsing or local sampling continuity is still invalid."
        )

    negatives = near_negatives + far_negatives

    # Keep per-well contribution comparable to the original 107 negatives.
    # Prefer all hard negatives, then spread remaining selections across far
    # history so one dense time region cannot dominate the model.
    if len(negatives) > TARGET_NEG_PER_WELL:
        keep_near = min(len(near_negatives), TARGET_NEG_PER_WELL)
        selected = list(near_negatives[:keep_near])
        remaining = TARGET_NEG_PER_WELL - len(selected)
        if remaining > 0 and far_negatives:
            ii = np.linspace(0, len(far_negatives) - 1, remaining)
            ii = np.unique(np.rint(ii).astype(int))
            selected.extend(far_negatives[i] for i in ii)
        negatives = selected[:TARGET_NEG_PER_WELL]

    # We do NOT fabricate post-event negatives. If a well has sparse earlier
    # history, the other nine wells still provide abundant negatives. LOSO is
    # made robust below for a held-out well with few negatives.
    if len(negatives) < 5:
        print(
            f"[warning] {well}: only {len(negatives)} valid negatives; "
            "continuing without post-event leakage"
        )

    items = positives + negatives
    X = np.vstack([x[0] for x in items]).astype(np.float32)
    y = np.asarray([x[1] for x in items], dtype=np.int8)
    regimes = np.asarray([x[2] for x in items], dtype=np.int8)

    print(
        f"[samples] {well}: n={len(y)}, pos={int(y.sum())}, "
        f"neg={int((y == 0).sum())}, "
        f"hardNeg={len(near_negatives)}, farNeg={len(far_negatives)}"
    )
    return X, y, regimes


def build_train_features_adaptive(train_dir, work_dir, force=False):
    train_dir = Path(train_dir)
    work_dir = Path(work_dir)
    cache = work_dir / ADAPTIVE_TRAIN_CACHE

    if cache.exists() and not force:
        print(f"[cache] {cache}")
        z = np.load(cache, allow_pickle=True)
        return z["X"], z["y"], z["groups"], z["regimes"]

    Xs, ys, gs, rs = [], [], [], []

    for gi in range(1, 11):
        well = f"WELL_{gi:06d}"
        reduced = work_dir / f"{well}_reduced_{OLD_REDUCED_TAG}.pkl"

        if reduced.exists() and not force:
            print(f"[cache] {reduced}")
            df = pd.read_pickle(reduced)
            base.validate_event_range(df, well)
        else:
            # Only reparse a well if its already validated reduced cache is
            # unavailable. This keeps the successful event-aware scan reusable.
            df = base.load_well_from_train(train_dir, well)
            df.to_pickle(reduced)

        X, y, regimes = build_well_samples_adaptive(df, well)
        Xs.append(X)
        ys.append(y)
        rs.append(regimes)
        gs.append(np.full(len(y), gi - 1, dtype=np.int8))

    X = np.vstack(Xs).astype(np.float32)
    y = np.concatenate(ys)
    groups = np.concatenate(gs)
    regimes = np.concatenate(rs)

    if np.unique(y).size != 2:
        raise RuntimeError("Global training set does not contain both classes")

    np.savez_compressed(
        cache,
        X=X,
        y=y,
        groups=groups,
        regimes=regimes,
    )

    print(
        f"[train] X={X.shape}, positive={int(y.sum())}, "
        f"negative={int((y == 0).sum())}"
    )
    return X, y, groups, regimes


def loso_oof_robust(X, y, groups, pos_weight):
    oof = np.zeros(len(y), dtype=float)
    per_well = []

    for g in np.unique(groups):
        tr = groups != g
        va = groups == g
        yv = y[va]

        print()
        print(f"[LOSO] WELL_{int(g)+1:06d}")
        print(
            f"       train={int(tr.sum())} val={int(va.sum())} "
            f"valPos={int(yv.sum())} valNeg={int((yv == 0).sum())}"
        )

        model = base.fit_ensemble(
            X[tr], y[tr], pos_weight=pos_weight, seed=3407 + int(g)
        )
        p = base.predict_ensemble(model, X[va])
        oof[va] = p

        if np.unique(yv).size >= 2:
            try:
                auc = float(roc_auc_score(yv, p))
            except Exception:
                auc = float("nan")
            local = base.search_threshold(yv, p)
            row = {"well": int(g) + 1, "auc": auc, **local}
            print(
                f"       AUC={auc:.5f} Score={local['score']:.5f} "
                f"Recall={local['recall']:.5f} Spec={local['specificity']:.5f}"
            )
        else:
            # Do not invent specificity for a validation group containing one
            # class only. Overall OOF below remains valid because other wells
            # supply both classes.
            auc = float("nan")
            row = {
                "well": int(g) + 1,
                "auc": auc,
                "score": None,
                "recall": None,
                "specificity": None,
                "threshold": None,
                "positive_rate": None,
                "note": "single-class held-out fold",
            }
            print("       single-class held-out fold; local AUC/Score skipped")

        per_well.append(row)

    overall = base.search_threshold(y, oof)
    print()
    print("=" * 70)
    print(f"[OOF] Score={overall['score']:.6f}")
    print(f"[OOF] Recall={overall['recall']:.6f}")
    print(f"[OOF] Specificity={overall['specificity']:.6f}")
    print(f"[OOF] Threshold={overall['threshold']:.8f}")
    print(f"[OOF] PositiveRate={overall['positive_rate']:.6f}")
    print("=" * 70)
    return oof, overall, per_well


def main():
    # Monkey-patch only the two pieces that need adaptive handling. The rest of
    # the complete Temporal-Graph V7 pipeline remains exactly the validated
    # EventFix implementation.
    base.build_train_features = build_train_features_adaptive
    base.loso_oof = loso_oof_robust

    parser = base.build_parser()
    args = parser.parse_args()
    base.run(args)


if __name__ == "__main__":
    main()
