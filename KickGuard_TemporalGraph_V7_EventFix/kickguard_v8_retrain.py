#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KickGuard V8 ReTrain
====================

A clean retraining pipeline built after V7/V7.1 diagnostics.

Key changes:
- reuses validated event-aware reduced caches from V7 EventFix
- no post-event leakage
- per-well balanced sampling and sample weights
- signed + direction-invariant feature branches
- continuous event-proximity regression branch
- strict leave-one-well-out diagnostics
- branch weights chosen only from OOF performance
- Temporal Graph is optional and weak; it is not the main model
- fixed Top-K submissions around the historically useful 116-124 region

Run:
python kickguard_v8_retrain.py \
  --train-dir ../train \
  --test-dir ../test \
  --work-dir ./work_v7 \
  --output-dir ./v8_submit
"""
from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import kickguard_v7 as base

CACHE_TAG = "v8_retrain_1"
OLD_REDUCED_TAG = "v7_eventfix_1"
TOPKS = [112, 116, 120, 124, 128]

POS_DELTAS = np.arange(60, 1800 + 1, 60, dtype=int)
HARD_NEG_DELTAS = np.arange(1860, 3600 + 1, 60, dtype=int)
FAR_NEG_DELTAS = np.arange(3900, 21600 + 1, 300, dtype=int)
MAX_FAR_NEG = 30


def finite(x, default=0.0):
    try:
        return float(x) if np.isfinite(x) else default
    except Exception:
        return default


def rank01(x):
    x = np.asarray(x, dtype=float)
    if len(x) <= 1:
        return np.zeros_like(x)
    bad = ~np.isfinite(x)
    if bad.any():
        med = np.nanmedian(x)
        if not np.isfinite(med):
            med = 0.0
        x = x.copy()
        x[bad] = med
    order = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort")
    return order.astype(float) / float(len(x) - 1)


def topk_pred(score, k):
    score = np.asarray(score, dtype=float)
    k = int(np.clip(k, 1, len(score) - 1))
    idx = np.argsort(score, kind="mergesort")[::-1][:k]
    out = np.zeros(len(score), dtype=np.int8)
    out[idx] = 1
    return out


def competition_metric(y, pred):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    recall = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return 100.0 * recall * recall * spec, recall, spec


def risk_target(delta):
    d = float(delta)
    if d <= 300:
        return 1.00
    if d <= 600:
        return 0.90
    if d <= 1200:
        return 0.72
    if d <= 1800:
        return 0.55
    if d <= 2700:
        return 0.18
    if d <= 3600:
        return 0.08
    return 0.0


def _sample_one(df, event, delta, label, sample_type):
    end_time = event - pd.Timedelta(seconds=int(delta))
    tmin = df["__time"].min()
    tmax = df["__time"].max()
    if end_time <= tmin or end_time > tmax:
        return None
    seg = base.slice_by_end_time(df, end_time, history=1800)
    if seg is None:
        return None
    try:
        feat, _, _, _, regime, _ = base.extract_features(seg)
    except Exception:
        return None
    return {
        "x": np.asarray(feat, dtype=np.float32),
        "y": int(label),
        "risk": float(risk_target(delta)),
        "delta": int(delta),
        "regime": int(regime),
        "type": sample_type,
    }


def build_well_samples(df, well):
    event = pd.Timestamp(base.EVENT_TIMES[well])
    pos, hard, far = [], [], []

    for d in POS_DELTAS:
        z = _sample_one(df, event, d, 1, "pos")
        if z is not None:
            pos.append(z)
    for d in HARD_NEG_DELTAS:
        z = _sample_one(df, event, d, 0, "hard")
        if z is not None:
            hard.append(z)
    for d in FAR_NEG_DELTAS:
        z = _sample_one(df, event, d, 0, "far")
        if z is not None:
            far.append(z)

    if len(pos) < 10:
        raise RuntimeError(f"{well}: too few positives: {len(pos)}")

    if len(far) > MAX_FAR_NEG:
        ii = np.unique(np.rint(np.linspace(0, len(far) - 1, MAX_FAR_NEG)).astype(int))
        far = [far[i] for i in ii]

    samples = pos + hard + far
    nneg = sum(s["y"] == 0 for s in samples)
    if nneg < 5:
        print(f"[warning] {well}: only {nneg} valid negatives; keep training without post-event leakage")

    print(
        f"[V8 samples] {well}: n={len(samples)}, pos={len(pos)}, "
        f"hardNeg={len(hard)}, farNeg={len(far)}"
    )
    return samples


def build_train_cache(train_dir, work_dir, force=False):
    cache = work_dir / f"train_{CACHE_TAG}.npz"
    if cache.exists() and not force:
        print(f"[cache] {cache}")
        z = np.load(cache, allow_pickle=True)
        return (
            z["X"], z["y"], z["risk"], z["groups"],
            z["regimes"], z["deltas"], z["types"].astype(str),
        )

    Xs, ys, rs, gs, regimes, deltas, types = [], [], [], [], [], [], []
    for gi in range(1, 11):
        well = f"WELL_{gi:06d}"
        reduced = work_dir / f"{well}_reduced_{OLD_REDUCED_TAG}.pkl"
        if reduced.exists() and not force:
            print(f"[cache] {reduced}")
            df = pd.read_pickle(reduced)
            base.validate_event_range(df, well)
        else:
            df = base.load_well_from_train(train_dir, well)
            df.to_pickle(reduced)

        samples = build_well_samples(df, well)
        Xs.extend(s["x"] for s in samples)
        ys.extend(s["y"] for s in samples)
        rs.extend(s["risk"] for s in samples)
        gs.extend([gi - 1] * len(samples))
        regimes.extend(s["regime"] for s in samples)
        deltas.extend(s["delta"] for s in samples)
        types.extend(s["type"] for s in samples)

    X = np.vstack(Xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.int8)
    risk = np.asarray(rs, dtype=np.float32)
    groups = np.asarray(gs, dtype=np.int8)
    regimes = np.asarray(regimes, dtype=np.int8)
    deltas = np.asarray(deltas, dtype=np.int32)
    types = np.asarray(types, dtype=object)

    np.savez_compressed(
        cache, X=X, y=y, risk=risk, groups=groups,
        regimes=regimes, deltas=deltas, types=types,
    )
    print(f"[V8 train] X={X.shape}, pos={int(y.sum())}, neg={int((y==0).sum())}")
    return X, y, risk, groups, regimes, deltas, types.astype(str)


def fill_matrix(X, med=None):
    X = np.asarray(X, dtype=float)
    if med is None:
        med = np.nanmedian(X, axis=0)
        med[~np.isfinite(med)] = 0.0
    out = X.copy()
    rr, cc = np.where(~np.isfinite(out))
    if len(rr):
        out[rr, cc] = med[cc]
    out = np.clip(out, -1e8, 1e8)
    return out, med


def signed_log_view(X):
    X = np.asarray(X, dtype=float)
    return np.sign(X) * np.log1p(np.abs(X))


def abs_log_view(X):
    X = np.asarray(X, dtype=float)
    return np.log1p(np.abs(X))


def balanced_weights(groups, y):
    groups = np.asarray(groups)
    y = np.asarray(y)
    w = np.zeros(len(y), dtype=float)
    unique_groups = np.unique(groups)
    for g in unique_groups:
        idx = np.where(groups == g)[0]
        pos = idx[y[idx] == 1]
        neg = idx[y[idx] == 0]
        if len(pos):
            w[pos] = 0.5 / len(pos)
        if len(neg):
            w[neg] = 0.5 / len(neg)
        if len(pos) == 0:
            w[neg] = 1.0 / max(len(neg), 1)
        if len(neg) == 0:
            w[pos] = 1.0 / max(len(pos), 1)
    if w.mean() > 0:
        w /= w.mean()
    else:
        w[:] = 1.0
    return w


def fit_models(X, y, risk, groups, seed=3407):
    X0, med = fill_matrix(X)
    Xs = signed_log_view(X0)
    Xa = abs_log_view(X0)
    sw = balanced_weights(groups, y)

    scaler = StandardScaler()
    Xlr = scaler.fit_transform(Xs)

    et_signed = ExtraTreesClassifier(
        n_estimators=700,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight=None,
        random_state=seed,
        n_jobs=-1,
    )
    et_signed.fit(Xs, y, sample_weight=sw)

    hgb_signed = HistGradientBoostingClassifier(
        learning_rate=0.035,
        max_iter=320,
        max_leaf_nodes=31,
        min_samples_leaf=15,
        l2_regularization=2.0,
        random_state=seed + 1,
    )
    hgb_signed.fit(Xs, y, sample_weight=sw)

    lr = LogisticRegression(
        C=0.20,
        max_iter=2500,
        solver="liblinear",
        random_state=seed + 2,
    )
    lr.fit(Xlr, y, sample_weight=sw)

    et_abs = ExtraTreesClassifier(
        n_estimators=700,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight=None,
        random_state=seed + 3,
        n_jobs=-1,
    )
    et_abs.fit(Xa, y, sample_weight=sw)

    hgb_abs = HistGradientBoostingClassifier(
        learning_rate=0.035,
        max_iter=320,
        max_leaf_nodes=31,
        min_samples_leaf=15,
        l2_regularization=2.0,
        random_state=seed + 4,
    )
    hgb_abs.fit(Xa, y, sample_weight=sw)

    reg = ExtraTreesRegressor(
        n_estimators=700,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=seed + 5,
        n_jobs=-1,
    )
    reg.fit(Xa, risk, sample_weight=sw)

    return {
        "med": med,
        "scaler": scaler,
        "et_signed": et_signed,
        "hgb_signed": hgb_signed,
        "lr": lr,
        "et_abs": et_abs,
        "hgb_abs": hgb_abs,
        "reg_abs": reg,
    }


def predict_models(model, X):
    X0, _ = fill_matrix(X, model["med"])
    Xs = signed_log_view(X0)
    Xa = abs_log_view(X0)
    Xlr = model["scaler"].transform(Xs)

    out = {
        "et_signed": model["et_signed"].predict_proba(Xs)[:, 1],
        "hgb_signed": model["hgb_signed"].predict_proba(Xs)[:, 1],
        "lr": model["lr"].predict_proba(Xlr)[:, 1],
        "et_abs": model["et_abs"].predict_proba(Xa)[:, 1],
        "hgb_abs": model["hgb_abs"].predict_proba(Xa)[:, 1],
        "reg_abs": np.clip(model["reg_abs"].predict(Xa), 0.0, 1.0),
    }
    return out


def oof_branch_predictions(X, y, risk, groups):
    branch_names = ["et_signed", "hgb_signed", "lr", "et_abs", "hgb_abs", "reg_abs"]
    oof = {k: np.zeros(len(y), dtype=float) for k in branch_names}
    rows = []

    for g in np.unique(groups):
        tr = groups != g
        va = groups == g
        print(f"\n[V8 LOSO] WELL_{int(g)+1:06d} train={int(tr.sum())} val={int(va.sum())}")
        model = fit_models(X[tr], y[tr], risk[tr], groups[tr], seed=5000 + int(g))
        pred = predict_models(model, X[va])
        for k in branch_names:
            oof[k][va] = pred[k]

        local_aucs = {}
        for k in branch_names:
            try:
                local_aucs[k] = float(roc_auc_score(y[va], pred[k]))
            except Exception:
                local_aucs[k] = float("nan")

        local_blend = np.mean(np.vstack([rank01(pred[k]) for k in branch_names]), axis=0)
        try:
            blend_auc = float(roc_auc_score(y[va], local_blend))
        except Exception:
            blend_auc = float("nan")
        kdiag = max(int(y[va].sum()), round(int(va.sum()) * 0.38))
        m = competition_metric(y[va], topk_pred(local_blend, kdiag))
        print(
            f"         blendAUC={blend_auc:.5f} RankScore@{kdiag}={m[0]:.5f} "
            f"Recall={m[1]:.5f} Spec={m[2]:.5f}"
        )
        rows.append({
            "well": int(g) + 1,
            "blend_auc": blend_auc,
            "rank_score": m[0],
            "recall": m[1],
            "specificity": m[2],
            **{f"auc_{k}": local_aucs[k] for k in branch_names},
        })

    branch_auc = {}
    for k in branch_names:
        try:
            branch_auc[k] = float(roc_auc_score(y, oof[k]))
        except Exception:
            branch_auc[k] = float("nan")

    # OOF-only branch weighting: branches at/below random receive zero weight.
    raw_w = {}
    for k in branch_names:
        a = branch_auc[k]
        edge = max((a if np.isfinite(a) else 0.5) - 0.5, 0.0)
        raw_w[k] = edge * edge
    if sum(raw_w.values()) <= 1e-12:
        raw_w = {k: 1.0 for k in branch_names}
    s = sum(raw_w.values())
    weights = {k: raw_w[k] / s for k in branch_names}

    oof_blend = np.zeros(len(y), dtype=float)
    for k in branch_names:
        oof_blend += weights[k] * rank01(oof[k])
    oof_blend = rank01(oof_blend)
    try:
        blend_auc = float(roc_auc_score(y, oof_blend))
    except Exception:
        blend_auc = float("nan")

    print("\n" + "=" * 72)
    print("[V8 OOF] branch AUC:")
    for k in branch_names:
        print(f"  {k:12s}: AUC={branch_auc[k]:.6f} weight={weights[k]:.4f}")
    print(f"[V8 OOF] weighted blend AUC={blend_auc:.6f}")
    print("=" * 72)
    return oof, oof_blend, branch_auc, weights, rows, blend_auc


def graph_adjust(score, starts, ends, missing, anchors, n_domains):
    try:
        domains = base.infer_hidden_domains(starts, ends, missing, n_domains)
        edges = base.build_temporal_graph(starts, ends, domains)
        pgraph = base.graph_propagation(score, anchors, edges)
        out = 0.95 * rank01(score) + 0.05 * rank01(pgraph)
        return rank01(out), domains, edges
    except Exception as e:
        print(f"[graph warning] {e}; graph disabled")
        return rank01(score), np.zeros(len(score), dtype=int), []


def write_zip(ids, pred, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"切片ID": ids, "溢流判断": np.asarray(pred, dtype=int)})
    if len(df) != 324 or df["切片ID"].nunique() != 324:
        raise RuntimeError("submission must contain 324 unique IDs")
    if not set(df["溢流判断"].unique()).issubset({0, 1}):
        raise RuntimeError("labels must be binary")
    raw = df.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("result.csv", raw)
    print(f"[output] {path} positive={int(df['溢流判断'].sum())}")


def run(args):
    train_dir = Path(args.train_dir).resolve()
    test_dir = Path(args.test_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[path] train={train_dir}")
    print(f"[path] test ={test_dir}")
    print(f"[path] work ={work_dir}")
    print(f"[path] out  ={out_dir}")

    X, y, risk, groups, train_regimes, deltas, types = build_train_cache(
        train_dir, work_dir, force=args.force_rebuild
    )
    XT, ids, starts, ends, missing, test_regimes, anchors = base.build_test_features(
        test_dir, work_dir, force=False
    )
    if X.shape[1] != XT.shape[1]:
        raise RuntimeError(f"feature mismatch train={X.shape} test={XT.shape}")

    oof_branches, oof_blend, branch_auc, weights, per_well, oof_auc = oof_branch_predictions(
        X, y, risk, groups
    )

    oof_df = pd.DataFrame({
        "well": groups + 1,
        "label": y,
        "risk_target": risk,
        "delta_sec": deltas,
        "sample_type": types,
        "oof_blend": oof_blend,
    })
    for k, v in oof_branches.items():
        oof_df[k] = v
    oof_df.to_csv(work_dir / "V8_oof.csv", index=False)

    print("[V8 final] fitting all wells")
    model = fit_models(X, y, risk, groups, seed=8808)
    p = predict_models(model, XT)

    model_score = np.zeros(len(ids), dtype=float)
    for k, w in weights.items():
        model_score += w * rank01(p[k])
    model_score = rank01(model_score)

    graph_score, domains, edges = graph_adjust(
        model_score, starts, ends, missing, anchors, args.hidden_domains
    )

    final_score = rank01(
        (1.0 - args.graph_weight) * rank01(model_score)
        + args.graph_weight * rank01(graph_score)
    )

    debug = pd.DataFrame({
        "切片ID": ids,
        "model_score": model_score,
        "graph_score": graph_score,
        "final_score": final_score,
        "physical_anchor": anchors,
        "domain": domains,
        "regime": test_regimes,
    })
    for k in p:
        debug[k] = p[k]
    debug["rank"] = debug["final_score"].rank(method="first", ascending=False).astype(int)
    debug.sort_values("rank").to_csv(work_dir / "V8_test_debug.csv", index=False)

    candidate_ks = sorted(set(TOPKS + [args.main_k]))
    outputs = []
    for k in candidate_ks:
        path = out_dir / f"result_v8_top{k}.zip"
        write_zip(ids, topk_pred(final_score, k), path)
        outputs.append(str(path))

    main_path = out_dir / "result_v8.zip"
    write_zip(ids, topk_pred(final_score, args.main_k), main_path)

    # Also emit pure-model versions so graph can be ablated without retraining.
    for k in [116, 120, 124]:
        path = out_dir / f"result_v8_nograph_top{k}.zip"
        write_zip(ids, topk_pred(model_score, k), path)
        outputs.append(str(path))

    edge_rows = []
    for e in edges:
        try:
            edge_rows.append({
                "src_id": ids[e["src"]],
                "dst_id": ids[e["dst"]],
                "domain": e.get("domain", -1),
                "distance": e.get("distance", np.nan),
                "confidence": e.get("confidence", np.nan),
            })
        except Exception:
            pass
    pd.DataFrame(edge_rows).to_csv(work_dir / "V8_graph_edges.csv", index=False)

    summary = {
        "version": "KickGuard V8 ReTrain",
        "train_shape": list(X.shape),
        "test_shape": list(XT.shape),
        "oof_auc": finite(oof_auc, -1),
        "branch_auc": {k: finite(v, -1) for k, v in branch_auc.items()},
        "branch_weights": {k: finite(v, 0) for k, v in weights.items()},
        "per_well": per_well,
        "graph_weight": float(args.graph_weight),
        "graph_edges": int(len(edges)),
        "domain_counts": np.bincount(domains).tolist() if len(domains) else [],
        "main_k": int(args.main_k),
        "candidate_ks": candidate_ks,
        "outputs": outputs + [str(main_path)],
    }
    with open(work_dir / "V8_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("KickGuard V8 ReTrain completed")
    print(f"OOF AUC      : {oof_auc:.6f}")
    print(f"Graph edges  : {len(edges)}")
    print(f"Graph weight : {args.graph_weight:.3f}")
    print(f"MAIN         : {main_path} (K={args.main_k})")
    print(f"SUMMARY      : {work_dir/'V8_summary.json'}")
    print("Candidates   : " + ", ".join(str(k) for k in candidate_ks))
    print("=" * 72)


def parser():
    ap = argparse.ArgumentParser(description="KickGuard V8 robust retraining")
    ap.add_argument("--train-dir", default="../train")
    ap.add_argument("--test-dir", default="../test")
    ap.add_argument("--work-dir", default="./work_v7")
    ap.add_argument("--output-dir", default="./v8_submit")
    ap.add_argument("--main-k", type=int, default=120)
    ap.add_argument("--hidden-domains", type=int, default=7)
    ap.add_argument("--graph-weight", type=float, default=0.03)
    ap.add_argument("--force-rebuild", action="store_true")
    return ap


if __name__ == "__main__":
    args = parser().parse_args()
    if not (0.0 <= args.graph_weight <= 0.10):
        raise SystemExit("--graph-weight should be between 0 and 0.10")
    run(args)
