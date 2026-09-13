#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KickGuard V10 Baseline-Rebuild / Well-Adaptive Mixture-of-Experts
=================================================================

Purpose
-------
Rebuild a strong baseline from the original training data when historical
submission ZIPs are no longer available.

Design
------
* reuse validated EventFix reduced caches (v7_eventfix_1)
* strict leave-one-well-out validation
* no post-event leakage
* balanced per-well sampling
* global HGB classifier as stable backbone
* one source-well-biased HGB expert per available training well
* router based on stable between-well / within-well feature separation
* router is rebuilt inside each LOSO fold, so held-out well never leaks
* test routing uses all ten source-well experts
* fixed Top-K candidates around the historically useful 112-128 range

Run
---
python kickguard_v10_rebuild.py \
  --train-dir ../train \
  --test-dir ../test \
  --work-dir ./work_v7 \
  --output-dir ./v10_submit
"""
from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

import kickguard_v7 as base

CACHE_TAG = "v10_rebuild_1"
REDUCED_TAG = "v7_eventfix_1"
TOPKS = [112, 116, 120, 124, 128]

POS_DELTAS = np.arange(60, 1800 + 1, 60, dtype=int)
HARD_NEG_DELTAS = np.arange(1860, 3600 + 1, 60, dtype=int)
FAR_NEG_DELTAS = np.arange(3900, 21600 + 1, 300, dtype=int)
MAX_FAR_NEG = 30


def finite(x, default=0.0):
    try:
        x = float(x)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def rank01(x):
    x = np.asarray(x, dtype=float)
    if len(x) <= 1:
        return np.zeros_like(x, dtype=float)
    bad = ~np.isfinite(x)
    if bad.any():
        med = np.nanmedian(x)
        if not np.isfinite(med):
            med = 0.0
        x = x.copy(); x[bad] = med
    order = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort")
    return order.astype(float) / float(len(x) - 1)


def topk_pred(score, k):
    score = np.asarray(score, dtype=float)
    k = int(np.clip(k, 1, len(score)-1))
    idx = np.argsort(score, kind="mergesort")[::-1][:k]
    out = np.zeros(len(score), dtype=np.int8)
    out[idx] = 1
    return out


def competition_metric(y, pred):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    tp = int(((y==1)&(pred==1)).sum())
    fn = int(((y==1)&(pred==0)).sum())
    tn = int(((y==0)&(pred==0)).sum())
    fp = int(((y==0)&(pred==1)).sum())
    rec = tp / max(tp+fn, 1)
    spec = tn / max(tn+fp, 1)
    return 100.0 * rec * rec * spec, rec, spec


def _sample(df, event, delta, label, kind):
    end_time = event - pd.Timedelta(seconds=int(delta))
    if end_time <= df["__time"].min() or end_time > df["__time"].max():
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
        "delta": int(delta),
        "kind": str(kind),
        "regime": int(regime),
    }


def build_well_samples(df, well):
    event = pd.Timestamp(base.EVENT_TIMES[well])
    pos, hard, far = [], [], []

    for d in POS_DELTAS:
        z = _sample(df, event, d, 1, "pos")
        if z is not None:
            pos.append(z)
    for d in HARD_NEG_DELTAS:
        z = _sample(df, event, d, 0, "hard")
        if z is not None:
            hard.append(z)
    for d in FAR_NEG_DELTAS:
        z = _sample(df, event, d, 0, "far")
        if z is not None:
            far.append(z)

    if len(pos) < 10:
        raise RuntimeError(f"{well}: too few positives={len(pos)}")

    if len(far) > MAX_FAR_NEG:
        ii = np.unique(np.rint(np.linspace(0, len(far)-1, MAX_FAR_NEG)).astype(int))
        far = [far[i] for i in ii]

    samples = pos + hard + far
    nneg = sum(z["y"] == 0 for z in samples)
    if nneg < 5:
        print(f"[warning] {well}: only {nneg} valid negatives; no post-event negatives will be fabricated")

    print(f"[V10 samples] {well}: n={len(samples)}, pos={len(pos)}, hard={len(hard)}, far={len(far)}")
    return samples


def build_train_cache(train_dir, work_dir, force=False):
    cache = work_dir / f"train_{CACHE_TAG}.npz"
    if cache.exists() and not force:
        print(f"[cache] {cache}")
        z = np.load(cache, allow_pickle=True)
        return z["X"], z["y"], z["groups"], z["deltas"], z["kinds"].astype(str), z["regimes"]

    Xs, ys, gs, ds, ks, rs = [], [], [], [], [], []
    for gi in range(1, 11):
        well = f"WELL_{gi:06d}"
        reduced = work_dir / f"{well}_reduced_{REDUCED_TAG}.pkl"
        if reduced.exists() and not force:
            print(f"[cache] {reduced}")
            df = pd.read_pickle(reduced)
            base.validate_event_range(df, well)
        else:
            df = base.load_well_from_train(train_dir, well)
            df.to_pickle(reduced)

        samples = build_well_samples(df, well)
        Xs.extend(z["x"] for z in samples)
        ys.extend(z["y"] for z in samples)
        gs.extend([gi-1] * len(samples))
        ds.extend(z["delta"] for z in samples)
        ks.extend(z["kind"] for z in samples)
        rs.extend(z["regime"] for z in samples)

    X = np.vstack(Xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.int8)
    groups = np.asarray(gs, dtype=np.int8)
    deltas = np.asarray(ds, dtype=np.int32)
    kinds = np.asarray(ks, dtype=object)
    regimes = np.asarray(rs, dtype=np.int8)
    np.savez_compressed(cache, X=X, y=y, groups=groups, deltas=deltas, kinds=kinds, regimes=regimes)
    print(f"[V10 train] X={X.shape}, pos={int(y.sum())}, neg={int((y==0).sum())}")
    return X, y, groups, deltas, kinds.astype(str), regimes


def fill_matrix(X, med=None):
    X = np.asarray(X, dtype=float)
    if med is None:
        med = np.nanmedian(X, axis=0)
        med[~np.isfinite(med)] = 0.0
    out = X.copy()
    rr, cc = np.where(~np.isfinite(out))
    if len(rr):
        out[rr, cc] = med[cc]
    return np.clip(out, -1e8, 1e8), med


def signed_log(X):
    X = np.asarray(X, dtype=float)
    return np.sign(X) * np.log1p(np.abs(X))


def balanced_weights(groups, y, target_group=None, target_boost=3.5, background=0.65):
    groups = np.asarray(groups)
    y = np.asarray(y)
    w = np.zeros(len(y), dtype=float)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        pos = idx[y[idx] == 1]
        neg = idx[y[idx] == 0]
        group_scale = target_boost if target_group is not None and int(g) == int(target_group) else background
        if target_group is None:
            group_scale = 1.0
        if len(pos):
            w[pos] = group_scale * 0.5 / len(pos)
        if len(neg):
            w[neg] = group_scale * 0.5 / len(neg)
        if len(pos) == 0 and len(neg):
            w[neg] = group_scale / len(neg)
        if len(neg) == 0 and len(pos):
            w[pos] = group_scale / len(pos)
    if w.mean() > 0:
        w /= w.mean()
    else:
        w[:] = 1.0
    return w


def fit_hgb(X, y, sample_weight, seed):
    m = HistGradientBoostingClassifier(
        learning_rate=0.032,
        max_iter=360,
        max_leaf_nodes=23,
        min_samples_leaf=14,
        l2_regularization=2.5,
        random_state=seed,
    )
    m.fit(X, y, sample_weight=sample_weight)
    return m


def choose_router_features(X, groups, y, max_features=96):
    """
    Select features that distinguish wells stably rather than event labels.
    Score = between-well centroid variance / within-well variance, with a
    penalty for strong positive-vs-negative separation so the router focuses
    on domain identity, not target leakage.
    """
    X0, med = fill_matrix(X)
    Z = signed_log(X0)
    uniq = np.unique(groups)
    centroids = []
    within = np.zeros(Z.shape[1], dtype=float)
    denom = 0
    for g in uniq:
        idx = np.where(groups == g)[0]
        zg = Z[idx]
        c = np.nanmedian(zg, axis=0)
        centroids.append(c)
        within += np.nanmedian(np.abs(zg - c), axis=0)
        denom += 1
    centroids = np.vstack(centroids)
    within /= max(denom, 1)
    between = np.nanstd(centroids, axis=0)

    pos = Z[y == 1]
    neg = Z[y == 0]
    if len(pos) and len(neg):
        target_gap = np.abs(np.nanmedian(pos, axis=0) - np.nanmedian(neg, axis=0))
    else:
        target_gap = np.zeros(Z.shape[1], dtype=float)

    score = between / (within + 1e-3)
    score = score / (1.0 + 0.5 * target_gap)
    score[~np.isfinite(score)] = -np.inf
    keep = np.argsort(score)[::-1][:min(max_features, Z.shape[1])]
    return keep.astype(int), med


def router_centroids(X, groups, keep, med, allowed_groups=None):
    X0, _ = fill_matrix(X, med)
    Z = signed_log(X0)[:, keep]
    if allowed_groups is None:
        allowed_groups = np.unique(groups)
    cents = {}
    scales = {}
    for g in allowed_groups:
        idx = np.where(groups == g)[0]
        if len(idx) == 0:
            continue
        zg = Z[idx]
        c = np.nanmedian(zg, axis=0)
        s = np.nanmedian(np.abs(zg - c), axis=0)
        s[~np.isfinite(s)] = 1.0
        s = np.maximum(s, 0.05)
        cents[int(g)] = c
        scales[int(g)] = s
    return cents, scales


def route_weights(Xq, keep, med, cents, scales, temperature=1.25, top_experts=4):
    X0, _ = fill_matrix(Xq, med)
    Z = signed_log(X0)[:, keep]
    expert_ids = sorted(cents)
    if not expert_ids:
        raise RuntimeError("No router centroids")
    D = np.zeros((len(Z), len(expert_ids)), dtype=float)
    for j, g in enumerate(expert_ids):
        d = np.abs((Z - cents[g]) / scales[g])
        D[:, j] = np.nanmedian(d, axis=1)
    logits = -D / max(float(temperature), 1e-6)

    if top_experts and top_experts < len(expert_ids):
        order = np.argsort(logits, axis=1)[:, ::-1]
        mask = np.ones_like(logits, dtype=bool)
        for i in range(len(logits)):
            mask[i, order[i, :top_experts]] = False
        logits[mask] = -1e9

    logits = logits - np.max(logits, axis=1, keepdims=True)
    W = np.exp(np.clip(logits, -50, 0))
    W /= np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
    return W, expert_ids, D


def fit_global_and_experts(X, y, groups, available_groups, seed=7000, target_boost=3.5, background=0.65):
    X0, med = fill_matrix(X)
    Z = signed_log(X0)

    wg = balanced_weights(groups, y, target_group=None)
    global_model = fit_hgb(Z, y, wg, seed)

    experts = {}
    for g in available_groups:
        we = balanced_weights(groups, y, target_group=int(g), target_boost=target_boost, background=background)
        experts[int(g)] = fit_hgb(Z, y, we, seed + 100 + int(g))
    return {"med": med, "global": global_model, "experts": experts}


def predict_global_and_experts(model, X):
    X0, _ = fill_matrix(X, model["med"])
    Z = signed_log(X0)
    pg = model["global"].predict_proba(Z)[:, 1]
    pe = {g: m.predict_proba(Z)[:, 1] for g, m in model["experts"].items()}
    return pg, pe


def optimize_global_moe_weight(y, pg, pm):
    best = None
    for a in np.linspace(0.0, 1.0, 21):
        s = a * rank01(pg) + (1.0-a) * rank01(pm)
        try:
            auc = float(roc_auc_score(y, s))
        except Exception:
            auc = -1.0
        if best is None or auc > best["auc"]:
            best = {"global_weight": float(a), "auc": auc}
    return best


def loso(X, y, groups, router_features, router_med, args):
    oof_global = np.zeros(len(y), dtype=float)
    oof_moe = np.zeros(len(y), dtype=float)
    rows = []

    for held in np.unique(groups):
        tr = groups != held
        va = groups == held
        available = [int(g) for g in np.unique(groups[tr])]
        model = fit_global_and_experts(
            X[tr], y[tr], groups[tr], available,
            seed=8000 + int(held),
            target_boost=args.expert_boost,
            background=args.background_weight,
        )
        pg, pe = predict_global_and_experts(model, X[va])

        cents, scales = router_centroids(X[tr], groups[tr], router_features, router_med, available)
        W, expert_ids, _ = route_weights(
            X[va], router_features, router_med, cents, scales,
            temperature=args.router_temperature,
            top_experts=args.top_experts,
        )
        pm = np.zeros(int(va.sum()), dtype=float)
        for j, g in enumerate(expert_ids):
            pm += W[:, j] * pe[g]

        oof_global[va] = pg
        oof_moe[va] = pm

        try:
            auc_g = float(roc_auc_score(y[va], pg))
            auc_m = float(roc_auc_score(y[va], pm))
        except Exception:
            auc_g = auc_m = float("nan")
        kdiag = max(int(y[va].sum()), round(int(va.sum()) * 0.38))
        mg = competition_metric(y[va], topk_pred(pg, kdiag))
        mm = competition_metric(y[va], topk_pred(pm, kdiag))
        print(
            f"[V10 LOSO] WELL_{int(held)+1:06d} "
            f"globalAUC={auc_g:.5f} moeAUC={auc_m:.5f} "
            f"globalScore={mg[0]:.3f} moeScore={mm[0]:.3f}"
        )
        rows.append({
            "well": int(held)+1,
            "global_auc": auc_g,
            "moe_auc": auc_m,
            "global_score": mg[0],
            "moe_score": mm[0],
            "n": int(va.sum()),
            "pos": int(y[va].sum()),
        })

    mix = optimize_global_moe_weight(y, oof_global, oof_moe)
    final = mix["global_weight"] * rank01(oof_global) + (1.0-mix["global_weight"]) * rank01(oof_moe)
    final = rank01(final)
    try:
        auc_g = float(roc_auc_score(y, oof_global))
        auc_m = float(roc_auc_score(y, oof_moe))
        auc_f = float(roc_auc_score(y, final))
    except Exception:
        auc_g = auc_m = auc_f = float("nan")

    print("="*72)
    print(f"[V10 OOF] global AUC={auc_g:.6f}")
    print(f"[V10 OOF] MoE    AUC={auc_m:.6f}")
    print(f"[V10 OOF] blend  AUC={auc_f:.6f}, global_weight={mix['global_weight']:.2f}")
    print("="*72)
    return oof_global, oof_moe, final, rows, {
        "global_auc": auc_g,
        "moe_auc": auc_m,
        "blend_auc": auc_f,
        "global_weight": mix["global_weight"],
    }


def write_zip(ids, pred, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"切片ID": ids, "溢流判断": np.asarray(pred, dtype=int)})
    if len(df) != 324 or df["切片ID"].nunique() != 324:
        raise RuntimeError("Submission must have exactly 324 unique IDs")
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

    X, y, groups, deltas, kinds, train_regimes = build_train_cache(train_dir, work_dir, args.force_rebuild)
    XT, ids, starts, ends, missing, test_regimes, anchors = base.build_test_features(test_dir, work_dir, force=False)
    if X.shape[1] != XT.shape[1]:
        raise RuntimeError(f"Feature mismatch train={X.shape} test={XT.shape}")

    router_features, router_med = choose_router_features(X, groups, y, max_features=args.router_features)
    np.save(work_dir / "V10_router_features.npy", router_features)
    print(f"[router] selected_features={len(router_features)}")

    oof_g, oof_m, oof_f, per_well, metrics = loso(X, y, groups, router_features, router_med, args)
    pd.DataFrame({
        "well": groups+1,
        "label": y,
        "delta_sec": deltas,
        "sample_type": kinds,
        "global_prob": oof_g,
        "moe_prob": oof_m,
        "blend_score": oof_f,
    }).to_csv(work_dir / "V10_oof.csv", index=False)

    print("[V10 final] fitting global + 10 experts")
    available = [int(g) for g in np.unique(groups)]
    model = fit_global_and_experts(
        X, y, groups, available,
        seed=10010,
        target_boost=args.expert_boost,
        background=args.background_weight,
    )
    pg, pe = predict_global_and_experts(model, XT)

    cents, scales = router_centroids(X, groups, router_features, router_med, available)
    W, expert_ids, D = route_weights(
        XT, router_features, router_med, cents, scales,
        temperature=args.router_temperature,
        top_experts=args.top_experts,
    )
    pm = np.zeros(len(ids), dtype=float)
    for j, g in enumerate(expert_ids):
        pm += W[:, j] * pe[g]

    a = metrics["global_weight"]
    pblend = rank01(a * rank01(pg) + (1.0-a) * rank01(pm))

    # Graph is deliberately excluded from the default score because previous
    # domain clustering collapsed most test slices into one giant cluster.
    debug = pd.DataFrame({
        "切片ID": ids,
        "global_prob": pg,
        "moe_prob": pm,
        "blend_score": pblend,
        "regime": test_regimes,
        "physical_anchor": anchors,
    })
    for j, g in enumerate(expert_ids):
        debug[f"router_well_{g+1:02d}"] = W[:, j]
        debug[f"router_dist_{g+1:02d}"] = D[:, j]
        debug[f"expert_{g+1:02d}"] = pe[g]
    debug["rank"] = debug["blend_score"].rank(method="first", ascending=False).astype(int)
    debug.sort_values("rank").to_csv(work_dir / "V10_test_debug.csv", index=False)

    outputs = []
    for k in TOPKS:
        p = out_dir / f"result_v10_top{k}.zip"
        write_zip(ids, topk_pred(pblend, k), p)
        outputs.append(str(p))

    # Useful ablations: global-only and MoE-only at the central K values.
    for k in [116, 120, 124]:
        p = out_dir / f"result_v10_global_top{k}.zip"
        write_zip(ids, topk_pred(pg, k), p); outputs.append(str(p))
        p = out_dir / f"result_v10_moe_top{k}.zip"
        write_zip(ids, topk_pred(pm, k), p); outputs.append(str(p))

    main_k = int(args.main_k)
    main = out_dir / "result_v10.zip"
    write_zip(ids, topk_pred(pblend, main_k), main)

    summary = {
        "version": "KickGuard V10 Baseline-Rebuild / Well-Adaptive MoE",
        "train_shape": list(X.shape),
        "test_shape": list(XT.shape),
        "router_features": int(len(router_features)),
        "router_temperature": float(args.router_temperature),
        "top_experts": int(args.top_experts),
        "expert_boost": float(args.expert_boost),
        "background_weight": float(args.background_weight),
        "oof": metrics,
        "per_well": per_well,
        "main_k": main_k,
        "outputs": outputs + [str(main)],
    }
    with open(work_dir / "V10_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("="*72)
    print("KickGuard V10 Baseline-Rebuild completed")
    print(f"OOF global AUC : {metrics['global_auc']:.6f}")
    print(f"OOF MoE AUC    : {metrics['moe_auc']:.6f}")
    print(f"OOF blend AUC  : {metrics['blend_auc']:.6f}")
    print(f"Global weight  : {metrics['global_weight']:.2f}")
    print(f"MAIN           : {main} (K={main_k})")
    print(f"SUMMARY        : {work_dir/'V10_summary.json'}")
    print("Candidates     : " + ", ".join(map(str, TOPKS)))
    print("="*72)


def parser():
    ap = argparse.ArgumentParser(description="KickGuard V10 Baseline-Rebuild / Well-Adaptive MoE")
    ap.add_argument("--train-dir", default="../train")
    ap.add_argument("--test-dir", default="../test")
    ap.add_argument("--work-dir", default="./work_v7")
    ap.add_argument("--output-dir", default="./v10_submit")
    ap.add_argument("--router-features", type=int, default=96)
    ap.add_argument("--router-temperature", type=float, default=1.25)
    ap.add_argument("--top-experts", type=int, default=4)
    ap.add_argument("--expert-boost", type=float, default=3.5)
    ap.add_argument("--background-weight", type=float, default=0.65)
    ap.add_argument("--main-k", type=int, default=120)
    ap.add_argument("--force-rebuild", action="store_true")
    return ap


if __name__ == "__main__":
    run(parser().parse_args())
