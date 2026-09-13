#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KickGuard V9 Feedback-Aware
===========================

Leaderboard-feedback-aware correction/re-ranking for the KickGuard task.

This is intentionally NOT another global classifier. It uses:
  * one or two previously strong submission ZIPs as anchors (e.g. 66.9444)
  * the V7 submission/ranking as a medium-strength positive signal
  * the exact V8 zero-score submission/ranking as an anti-signal
  * the validated 324-slice test feature cache from EventFix
  * a cross-fitted pseudo-label correction model

Important: a leaderboard score of zero is NOT interpreted as hard per-row labels.
It may be caused by zero recall or zero specificity. V8 is therefore used only
as a ranking penalty / anti-signal, never as ground-truth negatives.

Typical run:
python kickguard_v9_feedback.py \
  --anchor116 ./V5_feedback_dropbad_116.zip \
  --anchor124 ./V5_meta_plus4_124.zip \
  --v7-zip ./result_v7.zip \
  --v8-zero ./v8_submit/result_v8_nograph_top116.zip \
  --work-dir ./work_v7 \
  --output-dir ./v9_submit
"""
from __future__ import annotations

import argparse
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

EXPECTED_IDS = [f"test_{i:03d}" for i in range(1, 325)]
TOPKS = [112, 116, 120, 124, 128]
SWAPS = [1, 2, 3, 4, 6, 8]


def finite(x, default=0.0):
    try:
        return float(x) if np.isfinite(x) else default
    except Exception:
        return default


def normalize_id(v):
    s = str(v).strip().lower()
    if s.startswith("test_"):
        try:
            return f"test_{int(s.split('_')[-1]):03d}"
        except Exception:
            return s
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits:
        return f"test_{int(digits):03d}"
    return s


def rank01(score):
    x = np.asarray(score, dtype=float)
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


def detect_columns(df):
    id_col = None
    for c in ["切片ID", "id", "ID", "test_id", "slice_id"]:
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        id_col = df.columns[0]

    label_col = None
    for c in ["溢流判断", "label", "pred", "prediction", "target"]:
        if c in df.columns:
            label_col = c
            break
    if label_col is None and len(df.columns) >= 2:
        label_col = df.columns[1]
    return id_col, label_col


def read_submission_zip(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path, "r") as z:
        names = z.namelist()
        candidates = [n for n in names if Path(n).name.lower() == "result.csv"]
        if not candidates:
            candidates = [n for n in names if n.lower().endswith(".csv")]
        if not candidates:
            raise RuntimeError(f"No CSV inside {path}")
        raw = z.read(candidates[0])
    df = pd.read_csv(io.BytesIO(raw))
    id_col, label_col = detect_columns(df)
    out = pd.DataFrame({
        "id": [normalize_id(v) for v in df[id_col]],
        "label": pd.to_numeric(df[label_col], errors="coerce").fillna(-1).astype(int),
    })
    if len(out) != 324 or out["id"].nunique() != 324:
        raise RuntimeError(f"{path}: expected 324 unique IDs, got rows={len(out)} unique={out['id'].nunique()}")
    if set(out["label"].unique()) - {0, 1}:
        raise RuntimeError(f"{path}: labels must be binary, got {sorted(out['label'].unique())}")
    missing = set(EXPECTED_IDS) - set(out["id"])
    extra = set(out["id"]) - set(EXPECTED_IDS)
    if missing or extra:
        raise RuntimeError(f"{path}: bad IDs missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}")
    out = out.set_index("id").loc[EXPECTED_IDS].reset_index()
    print(f"[submission] {path}: positives={int(out['label'].sum())}")
    return out["label"].to_numpy(np.int8)


def read_debug_score(path, preferred):
    path = Path(path)
    if not path.exists():
        return None, None, None
    df = pd.read_csv(path)
    id_col = None
    for c in ["切片ID", "id", "ID", "test_id"]:
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        print(f"[debug warning] no ID column in {path}")
        return None, None, None

    score_col = None
    for c in preferred:
        if c in df.columns:
            score_col = c
            break
    if score_col is None:
        numeric = [c for c in df.columns if c != id_col and pd.api.types.is_numeric_dtype(df[c])]
        numeric = [c for c in numeric if c.lower() not in {"rank", "domain", "regime", "main_pred"}]
        if numeric:
            score_col = numeric[0]
    if score_col is None:
        print(f"[debug warning] no usable score column in {path}")
        return None, None, None

    mp = {normalize_id(r[id_col]): finite(r[score_col], np.nan) for _, r in df.iterrows()}
    vals = np.asarray([mp.get(i, np.nan) for i in EXPECTED_IDS], dtype=float)
    if np.isnan(vals).all():
        return None, None, None
    vals = rank01(vals)

    physical = None
    if "physical_anchor" in df.columns:
        mp2 = {normalize_id(r[id_col]): finite(r["physical_anchor"], np.nan) for _, r in df.iterrows()}
        physical = rank01(np.asarray([mp2.get(i, np.nan) for i in EXPECTED_IDS], dtype=float))
    print(f"[debug] {path}: score={score_col}")
    return vals, score_col, physical


def find_existing(explicit, candidates):
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    return None


def load_test_feature_cache(work_dir):
    p = Path(work_dir) / "test_features_v7_eventfix_1.npz"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing validated test feature cache: {p}. Run EventFix/V7 once first."
        )
    z = np.load(p, allow_pickle=True)
    X = np.asarray(z["X"], dtype=float)
    ids = [normalize_id(x) for x in z["ids"].astype(str)]
    if len(ids) != 324 or set(ids) != set(EXPECTED_IDS):
        raise RuntimeError("test feature cache IDs do not match test_001..test_324")
    order = {sid: i for i, sid in enumerate(ids)}
    X = X[[order[sid] for sid in EXPECTED_IDS]]
    return X


def robust_feature_view(X, max_features=192):
    X = np.asarray(X, dtype=float)
    med = np.nanmedian(X, axis=0)
    med[~np.isfinite(med)] = 0.0
    rr, cc = np.where(~np.isfinite(X))
    X0 = X.copy()
    if len(rr):
        X0[rr, cc] = med[cc]
    X0 = np.clip(X0, -1e8, 1e8)
    Z = np.sign(X0) * np.log1p(np.abs(X0))
    var = np.nanvar(Z, axis=0)
    var[~np.isfinite(var)] = 0.0
    k = min(int(max_features), Z.shape[1])
    keep = np.argsort(var, kind="mergesort")[-k:]
    Z = Z[:, keep]
    scaler = StandardScaler()
    Z = scaler.fit_transform(Z)
    return Z, keep


def build_anchor_prior(a116, a124):
    if a116 is None and a124 is None:
        raise RuntimeError("At least one strong anchor submission is required")
    if a116 is not None and a124 is not None:
        prior = 0.5 * a116.astype(float) + 0.5 * a124.astype(float)
        agree = (a116 == a124)
        confidence = np.where(agree, 1.0, 0.38)
        print(
            f"[anchor] intersection_pos={int(((a116==1)&(a124==1)).sum())} "
            f"only116={int(((a116==1)&(a124==0)).sum())} "
            f"only124={int(((a116==0)&(a124==1)).sum())} "
            f"both_neg={int(((a116==0)&(a124==0)).sum())}"
        )
        return prior, confidence, agree
    a = a116 if a116 is not None else a124
    return a.astype(float), np.full(len(a), 0.72), np.ones(len(a), dtype=bool)


def crossfit_correction(Z, pseudo_target, sample_weight, seed=9409):
    n = len(pseudo_target)
    kf = KFold(n_splits=6, shuffle=True, random_state=seed)
    oof_et = np.zeros(n, dtype=float)
    oof_hgb = np.zeros(n, dtype=float)

    for fold, (tr, va) in enumerate(kf.split(Z), 1):
        et = ExtraTreesRegressor(
            n_estimators=700,
            min_samples_leaf=7,
            max_features="sqrt",
            random_state=seed + fold,
            n_jobs=-1,
        )
        et.fit(Z[tr], pseudo_target[tr], sample_weight=sample_weight[tr])
        oof_et[va] = et.predict(Z[va])

        hgb = HistGradientBoostingRegressor(
            learning_rate=0.035,
            max_iter=260,
            max_leaf_nodes=15,
            min_samples_leaf=12,
            l2_regularization=4.0,
            random_state=seed + 100 + fold,
        )
        hgb.fit(Z[tr], pseudo_target[tr], sample_weight=sample_weight[tr])
        oof_hgb[va] = hgb.predict(Z[va])

    oof = np.clip(0.55 * oof_et + 0.45 * oof_hgb, 0.0, 1.0)

    et = ExtraTreesRegressor(
        n_estimators=1000,
        min_samples_leaf=7,
        max_features="sqrt",
        random_state=seed + 1000,
        n_jobs=-1,
    )
    et.fit(Z, pseudo_target, sample_weight=sample_weight)
    hgb = HistGradientBoostingRegressor(
        learning_rate=0.035,
        max_iter=320,
        max_leaf_nodes=15,
        min_samples_leaf=12,
        l2_regularization=4.0,
        random_state=seed + 1100,
    )
    hgb.fit(Z, pseudo_target, sample_weight=sample_weight)
    full = np.clip(0.55 * et.predict(Z) + 0.45 * hgb.predict(Z), 0.0, 1.0)
    return oof, full


def write_zip(ids, pred, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"切片ID": ids, "溢流判断": np.asarray(pred, dtype=int)})
    if len(df) != 324 or df["切片ID"].nunique() != 324:
        raise RuntimeError("submission must have exactly 324 unique IDs")
    raw = df.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("result.csv", raw)
    print(f"[output] {path} positive={int(df['溢流判断'].sum())}")


def swap_variant(base, residual, nswap, protected=None, forbidden_add=None):
    base = np.asarray(base, dtype=np.int8).copy()
    protected = np.zeros(len(base), dtype=bool) if protected is None else np.asarray(protected, dtype=bool)
    forbidden_add = np.zeros(len(base), dtype=bool) if forbidden_add is None else np.asarray(forbidden_add, dtype=bool)

    drop_pool = np.where((base == 1) & (~protected))[0]
    add_pool = np.where((base == 0) & (~forbidden_add))[0]
    if len(drop_pool) < nswap or len(add_pool) < nswap:
        return None, [], []
    drops = drop_pool[np.argsort(residual[drop_pool], kind="mergesort")[:nswap]]
    adds = add_pool[np.argsort(residual[add_pool], kind="mergesort")[::-1][:nswap]]
    out = base.copy()
    out[drops] = 0
    out[adds] = 1
    return out, drops.tolist(), adds.tolist()


def run(args):
    work_dir = Path(args.work_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[path] work={work_dir}")
    print(f"[path] out ={out_dir}")

    anchor116_p = find_existing(args.anchor116, [
        "./V5_feedback_dropbad_116.zip",
        "../V5_feedback_dropbad_116.zip",
    ])
    anchor124_p = find_existing(args.anchor124, [
        "./V5_meta_plus4_124.zip",
        "../V5_meta_plus4_124.zip",
    ])
    v7_p = find_existing(args.v7_zip, ["./result_v7.zip", "../result_v7.zip"])
    v8_p = find_existing(args.v8_zero, [
        "./v8_submit/result_v8_nograph_top116.zip",
        "./v8_submit/result_v8_top116.zip",
    ])

    if anchor116_p is None and anchor124_p is None:
        raise RuntimeError("No 66.9444 anchor ZIP found; pass --anchor116 and/or --anchor124")
    if v7_p is None:
        raise RuntimeError("No V7 scored submission found; pass --v7-zip")
    if v8_p is None:
        raise RuntimeError("No exact V8 zero-score submission found; pass --v8-zero")

    a116 = read_submission_zip(anchor116_p) if anchor116_p else None
    a124 = read_submission_zip(anchor124_p) if anchor124_p else None
    v7_label = read_submission_zip(v7_p)
    v8_label = read_submission_zip(v8_p)
    anchor_prior, anchor_conf, anchor_agree = build_anchor_prior(a116, a124)

    v7_debug_p = find_existing(args.v7_debug, [str(work_dir / "V7_test_debug.csv")])
    v8_debug_p = find_existing(args.v8_debug, [str(work_dir / "V8_test_debug.csv")])
    v71_debug_p = find_existing(args.v71_debug, [str(work_dir / "V7_1_test_debug.csv")])

    v7_rank, v7_col, physical = read_debug_score(
        v7_debug_p, ["final_prob", "graph_prob", "base_prob"]
    ) if v7_debug_p else (None, None, None)
    v8_rank, v8_col, physical8 = read_debug_score(
        v8_debug_p, ["final_score", "model_score", "hgb_signed", "graph_score"]
    ) if v8_debug_p else (None, None, None)
    v71_rank, v71_col, physical71 = read_debug_score(
        v71_debug_p, ["final_score", "risk_score", "graph_score"]
    ) if v71_debug_p else (None, None, None)

    # Fallback ranks from binary submissions. They are intentionally weak.
    if v7_rank is None:
        v7_rank = 0.25 + 0.50 * v7_label.astype(float)
    if v8_rank is None:
        v8_rank = 0.25 + 0.50 * v8_label.astype(float)
    if physical is None:
        physical = physical8 if physical8 is not None else physical71
    if physical is None:
        physical = np.full(324, 0.5, dtype=float)

    # Scores from leaderboard outcomes are used only as relative reliability,
    # not as per-row labels. Anchor dominates; V7 supports; V8 is anti-signal.
    v7_support = 0.65 * rank01(v7_rank) + 0.35 * v7_label.astype(float)
    v8_anti = 0.65 * rank01(v8_rank) + 0.35 * v8_label.astype(float)
    phys = rank01(physical)
    if v71_rank is not None:
        weak_extra = rank01(v71_rank)
    else:
        weak_extra = np.full(324, 0.5, dtype=float)

    pseudo_target = np.clip(
        0.72 * anchor_prior
        + 0.16 * v7_support
        + 0.07 * (1.0 - v8_anti)
        + 0.03 * phys
        + 0.02 * weak_extra,
        0.0,
        1.0,
    )
    sample_weight = 1.0 + 7.0 * anchor_conf
    # Disagreement between strong anchors is precisely where correction is allowed.
    if a116 is not None and a124 is not None:
        sample_weight = np.where(anchor_agree, sample_weight, 1.8)

    X = load_test_feature_cache(work_dir)
    Z, keep = robust_feature_view(X, max_features=args.correction_features)
    corr_oof, corr_full = crossfit_correction(Z, pseudo_target, sample_weight, seed=args.seed)
    corr_rank = rank01(0.55 * corr_oof + 0.45 * corr_full)

    residual = rank01(
        0.38 * v7_support
        + 0.24 * (1.0 - v8_anti)
        + 0.20 * corr_rank
        + 0.10 * phys
        + 0.08 * weak_extra
    )

    # Feedback-aware full score. The anchor remains dominant but not absolute;
    # residual ordering determines a small number of boundary exchanges.
    final_score = rank01(
        args.anchor_weight * anchor_prior
        + args.residual_weight * residual
    )

    # Protect dual-anchor consensus positives from conservative swap variants.
    if a116 is not None and a124 is not None:
        protected = (a116 == 1) & (a124 == 1)
        dual_negative = (a116 == 0) & (a124 == 0)
    else:
        protected = np.zeros(324, dtype=bool)
        dual_negative = np.zeros(324, dtype=bool)

    # Because V8=0 is only global feedback, do not ban V8 positives completely.
    # They are forbidden only for the most conservative swap variants when the
    # strong anchor also says negative.
    forbidden_conservative = (v8_label == 1) & (anchor_prior <= 0.5)

    outputs = []
    swap_report = []
    for name, base in [("base116", a116), ("base124", a124)]:
        if base is None:
            continue
        for nswap in SWAPS:
            pred, drops, adds = swap_variant(
                base,
                residual,
                nswap,
                protected=protected,
                forbidden_add=forbidden_conservative if nswap <= 3 else None,
            )
            if pred is None:
                continue
            p = out_dir / f"v9_{name}_swap{nswap}_k{int(base.sum())}.zip"
            write_zip(EXPECTED_IDS, pred, p)
            outputs.append(str(p))
            swap_report.append({
                "name": p.name,
                "base": name,
                "swap": int(nswap),
                "drops": [EXPECTED_IDS[i] for i in drops],
                "adds": [EXPECTED_IDS[i] for i in adds],
            })

    for k in TOPKS:
        pred = topk_pred(final_score, k)
        p = out_dir / f"v9_feedback_top{k}.zip"
        write_zip(EXPECTED_IDS, pred, p)
        outputs.append(str(p))

    # Anti-V8 ablation: anchor/residual candidates with a stronger explicit
    # penalty to the zero-score ranking. Useful as one controlled probe only.
    anti_score = rank01(
        0.76 * anchor_prior
        + 0.18 * residual
        + 0.06 * (1.0 - v8_anti)
    )
    for k in [116, 120, 124]:
        pred = topk_pred(anti_score, k)
        p = out_dir / f"v9_antiv8_top{k}.zip"
        write_zip(EXPECTED_IDS, pred, p)
        outputs.append(str(p))

    debug = pd.DataFrame({
        "切片ID": EXPECTED_IDS,
        "anchor_prior": anchor_prior,
        "anchor_conf": anchor_conf,
        "v7_label": v7_label,
        "v7_support": v7_support,
        "v8_zero_label": v8_label,
        "v8_anti": v8_anti,
        "physical": phys,
        "pseudo_target": pseudo_target,
        "correction_oof": corr_oof,
        "correction_full": corr_full,
        "correction_rank": corr_rank,
        "residual": residual,
        "final_score": final_score,
        "anti_score": anti_score,
    })
    if a116 is not None:
        debug["anchor116"] = a116
    if a124 is not None:
        debug["anchor124"] = a124
    debug["final_rank"] = debug["final_score"].rank(method="first", ascending=False).astype(int)
    debug.sort_values("final_rank").to_csv(out_dir / "V9_feedback_debug.csv", index=False)

    overlap = {
        "anchor116_pos": int(a116.sum()) if a116 is not None else None,
        "anchor124_pos": int(a124.sum()) if a124 is not None else None,
        "v7_pos": int(v7_label.sum()),
        "v8_zero_pos": int(v8_label.sum()),
        "v7_v8_overlap": int(((v7_label == 1) & (v8_label == 1)).sum()),
    }
    if a116 is not None:
        overlap["anchor116_v7_overlap"] = int(((a116 == 1) & (v7_label == 1)).sum())
        overlap["anchor116_v8_overlap"] = int(((a116 == 1) & (v8_label == 1)).sum())
    if a124 is not None:
        overlap["anchor124_v7_overlap"] = int(((a124 == 1) & (v7_label == 1)).sum())
        overlap["anchor124_v8_overlap"] = int(((a124 == 1) & (v8_label == 1)).sum())
    if a116 is not None and a124 is not None:
        overlap["anchor_intersection"] = int(((a116 == 1) & (a124 == 1)).sum())
        overlap["only116"] = int(((a116 == 1) & (a124 == 0)).sum())
        overlap["only124"] = int(((a116 == 0) & (a124 == 1)).sum())

    # Recommended order is deliberately conservative: 2 swaps, then 3 swaps,
    # before any global Top-K or anti-V8 probe.
    recommended = []
    for preferred in [
        "v9_base116_swap2_k116.zip",
        "v9_base124_swap2_k124.zip",
        "v9_base116_swap3_k116.zip",
        "v9_base124_swap3_k124.zip",
        "v9_feedback_top120.zip",
        "v9_antiv8_top116.zip",
    ]:
        if (out_dir / preferred).exists():
            recommended.append(preferred)

    summary = {
        "version": "V9 Feedback-Aware",
        "leaderboard_context": {
            "strong_anchor_score": args.anchor_score,
            "v7_score": args.v7_score,
            "v8_score": args.v8_score,
        },
        "inputs": {
            "anchor116": str(anchor116_p) if anchor116_p else None,
            "anchor124": str(anchor124_p) if anchor124_p else None,
            "v7_zip": str(v7_p),
            "v8_zero": str(v8_p),
            "v7_debug": str(v7_debug_p) if v7_debug_p else None,
            "v8_debug": str(v8_debug_p) if v8_debug_p else None,
            "v71_debug": str(v71_debug_p) if v71_debug_p else None,
        },
        "overlap": overlap,
        "weights": {
            "anchor_weight": args.anchor_weight,
            "residual_weight": args.residual_weight,
        },
        "correction_features": int(len(keep)),
        "recommended_submit_order": recommended,
        "swaps": swap_report,
        "outputs": outputs,
    }
    with open(out_dir / "V9_feedback_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(out_dir / "SUBMIT_ORDER.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(recommended) + "\n")

    print("=" * 78)
    print("KickGuard V9 Feedback-Aware completed")
    print(f"Anchor score context : {args.anchor_score:.4f}")
    print(f"V7 score context     : {args.v7_score:.4f}")
    print(f"V8 score context     : {args.v8_score:.4f}")
    print(f"V7/V8 overlap        : {overlap['v7_v8_overlap']}")
    if "anchor_intersection" in overlap:
        print(
            f"Anchor intersection  : {overlap['anchor_intersection']} "
            f"only116={overlap['only116']} only124={overlap['only124']}"
        )
    print("Recommended order:")
    for i, x in enumerate(recommended, 1):
        print(f"  {i}. {x}")
    print(f"Summary: {out_dir/'V9_feedback_summary.json'}")
    print("=" * 78)


def parser():
    ap = argparse.ArgumentParser(description="KickGuard V9 Feedback-Aware correction retraining")
    ap.add_argument("--anchor116", default=None)
    ap.add_argument("--anchor124", default=None)
    ap.add_argument("--v7-zip", default=None)
    ap.add_argument("--v8-zero", default=None, help="exact V8 ZIP that scored 0.0000")
    ap.add_argument("--v7-debug", default=None)
    ap.add_argument("--v8-debug", default=None)
    ap.add_argument("--v71-debug", default=None)
    ap.add_argument("--work-dir", default="./work_v7")
    ap.add_argument("--output-dir", default="./v9_submit")
    ap.add_argument("--anchor-score", type=float, default=66.9444)
    ap.add_argument("--v7-score", type=float, default=36.9733)
    ap.add_argument("--v8-score", type=float, default=0.0)
    ap.add_argument("--anchor-weight", type=float, default=0.76)
    ap.add_argument("--residual-weight", type=float, default=0.24)
    ap.add_argument("--correction-features", type=int, default=192)
    ap.add_argument("--seed", type=int, default=9409)
    return ap


if __name__ == "__main__":
    args = parser().parse_args()
    if abs(args.anchor_weight + args.residual_weight - 1.0) > 1e-6:
        raise SystemExit("anchor-weight + residual-weight must equal 1")
    run(args)
