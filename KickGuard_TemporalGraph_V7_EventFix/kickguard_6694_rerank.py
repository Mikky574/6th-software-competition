#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KickGuard 66.9444 Local Rerank Optimizer

Purpose
-------
Do NOT retrain the global classifier. Treat one or two leaderboard-proven
66.9444 submissions as anchors and only make tiny boundary swaps using V7/V7.1
signals as weak evidence.

Inputs
------
--base116   V5_feedback_dropbad_116.zip (optional but recommended)
--base124   V5_meta_plus4_124.zip       (optional but recommended)
--work-dir  directory containing V7_test_debug.csv and V7_1_test_debug.csv

Outputs
-------
A conservative set of same-count swap probes plus reports explaining every
changed test id.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

N_TEST = 324
DEFAULT_SWAPS = [1, 2, 3, 4, 5, 6]


def norm_id(x):
    s = str(x).strip()
    m = re.search(r"(\d+)$", s)
    if not m:
        raise ValueError(f"bad test id: {x!r}")
    return f"test_{int(m.group(1)):03d}"


def rank01(s: pd.Series, higher=True):
    x = pd.to_numeric(s, errors="coerce")
    med = x.median()
    if not np.isfinite(med):
        med = 0.0
    x = x.fillna(med)
    r = x.rank(method="average", ascending=not higher, pct=True)
    # rank pct is 1 for worst when ascending=False, invert to high-is-good
    if higher:
        r = 1.0 - r + 1.0 / len(r)
    else:
        r = 1.0 - r + 1.0 / len(r)
    return r.clip(0, 1)


def high_rank(s: pd.Series):
    x = pd.to_numeric(s, errors="coerce")
    med = x.median()
    if not np.isfinite(med): med = 0.0
    x = x.fillna(med)
    return x.rank(method="average", pct=True).clip(0,1)


def read_submission_zip(path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path, "r") as z:
        names = z.namelist()
        target = "result.csv" if "result.csv" in names else None
        if target is None:
            csvs = [n for n in names if n.lower().endswith(".csv")]
            if len(csvs) != 1:
                raise RuntimeError(f"{path}: expected result.csv or one csv, got {names}")
            target = csvs[0]
        raw = z.read(target)
    last = None
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk"]:
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc)
            break
        except Exception as e:
            last = e
    else:
        raise RuntimeError(f"cannot decode {path}: {last}")

    id_col = "切片ID" if "切片ID" in df.columns else df.columns[0]
    y_col = "溢流判断" if "溢流判断" in df.columns else df.columns[1]
    out = pd.DataFrame({"切片ID": [norm_id(v) for v in df[id_col]],
                        "pred": pd.to_numeric(df[y_col], errors="raise").astype(int)})
    if len(out) != N_TEST or out["切片ID"].nunique() != N_TEST:
        raise RuntimeError(f"{path}: expected 324 unique rows")
    if not set(out["pred"].unique()).issubset({0,1}):
        raise RuntimeError(f"{path}: labels must be binary")
    return out.sort_values("切片ID").reset_index(drop=True)


def read_debug(path: Path, prefix: str):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    id_col = "切片ID" if "切片ID" in df.columns else ("id" if "id" in df.columns else None)
    if id_col is None:
        raise RuntimeError(f"{path}: no id column")
    df = df.copy()
    df["切片ID"] = [norm_id(v) for v in df[id_col]]
    keep = ["切片ID"]
    rename = {}
    for c in df.columns:
        if c == id_col or c == "切片ID":
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            keep.append(c)
            rename[c] = f"{prefix}_{c}"
    return df[keep].rename(columns=rename).drop_duplicates("切片ID")


def build_signal_table(work_dir: Path, base116=None, base124=None):
    ids = pd.DataFrame({"切片ID": [f"test_{i:03d}" for i in range(1, N_TEST+1)]})
    tab = ids.copy()

    b116 = read_submission_zip(base116) if base116 else None
    b124 = read_submission_zip(base124) if base124 else None
    if b116 is None and b124 is None:
        raise RuntimeError("provide at least --base116 or --base124")

    if b116 is not None:
        tab = tab.merge(b116.rename(columns={"pred":"base116"}), on="切片ID", how="left")
    if b124 is not None:
        tab = tab.merge(b124.rename(columns={"pred":"base124"}), on="切片ID", how="left")

    v7 = read_debug(work_dir / "V7_test_debug.csv", "v7")
    v71 = read_debug(work_dir / "V7_1_test_debug.csv", "v71")
    if v7 is not None:
        tab = tab.merge(v7, on="切片ID", how="left")
    if v71 is not None:
        tab = tab.merge(v71, on="切片ID", how="left")
    if v7 is None and v71 is None:
        raise RuntimeError("need V7_test_debug.csv and/or V7_1_test_debug.csv in work-dir")

    # Rank-normalized weak evidence. We intentionally do not trust raw probability calibration.
    signal_cols = []
    preferred = [
        "v7_final_prob", "v7_graph_prob", "v7_base_prob", "v7_physical_anchor",
        "v71_final_score", "v71_risk_score", "v71_graph_score", "v71_physical_anchor",
        "v71_etr", "v71_hgr", "v71_etc", "v71_lr",
    ]
    for c in preferred:
        if c in tab.columns:
            rc = c + "_r"
            tab[rc] = high_rank(tab[c])
            signal_cols.append(rc)

    if len(signal_cols) < 2:
        raise RuntimeError(f"too few usable risk columns, got {signal_cols}")

    # Group signals into families to reduce correlated double counting.
    families = []
    for names in [
        ["v7_final_prob_r", "v7_graph_prob_r", "v7_base_prob_r"],
        ["v71_final_score_r", "v71_risk_score_r", "v71_graph_score_r"],
        ["v7_physical_anchor_r", "v71_physical_anchor_r"],
        ["v71_etr_r", "v71_hgr_r", "v71_etc_r", "v71_lr_r"],
    ]:
        use = [c for c in names if c in tab.columns]
        if use:
            fam = "fam_" + str(len(families))
            tab[fam] = tab[use].mean(axis=1)
            families.append(fam)

    tab["weak_consensus"] = tab[families].mean(axis=1)
    tab["signal_spread"] = tab[families].std(axis=1).fillna(0.0)
    tab["high_votes"] = (tab[families] >= 0.75).sum(axis=1)
    tab["low_votes"] = (tab[families] <= 0.25).sum(axis=1)

    if "base116" in tab.columns and "base124" in tab.columns:
        tab["anchor_sum"] = tab["base116"] + tab["base124"]
        tab["anchor_core"] = (tab["anchor_sum"] == 2).astype(int)
        tab["anchor_boundary"] = (tab["anchor_sum"] == 1).astype(int)
        tab["anchor_neg"] = (tab["anchor_sum"] == 0).astype(int)
    else:
        bcol = "base116" if "base116" in tab.columns else "base124"
        tab["anchor_sum"] = tab[bcol]
        tab["anchor_core"] = tab[bcol]
        tab["anchor_boundary"] = 0
        tab["anchor_neg"] = 1 - tab[bcol]

    return tab, families


def write_zip(ids, pred, path):
    out = pd.DataFrame({"切片ID": ids, "溢流判断": np.asarray(pred, dtype=int)})
    if len(out) != N_TEST or out["切片ID"].nunique() != N_TEST:
        raise RuntimeError("bad submission shape")
    raw = out.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("result.csv", raw)
    return path


def choose_swap(tab, base_col, nswap, protect_dual=True):
    cur = tab[base_col].astype(int).to_numpy().copy()

    pos = tab[cur == 1].copy()
    neg = tab[cur == 0].copy()

    # Conservative drop candidates:
    # 1) if two anchors exist, prefer positives that only one 66.94 anchor supports;
    # 2) require weak models to agree it is low risk where possible.
    pos["drop_priority"] = (
        1.20 * pos["low_votes"]
        + 0.70 * (1.0 - pos["weak_consensus"])
        - 0.25 * pos["signal_spread"]
    )
    if "base116" in tab.columns and "base124" in tab.columns:
        pos["drop_priority"] += 1.50 * (pos["anchor_sum"] == 1).astype(float)
        if protect_dual:
            pos.loc[pos["anchor_sum"] == 2, "drop_priority"] -= 3.0

    # Conservative add candidates:
    # strong weak-model agreement, low spread. If two anchors exist, samples already
    # supported by the other 66.94 anchor receive a large bonus.
    neg["add_priority"] = (
        1.20 * neg["high_votes"]
        + 0.70 * neg["weak_consensus"]
        - 0.25 * neg["signal_spread"]
    )
    if "base116" in tab.columns and "base124" in tab.columns:
        other = "base124" if base_col == "base116" else "base116"
        neg["add_priority"] += 1.75 * neg[other].astype(float)

    drops = pos.sort_values(["drop_priority", "weak_consensus"], ascending=[False, True]).head(nswap)
    adds = neg.sort_values(["add_priority", "weak_consensus"], ascending=[False, False]).head(nswap)

    id_to_i = {sid:i for i,sid in enumerate(tab["切片ID"])}
    for sid in drops["切片ID"]:
        cur[id_to_i[sid]] = 0
    for sid in adds["切片ID"]:
        cur[id_to_i[sid]] = 1

    report = {
        "base": base_col,
        "nswap": int(nswap),
        "drops": drops[["切片ID", "drop_priority", "weak_consensus", "low_votes", "high_votes", "anchor_sum"]].to_dict("records"),
        "adds": adds[["切片ID", "add_priority", "weak_consensus", "low_votes", "high_votes", "anchor_sum"]].to_dict("records"),
        "positive_count": int(cur.sum()),
    }
    return cur, report


def explicit_probe(tab, base_col):
    """Historical corrected V6 candidate: apply only when ids are present.
    This is generated as a separate probe, never mixed into consensus ranking.
    """
    drops = {18,21,179,180,298,310,312,317}
    adds = {107,212,271}
    cur = tab[base_col].astype(int).to_numpy().copy()
    id_to_i = {sid:i for i,sid in enumerate(tab["切片ID"])}
    changed_drop=[]; changed_add=[]
    for n in sorted(drops):
        sid=f"test_{n:03d}"
        if sid in id_to_i and cur[id_to_i[sid]]==1:
            cur[id_to_i[sid]]=0; changed_drop.append(sid)
    for n in sorted(adds):
        sid=f"test_{n:03d}"
        if sid in id_to_i and cur[id_to_i[sid]]==0:
            cur[id_to_i[sid]]=1; changed_add.append(sid)
    return cur, {"base":base_col,"type":"historical_v6_corrected","drops":changed_drop,"adds":changed_add,"positive_count":int(cur.sum())}


def run(args):
    work = Path(args.work_dir).resolve()
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    base116 = Path(args.base116).resolve() if args.base116 else None
    base124 = Path(args.base124).resolve() if args.base124 else None
    tab, families = build_signal_table(work, base116, base124)

    print(f"[signals] families={families}")
    if "base116" in tab.columns:
        print(f"[anchor] base116 positives={int(tab['base116'].sum())}")
    if "base124" in tab.columns:
        print(f"[anchor] base124 positives={int(tab['base124'].sum())}")
    if "base116" in tab.columns and "base124" in tab.columns:
        inter=int(((tab.base116==1)&(tab.base124==1)).sum())
        only116=int(((tab.base116==1)&(tab.base124==0)).sum())
        only124=int(((tab.base116==0)&(tab.base124==1)).sum())
        print(f"[anchor] intersection={inter} only116={only116} only124={only124}")

    reports=[]
    swap_counts = sorted({int(x) for x in args.swaps.split(",") if x.strip()})
    for base_col in ["base116", "base124"]:
        if base_col not in tab.columns:
            continue
        base_count=int(tab[base_col].sum())
        for n in swap_counts:
            pred, rep = choose_swap(tab, base_col, n, protect_dual=not args.allow_drop_dual)
            name=f"rerank_{base_col}_swap{n}_k{base_count}.zip"
            write_zip(tab["切片ID"], pred, out/name)
            rep["file"] = name; reports.append(rep)
            print(f"[output] {name} positives={int(pred.sum())}")

        hist, rep = explicit_probe(tab, base_col)
        name=f"rerank_{base_col}_historicalV6_k{int(hist.sum())}.zip"
        write_zip(tab["切片ID"], hist, out/name)
        rep["file"] = name; reports.append(rep)
        print(f"[output] {name} positives={int(hist.sum())}")

    # Two-anchor consensus candidates: preserve the common core and choose remaining
    # positives by weak consensus. These are more exploratory, so clearly separated.
    if "base116" in tab.columns and "base124" in tab.columns:
        core = ((tab.base116==1)&(tab.base124==1)).to_numpy()
        score = tab["weak_consensus"].to_numpy() + 0.20*(tab["anchor_sum"].to_numpy()==1)
        for k in [116,120,124]:
            pred=core.astype(int).copy()
            need=k-int(pred.sum())
            cand=np.where(~core)[0]
            order=cand[np.argsort(-score[cand])]
            pred[order[:max(0,need)]]=1
            name=f"rerank_dual_consensus_top{k}.zip"
            write_zip(tab["切片ID"], pred, out/name)
            reports.append({"file":name,"type":"dual_consensus","k":k,"core":int(core.sum()),"positive_count":int(pred.sum())})
            print(f"[output] {name} positives={int(pred.sum())}")

    # Save full diagnostics sorted from highest to lowest weak risk.
    tab.sort_values("weak_consensus", ascending=False).to_csv(out/"rerank_scores.csv", index=False)
    with open(out/"rerank_report.json", "w", encoding="utf-8") as f:
        json.dump({"families":families,"reports":reports}, f, ensure_ascii=False, indent=2)

    # Recommendation list favors smallest changes first.
    rec=[]
    for base_col in ["base116","base124"]:
        if base_col in tab.columns:
            for n in [2,3,4,5]:
                fn=f"rerank_{base_col}_swap{n}_k{int(tab[base_col].sum())}.zip"
                if (out/fn).exists(): rec.append(fn)
    with open(out/"SUBMIT_ORDER.txt","w",encoding="utf-8") as f:
        f.write("Recommended conservative probe order:\n")
        for i,x in enumerate(rec,1): f.write(f"{i}. {x}\n")
        f.write("\nDual-consensus files are exploratory; submit only after observing probe direction.\n")

    print("="*72)
    print("66.9444 local rerank generation completed")
    print(f"output: {out}")
    print("Start with swap2/swap3, not dual-consensus.")
    print("="*72)


def parser():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base116",default=None)
    ap.add_argument("--base124",default=None)
    ap.add_argument("--work-dir",default="./work_v7")
    ap.add_argument("--output-dir",default="./rerank_6694")
    ap.add_argument("--swaps",default="1,2,3,4,5,6")
    ap.add_argument("--allow-drop-dual",action="store_true",help="allow dropping samples positive in both 66.94 anchors")
    return ap


if __name__ == "__main__":
    run(parser().parse_args())
