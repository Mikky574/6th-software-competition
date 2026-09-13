#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

CACHE_TAG = "v7_eventaware_1"

EVENT_TIMES = {
    "WELL_000001": "2025-01-17 07:30:00",
    "WELL_000002": "2025-01-20 14:58:00",
    "WELL_000003": "2025-05-04 22:10:00",
    "WELL_000004": "2025-07-01 12:36:00",
    "WELL_000005": "2025-07-18 00:55:00",
    "WELL_000006": "2025-07-19 09:52:00",
    "WELL_000007": "2025-07-31 14:32:00",
    "WELL_000008": "2023-10-17 20:42:00",
    "WELL_000009": "2023-09-20 00:40:00",
    "WELL_000010": "2023-07-24 07:38:00",
}

SELECTED = [
    4, 5, 10, 11, 12, 18, 19, 28, 32, 47,
    59, 60, 61, 62, 66, 67, 72, 73, 74, 87, 88,
    98, 99, 111, 112, 113, 114, 115, 116, 117, 118, 125,
]

GRAPH_CHANNELS = [4, 5, 10, 12, 18, 19, 62, 66, 67, 72, 73, 74, 87, 99, 118, 125]
WINDOWS = [60, 180, 300, 600, 900, 1800]
SUPPORTED = {".csv", ".txt", ".xls", ".xlsx", ".xlsm"}

POSITIVE_HORIZON = 1800
AMBIGUOUS_GAP = 2400
NEGATIVE_LOOKBACK = 6 * 3600
POSITIVE_STRIDE = 60
NEGATIVE_STRIDE = 180
TRAIN_KEEP_BEFORE = 9 * 3600
TRAIN_KEEP_AFTER = 2 * 3600


def finite(x, default=0.0):
    try:
        return float(x) if np.isfinite(x) else default
    except Exception:
        return default


def signed_log(x):
    x = finite(x, 0.0)
    return math.copysign(math.log1p(abs(x)), x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def rank01(x):
    x = np.asarray(x, dtype=float)
    if len(x) <= 1:
        return np.zeros_like(x)
    order = np.argsort(np.argsort(x))
    return order.astype(float) / float(len(x) - 1)


def slope(t, x):
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    m = np.isfinite(t) & np.isfinite(x)
    if m.sum() < 3:
        return 0.0
    tt = t[m]
    xx = x[m]
    tt = tt - tt.mean()
    xx = xx - xx.mean()
    den = np.sum(tt * tt)
    if den <= 1e-12:
        return 0.0
    return finite(np.sum(tt * xx) / den)


def find_well_directory(train_dir: Path, well: str):
    aliases = {
        "WELL_000001": ["WELL_000001"],
        "WELL_000002": ["WELL_000002"],
        "WELL_000003": ["WELL_000003"],
        "WELL_000004": ["WELL_000004"],
        "WELL_000005": ["WELL_000005"],
        "WELL_000006": ["WELL_000006"],
        "WELL_000007": ["WELL_000007"],
        "WELL_000008": ["WELL_000008"],
        "WELL_000009": ["WELL_000009"],
        "WELL_000010": ["WELL_000010", "WELL_0000010"],
    }
    for name in aliases[well]:
        p = train_dir / name
        if p.exists() and p.is_dir():
            print(f"[well-dir] {well} -> {p}")
            return p
    raise FileNotFoundError(f"Cannot find {well} under {train_dir}; tried {aliases[well]}")


def iter_table_files(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED:
            yield p


def read_csv_robust(path: Path):
    last = None
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]:
        try:
            df = pd.read_csv(path, header=None, encoding=enc, low_memory=False)
            if df.shape[1] >= 10:
                return df
        except Exception as e:
            last = e
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]:
        try:
            df = pd.read_csv(path, header=None, encoding=enc, sep=None, engine="python")
            if df.shape[1] >= 10:
                return df
        except Exception as e:
            last = e
    raise RuntimeError(f"Cannot read CSV {path}: {last}")


def largest_sheet(path: Path, engine: str):
    books = pd.read_excel(path, sheet_name=None, header=None, engine=engine)
    if not books:
        raise RuntimeError(f"No sheets in {path}")
    return max(books.values(), key=lambda d: d.shape[0] * max(d.shape[1], 1))


def read_table_file(path: Path):
    ext = path.suffix.lower()
    if ext in {".csv", ".txt"}:
        return read_csv_robust(path)
    if ext == ".xls":
        return largest_sheet(path, "xlrd")
    if ext in {".xlsx", ".xlsm"}:
        return largest_sheet(path, "openpyxl")
    raise RuntimeError(f"Unsupported table: {path}")


def date_from_text(text: str):
    text = str(text)
    for pat in [r"(20\d{2})[-_/年](\d{1,2})[-_/月](\d{1,2})", r"(20\d{2})(\d{2})(\d{2})"]:
        m = re.search(pat, text)
        if not m:
            continue
        y, mo, d = map(int, m.groups())
        try:
            return pd.Timestamp(y, mo, d)
        except Exception:
            pass
    return None


def real_date_ratio(s):
    vals = pd.Series(s).dropna().astype(str).head(100)
    if len(vals) == 0:
        return 0.0
    hit = 0
    for v in vals:
        if re.search(r"20\d{2}[-/年]\d{1,2}[-/月]\d{1,2}", v) or re.search(r"\b20\d{6}\b", v):
            hit += 1
    return hit / len(vals)


def timestamp_candidate_score(dt, event):
    dt = pd.Series(pd.to_datetime(dt, errors="coerce"))
    valid = dt.notna()
    if valid.mean() < 0.30:
        return None
    x = dt[valid]
    if len(x) < 20:
        return None

    lo = event - pd.Timedelta(seconds=TRAIN_KEEP_BEFORE)
    hi = event + pd.Timedelta(seconds=TRAIN_KEEP_AFTER)
    inside = (x >= lo) & (x <= hi)
    n_inside = int(inside.sum())
    if n_inside < 20:
        return None

    near = x[inside].sort_values()
    ns = near.astype("int64").to_numpy()
    if len(ns) < 3:
        return None
    dif = np.diff(ns) / 1e9
    positive_dt = dif[(dif > 0) & np.isfinite(dif)]
    median_dt = float(np.median(positive_dt)) if len(positive_dt) else 999999.0
    if median_dt > 600 and n_inside < 100:
        return None

    nearest_event_sec = float(np.min(np.abs((x - event).dt.total_seconds())))
    total_span_days = float((x.max() - x.min()).total_seconds() / 86400.0)
    monotonic = float(np.mean(dif >= 0)) if len(dif) else 0.0

    score = 12.0 * math.log1p(n_inside) + 5.0 * monotonic - 0.20 * math.log1p(nearest_event_sec)
    if total_span_days > 180:
        score -= 2.0 * math.log1p(total_span_days)
    if 0.05 <= median_dt <= 30:
        score += 5.0
    elif median_dt <= 120:
        score += 2.0

    return {
        "score": float(score),
        "n_inside": n_inside,
        "median_dt": median_dt,
        "nearest_event_sec": nearest_event_sec,
        "span_days": total_span_days,
    }


def detect_timestamp_train(df, filename, event):
    event = pd.Timestamp(event)
    scan_cols = min(df.shape[1], 20)
    candidates = []

    def add(dt, source):
        try:
            info = timestamp_candidate_score(dt, event)
            if info is not None:
                candidates.append((info["score"], source, pd.Series(dt), info))
        except Exception:
            pass

    numeric = {}
    for c in range(scan_cols):
        s = df.iloc[:, c]
        if real_date_ratio(s) >= 0.10:
            add(pd.to_datetime(s, errors="coerce"), f"datetime_string_col_{c}")
        numeric[c] = pd.to_numeric(s, errors="coerce")

    for c, z in numeric.items():
        if z.notna().mean() < 0.50:
            continue
        med = np.nanmedian(z)
        if np.isfinite(med) and 30000 < med < 80000:
            dt = pd.Timestamp("1899-12-30") + pd.to_timedelta(z, unit="D")
            add(dt, f"excel_serial_col_{c}")

    pair_cols = min(scan_cols, 10)
    for a in range(pair_cols):
        da = numeric[a]
        if da.notna().mean() < 0.50:
            continue
        med_a = np.nanmedian(da)
        if not (30000 < med_a < 80000):
            continue
        for b in range(pair_cols):
            if a == b:
                continue
            tb = numeric[b]
            if tb.notna().mean() < 0.50:
                continue
            med_b = np.nanmedian(tb)
            if -0.01 <= med_b <= 1.5:
                dt = pd.Timestamp("1899-12-30") + pd.to_timedelta(da + tb, unit="D")
                add(dt, f"excel_date_{a}_time_{b}")

    file_date = date_from_text(filename)
    if file_date is not None:
        for c in range(scan_cols):
            s = df.iloc[:, c]
            try:
                td = pd.to_timedelta(s.astype(str), errors="coerce")
                if td.notna().mean() >= 0.50:
                    sec = td.dt.total_seconds()
                    if sec.between(0, 86400 * 1.1).mean() >= 0.50:
                        add(file_date + td, f"filename_date_time_col_{c}")
            except Exception:
                pass

            z = numeric[c]
            if z.notna().mean() < 0.50:
                continue
            med = np.nanmedian(z)
            if 1 <= med <= 90000:
                add(file_date + pd.to_timedelta(z, unit="s"), f"filename_date_seconds_col_{c}")
            if 0 <= med <= 1.5:
                add(file_date + pd.to_timedelta(z, unit="D"), f"filename_date_fraction_col_{c}")

    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, source, dt, info = candidates[0]
    return pd.to_datetime(dt, errors="coerce"), {"source": source, **info}


def reduce_frame(df, filename="", expected_event=None, synthetic_test=False):
    if df is None or len(df) == 0:
        return None, None
    out = pd.DataFrame()
    info = None

    if synthetic_test:
        n = len(df)
        if n < 10:
            return None, None
        step = 1799.0 / max(n - 1, 1)
        out["__time"] = np.arange(n, dtype=float) * step
    else:
        if expected_event is None:
            raise ValueError("expected_event required for train")
        dt, info = detect_timestamp_train(df, filename, pd.Timestamp(expected_event))
        if dt is None:
            return None, None
        out["__time"] = dt

    for idx in SELECTED:
        if idx < df.shape[1]:
            out[f"c{idx}"] = pd.to_numeric(df.iloc[:, idx], errors="coerce")
        else:
            out[f"c{idx}"] = np.nan

    value_cols = [f"c{i}" for i in SELECTED]
    good = out[value_cols].notna().sum(axis=1) >= 3
    out = out.loc[good].reset_index(drop=True)
    if len(out) < 10:
        return None, None

    if not synthetic_test:
        event = pd.Timestamp(expected_event)
        lo = event - pd.Timedelta(seconds=TRAIN_KEEP_BEFORE)
        hi = event + pd.Timedelta(seconds=TRAIN_KEEP_AFTER)
        out = out[(out["__time"] >= lo) & (out["__time"] <= hi)].reset_index(drop=True)
        if len(out) < 20:
            return None, None

    return out, info


def validate_event_range(df, well):
    event = pd.Timestamp(EVENT_TIMES[well])
    tmin = df["__time"].min()
    tmax = df["__time"].max()
    nearest = float(np.min(np.abs((df["__time"] - event).dt.total_seconds())))
    before_h = float((event - tmin).total_seconds() / 3600.0)
    after_h = float((tmax - event).total_seconds() / 3600.0)
    print(f"[check] {well}: {tmin} -> {tmax}, event={event}, before={before_h:.2f}h, after={after_h:.2f}h, nearest={nearest:.1f}s")
    if nearest > 600:
        raise RuntimeError(f"{well}: no valid data within 10 minutes of event; nearest={nearest:.1f}s")
    if before_h < 1.0:
        raise RuntimeError(f"{well}: less than 1 hour pre-event data")


def load_well_dir(well_dir: Path, well: str):
    files = list(iter_table_files(well_dir))
    print(f"[load] {well_dir}\n       table files={len(files)}")
    if not files:
        raise RuntimeError(f"No table files under {well_dir}")

    event = pd.Timestamp(EVENT_TIMES[well])
    pieces = []
    failed = 0
    accepted = []

    for i, path in enumerate(files):
        try:
            raw = read_table_file(path)
            small, info = reduce_frame(raw, filename=str(path), expected_event=event, synthetic_test=False)
            if small is not None and len(small) > 5:
                pieces.append(small)
                accepted.append((path.name, len(small), info))
            else:
                failed += 1
        except Exception:
            failed += 1
        if (i + 1) % 20 == 0 or i + 1 == len(files):
            print(f"       parsed={i+1}/{len(files)}, usable={len(pieces)}, failed={failed}")

    if not pieces:
        raise RuntimeError(f"No event-aware timestamped data found in {well_dir}")

    x = pd.concat(pieces, ignore_index=True)
    x["__time"] = pd.to_datetime(x["__time"], errors="coerce")
    x = x.dropna(subset=["__time"]).sort_values("__time").drop_duplicates(subset=["__time"], keep="last").reset_index(drop=True)
    print(f"       retained rows={len(x):,}")
    for name, n, info in accepted[:5]:
        if info:
            print(f"       time-source {name}: {info['source']}, rowsNearEvent={info['n_inside']}, dt={info['median_dt']:.3f}s")
    validate_event_range(x, well)
    return x


def load_well_from_train(train_dir: Path, well: str):
    return load_well_dir(find_well_directory(train_dir, well), well)


def channel_array(df, idx):
    return pd.to_numeric(df[f"c{idx}"], errors="coerce").to_numpy(float)


def relative_seconds(df):
    t = df["__time"]
    if pd.api.types.is_datetime64_any_dtype(t):
        end = t.iloc[-1]
        return (t - end).dt.total_seconds().to_numpy(float)
    x = pd.to_numeric(t, errors="coerce").to_numpy(float)
    return x - np.nanmax(x)


def one_channel_features(rel, x):
    x = np.asarray(x, dtype=float)
    out = [finite(np.mean(~np.isfinite(x)))]
    valid = np.where(np.isfinite(x))[0]
    if len(valid) < 3:
        return out + [0.0] * (3 + 5 * len(WINDOWS) + 3)

    med = np.nanmedian(x)
    q25, q75 = np.nanpercentile(x, [25, 75])
    scale = q75 - q25
    if not np.isfinite(scale) or scale < 1e-9:
        scale = np.nanstd(x)
    if not np.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    z = (x - med) / scale
    first, last = z[valid[0]], z[valid[-1]]
    out += [finite(np.clip(last, -20, 20)), finite(np.clip(last-first, -20, 20)), finite(np.clip(slope(rel, z)*300.0, -20, 20))]

    for w in WINDOWS:
        m = rel >= -w
        zz, tt = z[m], rel[m]
        good = np.isfinite(zz)
        zz, tt = zz[good], tt[good]
        if len(zz) < 3:
            out += [0.0] * 5
            continue
        mid = len(zz) // 2
        half = 0.0 if mid < 1 or mid >= len(zz) else np.nanmean(zz[mid:]) - np.nanmean(zz[:mid])
        out += [
            finite(np.nanmean(zz)),
            finite(np.nanstd(zz)),
            finite(np.nanpercentile(zz, 90) - np.nanpercentile(zz, 10)),
            finite(np.clip(half, -20, 20)),
            finite(np.clip(slope(tt, zz)*300.0, -20, 20)),
        ]

    raw_first, raw_last = x[valid[0]], x[valid[-1]]
    out += [signed_log(med), signed_log(raw_last), signed_log(raw_last-raw_first)]
    return out


def physical_channels(df):
    def a(i): return channel_array(df, i)
    flow_in, flow_out, flow_out2 = a(72), a(73), a(99)
    effective_out = flow_out2 if np.isfinite(flow_out2).mean() > 0.40 else flow_out
    flow_balance = effective_out - flow_in
    gas_sum = np.zeros(len(df), dtype=float)
    for idx in [111, 112, 113, 114, 115, 116, 117]:
        gas_sum += np.nan_to_num(a(idx))
    return [flow_balance, a(88), a(87), a(67)-a(66), a(118), gas_sum, a(125), a(62), a(18), a(98), a(19), a(74)]


def infer_regime(df):
    pump, rpm, bit, hook = channel_array(df,62), channel_array(df,12), channel_array(df,5), channel_array(df,10)
    pump90 = finite(np.nanpercentile(np.abs(pump), 90))
    rpm90 = finite(np.nanpercentile(np.abs(rpm), 90))
    bit_range = finite(np.nanpercentile(bit,90)-np.nanpercentile(bit,10))
    hook_std = finite(np.nanstd(hook))
    if pump90 > 1e-6 and rpm90 > 1e-6: return 0
    if pump90 > 1e-6: return 1
    if abs(bit_range) > 1e-5 or hook_std > 1e-5: return 2
    return 3


def endpoint_vector(df, where):
    rel = relative_seconds(df)
    if where == "start":
        mask = rel <= np.nanmin(rel) + 60
    else:
        mask = rel >= -60
    vals = []
    for idx in GRAPH_CHANNELS:
        z = channel_array(df, idx)[mask]
        vals.append(signed_log(np.nanmedian(z)) if np.isfinite(z).sum() else np.nan)
    return np.asarray(vals, dtype=np.float32)


def anchor_score(df):
    rel = relative_seconds(df)
    recent = rel >= -300
    earlier = (rel >= -1800) & (rel < -600)
    def change(x):
        if np.isfinite(x[recent]).sum() < 3 or np.isfinite(x[earlier]).sum() < 3:
            return 0.0
        base = np.nanmedian(x[earlier])
        scale = np.nanpercentile(x[earlier],75)-np.nanpercentile(x[earlier],25)
        if not np.isfinite(scale) or abs(scale) < 1e-9:
            scale = np.nanstd(x[earlier])
        if not np.isfinite(scale) or scale < 1e-9:
            scale = 1.0
        return finite((np.nanmedian(x[recent])-base)/scale)
    flow = channel_array(df,99)-channel_array(df,72)
    parts = [flow, channel_array(df,88), channel_array(df,87), channel_array(df,118), channel_array(df,98)]
    weights = [0.30,0.20,0.15,0.20,0.15]
    s = sum(w*max(change(x),0.0) for w,x in zip(weights,parts))
    return float(np.clip(s,0,20))


def extract_features(df):
    rel = relative_seconds(df)
    duration = np.nanmax(rel)-np.nanmin(rel)
    if duration < 300:
        raise ValueError(f"slice too short: {duration:.1f}s")
    feats, missing = [], []
    for idx in SELECTED:
        x = channel_array(df, idx)
        missing.append(float(np.mean(~np.isfinite(x))))
        feats += one_channel_features(rel, x)
    for x in physical_channels(df):
        feats += one_channel_features(rel, x)
    regime = infer_regime(df)
    feats += [1.0 if regime == r else 0.0 for r in range(4)]
    return (
        np.asarray(feats,dtype=np.float32),
        endpoint_vector(df,"start"),
        endpoint_vector(df,"end"),
        np.asarray(missing,dtype=np.float32),
        int(regime),
        anchor_score(df),
    )


def slice_by_end_time(df, end_time, history=1800):
    times = df["__time"].astype("int64").to_numpy()
    end_ns = pd.Timestamp(end_time).value
    start_ns = end_ns - int(history*1e9)
    lo = np.searchsorted(times,start_ns,side="left")
    hi = np.searchsorted(times,end_ns,side="right")
    if hi-lo < 50:
        return None
    seg = df.iloc[lo:hi].copy()
    duration = (seg["__time"].iloc[-1]-seg["__time"].iloc[0]).total_seconds()
    return seg if duration >= 600 else None


def build_well_samples(df, well):
    event = pd.Timestamp(EVENT_TIMES[well])
    tmin, tmax = df["__time"].min(), df["__time"].max()
    X,y,regimes = [],[],[]
    plans = [
        (1, np.arange(30, POSITIVE_HORIZON+1, POSITIVE_STRIDE)),
        (0, np.arange(AMBIGUOUS_GAP, NEGATIVE_LOOKBACK+1, NEGATIVE_STRIDE)),
    ]
    for label,deltas in plans:
        for delta in deltas:
            end_time = event - pd.Timedelta(seconds=int(delta))
            if end_time <= tmin or end_time > tmax:
                continue
            seg = slice_by_end_time(df,end_time)
            if seg is None:
                continue
            try:
                f,_,_,_,regime,_ = extract_features(seg)
            except Exception:
                continue
            X.append(f); y.append(label); regimes.append(regime)
    if not X:
        raise RuntimeError(f"No samples for {well}")
    X = np.vstack(X).astype(np.float32)
    y = np.asarray(y,dtype=np.int8)
    regimes = np.asarray(regimes,dtype=np.int8)
    print(f"[samples] {well}: n={len(y)} pos={int(y.sum())} neg={int((y==0).sum())}")
    if y.sum() < 5 or (y==0).sum() < 5:
        raise RuntimeError(f"Insufficient class samples for {well}")
    return X,y,regimes


def build_train_features(train_dir, work_dir, force=False):
    cache = work_dir / f"train_features_{CACHE_TAG}.npz"
    if cache.exists() and not force:
        print(f"[cache] {cache}")
        z=np.load(cache,allow_pickle=True)
        return z["X"],z["y"],z["groups"],z["regimes"]
    Xa,ya,ga,ra=[],[],[],[]
    for gi in range(1,11):
        well=f"WELL_{gi:06d}"
        reduced=work_dir/f"{well}_{CACHE_TAG}_reduced.pkl"
        if reduced.exists() and not force:
            print(f"[cache] {reduced}")
            df=pd.read_pickle(reduced)
            validate_event_range(df,well)
        else:
            df=load_well_from_train(train_dir,well)
            df.to_pickle(reduced)
        X,y,r=build_well_samples(df,well)
        Xa.append(X); ya.append(y); ra.append(r); ga.append(np.full(len(y),gi-1,dtype=np.int8))
    X=np.vstack(Xa).astype(np.float32); y=np.concatenate(ya); groups=np.concatenate(ga); regimes=np.concatenate(ra)
    np.savez_compressed(cache,X=X,y=y,groups=groups,regimes=regimes)
    print(f"[train features] X={X.shape} positive={int(y.sum())} negative={int((y==0).sum())}")
    return X,y,groups,regimes


def test_id(path):
    m=re.search(r"(test_\d+)",str(path),re.I)
    if not m: raise ValueError(f"Bad test path: {path}")
    return m.group(1).lower()


def test_num(s):
    m=re.search(r"(\d+)$",s)
    return int(m.group(1)) if m else 10**9


def build_test_features(test_dir, work_dir, force=False):
    cache=work_dir/f"test_features_{CACHE_TAG}.npz"
    if cache.exists() and not force:
        print(f"[cache] {cache}")
        z=np.load(cache,allow_pickle=True)
        return z["X"],z["ids"].astype(str),z["starts"],z["ends"],z["missing"],z["regimes"],z["anchors"]
    files=[p for p in iter_table_files(test_dir) if re.search(r"test_\d+",str(p),re.I)]
    print(f"[test] files={len(files)}")
    if len(files)!=324:
        raise RuntimeError(f"Expected 324 test slices, found {len(files)}")
    rows=[]
    for i,path in enumerate(sorted(files,key=lambda p:test_num(test_id(p)))):
        sid=test_id(path)
        raw=read_table_file(path)
        df,_=reduce_frame(raw,filename=str(path),synthetic_test=True)
        if df is None: raise RuntimeError(f"Cannot parse {path}")
        f,start,end,miss,regime,anchor=extract_features(df)
        rows.append((sid,f,start,end,miss,regime,anchor))
        if (i+1)%25==0 or i+1==len(files): print(f"[test] {i+1}/{len(files)}")
    rows.sort(key=lambda x:test_num(x[0]))
    ids=np.asarray([r[0] for r in rows],dtype=object)
    expected={f"test_{i:03d}" for i in range(1,325)}
    if set(ids)!=expected: raise RuntimeError(f"Test IDs mismatch; missing={sorted(expected-set(ids))}, extra={sorted(set(ids)-expected)}")
    X=np.vstack([r[1] for r in rows]).astype(np.float32)
    starts=np.vstack([r[2] for r in rows]).astype(np.float32)
    ends=np.vstack([r[3] for r in rows]).astype(np.float32)
    missing=np.vstack([r[4] for r in rows]).astype(np.float32)
    regimes=np.asarray([r[5] for r in rows],dtype=np.int8)
    anchors=np.asarray([r[6] for r in rows],dtype=np.float32)
    np.savez_compressed(cache,X=X,ids=ids,starts=starts,ends=ends,missing=missing,regimes=regimes,anchors=anchors)
    return X,ids.astype(str),starts,ends,missing,regimes,anchors


def fill_matrix(X, med=None):
    X=np.asarray(X,dtype=float)
    if med is None:
        med=np.nanmedian(X,axis=0); med[~np.isfinite(med)]=0.0
    out=X.copy(); rr,cc=np.where(~np.isfinite(out))
    if len(rr): out[rr,cc]=med[cc]
    return np.clip(out,-1e8,1e8),med


def fit_ensemble(X,y,pos_weight=1.6,seed=3407):
    Xf,med=fill_matrix(X)
    scaler=StandardScaler(); Xs=scaler.fit_transform(Xf)
    lr=LogisticRegression(C=0.25,max_iter=2000,class_weight={0:1.0,1:pos_weight},random_state=seed).fit(Xs,y)
    et=ExtraTreesClassifier(n_estimators=500,min_samples_leaf=2,max_features="sqrt",class_weight={0:1.0,1:pos_weight},n_jobs=-1,random_state=seed).fit(Xf,y)
    sw=np.where(y==1,pos_weight,1.0)
    hgb=HistGradientBoostingClassifier(learning_rate=0.04,max_iter=300,max_leaf_nodes=31,min_samples_leaf=12,l2_regularization=1.5,random_state=seed).fit(Xf,y,sample_weight=sw)
    cpos=np.median(Xs[y==1],axis=0); cneg=np.median(Xs[y==0],axis=0)
    dpos=np.mean((Xs-cpos)**2,axis=1); dneg=np.mean((Xs-cneg)**2,axis=1)
    temp=np.nanstd(dneg-dpos)
    if not np.isfinite(temp) or temp<1e-6: temp=1.0
    return {"med":med,"scaler":scaler,"lr":lr,"et":et,"hgb":hgb,"cpos":cpos,"cneg":cneg,"temp":temp}


def predict_ensemble(model,X):
    Xf,_=fill_matrix(X,model["med"]); Xs=model["scaler"].transform(Xf)
    p_lr=model["lr"].predict_proba(Xs)[:,1]; p_et=model["et"].predict_proba(Xf)[:,1]; p_hgb=model["hgb"].predict_proba(Xf)[:,1]
    dpos=np.mean((Xs-model["cpos"])**2,axis=1); dneg=np.mean((Xs-model["cneg"])**2,axis=1)
    p_proto=sigmoid((dneg-dpos)/model["temp"])
    return np.clip(0.50*p_et+0.25*p_hgb+0.15*p_lr+0.10*p_proto,0,1)


def competition_metric(y,pred):
    y=np.asarray(y,dtype=int); pred=np.asarray(pred,dtype=int)
    tp=int(np.sum((y==1)&(pred==1))); fn=int(np.sum((y==1)&(pred==0))); tn=int(np.sum((y==0)&(pred==0))); fp=int(np.sum((y==0)&(pred==1)))
    recall=tp/max(tp+fn,1); specificity=tn/max(tn+fp,1); score=100.0*recall*recall*specificity
    return {"score":float(score),"recall":float(recall),"specificity":float(specificity),"tp":tp,"fn":fn,"fp":fp,"tn":tn}


def search_threshold(y,p):
    ts=np.unique(np.quantile(p,np.linspace(0.01,0.99,600)))
    best=None
    for t in ts:
        pred=(p>=t).astype(int); m=competition_metric(y,pred); m.update({"threshold":float(t),"positive_rate":float(pred.mean())})
        if best is None or m["score"]>best["score"]: best=m
    return best


def loso_oof(X,y,groups,pos_weight):
    oof=np.zeros(len(y),dtype=float); per=[]
    for g in np.unique(groups):
        tr=groups!=g; va=groups==g
        print(f"[LOSO] well={int(g)+1} train={tr.sum()} val={va.sum()}")
        model=fit_ensemble(X[tr],y[tr],pos_weight=pos_weight,seed=3407+int(g))
        p=predict_ensemble(model,X[va]); oof[va]=p
        try: auc=float(roc_auc_score(y[va],p))
        except Exception: auc=-1.0
        local=search_threshold(y[va],p); per.append({"well":int(g)+1,"auc":auc,**local})
        print(f"       auc={auc:.5f} score={local['score']:.5f} recall={local['recall']:.5f} spec={local['specificity']:.5f}")
    overall=search_threshold(y,oof)
    print(f"[OOF] score={overall['score']:.6f} recall={overall['recall']:.6f} spec={overall['specificity']:.6f} thr={overall['threshold']:.8f} posRate={overall['positive_rate']:.6f}")
    return oof,overall,per


def robust_matrix(Z):
    Z=np.asarray(Z,dtype=float); med=np.nanmedian(Z,axis=0); q25=np.nanpercentile(Z,25,axis=0); q75=np.nanpercentile(Z,75,axis=0); scale=q75-q25
    med[~np.isfinite(med)]=0.0; scale[(~np.isfinite(scale))|(scale<1e-6)]=1.0
    X=Z.copy()
    for j in range(X.shape[1]):
        bad=~np.isfinite(X[:,j]); X[bad,j]=med[j]
    return (X-med)/scale


def infer_hidden_domains(starts,ends,missing,n_domains):
    stable=(np.nan_to_num(starts)+np.nan_to_num(ends))/2.0
    Z=robust_matrix(np.concatenate([stable,3.0*missing],axis=1))
    domains=KMeans(n_clusters=n_domains,n_init=50,random_state=3407).fit_predict(Z)
    print(f"[domains] counts={np.bincount(domains).tolist()}")
    return domains


def pairwise_sqdist(A,B):
    aa=np.sum(A*A,axis=1,keepdims=True); bb=np.sum(B*B,axis=1,keepdims=True).T
    return np.maximum(aa+bb-2.0*A@B.T,0.0)


def build_temporal_graph(starts,ends,domains):
    n=len(starts); both=robust_matrix(np.vstack([starts,ends])); S=both[:n]; E=both[n:]; edges=[]
    for domain in np.unique(domains):
        idx=np.where(domains==domain)[0]
        if len(idx)<3: continue
        D=pairwise_sqdist(E[idx],S[idx]); np.fill_diagonal(D,np.inf)
        succ=np.argmin(D,axis=1); pred=np.argmin(D,axis=0); sd=D[np.arange(len(idx)),succ]; fd=sd[np.isfinite(sd)]
        if len(fd)<2: continue
        limit=np.quantile(fd,0.65)
        for i in range(len(idx)):
            j=succ[i]; d=sd[i]
            if pred[j]!=i or not np.isfinite(d) or d>limit: continue
            edges.append({"src":int(idx[i]),"dst":int(idx[j]),"domain":int(domain),"distance":float(d),"confidence":float(math.exp(-d/max(limit,1e-6)))})
    print(f"[graph] strong edges={len(edges)}")
    return edges


def graph_propagation(base_prob,anchors,edges):
    p=np.asarray(base_prob,dtype=float).copy(); p=0.96*p+0.04*rank01(anchors)
    for _ in range(3):
        old=p.copy()
        for e in edges:
            s,d,c=e["src"],e["dst"],e["confidence"]
            if old[d]>old[s]: p[s]+=0.16*c*(old[d]-old[s])
            if old[s]>old[d]: p[d]+=0.035*c*(old[s]-old[d])
        p=np.clip(p,0,1)
    return p


def topk_prediction(p,k):
    k=int(np.clip(k,1,len(p)-1)); pred=np.zeros(len(p),dtype=np.int8); pred[np.argsort(-p)[:k]]=1; return pred


def write_zip(ids,pred,path):
    df=pd.DataFrame({"切片ID":ids,"溢流判断":np.asarray(pred,dtype=int)})
    if len(df)!=324 or df["切片ID"].nunique()!=324 or not set(df["溢流判断"].unique()).issubset({0,1}): raise RuntimeError("Invalid submission")
    data=df.to_csv(index=False,lineterminator="\n").encode("utf-8-sig")
    path=Path(path)
    with zipfile.ZipFile(path,"w",compression=zipfile.ZIP_DEFLATED) as zf: zf.writestr("result.csv",data)
    print(f"[output] {path} positive={int(df['溢流判断'].sum())}")


def scan_only(train_dir,work_dir,force):
    print("Running event-aware train validation only")
    for gi in range(1,11):
        well=f"WELL_{gi:06d}"; cache=work_dir/f"{well}_{CACHE_TAG}_reduced.pkl"
        if cache.exists() and not force:
            df=pd.read_pickle(cache); validate_event_range(df,well)
        else:
            df=load_well_from_train(train_dir,well); df.to_pickle(cache)
    print("All 10 wells passed event-time validation")


def run(args):
    train_dir=Path(args.train_dir).resolve(); test_dir=Path(args.test_dir).resolve(); work_dir=Path(args.work_dir).resolve(); output=Path(args.output).resolve()
    work_dir.mkdir(parents=True,exist_ok=True)
    print(f"[path] train={train_dir}\n[path] test ={test_dir}\n[path] work ={work_dir}")
    if args.scan_only:
        scan_only(train_dir,work_dir,args.force_rebuild); return

    X,y,groups,train_regimes=build_train_features(train_dir,work_dir,args.force_rebuild)
    XT,ids,starts,ends,missing,test_regimes,anchors=build_test_features(test_dir,work_dir,args.force_rebuild)
    if X.shape[1]!=XT.shape[1]: raise RuntimeError(f"Feature mismatch train={X.shape} test={XT.shape}")

    oof,best,per_well=loso_oof(X,y,groups,args.pos_weight)
    pd.DataFrame({"well":groups+1,"label":y,"regime":train_regimes,"oof_prob":oof,"oof_pred":(oof>=best['threshold']).astype(int)}).to_csv(work_dir/"V7_oof.csv",index=False)

    print("[final] fitting all wells")
    model=fit_ensemble(X,y,args.pos_weight,7777); p_base=predict_ensemble(model,XT)
    domains=infer_hidden_domains(starts,ends,missing,args.hidden_domains)
    if args.no_graph:
        edges=[]; p_graph=p_base.copy()
    else:
        edges=build_temporal_graph(starts,ends,domains); p_graph=graph_propagation(p_base,anchors,edges)
    p_final=np.clip(0.94*p_graph+0.06*rank01(p_graph),0,1)

    oof_rate=float(np.clip(best["positive_rate"],args.min_positive_rate,args.max_positive_rate))
    expected_count=round(len(ids)*oof_rate*args.count_scale)
    threshold_pred=(p_final>=best["threshold"]).astype(np.int8); threshold_count=int(threshold_pred.sum())
    main_count=round(0.65*expected_count+0.35*threshold_count)
    main_count=int(np.clip(main_count,round(len(ids)*args.min_positive_rate),round(len(ids)*args.max_positive_rate)))
    precision_count=max(1,round(main_count*0.92)); recall_count=min(len(ids)-1,round(main_count*1.08))
    main_pred=topk_prediction(p_final,main_count); precision_pred=topk_prediction(p_final,precision_count); recall_pred=topk_prediction(p_final,recall_count)

    write_zip(ids,main_pred,output)
    precision_path=output.with_name(output.stem+"_precision.zip"); recall_path=output.with_name(output.stem+"_recall.zip"); threshold_path=output.with_name(output.stem+"_threshold.zip")
    write_zip(ids,precision_pred,precision_path); write_zip(ids,recall_pred,recall_path); write_zip(ids,threshold_pred,threshold_path)

    debug=pd.DataFrame({"切片ID":ids,"base_prob":p_base,"graph_prob":p_graph,"final_prob":p_final,"physical_anchor":anchors,"domain":domains,"regime":test_regimes,"main_pred":main_pred})
    debug["rank"]=debug["final_prob"].rank(method="first",ascending=False).astype(int); debug.sort_values("rank").to_csv(work_dir/"V7_test_debug.csv",index=False)
    pd.DataFrame([{"src_id":ids[e['src']],"dst_id":ids[e['dst']],"domain":e['domain'],"distance":e['distance'],"confidence":e['confidence']} for e in edges]).to_csv(work_dir/"V7_graph_edges.csv",index=False)
    summary={"cache_tag":CACHE_TAG,"train_shape":list(X.shape),"test_shape":list(XT.shape),"oof":best,"per_well":per_well,"hidden_domains":args.hidden_domains,"graph_edges":len(edges),"expected_count":int(expected_count),"threshold_count":threshold_count,"main_count":main_count,"precision_count":precision_count,"recall_count":recall_count}
    with open(work_dir/"V7_summary.json","w",encoding="utf-8") as f: json.dump(summary,f,ensure_ascii=False,indent=2)

    print("="*70)
    print("Temporal-Graph V7 completed")
    print(f"OOF Score       : {best['score']:.6f}")
    print(f"OOF Recall      : {best['recall']:.6f}")
    print(f"OOF Specificity : {best['specificity']:.6f}")
    print(f"OOF threshold   : {best['threshold']:.8f}")
    print(f"Graph edges     : {len(edges)}")
    print(f"Main count      : {main_count}")
    print(f"MAIN            : {output}")
    print("="*70)


def parser():
    ap=argparse.ArgumentParser(description="KickGuard Temporal-Graph V7")
    ap.add_argument("--train-dir",default="../train")
    ap.add_argument("--test-dir",default="../test")
    ap.add_argument("--work-dir",default="./work_v7")
    ap.add_argument("--output",default="./result_v7.zip")
    ap.add_argument("--force-rebuild",action="store_true")
    ap.add_argument("--scan-only",action="store_true")
    ap.add_argument("--no-graph",action="store_true")
    ap.add_argument("--hidden-domains",type=int,default=5)
    ap.add_argument("--pos-weight",type=float,default=1.60)
    ap.add_argument("--count-scale",type=float,default=1.00)
    ap.add_argument("--min-positive-rate",type=float,default=0.28)
    ap.add_argument("--max-positive-rate",type=float,default=0.48)
    return ap


if __name__ == "__main__":
    args=parser().parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("Interrupted")
        sys.exit(130)
    except Exception:
        print("="*70)
        print("V7 FAILED")
        print("="*70)
        raise
