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

CACHE_TAG = "v7_eventfix_1"

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
    59, 60, 61, 62, 66, 67, 72, 73, 74,
    87, 88, 98, 99, 111, 112, 113, 114,
    115, 116, 117, 118, 125,
]

GRAPH_CHANNELS = [4, 5, 10, 12, 18, 19, 62, 66, 67, 72, 73, 74, 87, 99, 118, 125]
WINDOWS = [60, 180, 300, 600, 900, 1800]

TRAIN_KEEP_BEFORE = 9 * 3600
TRAIN_KEEP_AFTER = 2 * 3600
POSITIVE_HORIZON = 1800
AMBIGUOUS_GAP = 2400
NEGATIVE_LOOKBACK = 6 * 3600
POSITIVE_STRIDE = 60
NEGATIVE_STRIDE = 180

SUPPORTED = {".csv", ".txt", ".xls", ".xlsx", ".xlsm"}
FILENAME_RANGE_RE = re.compile(r"(20\d{6})T(\d{6})\s*[~_-]+\s*(20\d{6})T(\d{6})", re.I)


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


def parse_filename_interval(text: str):
    m = FILENAME_RANGE_RE.search(str(text))
    if not m:
        return None
    try:
        start = pd.to_datetime(m.group(1) + m.group(2), format="%Y%m%d%H%M%S")
        end = pd.to_datetime(m.group(3) + m.group(4), format="%Y%m%d%H%M%S")
        if end <= start:
            return None
        return pd.Timestamp(start), pd.Timestamp(end)
    except Exception:
        return None


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
    raise RuntimeError(f"Unsupported file: {path}")


def _time_string_ratio(s):
    vals = pd.Series(s).dropna().astype(str).head(100)
    if len(vals) == 0:
        return 0.0
    hit = 0
    for v in vals:
        if re.search(r"\d{1,2}:\d{2}:\d{2}", v) or re.search(r"20\d{2}[-/]\d{1,2}[-/]\d{1,2}", v):
            hit += 1
    return hit / len(vals)


def _score_timestamp_candidate(dt, file_start, file_end, event):
    dt = pd.Series(pd.to_datetime(dt, errors="coerce"))
    valid = dt.notna()
    if valid.mean() < 0.5:
        return None
    x = dt[valid]
    if len(x) < 20:
        return None
    inside_file = (x >= file_start - pd.Timedelta(minutes=5)) & (x <= file_end + pd.Timedelta(minutes=5))
    file_ratio = float(inside_file.mean())
    if file_ratio < 0.80:
        return None
    ns = x.astype("int64").to_numpy()
    d = np.diff(ns) / 1e9
    mono = float(np.mean(d >= -1e-6)) if len(d) else 0.0
    if mono < 0.80:
        return None
    pos = d[(d > 0) & np.isfinite(d)]
    med_dt = float(np.median(pos)) if len(pos) else np.inf
    if event is not None and file_start <= event <= file_end:
        nearest = float(np.min(np.abs((x - event).dt.total_seconds())))
        if nearest > 1800:
            return None
    else:
        nearest = 0.0
    score = 10.0 * file_ratio + 4.0 * mono - math.log1p(max(nearest, 0.0)) * 0.1
    if 0.02 <= med_dt <= 30:
        score += 4.0
    elif med_dt <= 120:
        score += 2.0
    return score, med_dt, nearest


def detect_timestamp_from_first4(df, file_start, file_end, event):
    ncols = min(4, df.shape[1])
    duration = (file_end - file_start).total_seconds()
    candidates = []

    def add(dt, source):
        try:
            scored = _score_timestamp_candidate(dt, file_start, file_end, event)
            if scored is not None:
                score, med_dt, nearest = scored
                candidates.append((score, source, pd.Series(dt), med_dt, nearest))
        except Exception:
            pass

    numerics = {}
    for c in range(ncols):
        s = df.iloc[:, c]
        if _time_string_ratio(s) >= 0.10:
            try:
                add(pd.to_datetime(s, errors="coerce"), f"datetime_col_{c}")
            except Exception:
                pass
            try:
                td = pd.to_timedelta(s.astype(str), errors="coerce")
                sec = td.dt.total_seconds()
                if sec.notna().mean() >= 0.5:
                    add(file_start.normalize() + td, f"timeofday_col_{c}")
            except Exception:
                pass

        z = pd.to_numeric(s, errors="coerce")
        numerics[c] = z
        if z.notna().mean() < 0.5:
            continue
        vv = z.dropna().to_numpy(float)
        med = np.nanmedian(vv)
        q01 = np.nanpercentile(vv, 1)
        q99 = np.nanpercentile(vv, 99)
        mono = float(np.mean(np.diff(vv) >= -1e-8)) if len(vv) > 1 else 0.0

        if 30000 < med < 80000:
            add(pd.Timestamp("1899-12-30") + pd.to_timedelta(z, unit="D"), f"excel_serial_col_{c}")

        if mono >= 0.90 and q01 >= -5 and q99 <= max(duration * 1.2, 90000):
            if q99 <= 90000:
                event_sod = None if event is None else event.hour * 3600 + event.minute * 60 + event.second
                if event_sod is None or not (file_start <= event <= file_end) or (q01 - 300 <= event_sod <= q99 + 300):
                    add(file_start.normalize() + pd.to_timedelta(z, unit="s"), f"seconds_midnight_col_{c}")
            if q01 <= 60 and q99 <= duration * 1.2:
                add(file_start + pd.to_timedelta(z - np.nanmin(vv), unit="s"), f"elapsed_seconds_col_{c}")

        if mono >= 0.90 and q01 >= -0.01 and q99 <= 1.05:
            add(file_start.normalize() + pd.to_timedelta(z, unit="D"), f"day_fraction_col_{c}")

    for a in range(ncols):
        da = numerics.get(a)
        if da is None or da.notna().mean() < 0.5:
            continue
        ma = np.nanmedian(da)
        if not (30000 < ma < 80000):
            continue
        for b in range(ncols):
            if a == b:
                continue
            tb = numerics.get(b)
            if tb is None or tb.notna().mean() < 0.5:
                continue
            mb = np.nanmedian(tb)
            if -0.01 <= mb <= 1.05:
                add(pd.Timestamp("1899-12-30") + pd.to_timedelta(da + tb, unit="D"), f"excel_date_{a}_fraction_{b}")

    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[0], reverse=True)
    score, source, dt, med_dt, nearest = candidates[0]
    return pd.to_datetime(dt, errors="coerce"), {
        "source": source,
        "score": float(score),
        "median_dt": float(med_dt),
        "nearest_event": float(nearest),
    }


def filename_linear_timestamp(n, start, end):
    if n <= 0:
        return pd.Series([], dtype="datetime64[ns]")
    total_ns = int((end.value - start.value))
    offsets = (np.arange(n, dtype=np.float64) / max(n, 1) * total_ns).astype(np.int64)
    return pd.Series(pd.to_datetime(start.value + offsets))


def timestamp_for_train_file(df, path, event):
    interval = parse_filename_interval(str(path))
    if interval is None:
        return None, None
    start, end = interval
    dt, info = detect_timestamp_from_first4(df, start, end, event)
    if dt is not None:
        return dt, info
    dt = filename_linear_timestamp(len(df), start, end)
    step = (end - start).total_seconds() / max(len(df), 1)
    nearest = float(np.min(np.abs((dt - event).dt.total_seconds()))) if start <= event <= end else 0.0
    return dt, {
        "source": "filename_range_linear",
        "score": 0.0,
        "median_dt": float(step),
        "nearest_event": nearest,
    }


def reduce_train_frame(df, path, event):
    dt, info = timestamp_for_train_file(df, path, event)
    if dt is None:
        return None, None
    out = pd.DataFrame({"__time": pd.to_datetime(dt, errors="coerce")})
    for idx in SELECTED:
        out[f"c{idx}"] = pd.to_numeric(df.iloc[:, idx], errors="coerce") if idx < df.shape[1] else np.nan
    value_cols = [f"c{i}" for i in SELECTED]
    good = out[value_cols].notna().sum(axis=1) >= 3
    out = out.loc[good].copy()
    lo = event - pd.Timedelta(seconds=TRAIN_KEEP_BEFORE)
    hi = event + pd.Timedelta(seconds=TRAIN_KEEP_AFTER)
    out = out[(out["__time"] >= lo) & (out["__time"] <= hi)]
    out = out.dropna(subset=["__time"]).reset_index(drop=True)
    if len(out) < 10:
        return None, info
    return out, info


def reduce_test_frame(df):
    n = len(df)
    if n < 10:
        return None
    step = 1799.0 / max(n - 1, 1)
    out = pd.DataFrame({"__time": np.arange(n, dtype=float) * step})
    for idx in SELECTED:
        out[f"c{idx}"] = pd.to_numeric(df.iloc[:, idx], errors="coerce") if idx < df.shape[1] else np.nan
    value_cols = [f"c{i}" for i in SELECTED]
    good = out[value_cols].notna().sum(axis=1) >= 3
    out = out.loc[good].reset_index(drop=True)
    return out if len(out) >= 10 else None


def validate_event_range(df, well):
    event = pd.Timestamp(EVENT_TIMES[well])
    tmin = df["__time"].min()
    tmax = df["__time"].max()
    nearest = float(np.min(np.abs((df["__time"] - event).dt.total_seconds())))
    before = (event - tmin).total_seconds() / 3600.0
    after = (tmax - event).total_seconds() / 3600.0
    print(f"[check] {well}: {tmin} -> {tmax}, event={event}, before={before:.2f}h, after={after:.2f}h, nearest={nearest:.1f}s")
    if nearest > 120:
        raise RuntimeError(f"{well}: no valid data within 120s of event; nearest={nearest:.1f}s")
    if before < 6.5:
        raise RuntimeError(f"{well}: only {before:.2f}h pre-event data; need >= 6.5h")


def load_well_dir(well_dir: Path, well: str):
    files = list(iter_table_files(well_dir))
    event = pd.Timestamp(EVENT_TIMES[well])
    keep_lo = event - pd.Timedelta(seconds=TRAIN_KEEP_BEFORE)
    keep_hi = event + pd.Timedelta(seconds=TRAIN_KEEP_AFTER)
    print(f"[load] {well_dir}")
    print(f"       table files={len(files)}")
    pieces = []
    sources = []
    failed = 0
    skipped = 0

    for i, path in enumerate(files):
        interval = parse_filename_interval(str(path))
        if interval is not None:
            fs, fe = interval
            if fe < keep_lo or fs > keep_hi:
                skipped += 1
                continue
        try:
            raw = read_table_file(path)
            small, info = reduce_train_frame(raw, path, event)
            if small is not None:
                pieces.append(small)
                sources.append((path.name, info, len(small)))
            else:
                failed += 1
        except Exception as e:
            failed += 1
            print(f"       [warn] {path.name}: {type(e).__name__}: {e}")
        if (i + 1) % 20 == 0 or i + 1 == len(files):
            print(f"       scanned={i+1}/{len(files)}, usable={len(pieces)}, skipped={skipped}, failed={failed}")

    if not pieces:
        raise RuntimeError(f"No usable event-neighborhood files for {well}")

    x = pd.concat(pieces, ignore_index=True)
    x["__time"] = pd.to_datetime(x["__time"], errors="coerce")
    x = x.dropna(subset=["__time"]).sort_values("__time").drop_duplicates(subset=["__time"], keep="last").reset_index(drop=True)
    print(f"       retained rows={len(x):,}")
    for name, info, rows in sources:
        if info:
            print(f"       time-source {name}: {info['source']}, rows={rows}, dt={info['median_dt']:.3f}s")
    validate_event_range(x, well)
    return x


def load_well_from_train(train_dir: Path, well: str):
    return load_well_dir(find_well_directory(train_dir, well), well)


def relative_seconds(df):
    t = df["__time"]
    if pd.api.types.is_datetime64_any_dtype(t):
        return (t - t.iloc[-1]).dt.total_seconds().to_numpy(float)
    x = pd.to_numeric(t, errors="coerce").to_numpy(float)
    return x - np.nanmax(x)


def channel_array(df, idx):
    return pd.to_numeric(df[f"c{idx}"], errors="coerce").to_numpy(float)


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
    out.extend([
        finite(np.clip(z[valid[-1]], -20, 20)),
        finite(np.clip(z[valid[-1]] - z[valid[0]], -20, 20)),
        finite(np.clip(slope(rel, z) * 300.0, -20, 20)),
    ])
    for w in WINDOWS:
        m = rel >= -w
        zz = z[m]
        tt = rel[m]
        good = np.isfinite(zz)
        zz, tt = zz[good], tt[good]
        if len(zz) < 3:
            out.extend([0.0] * 5)
            continue
        mid = len(zz) // 2
        half = 0.0 if mid < 1 or mid >= len(zz) else np.nanmean(zz[mid:]) - np.nanmean(zz[:mid])
        out.extend([
            finite(np.nanmean(zz)),
            finite(np.nanstd(zz)),
            finite(np.nanpercentile(zz, 90) - np.nanpercentile(zz, 10)),
            finite(np.clip(half, -20, 20)),
            finite(np.clip(slope(tt, zz) * 300.0, -20, 20)),
        ])
    out.extend([
        signed_log(med),
        signed_log(x[valid[-1]]),
        signed_log(x[valid[-1]] - x[valid[0]]),
    ])
    return out


def physical_channels(df):
    def a(i):
        return channel_array(df, i)
    flow_out2 = a(99)
    effective_out = flow_out2 if np.isfinite(flow_out2).mean() > 0.4 else a(73)
    gas_sum = np.zeros(len(df), dtype=float)
    for idx in [111, 112, 113, 114, 115, 116, 117]:
        gas_sum += np.nan_to_num(a(idx))
    return [
        effective_out - a(72),
        a(88), a(87), a(67) - a(66),
        a(118), gas_sum, a(125), a(62),
        a(18), a(98), a(19), a(74),
    ]


def infer_regime(df):
    pump = channel_array(df, 62)
    rpm = channel_array(df, 12)
    bit = channel_array(df, 5)
    hook = channel_array(df, 10)
    pump90 = finite(np.nanpercentile(np.abs(pump), 90))
    rpm90 = finite(np.nanpercentile(np.abs(rpm), 90))
    bit_range = finite(np.nanpercentile(bit, 90) - np.nanpercentile(bit, 10))
    hook_std = finite(np.nanstd(hook))
    if pump90 > 1e-6 and rpm90 > 1e-6:
        return 0
    if pump90 > 1e-6:
        return 1
    if abs(bit_range) > 1e-5 or hook_std > 1e-5:
        return 2
    return 3


def endpoint_vector(df, where):
    rel = relative_seconds(df)
    if where == "start":
        mask = rel <= np.nanmin(rel) + 60
    else:
        mask = rel >= -60
    vals = []
    for idx in GRAPH_CHANNELS:
        x = channel_array(df, idx)[mask]
        vals.append(signed_log(np.nanmedian(x)) if np.isfinite(x).sum() else np.nan)
    return np.asarray(vals, dtype=np.float32)


def anchor_score(df):
    rel = relative_seconds(df)
    recent = rel >= -300
    earlier = (rel >= -1800) & (rel < -600)
    def change(x):
        x = np.asarray(x, dtype=float)
        if np.isfinite(x[recent]).sum() < 3 or np.isfinite(x[earlier]).sum() < 3:
            return 0.0
        base = np.nanmedian(x[earlier])
        scale = np.nanpercentile(x[earlier], 75) - np.nanpercentile(x[earlier], 25)
        if not np.isfinite(scale) or abs(scale) < 1e-9:
            scale = np.nanstd(x[earlier])
        if not np.isfinite(scale) or scale < 1e-9:
            scale = 1.0
        return finite((np.nanmedian(x[recent]) - base) / scale)
    flow = channel_array(df, 99) - channel_array(df, 72)
    score = (
        0.30 * max(change(flow), 0) +
        0.20 * max(change(channel_array(df, 88)), 0) +
        0.15 * max(change(channel_array(df, 87)), 0) +
        0.20 * max(change(channel_array(df, 118)), 0) +
        0.15 * max(change(channel_array(df, 98)), 0)
    )
    return float(np.clip(score, 0, 20))


def extract_features(df):
    rel = relative_seconds(df)
    duration = np.nanmax(rel) - np.nanmin(rel)
    if duration < 300:
        raise ValueError(f"Duration too short: {duration}")
    feats, missing = [], []
    for idx in SELECTED:
        x = channel_array(df, idx)
        missing.append(np.mean(~np.isfinite(x)))
        feats.extend(one_channel_features(rel, x))
    for x in physical_channels(df):
        feats.extend(one_channel_features(rel, x))
    regime = infer_regime(df)
    feats.extend([1.0 if regime == r else 0.0 for r in range(4)])
    return (
        np.asarray(feats, dtype=np.float32),
        endpoint_vector(df, "start"),
        endpoint_vector(df, "end"),
        np.asarray(missing, dtype=np.float32),
        int(regime),
        anchor_score(df),
    )


def slice_by_end_time(df, end_time, history=1800):
    times = df["__time"].astype("int64").to_numpy()
    end_ns = pd.Timestamp(end_time).value
    start_ns = end_ns - int(history * 1e9)
    lo = np.searchsorted(times, start_ns, side="left")
    hi = np.searchsorted(times, end_ns, side="right")
    if hi - lo < 50:
        return None
    seg = df.iloc[lo:hi].copy()
    duration = (seg["__time"].iloc[-1] - seg["__time"].iloc[0]).total_seconds()
    return seg if duration >= 600 else None


def build_well_samples(df, well):
    event = pd.Timestamp(EVENT_TIMES[well])
    tmin, tmax = df["__time"].min(), df["__time"].max()
    X, y, regimes = [], [], []
    configs = [
        (1, np.arange(30, POSITIVE_HORIZON + 1, POSITIVE_STRIDE)),
        (0, np.arange(AMBIGUOUS_GAP, NEGATIVE_LOOKBACK + 1, NEGATIVE_STRIDE)),
    ]
    for label, deltas in configs:
        for delta in deltas:
            end_time = event - pd.Timedelta(seconds=int(delta))
            if end_time <= tmin or end_time > tmax:
                continue
            seg = slice_by_end_time(df, end_time)
            if seg is None:
                continue
            try:
                f, _, _, _, regime, _ = extract_features(seg)
            except Exception:
                continue
            X.append(f)
            y.append(label)
            regimes.append(regime)
    if not X:
        raise RuntimeError(f"No samples for {well}")
    X = np.vstack(X).astype(np.float32)
    y = np.asarray(y, dtype=np.int8)
    regimes = np.asarray(regimes, dtype=np.int8)
    print(f"[samples] {well}: n={len(y)}, pos={int(y.sum())}, neg={int((y==0).sum())}")
    if y.sum() < 5 or (y == 0).sum() < 5:
        raise RuntimeError(f"Too few class samples for {well}")
    return X, y, regimes


def build_train_features(train_dir, work_dir, force=False):
    cache = work_dir / f"train_features_{CACHE_TAG}.npz"
    if cache.exists() and not force:
        print(f"[cache] {cache}")
        z = np.load(cache, allow_pickle=True)
        return z["X"], z["y"], z["groups"], z["regimes"]
    Xs, ys, gs, rs = [], [], [], []
    for gi in range(1, 11):
        well = f"WELL_{gi:06d}"
        reduced = work_dir / f"{well}_reduced_{CACHE_TAG}.pkl"
        if reduced.exists() and not force:
            print(f"[cache] {reduced}")
            df = pd.read_pickle(reduced)
            validate_event_range(df, well)
        else:
            df = load_well_from_train(train_dir, well)
            df.to_pickle(reduced)
        X, y, regimes = build_well_samples(df, well)
        Xs.append(X); ys.append(y); rs.append(regimes)
        gs.append(np.full(len(y), gi - 1, dtype=np.int8))
    X = np.vstack(Xs).astype(np.float32)
    y = np.concatenate(ys)
    groups = np.concatenate(gs)
    regimes = np.concatenate(rs)
    np.savez_compressed(cache, X=X, y=y, groups=groups, regimes=regimes)
    print(f"[train] X={X.shape}, positive={int(y.sum())}, negative={int((y==0).sum())}")
    return X, y, groups, regimes


def test_id_from_path(path):
    m = re.search(r"(test_\d+)", str(path), re.I)
    if not m:
        raise ValueError(f"Cannot parse test id from {path}")
    return m.group(1).lower()


def numeric_test_id(s):
    m = re.search(r"(\d+)$", s)
    return int(m.group(1)) if m else 10**9


def build_test_features(test_dir, work_dir, force=False):
    cache = work_dir / f"test_features_{CACHE_TAG}.npz"
    if cache.exists() and not force:
        print(f"[cache] {cache}")
        z = np.load(cache, allow_pickle=True)
        return z["X"], z["ids"].astype(str), z["starts"], z["ends"], z["missing"], z["regimes"], z["anchors"]
    files = [p for p in iter_table_files(test_dir) if re.search(r"test_\d+", str(p), re.I)]
    if len(files) != 324:
        raise RuntimeError(f"Expected 324 test files, found {len(files)}")
    files.sort(key=lambda p: numeric_test_id(test_id_from_path(p)))
    rows = []
    for i, path in enumerate(files):
        sid = test_id_from_path(path)
        raw = read_table_file(path)
        df = reduce_test_frame(raw)
        if df is None:
            raise RuntimeError(f"Cannot parse test slice {path}")
        f, start, end, missing, regime, a = extract_features(df)
        rows.append((sid, f, start, end, missing, regime, a))
        if (i + 1) % 25 == 0 or i + 1 == len(files):
            print(f"[test] {i+1}/{len(files)}")
    ids = np.asarray([r[0] for r in rows], dtype=object)
    expected = {f"test_{i:03d}" for i in range(1, 325)}
    if set(ids) != expected:
        raise RuntimeError(f"Bad test IDs; missing={sorted(expected-set(ids))}, extra={sorted(set(ids)-expected)}")
    X = np.vstack([r[1] for r in rows]).astype(np.float32)
    starts = np.vstack([r[2] for r in rows]).astype(np.float32)
    ends = np.vstack([r[3] for r in rows]).astype(np.float32)
    missing = np.vstack([r[4] for r in rows]).astype(np.float32)
    regimes = np.asarray([r[5] for r in rows], dtype=np.int8)
    anchors = np.asarray([r[6] for r in rows], dtype=np.float32)
    np.savez_compressed(cache, X=X, ids=ids, starts=starts, ends=ends, missing=missing, regimes=regimes, anchors=anchors)
    print(f"[test] X={X.shape}")
    return X, ids.astype(str), starts, ends, missing, regimes, anchors


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


def fit_ensemble(X, y, pos_weight=1.6, seed=3407):
    Xf, med = fill_matrix(X)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xf)
    lr = LogisticRegression(C=0.25, max_iter=2000, class_weight={0: 1.0, 1: pos_weight}, random_state=seed)
    lr.fit(Xs, y)
    et = ExtraTreesClassifier(n_estimators=600, min_samples_leaf=2, max_features="sqrt", class_weight={0: 1.0, 1: pos_weight}, n_jobs=-1, random_state=seed)
    et.fit(Xf, y)
    sw = np.where(y == 1, pos_weight, 1.0)
    hgb = HistGradientBoostingClassifier(learning_rate=0.04, max_iter=320, max_leaf_nodes=31, min_samples_leaf=12, l2_regularization=1.5, random_state=seed)
    hgb.fit(Xf, y, sample_weight=sw)
    pos, neg = Xs[y == 1], Xs[y == 0]
    cpos, cneg = np.median(pos, axis=0), np.median(neg, axis=0)
    delta = np.mean((Xs - cneg) ** 2, axis=1) - np.mean((Xs - cpos) ** 2, axis=1)
    temp = np.nanstd(delta)
    if not np.isfinite(temp) or temp < 1e-6:
        temp = 1.0
    return {"med": med, "scaler": scaler, "lr": lr, "et": et, "hgb": hgb, "cpos": cpos, "cneg": cneg, "temp": temp}


def predict_ensemble(model, X):
    Xf, _ = fill_matrix(X, model["med"])
    Xs = model["scaler"].transform(Xf)
    p_lr = model["lr"].predict_proba(Xs)[:, 1]
    p_et = model["et"].predict_proba(Xf)[:, 1]
    p_hgb = model["hgb"].predict_proba(Xf)[:, 1]
    dpos = np.mean((Xs - model["cpos"]) ** 2, axis=1)
    dneg = np.mean((Xs - model["cneg"]) ** 2, axis=1)
    p_proto = sigmoid((dneg - dpos) / model["temp"])
    return np.clip(0.50*p_et + 0.25*p_hgb + 0.15*p_lr + 0.10*p_proto, 0, 1)


def competition_metric(y, pred):
    y = np.asarray(y, dtype=int); pred = np.asarray(pred, dtype=int)
    tp = int(np.sum((y == 1) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {"score": 100.0 * recall * recall * specificity, "recall": recall, "specificity": specificity, "tp": tp, "fn": fn, "fp": fp, "tn": tn}


def search_threshold(y, p):
    thresholds = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 700)))
    best = None
    for t in thresholds:
        pred = (p >= t).astype(int)
        m = competition_metric(y, pred)
        item = {**m, "threshold": float(t), "positive_rate": float(pred.mean())}
        if best is None or item["score"] > best["score"]:
            best = item
    return best


def loso_oof(X, y, groups, pos_weight):
    oof = np.zeros(len(y), dtype=float)
    per_well = []
    for g in np.unique(groups):
        tr, va = groups != g, groups == g
        print(f"[LOSO] WELL_{int(g)+1:06d}: train={tr.sum()} val={va.sum()}")
        model = fit_ensemble(X[tr], y[tr], pos_weight=pos_weight, seed=3407 + int(g))
        p = predict_ensemble(model, X[va])
        oof[va] = p
        try:
            auc = roc_auc_score(y[va], p)
        except Exception:
            auc = float("nan")
        local = search_threshold(y[va], p)
        per_well.append({"well": int(g)+1, "auc": finite(auc, -1), **local})
        print(f"       AUC={auc:.5f} Score={local['score']:.5f} Recall={local['recall']:.5f} Spec={local['specificity']:.5f}")
    overall = search_threshold(y, oof)
    print(f"[OOF] Score={overall['score']:.6f} Recall={overall['recall']:.6f} Specificity={overall['specificity']:.6f} Threshold={overall['threshold']:.8f} PositiveRate={overall['positive_rate']:.6f}")
    return oof, overall, per_well


def robust_matrix(Z):
    Z = np.asarray(Z, dtype=float)
    med = np.nanmedian(Z, axis=0)
    q25 = np.nanpercentile(Z, 25, axis=0)
    q75 = np.nanpercentile(Z, 75, axis=0)
    scale = q75 - q25
    med[~np.isfinite(med)] = 0.0
    scale[(~np.isfinite(scale)) | (scale < 1e-6)] = 1.0
    X = Z.copy()
    for j in range(X.shape[1]):
        bad = ~np.isfinite(X[:, j])
        X[bad, j] = med[j]
    return (X - med) / scale


def infer_hidden_domains(starts, ends, missing, n_domains):
    stable = (np.nan_to_num(starts) + np.nan_to_num(ends)) / 2.0
    Z = robust_matrix(np.concatenate([stable, 3.0 * missing], axis=1))
    km = KMeans(n_clusters=n_domains, n_init=50, random_state=3407)
    domains = km.fit_predict(Z)
    print(f"[domains] counts={np.bincount(domains).tolist()}")
    return domains


def pairwise_sqdist(A, B):
    aa = np.sum(A*A, axis=1, keepdims=True)
    bb = np.sum(B*B, axis=1, keepdims=True).T
    return np.maximum(aa + bb - 2.0 * A @ B.T, 0.0)


def build_temporal_graph(starts, ends, domains):
    n = len(starts)
    both = robust_matrix(np.vstack([starts, ends]))
    S, E = both[:n], both[n:]
    edges = []
    for domain in np.unique(domains):
        idx = np.where(domains == domain)[0]
        if len(idx) < 3:
            continue
        D = pairwise_sqdist(E[idx], S[idx])
        np.fill_diagonal(D, np.inf)
        succ = np.argmin(D, axis=1)
        pred = np.argmin(D, axis=0)
        sd = D[np.arange(len(idx)), succ]
        fd = sd[np.isfinite(sd)]
        if len(fd) < 2:
            continue
        limit = np.quantile(fd, 0.65)
        for i in range(len(idx)):
            j = succ[i]
            d = sd[i]
            if pred[j] != i or not np.isfinite(d) or d > limit:
                continue
            conf = math.exp(-d / max(limit, 1e-6))
            edges.append({"src": int(idx[i]), "dst": int(idx[j]), "domain": int(domain), "distance": float(d), "confidence": float(conf)})
    print(f"[graph] strong edges={len(edges)}")
    return edges


def graph_propagation(base_prob, anchors, edges):
    p = 0.96 * np.asarray(base_prob, dtype=float) + 0.04 * rank01(anchors)
    for _ in range(3):
        old = p.copy()
        for e in edges:
            s, d, c = e["src"], e["dst"], e["confidence"]
            if old[d] > old[s]:
                p[s] += 0.16 * c * (old[d] - old[s])
            if old[s] > old[d]:
                p[d] += 0.035 * c * (old[s] - old[d])
        p = np.clip(p, 0, 1)
    return p


def topk_prediction(p, k):
    k = int(np.clip(k, 1, len(p)-1))
    pred = np.zeros(len(p), dtype=np.int8)
    pred[np.argsort(-p)[:k]] = 1
    return pred


def write_zip(ids, pred, path):
    df = pd.DataFrame({"切片ID": ids, "溢流判断": np.asarray(pred, dtype=int)})
    if len(df) != 324 or df["切片ID"].nunique() != 324:
        raise RuntimeError("Invalid submission row/id count")
    if not set(df["溢流判断"].unique()).issubset({0, 1}):
        raise RuntimeError("Prediction must be binary")
    data = df.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")
    path = Path(path)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("result.csv", data)
    print(f"[output] {path} positive={int(df['溢流判断'].sum())}")


def scan_only(train_dir, work_dir, force):
    print("Running event-aware train validation only")
    for gi in range(1, 11):
        well = f"WELL_{gi:06d}"
        cache = work_dir / f"{well}_reduced_{CACHE_TAG}.pkl"
        if cache.exists() and not force:
            df = pd.read_pickle(cache)
            validate_event_range(df, well)
        else:
            df = load_well_from_train(train_dir, well)
            df.to_pickle(cache)
    print("All 10 wells passed event-time validation.")


def run(args):
    train_dir = Path(args.train_dir).resolve()
    test_dir = Path(args.test_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    output = Path(args.output).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"[path] train={train_dir}")
    print(f"[path] test ={test_dir}")
    print(f"[path] work ={work_dir}")
    if args.scan_only:
        scan_only(train_dir, work_dir, args.force_rebuild)
        return

    X, y, groups, train_regimes = build_train_features(train_dir, work_dir, args.force_rebuild)
    XT, ids, starts, ends, missing, test_regimes, anchors = build_test_features(test_dir, work_dir, args.force_rebuild)
    if X.shape[1] != XT.shape[1]:
        raise RuntimeError(f"Feature dimension mismatch: {X.shape} vs {XT.shape}")

    oof, best, per_well = loso_oof(X, y, groups, args.pos_weight)
    pd.DataFrame({"well": groups+1, "label": y, "regime": train_regimes, "oof_prob": oof, "oof_pred": (oof >= best['threshold']).astype(int)}).to_csv(work_dir / "V7_oof.csv", index=False)

    print("[final] fitting all wells")
    model = fit_ensemble(X, y, pos_weight=args.pos_weight, seed=7777)
    p_base = predict_ensemble(model, XT)
    domains = infer_hidden_domains(starts, ends, missing, args.hidden_domains)
    if args.no_graph:
        edges = []
        p_graph = p_base.copy()
    else:
        edges = build_temporal_graph(starts, ends, domains)
        p_graph = graph_propagation(p_base, anchors, edges)
    p_final = 0.94 * p_graph + 0.06 * rank01(p_graph)

    oof_rate = float(np.clip(best["positive_rate"], args.min_positive_rate, args.max_positive_rate))
    expected_count = round(len(ids) * oof_rate * args.count_scale)
    threshold_pred = (p_final >= best["threshold"]).astype(np.int8)
    threshold_count = int(threshold_pred.sum())
    main_count = round(0.65 * expected_count + 0.35 * threshold_count)
    main_count = int(np.clip(main_count, round(len(ids)*args.min_positive_rate), round(len(ids)*args.max_positive_rate)))
    precision_count = max(1, round(main_count * 0.92))
    recall_count = min(len(ids)-1, round(main_count * 1.08))

    main_pred = topk_prediction(p_final, main_count)
    precision_pred = topk_prediction(p_final, precision_count)
    recall_pred = topk_prediction(p_final, recall_count)

    write_zip(ids, main_pred, output)
    precision_path = output.with_name(output.stem + "_precision.zip")
    recall_path = output.with_name(output.stem + "_recall.zip")
    threshold_path = output.with_name(output.stem + "_threshold.zip")
    write_zip(ids, precision_pred, precision_path)
    write_zip(ids, recall_pred, recall_path)
    write_zip(ids, threshold_pred, threshold_path)

    debug = pd.DataFrame({
        "切片ID": ids,
        "base_prob": p_base,
        "graph_prob": p_graph,
        "final_prob": p_final,
        "physical_anchor": anchors,
        "domain": domains,
        "regime": test_regimes,
        "main_pred": main_pred,
    })
    debug["rank"] = debug["final_prob"].rank(method="first", ascending=False).astype(int)
    debug.sort_values("rank").to_csv(work_dir / "V7_test_debug.csv", index=False)

    edge_rows = [{"src_id": ids[e['src']], "dst_id": ids[e['dst']], "domain": e['domain'], "distance": e['distance'], "confidence": e['confidence']} for e in edges]
    pd.DataFrame(edge_rows).to_csv(work_dir / "V7_graph_edges.csv", index=False)

    summary = {
        "cache_tag": CACHE_TAG,
        "train_shape": list(X.shape),
        "test_shape": list(XT.shape),
        "oof": best,
        "per_well": per_well,
        "hidden_domains": int(args.hidden_domains),
        "graph_edges": int(len(edges)),
        "expected_count": int(expected_count),
        "threshold_count": int(threshold_count),
        "main_count": int(main_count),
        "precision_count": int(precision_count),
        "recall_count": int(recall_count),
    }
    with open(work_dir / "V7_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("="*70)
    print("Temporal-Graph V7 EventFix completed")
    print(f"OOF Score       : {best['score']:.6f}")
    print(f"OOF Recall      : {best['recall']:.6f}")
    print(f"OOF Specificity : {best['specificity']:.6f}")
    print(f"OOF threshold   : {best['threshold']:.8f}")
    print(f"OOF positiveRate: {best['positive_rate']:.6f}")
    print(f"Graph edges     : {len(edges)}")
    print(f"Main count      : {main_count}")
    print(f"MAIN      : {output}")
    print(f"PRECISION : {precision_path}")
    print(f"RECALL    : {recall_path}")
    print(f"THRESHOLD : {threshold_path}")
    print("="*70)


def build_parser():
    ap = argparse.ArgumentParser(description="KickGuard Temporal-Graph V7 EventFix")
    ap.add_argument("--train-dir", default="../train")
    ap.add_argument("--test-dir", default="../test")
    ap.add_argument("--work-dir", default="./work_v7")
    ap.add_argument("--output", default="./result_v7.zip")
    ap.add_argument("--force-rebuild", action="store_true")
    ap.add_argument("--scan-only", action="store_true")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--hidden-domains", type=int, default=5)
    ap.add_argument("--pos-weight", type=float, default=1.60)
    ap.add_argument("--count-scale", type=float, default=1.00)
    ap.add_argument("--min-positive-rate", type=float, default=0.28)
    ap.add_argument("--max-positive-rate", type=float, default=0.48)
    return ap


if __name__ == "__main__":
    args = build_parser().parse_args()
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
