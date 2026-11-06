"""Leakage-checked feature construction for day-ahead forecasting.

Every feature column comes with an "as-of" timestamp (when the information it
uses became known); ``build_features`` runs ``check_features`` before returning.

As-of conventions:
- vendor day-ahead derived columns: ``issue_time(D)``;
- calendar columns: NaT (known in advance);
- error statistics and ``last_actual``: ``known_at(latest hour actually used)``
  (NaT when no valid hour was used, in which case the value is NaN too).

Error statistics use the capacity-corrected vendor error
``e(h) = da_total_adj(h) - actual_total(h)``, each historical hour corrected
with its own delivery day's capacity ratio, restricted to hours in
``(last_known_hour(D) - N days, last_known_hour(D)]``.

"Day" always means delivery day (Danish local date, ``cutoff.delivery_day``);
rows for day D are ``cutoff.delivery_hours(D)`` (23/24/25 tz-naive UTC hours).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from windfc.capacity import apply_capacity_correction, capacity_ratio
from windfc.cutoff import (
    check_features,
    delivery_day,
    delivery_hours,
    issue_time,
    known_at,
    last_known_hour,
)

BASE_FEATURES = [
    "da_on",
    "da_off_adj",
    "da_total_adj",
    "ramp_1h",
    "ramp_3h",
    "hour",
    "doy_sin",
    "doy_cos",
    "err_mean_7d",
    "err_mae_7d",
]
CANDIDATE_FEATURES = [
    "err_mean_14d",
    "err_mae_14d",
    "err_mean_30d",
    "err_mae_30d",
    "err_by_hour_30d",
    "level_x_hour",
    "off_share",
]
# Raw vendor columns (for VendorEmpirical baseline 2) and persistence input.
EXTRA_FEATURES = ["da_total", "da_off", "last_actual"]

_VENDOR = {"da_on", "da_off", "da_total", "da_off_adj", "da_total_adj",
           "ramp_1h", "ramp_3h", "off_share", "level_x_hour"}
_CALENDAR = {"hour", "doy_sin", "doy_cos"}
_ERR = {f"err_{k}_{n}d": (k, n) for k in ("mean", "mae") for n in (7, 14, 30)}
_KNOWN = set(BASE_FEATURES) | set(CANDIDATE_FEATURES) | set(EXTRA_FEATURES)

_H = pd.Timedelta(hours=1)


def _window_stats(values: np.ndarray, times: pd.DatetimeIndex, hi_time, lo_time):
    """Mean of finite ``values`` over ``times`` in (lo_time, hi_time] and the
    latest valid time used. ``times`` must be sorted. Vectorized over the
    query arrays ``hi_time`` / ``lo_time``."""
    ok = np.isfinite(values)
    zero = np.zeros(1)
    cs_v = np.concatenate([zero, np.cumsum(np.where(ok, values, 0.0))])
    cs_n = np.concatenate([zero, np.cumsum(ok.astype(np.int64))])
    # position of latest valid element at or before each position
    pos = np.where(ok, np.arange(len(values)), -1)
    last_valid = np.concatenate([[-1], np.maximum.accumulate(pos)]) if len(pos) else np.array([-1])
    hi = times.searchsorted(hi_time, side="right")
    lo = times.searchsorted(lo_time, side="right")
    n = cs_n[hi] - cs_n[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, (cs_v[hi] - cs_v[lo]) / np.maximum(n, 1), np.nan)
    lv = last_valid[hi]
    used = (n > 0) & (lv >= 0)
    tv = times.to_numpy()
    if len(tv):
        t_used = pd.DatetimeIndex(np.where(used, tv[np.maximum(lv, 0)], np.datetime64("NaT", "ns").astype(tv.dtype)))
    else:
        t_used = pd.DatetimeIndex(np.full(len(hi), np.datetime64("NaT", "ns")))
    asof = known_at(t_used)
    return mean, asof


def build_features(df: pd.DataFrame, target_days, feature_set: list[str]):
    """Features and as-of stamps for all hours of each target delivery day.

    Returns ``(X, asof)`` with identical index (tz-naive UTC hours) and columns.
    ``level_x_hour`` expands to ``level_x_hour_sin`` / ``level_x_hour_cos``.
    NaNs are kept.
    """
    unknown = [f for f in feature_set if f not in _KNOWN]
    if unknown:
        raise ValueError(f"Unknown feature(s): {unknown}")
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError("df index must be sorted and unique")

    days = pd.DatetimeIndex(target_days).normalize().unique().sort_values()
    idx = delivery_hours(days).as_unit(df.index.unit)
    idx.name = df.index.name

    # Capacity ratio for every delivery day of the history (so historical
    # errors use their own delivery day's ratio) plus the target days.
    all_days = delivery_day(df.index).unique().union(days)
    ratios = capacity_ratio(df, all_days)
    hist = apply_capacity_correction(df, ratios)
    # Vendor columns at target hours (reindex: target hours outside df -> NaN)
    full_idx = hist.index.union(idx).union(idx - 3 * _H)
    vend = hist[["da_on", "da_off", "da_total", "da_off_adj", "da_total_adj"]]
    vend = vend.reindex(full_idx)

    dday = delivery_day(idx)
    issue = issue_time(dday)
    nat = pd.DatetimeIndex([pd.NaT] * len(idx), dtype=issue.dtype)
    hours = idx.hour.to_numpy()
    lk = last_known_hour(dday)

    e = (hist["da_total_adj"] - hist["actual_total"]).to_numpy(dtype=float)
    etimes = hist.index

    cols, asofs = {}, {}

    def put(name, values, asof):
        cols[name] = np.asarray(values, dtype=float)
        asofs[name] = pd.DatetimeIndex(asof)

    for f in feature_set:
        if f in ("da_on", "da_off", "da_total", "da_off_adj", "da_total_adj"):
            put(f, vend[f].reindex(idx).to_numpy(), issue)
        elif f in ("ramp_1h", "ramp_3h"):
            k = 1 if f == "ramp_1h" else 3
            tot = vend["da_total_adj"]
            put(f, tot.reindex(idx).to_numpy() - tot.reindex(idx - k * _H).to_numpy(), issue)
        elif f == "off_share":
            off = vend["da_off_adj"].reindex(idx).to_numpy()
            tot = vend["da_total_adj"].reindex(idx).to_numpy()
            with np.errstate(invalid="ignore", divide="ignore"):
                put(f, np.where(tot > 0, off / tot, np.nan), issue)
        elif f == "level_x_hour":
            tot = vend["da_total_adj"].reindex(idx).to_numpy()
            ang = 2 * np.pi * hours / 24
            put("level_x_hour_sin", tot * np.sin(ang), issue)
            put("level_x_hour_cos", tot * np.cos(ang), issue)
        elif f == "hour":
            put(f, hours, nat)
        elif f in ("doy_sin", "doy_cos"):
            ang = 2 * np.pi * idx.dayofyear.to_numpy() / 365.25
            put(f, np.sin(ang) if f == "doy_sin" else np.cos(ang), nat)
        elif f in _ERR:
            kind, n = _ERR[f]
            vals = e if kind == "mean" else np.abs(e)
            m, a = _window_stats(vals, etimes, lk, lk - pd.Timedelta(days=n))
            put(f, m, a)
        elif f == "err_by_hour_30d":
            out = np.full(len(idx), np.nan)
            out_asof = nat.to_numpy().copy()
            for hh in range(24):
                tm = hours == hh
                if not tm.any():
                    continue
                sel = etimes.hour == hh
                m, a = _window_stats(e[sel], etimes[sel], lk[tm], lk[tm] - pd.Timedelta(days=30))
                out[tm] = m
                out_asof[tm] = a.to_numpy()
            put(f, out, out_asof)
        elif f == "last_actual":
            act = hist["actual_total"].reindex(lk).to_numpy(dtype=float)
            put(f, act, known_at(lk).where(np.isfinite(act)))

    X = pd.DataFrame(cols, index=idx)
    asof = pd.DataFrame({c: asofs[c] for c in X.columns}, index=idx)
    check_features(asof, X.index)
    return X, asof


def complete_rows(X: pd.DataFrame, y: pd.Series):
    """Drop rows with NaN in ``y`` or any column of ``X`` (y aligned to X)."""
    y = y.reindex(X.index)
    ok = y.notna().to_numpy() & X.notna().all(axis=1).to_numpy()
    return X[ok], y[ok]
