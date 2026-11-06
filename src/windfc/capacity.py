"""Offshore capacity correction (step 1).

The vendor day-ahead offshore forecast is scaled by a trailing ratio of actual
to vendor-"current" offshore output, using only hours known at issue time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from windfc.cutoff import last_known_hour


def capacity_ratio(
    df: pd.DataFrame,
    target_days,
    window_days: int = 30,
    min_cur_off: float = 100.0,
) -> pd.Series:
    """Trailing offshore capacity ratio per target day.

    For day D the window is the hours in (last_known_hour(D) - window_days,
    last_known_hour(D)]. Only hours with ``cur_off > min_cur_off`` and both
    ``actual_off`` and ``cur_off`` non-NaN count. ratio = sum(actual_off) /
    sum(cur_off). With no qualifying hours, the most recent valid ratio of an
    earlier day (from the full daily history, independent of the requested days) is used, else 1.0. The result is always finite.
    """
    if not df.index.is_monotonic_increasing:
        raise ValueError("df index must be monotonic increasing")
    days = pd.DatetimeIndex(target_days).normalize()
    # Full daily sequence so the fallback never depends on which days were requested.
    start = min(df.index[0].normalize(), days.min())
    uniq = pd.date_range(start, days.max(), freq="D")

    cur = df["cur_off"].to_numpy(dtype=float)
    act = df["actual_off"].to_numpy(dtype=float)
    q = np.isfinite(cur) & np.isfinite(act) & (cur > min_cur_off)
    zero = np.zeros(1)
    cs_num = np.concatenate([zero, np.cumsum(np.where(q, act, 0.0))])
    cs_den = np.concatenate([zero, np.cumsum(np.where(q, cur, 0.0))])
    cs_n = np.concatenate([zero, np.cumsum(q.astype(np.int64))])

    lk = last_known_hour(uniq)
    hi = df.index.searchsorted(lk, side="right")
    lo = df.index.searchsorted(lk - pd.Timedelta(days=window_days), side="right")
    n = cs_n[hi] - cs_n[lo]
    num = cs_num[hi] - cs_num[lo]
    den = cs_den[hi] - cs_den[lo]

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((n > 0) & (den > 0), num / den, np.nan)
    s = pd.Series(ratio, index=uniq).ffill().fillna(1.0)
    s = s.where(np.isfinite(s), 1.0)
    out = s.reindex(days)
    out.index.name = None
    out.name = "capacity_ratio"
    return out


def apply_capacity_correction(df: pd.DataFrame, ratios: pd.Series) -> pd.DataFrame:
    """Return a copy of ``df`` with ``da_off_adj`` and ``da_total_adj`` added.

    ``da_off_adj = ratio(day of hour) * da_off``; ``da_total_adj = da_on +
    da_off_adj`` (NaN if either part is NaN). Hours whose day has no ratio get
    NaN. ``da_on`` is unchanged.
    """
    out = df.copy()
    r = ratios.copy()
    r.index = pd.DatetimeIndex(r.index).normalize()
    r = r[~r.index.duplicated(keep="last")]
    per_hour = r.reindex(df.index.normalize()).to_numpy(dtype=float)
    out["da_off_adj"] = per_hour * out["da_off"].to_numpy(dtype=float)
    out["da_total_adj"] = out["da_on"] + out["da_off_adj"]
    return out
