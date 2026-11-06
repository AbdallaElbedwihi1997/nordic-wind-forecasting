"""Day-ahead information cutoff rules and leakage check.

Delivery day D is the Danish local (Europe/Copenhagen) calendar day, matching
how Energinet issues the vendor day-ahead forecast. All timestamps are tz-naive
UTC; a delivery day is represented as a tz-naive midnight Timestamp of the local
date (see ``delivery_day`` / ``delivery_hours``). ``issue_time`` and
``last_known_hour`` take delivery days, not hours.

"As-of" means the time a value became known:
- Actuals / ForecastCurrent for hour h become known at h + 1h (see ``known_at``).
- Vendor ForecastDayAhead values for day D carry as-of = ``issue_time(D)``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FORBIDDEN_COLUMNS = {"ForecastIntraday", "Forecast5Hour", "Forecast1Hour"}

_ISSUE_HOUR = 10
_LAST_HOUR = 9
_TZ = "Europe/Copenhagen"


class LeakageError(Exception):
    """Raised when a feature uses information unavailable at issue time."""


def _day(x):
    """Validate that ``x`` holds delivery days (exact midnights) and return it."""
    if isinstance(x, pd.Timestamp):
        if x != x.normalize():
            raise ValueError(f"expected a delivery day (midnight), got {x}; map hours with delivery_day")
        return x
    x = pd.DatetimeIndex(x)
    bad = x.notna() & (x != x.normalize())
    if bad.any():
        raise ValueError(
            f"expected delivery days (midnights), got e.g. {x[bad][0]}; map hours with delivery_day"
        )
    return x


def delivery_day(hours):
    """Delivery day (Europe/Copenhagen local date) of tz-naive UTC hour(s).

    Returns a tz-naive midnight Timestamp (for a Timestamp) or DatetimeIndex.
    """
    if isinstance(hours, pd.Timestamp):
        return hours.tz_localize("UTC").tz_convert(_TZ).normalize().tz_localize(None)
    idx = pd.DatetimeIndex(hours)
    return idx.tz_localize("UTC").tz_convert(_TZ).normalize().tz_localize(None)


def delivery_hours(days) -> pd.DatetimeIndex:
    """All tz-naive UTC hours whose delivery day is in ``days`` (sorted, unique).

    23 hours on the spring-forward day, 25 on the fall-back day, else 24.
    """
    if isinstance(days, (pd.Timestamp, str)):
        days = [days]
    days = pd.DatetimeIndex(days).normalize().unique().sort_values()
    if len(days) == 0:
        return pd.DatetimeIndex([], dtype=days.dtype)
    # Local midnight is never ambiguous / nonexistent in Europe/Copenhagen.
    start = days.tz_localize(_TZ).tz_convert("UTC").tz_localize(None)
    end = (days + pd.Timedelta(days=1)).tz_localize(_TZ).tz_convert("UTC").tz_localize(None)
    n = ((end - start) // pd.Timedelta(hours=1)).to_numpy().astype(int)
    offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    vals = np.repeat(start.to_numpy(), n) + offs * np.timedelta64(1, "h")
    return pd.DatetimeIndex(vals).as_unit(days.unit)


def issue_time(target_day):
    """Forecast issue time for delivery day D: D-1 10:00 UTC (Timestamp or
    DatetimeIndex). Raises ValueError unless every input is exactly midnight;
    map hours to ``delivery_day`` before calling."""
    return _day(target_day) - pd.Timedelta(days=1) + pd.Timedelta(hours=_ISSUE_HOUR)


def last_known_hour(target_day):
    """Last actual hour usable for delivery day D: D-1 09:00 UTC (delivery
    days only; non-midnight inputs raise ValueError)."""
    return _day(target_day) - pd.Timedelta(days=1) + pd.Timedelta(hours=_LAST_HOUR)


def known_at(hour):
    """Time at which the value for hour ``hour`` becomes known: hour + 1h."""
    return hour + pd.Timedelta(hours=1)


def check_features(feature_asof: pd.DataFrame, target_index: pd.DatetimeIndex) -> None:
    """Raise LeakageError if any feature leaks information past the issue time."""
    target_index = pd.DatetimeIndex(target_index)
    if not feature_asof.index.equals(target_index):
        raise ValueError("feature_asof index must equal target_index")

    bad_names = [
        c
        for c in feature_asof.columns
        if any(f in str(c) for f in FORBIDDEN_COLUMNS)
        or (str(c).startswith("cur_") and "_lag" not in str(c))
    ]
    if bad_names:
        raise LeakageError(f"Forbidden feature column(s): {bad_names}")

    limit = issue_time(delivery_day(target_index))
    bad_cols = []
    first_bad = None
    for c in feature_asof.columns:
        asof = pd.to_datetime(feature_asof[c])
        viol = (asof.to_numpy() > limit.to_numpy()) & asof.notna().to_numpy()
        if viol.any():
            bad_cols.append(c)
            h = target_index[viol.argmax()]
            first_bad = h if first_bad is None else min(first_bad, h)
    if bad_cols:
        raise LeakageError(
            f"Feature column(s) {bad_cols} have as-of after issue time "
            f"(delivery day D-1 10:00 UTC); first offending target hour: {first_bad}"
        )
