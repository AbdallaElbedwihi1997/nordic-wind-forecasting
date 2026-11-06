"""Day-ahead information cutoff rules and leakage check.

"As-of" means the time a value became known:
- Actuals / ForecastCurrent for hour h become known at h + 1h (see ``known_at``).
- Vendor ForecastDayAhead values for day D carry as-of = ``issue_time(D)``.
"""
from __future__ import annotations

import pandas as pd

FORBIDDEN_COLUMNS = {"ForecastIntraday", "Forecast5Hour", "Forecast1Hour"}

_ISSUE_HOUR = 10
_LAST_HOUR = 9


class LeakageError(Exception):
    """Raised when a feature uses information unavailable at issue time."""


def _day(x):
    return x.normalize()


def issue_time(target_day):
    """Forecast issue time for day D: D-1 10:00 (Timestamp or DatetimeIndex)."""
    return _day(target_day) - pd.Timedelta(days=1) + pd.Timedelta(hours=_ISSUE_HOUR)


def last_known_hour(target_day):
    """Last actual hour usable for day D: D-1 09:00."""
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

    limit = issue_time(target_index)
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
            f"(D-1 10:00); first offending target hour: {first_bad}"
        )
