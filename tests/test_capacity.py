from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from windfc.capacity import apply_capacity_correction, capacity_ratio
from windfc.cutoff import delivery_day

PARQUET = Path(__file__).resolve().parents[1] / "data" / "dk1_hourly.parquet"


def _frame(n_days=60, start="2026-01-01", cur=500.0, actual=500.0, da=400.0):
    idx = pd.date_range(start, periods=24 * n_days, freq="h", name="hour_utc")
    return pd.DataFrame(
        {
            "actual_off": actual,
            "cur_off": cur,
            "da_off": da,
            "da_on": 200.0,
            "actual_on": 100.0,
            "actual_total": 600.0,
            "da_total": 600.0,
            "cur_on": 100.0,
        },
        index=idx,
    )


def _days(*ds):
    return pd.DatetimeIndex([pd.Timestamp(d) for d in ds])


def test_ratio_uses_only_past_window():
    df = _frame()
    D = pd.Timestamp("2026-02-15")
    df.loc[df.index >= D, "actual_off"] = 1000.0  # doubles from D 00:00
    r = capacity_ratio(df, _days(D))
    assert r.name == "capacity_ratio"
    assert r.loc[D] == pytest.approx(1.0)


def test_ratio_window_edges():
    df = _frame()
    D = pd.Timestamp("2026-02-15")
    # D-1 09:00 is included, D-1 10:00 excluded
    df.loc[D - pd.Timedelta(hours=15), "actual_off"] = 1500.0
    df.loc[D - pd.Timedelta(hours=14), "actual_off"] = 9e6
    n = 24 * 30
    expected = (500.0 * (n - 1) + 1500.0) / (500.0 * n)
    assert capacity_ratio(df, _days(D)).loc[D] == pytest.approx(expected)


def test_ratio_threshold_excludes_low_hours():
    df = _frame()
    D = pd.Timestamp("2026-02-15")
    low = df.index[100:150]
    df.loc[low, "cur_off"] = 100.0  # == threshold -> excluded
    df.loc[low, "actual_off"] = 1e6
    df.loc[df.index[200:210], "cur_off"] = 50.0
    df.loc[df.index[200:210], "actual_off"] = 1e6
    assert capacity_ratio(df, _days(D)).loc[D] == pytest.approx(1.0)


def test_ratio_nan_hours_excluded():
    df = _frame()
    D = pd.Timestamp("2026-02-15")
    df.loc[df.index[500:510], "actual_off"] = np.nan
    assert capacity_ratio(df, _days(D)).loc[D] == pytest.approx(1.0)


def test_ratio_fallback_when_no_qualifying_hours():
    df = _frame(actual=600.0)
    # starts with qualifying hours (ratio 1.2) then everything drops below threshold
    df.loc[df.index >= "2026-01-20", "cur_off"] = 50.0
    days = _days("2026-01-01", "2026-01-10", "2026-02-25", "2026-02-28")
    r = capacity_ratio(df, days)
    assert r.loc["2026-01-01"] == 1.0  # nothing known yet -> 1.0
    assert r.loc["2026-01-10"] == pytest.approx(1.2)
    # window for 2026-02-25 has no qualifying hours -> last valid ratio
    assert r.loc["2026-02-25"] == pytest.approx(1.2)
    assert r.loc["2026-02-28"] == pytest.approx(1.2)
    assert np.isfinite(r).all()


def test_ratio_all_low_is_one():
    df = _frame(cur=50.0)
    r = capacity_ratio(df, _days("2026-02-01", "2026-02-10"))
    assert (r == 1.0).all()


def test_ratio_unsorted_targets_fallback_by_day_order():
    df = _frame(actual=600.0)
    df.loc[df.index >= "2026-01-20", "cur_off"] = 50.0
    r = capacity_ratio(df, _days("2026-02-25", "2026-01-10"))
    assert r.loc["2026-02-25"] == pytest.approx(1.2)


def test_correction_scales_offshore_only():
    df = _frame(n_days=3)
    df.loc[df.index[5], "da_off"] = np.nan
    ratios = pd.Series(
        [1.5, 2.0], index=_days("2026-01-01", "2026-01-02"), name="capacity_ratio"
    )
    out = apply_capacity_correction(df, ratios)
    assert "da_off_adj" not in df.columns  # input untouched
    dd = delivery_day(out.index)  # e.g. UTC 2026-01-01 23:00 -> delivery 2026-01-02
    d1 = dd == pd.Timestamp("2026-01-01")
    d2 = dd == pd.Timestamp("2026-01-02")
    d3 = dd == pd.Timestamp("2026-01-03")
    assert out.loc["2026-01-01 23:00", "da_off_adj"] == 800.0
    assert (out.loc[d2, "da_off_adj"] == 800.0).all()
    assert out.loc[d1, "da_off_adj"].iloc[0] == 600.0
    assert np.isnan(out["da_off_adj"].iloc[5]) and np.isnan(out["da_total_adj"].iloc[5])
    assert out.loc[d3, ["da_off_adj", "da_total_adj"]].isna().all().all()
    assert (out.loc[d2, "da_total_adj"] == 1000.0).all()
    pd.testing.assert_series_equal(out["da_on"], df["da_on"])


@pytest.mark.skipif(not PARQUET.exists(), reason="real dataset missing")
def test_real_data_ratio_sanity():
    df = pd.read_parquet(PARQUET)
    r = capacity_ratio(df, _days("2022-06-15", "2026-09-15"))
    assert 0.85 <= r.loc["2022-06-15"] <= 1.15
    assert r.loc["2026-09-15"] > 1.3


def test_fallback_independent_of_requested_days():
    df = _frame(actual=600.0)
    df.loc[df.index >= "2026-01-20", "cur_off"] = 50.0
    D = pd.Timestamp("2026-02-25")  # calm window: no qualifying hours
    alone = capacity_ratio(df, _days(D)).loc[D]
    together = capacity_ratio(df, _days("2026-01-10", D)).loc[D]
    assert alone == pytest.approx(together)
    assert alone == pytest.approx(1.2) and alone != 1.0


def test_non_monotonic_index_rejected():
    df = _frame(n_days=3).iloc[::-1]
    with pytest.raises(ValueError):
        capacity_ratio(df, _days("2026-01-02"))
