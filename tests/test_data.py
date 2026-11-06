from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from windfc.data import load_raw

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def df():
    return load_raw(FIX / "settlement.csv", FIX / "forecasts.csv")


def test_target_is_sum_of_four_wind_columns(df):
    row = df.loc["2022-01-01 00:00"]
    assert row.actual_on == pytest.approx(1.5 + 20.25)
    assert row.actual_off == pytest.approx(10.5)  # Lt100MW absent -> min_count=1
    assert row.actual_total == pytest.approx(1.5 + 20.25 + 10.5)
    assert df.loc["2022-01-01 01:00", "actual_total"] == pytest.approx(2 + 30 + 12)


def test_missing_actual_component_group_gives_nan_total(df):
    row = df.loc["2022-01-01 02:00"]
    assert np.isnan(row.actual_off) and np.isnan(row.actual_total)
    assert row.actual_on == pytest.approx(32.0)


def test_only_dk1_and_wind_types_kept(df):
    # DK2 / UNDEFINED / Solar values (99, 77, 88, 5) must not leak in
    assert df.loc["2022-01-01 00:00", "da_on"] == 20.0
    assert df.loc["2022-01-01 00:00", "da_off"] == 10.0
    assert df.loc["2022-01-01 00:00", "da_total"] == 30.0
    assert df.loc["2022-01-01 00:00", "cur_on"] == 24.0
    assert df.loc["2022-01-01 00:00", "cur_off"] == 14.0
    assert len(df) == 3


def test_missing_vendor_hour_is_nan_not_zero(df):
    row = df.loc["2022-01-01 01:00"]  # Onshore forecast row absent
    assert np.isnan(row.da_on) and np.isnan(row.da_total)
    assert row.da_off == 11.0
    assert np.isnan(row.cur_on)
    assert row.cur_off == 15.0


def test_index_is_unique_hourly_utc(df):
    assert df.index.name == "hour_utc"
    assert df.index.is_unique and df.index.is_monotonic_increasing
    assert df.index.tz is None
    assert (df.index.to_series().diff().dropna() == pd.Timedelta("1h")).all()
    assert list(df.columns) == [
        "actual_on", "actual_off", "actual_total",
        "da_on", "da_off", "da_total", "cur_on", "cur_off",
    ]
    assert all(df.dtypes == "float64")
