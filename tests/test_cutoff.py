import pandas as pd
import pytest

from windfc.cutoff import (
    LeakageError,
    check_features,
    issue_time,
    known_at,
    last_known_hour,
)

D = pd.Timestamp("2026-03-10")
IDX = pd.DatetimeIndex([D + pd.Timedelta(hours=h) for h in (0, 12, 23)])


def _asof(ts):
    return pd.DataFrame({"x_lag24": [ts] * len(IDX)}, index=IDX)


def test_issue_time():
    assert issue_time(D) == pd.Timestamp("2026-03-09 10:00")
    assert issue_time(pd.Timestamp("2026-03-10 17:45")) == pd.Timestamp("2026-03-09 10:00")
    out = issue_time(IDX)
    assert isinstance(out, pd.DatetimeIndex)
    assert (out == pd.Timestamp("2026-03-09 10:00")).all()


def test_last_known_hour_lag_range():
    lk = last_known_hour(D)
    assert lk == pd.Timestamp("2026-03-09 09:00")
    assert (D - lk) / pd.Timedelta(hours=1) == 15
    assert (D + pd.Timedelta(hours=23) - lk) / pd.Timedelta(hours=1) == 38
    assert isinstance(last_known_hour(IDX), pd.DatetimeIndex)


def test_known_at():
    h = pd.Timestamp("2026-03-09 09:00")
    assert known_at(h) == pd.Timestamp("2026-03-09 10:00")
    assert isinstance(known_at(IDX), pd.DatetimeIndex)


def test_check_features_accepts_actual_hour_0900():
    # actual at 09:00 D-1 is known at 10:00 D-1
    check_features(_asof(known_at(pd.Timestamp("2026-03-09 09:00"))), IDX)


def test_check_features_rejects_actual_hour_1000():
    with pytest.raises(LeakageError) as e:
        check_features(_asof(known_at(pd.Timestamp("2026-03-09 10:00"))), IDX)
    assert "x_lag24" in str(e.value)
    assert str(IDX[0]) in str(e.value)


def test_check_features_accepts_vendor_da_asof_issue_time():
    check_features(_asof(pd.Timestamp("2026-03-09 10:00")), IDX)


def test_check_features_allows_nat():
    check_features(_asof(pd.NaT), IDX)


def test_check_features_per_row_day():
    # row on day D+1 may use as-of up to D 10:00, which is forbidden for day D
    idx = pd.DatetimeIndex(["2026-03-10 05:00", "2026-03-11 05:00"])
    ok = pd.DataFrame({"f": [pd.Timestamp("2026-03-09 10:00"), pd.Timestamp("2026-03-10 10:00")]}, index=idx)
    check_features(ok, idx)
    bad = ok.copy()
    bad.iloc[0, 0] = pd.Timestamp("2026-03-10 10:00")
    with pytest.raises(LeakageError):
        check_features(bad, idx)


@pytest.mark.parametrize("name", ["cur_off", "ForecastIntraday_x", "my_Forecast5Hour", "Forecast1Hour"])
def test_forbidden_column_names_rejected(name):
    df = pd.DataFrame({name: [pd.NaT] * len(IDX)}, index=IDX)
    with pytest.raises(LeakageError) as e:
        check_features(df, IDX)
    assert name in str(e.value)


def test_cur_with_lag_suffix_allowed():
    df = pd.DataFrame({"cur_off_lag24": [pd.NaT] * len(IDX)}, index=IDX)
    check_features(df, IDX)
