import pandas as pd
import pytest

from windfc.cutoff import (
    LeakageError,
    check_features,
    delivery_day,
    delivery_hours,
    issue_time,
    known_at,
    last_known_hour,
)

D = pd.Timestamp("2026-03-10")  # delivery day (CET, UTC+1): UTC 2026-03-09 23:00 .. 03-10 22:00
IDX = pd.DatetimeIndex([D + pd.Timedelta(hours=h) for h in (-1, 12, 22)])


def _asof(ts):
    return pd.DataFrame({"x_lag24": [ts] * len(IDX)}, index=IDX)


def test_issue_time():
    assert issue_time(D) == pd.Timestamp("2026-03-09 10:00")
    assert issue_time(pd.Timestamp("2026-03-10 17:45")) == pd.Timestamp("2026-03-09 10:00")
    out = issue_time(delivery_day(IDX))
    assert isinstance(out, pd.DatetimeIndex)
    assert (out == pd.Timestamp("2026-03-09 10:00")).all()


def test_last_known_hour_lag_range():
    lk = last_known_hour(D)
    assert lk == pd.Timestamp("2026-03-09 09:00")
    hrs = delivery_hours([D])
    assert (hrs[0] - lk) / pd.Timedelta(hours=1) == 14
    assert (hrs[-1] - lk) / pd.Timedelta(hours=1) == 37
    assert isinstance(last_known_hour(delivery_day(IDX)), pd.DatetimeIndex)


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


def test_delivery_day_winter_and_summer():
    T = pd.Timestamp
    assert delivery_day(T("2026-01-15 23:00")) == T("2026-01-16")
    assert delivery_day(T("2026-01-15 22:00")) == T("2026-01-15")
    assert delivery_day(T("2026-07-15 22:00")) == T("2026-07-16")
    assert delivery_day(T("2026-07-15 21:00")) == T("2026-07-15")
    idx = pd.DatetimeIndex(["2026-01-15 22:00", "2026-01-15 23:00", "2026-07-15 21:00", "2026-07-15 22:00"])
    out = delivery_day(idx)
    assert isinstance(out, pd.DatetimeIndex) and out.tz is None
    assert list(out) == [T("2026-01-15"), T("2026-01-16"), T("2026-07-15"), T("2026-07-16")]


def test_delivery_hours_dst():
    spring = delivery_hours([pd.Timestamp("2026-03-29")])
    fall = delivery_hours([pd.Timestamp("2026-10-25")])
    winter = delivery_hours([pd.Timestamp("2026-01-16")])
    assert len(spring) == 23 and len(fall) == 25 and len(winter) == 24
    assert winter[0] == pd.Timestamp("2026-01-15 23:00")
    assert winter[-1] == pd.Timestamp("2026-01-16 22:00")
    assert winter.tz is None and winter.is_monotonic_increasing
    assert (winter.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()
    assert (delivery_day(fall) == pd.Timestamp("2026-10-25")).all()
    both = delivery_hours(pd.DatetimeIndex(["2026-01-17", "2026-01-16"]))
    assert len(both) == 48 and both.is_monotonic_increasing and both.is_unique


def test_check_features_uses_delivery_day():
    t = pd.DatetimeIndex(["2026-01-15 23:00"])  # delivery 2026-01-16
    check_features(pd.DataFrame({"f": [pd.Timestamp("2026-01-15 10:00")]}, index=t), t)
    with pytest.raises(LeakageError):
        check_features(pd.DataFrame({"f": [pd.Timestamp("2026-01-15 11:00")]}, index=t), t)
    t2 = pd.DatetimeIndex(["2026-01-15 22:00"])  # delivery 2026-01-15
    with pytest.raises(LeakageError):
        check_features(pd.DataFrame({"f": [pd.Timestamp("2026-01-14 11:00")]}, index=t2), t2)
