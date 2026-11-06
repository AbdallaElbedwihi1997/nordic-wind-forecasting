from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from windfc.capacity import apply_capacity_correction, capacity_ratio
from windfc.cutoff import issue_time, last_known_hour
from windfc.evaluate import FOLDS
from windfc.features import (
    BASE_FEATURES,
    CANDIDATE_FEATURES,
    build_features,
    complete_rows,
)

PARQUET = Path(__file__).resolve().parents[1] / "data" / "dk1_hourly.parquet"
ALL = BASE_FEATURES + CANDIDATE_FEATURES + ["last_actual"]


def _frame(n_days=80, start="2026-01-01", seed=0):
    idx = pd.date_range(start, periods=24 * n_days, freq="h", name="hour_utc")
    rng = np.random.default_rng(seed)
    n = len(idx)
    da_on = 1000 + 500 * np.sin(np.arange(n) / 7.0) + rng.normal(0, 50, n)
    da_off = 600 + 300 * np.cos(np.arange(n) / 11.0)
    return pd.DataFrame(
        {
            "actual_on": da_on,
            "actual_off": da_off,
            "actual_total": da_on + da_off,
            "da_on": da_on,
            "da_off": da_off,
            "da_total": da_on + da_off,
            "cur_on": da_on,
            "cur_off": da_off,  # ratio == 1 -> da_total_adj == da_total
        },
        index=idx,
    )


def _day_range(a, b):
    return pd.date_range(a, b, freq="D")


def test_exports():
    assert BASE_FEATURES == [
        "da_on", "da_off_adj", "da_total_adj", "ramp_1h", "ramp_3h",
        "hour", "doy_sin", "doy_cos", "err_mean_7d", "err_mae_7d",
    ]
    assert CANDIDATE_FEATURES == [
        "err_mean_14d", "err_mae_14d", "err_mean_30d", "err_mae_30d",
        "err_by_hour_30d", "level_x_hour", "off_share",
    ]


def test_shape_and_expansion():
    df = _frame()
    days = _day_range("2026-03-01", "2026-03-03")
    X, asof = build_features(df, days, ["da_on", "level_x_hour", "hour"])
    assert list(X.columns) == ["da_on", "level_x_hour_sin", "level_x_hour_cos", "hour"]
    assert len(X) == 72
    assert X.index.equals(pd.date_range("2026-03-01", periods=72, freq="h"))
    assert asof.index.equals(X.index) and list(asof.columns) == list(X.columns)
    assert asof["hour"].isna().all()
    assert (asof["da_on"] == issue_time(X.index)).all()
    assert X["hour"].tolist()[:24] == list(range(24))
    h = X.index.hour.to_numpy()
    np.testing.assert_allclose(
        X["level_x_hour_sin"], df.loc[X.index, "da_total"] * np.sin(2 * np.pi * h / 24)
    )


def test_unknown_feature_rejected():
    with pytest.raises(ValueError):
        build_features(_frame(), _day_range("2026-03-01", "2026-03-01"), ["cur_off"])


def test_ramps_within_day():
    df = _frame()
    days = _day_range("2026-03-01", "2026-03-02")
    X, _ = build_features(df, days, ["da_total_adj", "ramp_1h", "ramp_3h"])
    full = apply_capacity_correction(
        df, capacity_ratio(df, df.index.normalize().unique())
    )["da_total_adj"]
    for t in X.index:
        assert X.loc[t, "ramp_1h"] == pytest.approx(
            full[t] - full[t - pd.Timedelta(hours=1)]
        )
        assert X.loc[t, "ramp_3h"] == pytest.approx(
            full[t] - full[t - pd.Timedelta(hours=3)]
        )
    # within-day equality on the returned columns
    np.testing.assert_allclose(
        X["ramp_1h"].iloc[1:24].to_numpy(), np.diff(X["da_total_adj"].iloc[:24].to_numpy())
    )


def test_calendar_and_off_share():
    df = _frame()
    df.loc["2026-03-01 05:00", ["da_on", "da_off"]] = 0.0
    X, _ = build_features(df, _day_range("2026-03-01", "2026-03-01"),
                          ["doy_sin", "doy_cos", "off_share"])
    doy = 60
    assert X["doy_sin"].iloc[0] == pytest.approx(np.sin(2 * np.pi * doy / 365.25))
    assert X["doy_cos"].iloc[0] == pytest.approx(np.cos(2 * np.pi * doy / 365.25))
    assert np.isnan(X.loc["2026-03-01 05:00", "off_share"])
    t = pd.Timestamp("2026-03-01 06:00")
    assert X.loc[t, "off_share"] == pytest.approx(df.loc[t, "da_off"] / df.loc[t, "da_total"])


def test_err_mean_7d_uses_only_known_hours():
    df = _frame()
    D = pd.Timestamp("2026-03-10")
    start_err = issue_time(D)  # D-1 10:00
    df.loc[df.index >= start_err, "actual_total"] -= 500.0  # e = +500 from D-1 10:00
    X, asof = build_features(df, _day_range(D, D), ["err_mean_7d", "err_mae_7d"])
    np.testing.assert_allclose(X["err_mean_7d"], 0.0, atol=1e-9)
    np.testing.assert_allclose(X["err_mae_7d"], 0.0, atol=1e-9)
    assert (asof["err_mean_7d"] == issue_time(D)).all()
    # the next day sees the error
    X2, _ = build_features(df, _day_range(D + pd.Timedelta(days=1), D + pd.Timedelta(days=1)),
                           ["err_mean_7d"])
    assert X2["err_mean_7d"].iloc[0] == pytest.approx(500.0 * 24 / (24 * 7))


def test_planted_error_after_cutoff_does_not_affect_error_stats():
    df = _frame()
    D = pd.Timestamp("2026-03-10")
    feats = ["err_mean_7d", "err_mae_7d", "err_mean_30d", "err_by_hour_30d", "last_actual"]
    X0, _ = build_features(df, _day_range(D, D), feats)
    df.loc[issue_time(D), "actual_total"] = -1e9  # D-1 10:00
    X1, _ = build_features(df, _day_range(D, D), feats)
    pd.testing.assert_frame_equal(X0, X1)


def test_err_values_and_by_hour():
    df = _frame()
    D = pd.Timestamp("2026-03-10")
    L = last_known_hour(D)
    rng = np.random.default_rng(3)
    df["actual_total"] = df["da_total"] - rng.normal(0, 100, len(df))
    X, asof = build_features(df, _day_range(D, D),
                             ["err_mean_14d", "err_mae_30d", "err_by_hour_30d", "last_actual"])
    e = df["da_total"] - df["actual_total"]
    w14 = e[(e.index > L - pd.Timedelta(days=14)) & (e.index <= L)]
    w30 = e[(e.index > L - pd.Timedelta(days=30)) & (e.index <= L)]
    assert len(w14) == 14 * 24
    assert X["err_mean_14d"].iloc[0] == pytest.approx(w14.mean())
    assert X["err_mae_30d"].iloc[0] == pytest.approx(w30.abs().mean())
    for hh in (0, 9, 10, 23):
        sub = w30[w30.index.hour == hh]
        assert len(sub) == 30
        t = D + pd.Timedelta(hours=hh)
        assert X.loc[t, "err_by_hour_30d"] == pytest.approx(sub.mean())
        assert asof.loc[t, "err_by_hour_30d"] == sub.index.max() + pd.Timedelta(hours=1)
    assert (X["last_actual"] == df.loc[L, "actual_total"]).all()
    assert (asof["last_actual"] == L + pd.Timedelta(hours=1)).all()


def test_error_stats_use_corrected_vendor_with_own_day_ratio():
    df = _frame()
    df["cur_off"] = df["actual_off"] / 2.0  # ratio 2 everywhere once established
    df["actual_total"] = df["da_on"] + 2.0 * df["da_off"]  # perfect corrected forecast
    D = pd.Timestamp("2026-03-10")
    X, _ = build_features(df, _day_range(D, D), ["err_mean_7d", "da_off_adj"])
    np.testing.assert_allclose(X["err_mean_7d"], 0.0, atol=1e-6)
    np.testing.assert_allclose(X["da_off_adj"], 2.0 * df.loc[X.index, "da_off"])


def test_nan_actuals_skipped_and_kept():
    df = _frame()
    D = pd.Timestamp("2026-03-10")
    L = last_known_hour(D)
    df.loc[(df.index > L - pd.Timedelta(days=7)) & (df.index <= L), "actual_total"] = np.nan
    X, asof = build_features(df, _day_range(D, D), ["err_mean_7d", "err_mean_14d"])
    assert X["err_mean_7d"].isna().all()
    assert asof["err_mean_7d"].isna().all()
    assert X["err_mean_14d"].notna().all()
    assert (asof["err_mean_14d"] <= issue_time(X.index)).all()


def test_future_perturbation_invariance_real_data():
    df = pd.read_parquet(PARQUET)
    D = pd.Timestamp("2025-06-15")
    X0, _ = build_features(df, _day_range(D, D), ALL)
    bad = df.copy()
    after = bad.index > last_known_hour(D)
    for c in ["actual_on", "actual_off", "actual_total", "cur_on", "cur_off"]:
        bad.loc[after, c] = 12345.0
    later_days = bad.index >= D + pd.Timedelta(days=1)
    for c in ["da_on", "da_off", "da_total"]:
        bad.loc[later_days, c] = 777.0
    X1, _ = build_features(bad, _day_range(D, D), ALL)
    pd.testing.assert_frame_equal(X0, X1)


def test_build_features_passes_leakage_check():
    df = pd.read_parquet(PARQUET)
    a, b = FOLDS[0]
    days = _day_range(a, b)
    X, asof = build_features(df, days, ALL)
    assert len(X) == 24 * len(days)
    lim = issue_time(X.index)
    for c in asof.columns:
        s = asof[c]
        assert ((s <= lim) | s.isna()).all(), c
    assert X[BASE_FEATURES].notna().mean().min() > 0.95


def test_complete_rows():
    idx = pd.date_range("2026-01-01", periods=4, freq="h")
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    y = pd.Series([1.0, 2.0, np.nan, 4.0], index=idx)
    X2, y2 = complete_rows(X, y)
    assert list(X2.index) == [idx[0], idx[3]]
    assert y2.index.equals(X2.index)
