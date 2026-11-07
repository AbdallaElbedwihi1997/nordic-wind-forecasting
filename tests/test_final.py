"""score_test on synthetic data only (never the real parquet)."""
import numpy as np
import pandas as pd
import pytest

from windfc import final
from windfc.cutoff import delivery_day, delivery_hours
from windfc.evaluate import QCOLS, TEST_PERIOD, pinball, score
from windfc.final import score_test
from windfc.run import default_paths

TEST_START, TEST_END = (pd.Timestamp(d) for d in TEST_PERIOD)


def _frame():
    idx = pd.date_range("2021-12-31 23:00", "2026-11-05 22:00", freq="h", name="hour_utc")
    n = len(idx)
    rng = np.random.default_rng(1)
    t = np.arange(n)
    da_on = np.clip(1500 + 900 * np.sin(t / 37.0) + rng.normal(0, 80, n), 0, None)
    da_off = np.clip(700 + 400 * np.cos(t / 53.0) + rng.normal(0, 40, n), 0, None)
    act_on = np.clip(da_on + rng.normal(0, 150, n), 0, None)
    act_off = np.clip(da_off + rng.normal(0, 80, n), 0, None)
    df = pd.DataFrame({
        "actual_on": act_on, "actual_off": act_off, "actual_total": act_on + act_off,
        "da_on": da_on, "da_off": da_off, "da_total": da_on + da_off,
        "cur_on": da_on, "cur_off": da_off,
    }, index=idx)
    # actuals end at 2026-10-28 23:00 UTC like the real data
    df.loc[df.index > pd.Timestamp("2026-10-28 23:00"), ["actual_on", "actual_off", "actual_total"]] = np.nan
    # a 5-hour actual gap inside the test period (after 09:00 UTC, so last_actual of the next day is valid)
    gap = df.index.get_indexer([pd.Timestamp("2026-06-10 12:00")])[0]
    df.iloc[gap:gap + 5, df.columns.get_loc("actual_total")] = np.nan
    return df


@pytest.fixture(scope="module")
def frame():
    return _frame()


@pytest.fixture(scope="module")
def data_path(frame, tmp_path_factory):
    p = tmp_path_factory.mktemp("final") / "synthetic.parquet"
    frame.to_parquet(p)
    return p


def _cfg(model="vendor_raw", feature_set=("da_total",), params=None, hl=None, cqr=None):
    return dict(run_id=f"t_{model}", hypothesis="h", change="c", model=model,
                params=params or {}, feature_set=list(feature_set),
                half_life_days=hl, cqr_calib_days=cqr)


LGBM = dict(model="lgbm", params=dict(n_estimators=15, num_leaves=8),
            feature_set=["da_on", "da_off_adj", "da_total_adj", "hour", "err_mean_7d"],
            hl=365, cqr=30)


@pytest.fixture(scope="module")
def vendor_res(data_path):
    return score_test(_cfg(), data_path=data_path, check_freeze=False)


def test_scores_only_test_hours_with_common_mask(vendor_res):
    p = vendor_res["predictions"]
    dd = delivery_day(p.index)
    assert dd.min() == TEST_START and dd.max() == TEST_END
    n_test = len(delivery_hours(pd.date_range(TEST_START, TEST_END)))
    assert vendor_res["n_hours"] == len(p) == n_test - 5
    assert vendor_res["n_dropped"] == 5
    assert list(p.columns) == ["actual_total", *QCOLS]


def test_scores_are_frozen_score_and_pinball(vendor_res):
    p = vendor_res["predictions"]
    s = score(p["actual_total"], p)
    for k in ("crps", "mae_p50", "coverage_80", "bias", "crps_norm"):
        assert vendor_res["scores"][k] == pytest.approx(s[k])
    pl = pinball(p["actual_total"], p)
    pd.testing.assert_series_equal(vendor_res["pinball"], pl)
    assert pl.mean() == pytest.approx(vendor_res["scores"]["crps"])


def test_training_stops_before_test_period(data_path, monkeypatch):
    seen = {}
    real = final.make_model

    def spy(cfg):
        m = real(cfg)
        fit = m.fit

        def rec(X, y, sample_weight=None):
            seen["max_day"] = delivery_day(X.index).max()
            seen["min_day"] = delivery_day(X.index).min()
            seen["w_max"] = float(np.max(sample_weight))
            seen["w_min"] = float(np.min(sample_weight))
            return fit(X, y, sample_weight=sample_weight)

        m.fit = rec
        return m

    monkeypatch.setattr(final, "make_model", spy)
    cfg = _cfg(LGBM["model"], LGBM["feature_set"], LGBM["params"], LGBM["hl"], LGBM["cqr"])
    res = score_test(cfg, data_path=data_path, check_freeze=False)
    assert seen["max_day"] == TEST_START - pd.Timedelta(days=1)
    assert seen["min_day"] <= pd.Timestamp("2022-01-08")  # first days lack error-stat history
    assert seen["w_max"] == pytest.approx(1.0) and seen["w_min"] < 0.1  # recency weights used
    assert np.isfinite(res["scores"]["crps"])
    q = res["predictions"][QCOLS].to_numpy()
    assert (np.diff(q, axis=1) >= 0).all()


def test_predictions_ignore_future_actuals(frame, tmp_path):
    """Changing actuals from the last test day's issue time on cannot move any
    prediction (they are not known at any test issue time, nor used to train)."""
    cfg = _cfg(LGBM["model"], LGBM["feature_set"], LGBM["params"], LGBM["hl"], LGBM["cqr"])
    a = tmp_path / "a.parquet"
    frame.to_parquet(a)
    b = tmp_path / "b.parquet"
    late = frame.index >= pd.Timestamp("2026-10-27 10:00")  # issue time of local day 2026-10-28
    f2 = frame.copy()
    f2.loc[late, "actual_total"] = f2.loc[late, "actual_total"] + 5000.0
    f2.to_parquet(b)
    ra = score_test(cfg, data_path=a, check_freeze=False)
    rb = score_test(cfg, data_path=b, check_freeze=False)
    pd.testing.assert_frame_equal(ra["predictions"][QCOLS], rb["predictions"][QCOLS])


def test_does_not_touch_real_log(data_path):
    log = default_paths()[1]
    before = log.read_bytes()
    score_test(_cfg("vendor_capacity", ["da_total_adj"]), data_path=data_path, check_freeze=False)
    assert log.read_bytes() == before


def test_config_validated(data_path):
    with pytest.raises(ValueError):
        score_test(_cfg(model="nope"), data_path=data_path, check_freeze=False)
