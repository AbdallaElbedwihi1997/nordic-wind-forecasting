import numpy as np
import pandas as pd
import pytest

from windfc import QUANTILES
from windfc.models import (
    CQR,
    LightGBMQR,
    LinearQR,
    Persistence,
    VendorEmpirical,
    finalize,
    recency_weights,
)

QCOLS = [f"q{int(round(q * 100))}" for q in QUANTILES]


def _data(n_days=150, seed=0):
    idx = pd.date_range("2025-01-01", periods=24 * n_days, freq="h")
    rng = np.random.default_rng(seed)
    n = len(idx)
    base = np.clip(1500 + 1200 * np.sin(np.arange(n) / 30.0) + rng.normal(0, 100, n), 0, None)
    y = pd.Series(np.clip(base + rng.normal(0, 200, n), 0, None), index=idx)
    X = pd.DataFrame(
        {
            "da_total": base,
            "da_total_adj": base * 1.02,
            "hour": idx.hour,
            "last_actual": np.roll(y.to_numpy(), 24),
        },
        index=idx,
    )
    return X, y


def _check(p, X):
    assert list(p.columns) == QCOLS
    assert p.index.equals(X.index)
    v = p.to_numpy()
    assert np.isfinite(v).all()
    assert (v >= 0).all()
    assert (np.diff(v, axis=1) >= 0).all()


MODELS = [
    lambda: Persistence(),
    lambda: VendorEmpirical("da_total"),
    lambda: VendorEmpirical("da_total_adj", window_days=60, n_bins=3),
    lambda: LinearQR(),
    lambda: LightGBMQR(n_estimators=50),
    lambda: CQR(LightGBMQR(n_estimators=50), calib_days=30),
    lambda: CQR(LinearQR, calib_days=30),
]


@pytest.mark.parametrize("make", MODELS)
def test_predictions_non_crossing_and_non_negative(make):
    X, y = _data()
    tr = X.index < "2025-05-01"
    m = make().fit(X[tr], y[tr], None)
    p = m.predict(X[~tr])
    _check(p, X[~tr])
    # fit with weights too
    w = recency_weights(X.index[tr], 30.0)
    _check(make().fit(X[tr], y[tr], sample_weight=w).predict(X[~tr]), X[~tr])


def test_finalize_sorts_and_clips():
    p = pd.DataFrame([[5, 4, 3, 2, 1, 0, -1, -2, -3]], columns=QCOLS, dtype=float)
    out = finalize(p)
    assert out.iloc[0].tolist() == [0, 0, 0, 0, 1, 2, 3, 4, 5]


def test_vendor_empirical_median_tracks_vendor():
    X, _ = _data()
    y = pd.Series(X["da_total"].to_numpy(), index=X.index)
    m = VendorEmpirical("da_total").fit(X, y)
    p = m.predict(X.iloc[-500:])
    np.testing.assert_allclose(p["q50"], X["da_total"].iloc[-500:])
    np.testing.assert_allclose(p["q10"], X["da_total"].iloc[-500:])


def test_vendor_empirical_uses_bins_and_window():
    idx = pd.date_range("2025-01-01", periods=24 * 200, freq="h")
    rng = np.random.default_rng(1)
    base = rng.uniform(0, 1000, len(idx))
    # errors: +100 for low base, -100 for high base, in the last 90 days only
    y = base + np.where(base < 500, 100.0, -100.0)
    y[idx < idx.max() - pd.Timedelta(days=90)] = base[idx < idx.max() - pd.Timedelta(days=90)] + 5000
    X = pd.DataFrame({"da_total": base}, index=idx)
    m = VendorEmpirical("da_total", n_bins=2).fit(X, pd.Series(y, index=idx))
    Xt = pd.DataFrame({"da_total": [100.0, 900.0]}, index=idx[:2])
    p = m.predict(Xt)
    assert p["q50"].tolist() == pytest.approx([200.0, 800.0])


def test_persistence_offsets():
    X, y = _data()
    m = Persistence().fit(X, y)
    p = m.predict(X.iloc[-10:])
    last90 = X.index > X.index.max() - pd.Timedelta(days=90)
    r = (y - X["last_actual"])[last90]
    exp = X["last_actual"].iloc[-10:] + np.quantile(r, 0.5)
    np.testing.assert_allclose(p["q50"], np.clip(exp, 0, None))


def test_linear_qr_learns_median_slope():
    rng = np.random.default_rng(2)
    idx = pd.date_range("2025-01-01", periods=3000, freq="h")
    x = rng.uniform(0, 1000, len(idx))
    y = pd.Series(2 * x + 100 + rng.normal(0, 10, len(idx)), index=idx)
    X = pd.DataFrame({"x": x}, index=idx)
    p = LinearQR().fit(X, y).predict(pd.DataFrame({"x": [500.0]}, index=idx[:1]))
    assert p["q50"].iloc[0] == pytest.approx(1100, abs=5)


def test_linear_qr_auto_stride():
    assert LinearQR.auto_stride(5000, 6000) == 1
    assert LinearQR.auto_stride(28923, 6000) == 5
    assert LinearQR.auto_stride(35000, 6000) == 7  # 6 shares a factor with 24
    assert LinearQR.auto_stride(13000, 6000) == 5  # 3, 4 skipped
    X, y = _data(60)
    m = LinearQR(max_rows=500).fit(X, y)
    assert m.stride_ == LinearQR.auto_stride(len(X), 500)
    assert LinearQR(stride=1).fit(X.iloc[:200], y.iloc[:200]).stride_ == 1


def test_lightgbm_passes_sample_weight(monkeypatch):
    import lightgbm

    seen = []
    orig = lightgbm.LGBMRegressor.fit

    def spy(self, X, y, sample_weight=None, **kw):
        seen.append(sample_weight)
        return orig(self, X, y, sample_weight=sample_weight, **kw)

    monkeypatch.setattr(lightgbm.LGBMRegressor, "fit", spy)
    X, y = _data(30)
    w = recency_weights(X.index, 7.0)
    LightGBMQR(n_estimators=5).fit(X, y, sample_weight=w)
    assert len(seen) == len(QUANTILES)
    assert all(s is not None and len(s) == len(X) for s in seen)


def test_numpy_weights_equal_series_weights():
    X, y = _data(40)
    w = recency_weights(X.index, 7.0)
    a = LinearQR(max_rows=400).fit(X, y, sample_weight=w).predict(X.iloc[-24:])
    b = LinearQR(max_rows=400).fit(X, y, sample_weight=w.to_numpy()).predict(X.iloc[-24:])
    pd.testing.assert_frame_equal(a, b)
    c = CQR(LinearQR(max_rows=400), calib_days=10).fit(X, y, sample_weight=w.to_numpy())
    d = CQR(LinearQR(max_rows=400), calib_days=10).fit(X, y, sample_weight=w)
    pd.testing.assert_frame_equal(c.predict(X.iloc[-24:]), d.predict(X.iloc[-24:]))


class _Narrow:
    """Quantiles of a much-too-narrow N(0, 10) when the truth is N(0, 100)."""

    def fit(self, X, y, sample_weight=None):
        self.n_fit = len(X)
        return self

    def predict(self, X):
        from scipy.stats import norm

        cols = {c: 1000 + 10 * norm.ppf(q) for c, q in zip(QCOLS, QUANTILES)}
        return pd.DataFrame(cols, index=X.index)


def _coverage(p, y):
    return float(((y >= p["q10"]) & (y <= p["q90"])).mean())


def test_cqr_widens_undercovering_intervals():
    rng = np.random.default_rng(5)
    idx = pd.date_range("2025-01-01", periods=24 * 200, freq="h")
    y = pd.Series(1000 + rng.normal(0, 100, len(idx)), index=idx)
    X = pd.DataFrame({"f": 0.0}, index=idx)
    tr = idx < "2025-06-01"
    raw = _Narrow().fit(X[tr], y[tr]).predict(X[~tr])
    c = CQR(_Narrow, calib_days=60).fit(X[tr], y[tr])
    p = c.predict(X[~tr])
    cov_raw, cov_cqr = _coverage(raw, y[~tr]), _coverage(p, y[~tr])
    assert cov_raw < 0.2
    assert abs(cov_cqr - 0.8) < abs(cov_raw - 0.8)
    assert abs(cov_cqr - 0.8) < 0.05
    np.testing.assert_allclose(p["q50"], raw["q50"])
    _check(p, X[~tr])
    # wrapped model trained on proper-train only
    n_calib = int((X[tr].index >= X[tr].index.max().normalize() - pd.Timedelta(days=59)).sum())
    assert c.model_.n_fit == tr.sum() - n_calib


def test_cqr_does_not_mutate_template():
    X, y = _data(60)
    tmpl = LightGBMQR(n_estimators=5)
    CQR(tmpl, calib_days=10).fit(X, y)
    assert not hasattr(tmpl, "models_")


def test_recency_weights():
    idx = pd.date_range("2025-01-01", periods=24 * 21, freq="h")
    w = recency_weights(idx, None)
    assert (w == 1.0).all() and w.index.equals(idx)
    w = recency_weights(idx, 10.0)
    assert w.iloc[-1] == pytest.approx(1.0)
    t = idx.max() - pd.Timedelta(days=10)
    assert w[t] == pytest.approx(0.5)
