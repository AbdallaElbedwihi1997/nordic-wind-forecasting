"""Probabilistic models: baselines, linear QR, LightGBM QR, CQR wrapper.

Protocol: ``fit(X, y, sample_weight=None) -> self``; ``predict(X) -> DataFrame``
with columns ``q10 … q90`` (``windfc.QUANTILES``), sorted per row and clipped
at 0 (``finalize``). Rows whose inputs are NaN predict NaN.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from windfc import QUANTILES

QCOLS = [f"q{int(round(q * 100))}" for q in QUANTILES]
_Q = np.asarray(QUANTILES)


def finalize(preds: pd.DataFrame) -> pd.DataFrame:
    """Sort each row ascending across the quantile columns and clip at 0."""
    v = np.sort(preds[QCOLS].to_numpy(dtype=float), axis=1)  # NaN sorts last
    v = np.clip(v, 0.0, None)
    return pd.DataFrame(v, index=preds.index, columns=QCOLS)


def _last_days(index: pd.DatetimeIndex, days: float) -> np.ndarray:
    """Rows with index > index.max() - days."""
    return np.asarray(index > index.max() - pd.Timedelta(days=days))


def _weights(sample_weight, index) -> np.ndarray | None:
    """Sample weights as an array aligned to ``index`` (Series are reindexed)."""
    if sample_weight is None:
        return None
    if isinstance(sample_weight, pd.Series):
        w = sample_weight.reindex(index).to_numpy(dtype=float)
    else:
        w = np.asarray(sample_weight, dtype=float)
        if len(w) != len(index):
            raise ValueError("sample_weight length must match X")
    if not np.isfinite(w).all():
        raise ValueError("sample_weight has NaN / missing entries for some rows")
    return w


def _quantiles(r: np.ndarray) -> np.ndarray:
    return np.quantile(r, _Q)


class Persistence:
    """last_actual + empirical quantiles of (y - last_actual) over the last 90 days.

    ``sample_weight`` is accepted for protocol compatibility and ignored.
    """

    window_days = 90

    def fit(self, X, y, sample_weight=None):
        r = (y - X["last_actual"])[_last_days(X.index, self.window_days)]
        r = r.to_numpy(dtype=float)
        r = r[np.isfinite(r)]
        if len(r) == 0:
            raise ValueError("no valid residuals in the persistence window")
        self.offsets_ = _quantiles(r)
        return self

    def predict(self, X):
        base = X["last_actual"].to_numpy(dtype=float)[:, None]
        return finalize(pd.DataFrame(base + self.offsets_[None, :], index=X.index, columns=QCOLS))


class VendorEmpirical:
    """Vendor forecast + per-level-bin empirical error quantiles (last window_days).

    Bin edges are quantiles of ``base_col`` within the window (outer bins open);
    an empty bin (tied edges) falls back to the whole-window quantiles.
    ``sample_weight`` is accepted for protocol compatibility and ignored.
    """

    def __init__(self, base_col: str, window_days: int = 90, n_bins: int = 5):
        self.base_col = base_col
        self.window_days = window_days
        self.n_bins = n_bins

    def fit(self, X, y, sample_weight=None):
        w = _last_days(X.index, self.window_days)
        b = X[self.base_col].to_numpy(dtype=float)[w]
        r = y.reindex(X.index).to_numpy(dtype=float)[w] - b
        ok = np.isfinite(b) & np.isfinite(r)
        b, r = b[ok], r[ok]
        if len(r) == 0:
            raise ValueError("no valid rows in the vendor window")
        # interior edges only; outer bins are open
        self.edges_ = np.quantile(b, np.linspace(0, 1, self.n_bins + 1)[1:-1])
        bins = np.searchsorted(self.edges_, b, side="right")
        overall = _quantiles(r)
        self.bin_q_ = np.vstack(
            [_quantiles(r[bins == k]) if (bins == k).any() else overall for k in range(self.n_bins)]
        )
        return self

    def predict(self, X):
        b = X[self.base_col].to_numpy(dtype=float)
        bins = np.searchsorted(self.edges_, np.nan_to_num(b), side="right")
        v = b[:, None] + self.bin_q_[bins]
        return finalize(pd.DataFrame(v, index=X.index, columns=QCOLS))


class LinearQR:
    """Per-quantile sklearn QuantileRegressor (HiGHS) on standardized features.

    HiGHS fit time grows roughly quadratically in rows (~308 s for 9 quantiles
    on 29k rows, ~14 s on 5.8k with near-identical CRPS), so by default it fits
    on a deterministic stride subsample: every k-th complete row, k = the
    smallest integer >= ceil(n / max_rows) coprime with 24 (so the subsample
    cycles through all hours of day). ``stride`` overrides k (1 = all rows).
    """

    def __init__(self, alpha: float = 0.0, stride: int | None = None, max_rows: int = 6000):
        self.alpha = alpha
        self.stride = stride
        self.max_rows = max_rows

    @staticmethod
    def auto_stride(n: int, max_rows: int) -> int:
        k = max(1, -(-n // max_rows))
        while k > 1 and np.gcd(k, 24) != 1:
            k += 1
        return k

    def fit(self, X, y, sample_weight=None):
        from sklearn.linear_model import QuantileRegressor
        from sklearn.preprocessing import StandardScaler

        yv = y.reindex(X.index).to_numpy(dtype=float)
        Xv = X.to_numpy(dtype=float)
        ok = np.isfinite(yv) & np.isfinite(Xv).all(axis=1)
        Xv, yv = Xv[ok], yv[ok]
        sw = _weights(sample_weight, X.index)
        sw = None if sw is None else sw[ok]
        k = self.stride if self.stride is not None else self.auto_stride(len(yv), self.max_rows)
        self.stride_ = k
        if k > 1:
            Xv, yv = Xv[::k], yv[::k]
            sw = None if sw is None else sw[::k]
        self.columns_ = list(X.columns)
        self.scaler_ = StandardScaler().fit(Xv)
        Z = self.scaler_.transform(Xv)
        self.models_ = [
            QuantileRegressor(quantile=q, alpha=self.alpha, solver="highs").fit(Z, yv, sample_weight=sw)
            for q in QUANTILES
        ]
        return self

    def predict(self, X):
        Xv = X[self.columns_].to_numpy(dtype=float)
        ok = np.isfinite(Xv).all(axis=1)
        v = np.full((len(X), len(QCOLS)), np.nan)
        if ok.any():
            Z = self.scaler_.transform(Xv[ok])
            v[ok] = np.column_stack([m.predict(Z) for m in self.models_])
        return finalize(pd.DataFrame(v, index=X.index, columns=QCOLS))


class LightGBMQR:
    """One LGBMRegressor(objective="quantile", alpha=tau) per quantile."""

    def __init__(self, **params):
        self.params = params

    def fit(self, X, y, sample_weight=None):
        import lightgbm

        yv = y.reindex(X.index)
        ok = yv.notna().to_numpy()
        sw = _weights(sample_weight, X.index)
        sw = None if sw is None else sw[ok]
        self.columns_ = list(X.columns)
        self.models_ = []
        for q in QUANTILES:
            m = lightgbm.LGBMRegressor(objective="quantile", alpha=q, verbose=-1, **self.params)
            m.fit(X[ok], yv[ok], sample_weight=sw)
            self.models_.append(m)
        return self

    def predict(self, X):
        Xc = X[self.columns_]
        v = np.column_stack([m.predict(Xc) for m in self.models_])
        return finalize(pd.DataFrame(v, index=X.index, columns=QCOLS))


class CQR:
    """Conformalized quantile regression wrapper (symmetric pairwise adjustment).

    ``model`` is a factory (class / zero-arg callable) or an unfitted instance
    (deep-copied, never mutated). The wrapped model is fitted on rows before
    the last ``calib_days`` calendar days of the training index; those last days
    calibrate each pair (tau, 1 - tau), tau in {0.1, 0.2, 0.3, 0.4}.
    """

    def __init__(self, model, calib_days: int):
        self.model = model
        self.calib_days = calib_days

    def _new_model(self):
        m = self.model
        if isinstance(m, type) or (callable(m) and not hasattr(m, "fit")):
            return m()
        return copy.deepcopy(m)

    def fit(self, X, y, sample_weight=None):
        start = X.index.max().normalize() - pd.Timedelta(days=self.calib_days - 1)
        cal = np.asarray(X.index >= start)
        if cal.all() or not cal.any():
            raise ValueError("calibration split leaves no proper-train or calibration rows")
        y = y.reindex(X.index)
        sw = _weights(sample_weight, X.index)
        sw = None if sw is None else pd.Series(sw[~cal], index=X.index[~cal])
        self.model_ = self._new_model().fit(X[~cal], y[~cal], sample_weight=sw)
        p = self.model_.predict(X[cal])
        yc = y[cal].to_numpy(dtype=float)
        pv = p[QCOLS].to_numpy(dtype=float)
        ok = np.isfinite(yc) & np.isfinite(pv).all(axis=1)
        yc, pv = yc[ok], pv[ok]
        n = len(yc)
        if n == 0:
            raise ValueError("no valid calibration rows")
        self.adjust_ = {}
        k = len(QCOLS)
        for i in range(k // 2):
            tau = QUANTILES[i]
            lo, hi = pv[:, i], pv[:, k - 1 - i]
            E = np.maximum(lo - yc, yc - hi)
            level = min(1.0, (1 - 2 * tau) * (1 + 1 / n))
            self.adjust_[i] = float(np.quantile(E, level, method="higher"))
        self.n_calib_ = n
        return self

    def predict(self, X):
        v = self.model_.predict(X)[QCOLS].to_numpy(dtype=float).copy()
        k = len(QCOLS)
        for i, Q in self.adjust_.items():
            v[:, i] -= Q
            v[:, k - 1 - i] += Q
        return finalize(pd.DataFrame(v, index=X.index, columns=QCOLS))


def recency_weights(index, half_life_days: float | None) -> pd.Series:
    """Exponential recency weights: 0.5 ** (age_days / half_life_days); None -> ones."""
    index = pd.DatetimeIndex(index)
    if half_life_days is None:
        return pd.Series(1.0, index=index, name="weight")
    age = (index.max() - index) / pd.Timedelta(days=1)
    return pd.Series(0.5 ** (np.asarray(age, dtype=float) / half_life_days), index=index, name="weight")
