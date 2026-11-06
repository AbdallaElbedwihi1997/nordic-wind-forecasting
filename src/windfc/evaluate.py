"""Frozen evaluation: metrics, validation folds, Diebold-Mariano test.

Do not change after model optimization starts.
"""
import numpy as np
import pandas as pd
from scipy import stats

from windfc import QUANTILES
from windfc.cutoff import delivery_day

QCOLS = [f"q{int(round(q * 100))}" for q in QUANTILES]

# Inclusive delivery days (Danish local calendar days, see cutoff.delivery_day).
FOLDS = [
    ("2025-05-01", "2025-07-31"),
    ("2025-08-01", "2025-10-31"),
    ("2025-11-01", "2026-01-31"),
    ("2026-02-01", "2026-04-30"),
]
TEST_PERIOD = ("2026-05-01", "2026-10-28")


def period_mask(index, start, end) -> np.ndarray:
    """Boolean mask for start <= delivery_day(t) <= end (tz-naive UTC hours)."""
    dd = delivery_day(pd.DatetimeIndex(index))
    lo = pd.Timestamp(start).normalize()
    hi = pd.Timestamp(end).normalize()
    return np.asarray((dd >= lo) & (dd <= hi))


def _aligned(y: pd.Series, preds: pd.DataFrame):
    """Rows of y (with preds reindexed to y's index) where y and all preds are valid."""
    p = preds[QCOLS].reindex(y.index)
    ok = y.notna().to_numpy() & p.notna().all(axis=1).to_numpy()
    return y[ok], p[ok], ok


def pinball(y: pd.Series, preds: pd.DataFrame) -> pd.Series:
    """Per-hour pinball loss averaged over the 9 quantiles (intersection of indices)."""
    common = y.index.intersection(preds.index)
    y2, p2, _ = _aligned(y.loc[common], preds.loc[common])
    err = y2.to_numpy()[:, None] - p2.to_numpy()
    tau = np.asarray(QUANTILES)[None, :]
    loss = np.maximum(tau * err, (tau - 1.0) * err)
    return pd.Series(loss.mean(axis=1), index=y2.index)


def score(y: pd.Series, preds: pd.DataFrame) -> dict:
    y2, p2, ok = _aligned(y, preds)
    n_hours = int(ok.sum())
    n_dropped = int(len(y) - n_hours)
    if n_hours == 0:
        nan = float("nan")
        return dict(crps=nan, mae_p50=nan, coverage_80=nan, bias=nan,
                    crps_norm=nan, mae_norm=nan, n_hours=0, n_dropped=n_dropped)
    crps = float(pinball(y2, p2).mean())
    q50 = p2["q50"]
    mae = float((y2 - q50).abs().mean())
    mean_y = float(y2.mean())
    return dict(
        crps=crps,
        mae_p50=mae,
        coverage_80=float(((p2["q10"] <= y2) & (y2 <= p2["q90"])).mean()),
        bias=float((q50 - y2).mean()),
        crps_norm=crps / mean_y,
        mae_norm=mae / mean_y,
        n_hours=n_hours,
        n_dropped=n_dropped,
    )


def diebold_mariano(loss_a: pd.Series, loss_b: pd.Series, h: int = 1):
    """HLN-corrected DM test on delivery-day mean losses; negative stat => a better."""
    df = pd.concat([loss_a, loss_b], axis=1, join="inner").dropna()
    daily = df.groupby(delivery_day(df.index)).mean()
    d = (daily.iloc[:, 0] - daily.iloc[:, 1]).to_numpy()
    n = len(d)
    if n < 2:
        return (0.0, 1.0)
    dbar = d.mean()
    dc = d - dbar
    var = float(np.dot(dc, dc) / n)  # gamma_0
    for k in range(1, h):  # Newey-West, Bartlett weights, lag h-1
        gk = float(np.dot(dc[k:], dc[:-k]) / n)
        var += 2.0 * (1.0 - k / h) * gk
    if var <= 0.0 or np.isclose(var, 0.0, atol=1e-18):
        return (0.0, 1.0)
    dm = dbar / np.sqrt(var / n)
    stat = float(dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n))
    pval = float(2.0 * stats.t.sf(abs(stat), df=n - 1))
    return (stat, pval)
