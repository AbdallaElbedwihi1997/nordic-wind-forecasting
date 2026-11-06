import numpy as np
import pandas as pd
import pytest

from windfc import QUANTILES
from windfc.evaluate import (FOLDS, TEST_PERIOD, diebold_mariano, period_mask,
                             pinball, score)

COLS = [f"q{int(q * 100)}" for q in QUANTILES]


def _preds(index, values):
    arr = np.broadcast_to(np.asarray(values, dtype=float), (len(index), 9)).copy()
    return pd.DataFrame(arr, index=index, columns=COLS)


def _idx(n):
    return pd.date_range("2026-01-01", periods=n, freq="h")


def test_pinball_hand_value():
    idx = _idx(1)
    y = pd.Series([10.0], index=idx)
    out = pinball(y, _preds(idx, 8.0))
    assert out.iloc[0] == pytest.approx(1.0)


def test_pinball_hand_value_differing_quantiles():
    idx = _idx(1)
    y = pd.Series([10.0], index=idx)
    q = [4, 5, 6, 7, 8, 9, 10, 12, 14]
    # under-predictions (y-q>0): tau*(y-q); over: (1-tau)*(q-y)
    exp = (0.1 * 6 + 0.2 * 5 + 0.3 * 4 + 0.4 * 3 + 0.5 * 2 + 0.6 * 1 + 0.0
           + 0.2 * 2 + 0.1 * 4) / 9
    out = pinball(y, _preds(idx, q))
    assert out.iloc[0] == pytest.approx(exp)


def test_coverage():
    idx = _idx(4)
    y = pd.Series([5.0, 5.0, 5.0, 20.0], index=idx)
    preds = _preds(idx, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert score(y, preds)["coverage_80"] == pytest.approx(0.75)


def test_score_drops_nan_rows_and_counts_them():
    idx = _idx(4)
    y = pd.Series([5.0, np.nan, 5.0, 5.0], index=idx)
    s = score(y, _preds(idx, 5.0))
    assert s["n_dropped"] == 1 and s["n_hours"] == 3
    assert isinstance(s["n_hours"], int) and isinstance(s["n_dropped"], int)


def test_score_counts_nan_pred_and_missing_rows():
    idx = _idx(4)
    y = pd.Series([5.0, 6.0, 7.0, 8.0], index=idx)
    p = _preds(idx, 5.0).iloc[:3].copy()
    p.iloc[0, 2] = np.nan
    s = score(y, p)
    assert s["n_dropped"] == 2 and s["n_hours"] == 2


def test_score_values():
    idx = _idx(2)
    y = pd.Series([10.0, 20.0], index=idx)
    p = _preds(idx, [8, 8, 8, 8, 8, 8, 8, 8, 8])
    s = score(y, p)
    assert set(s) == {"crps", "mae_p50", "coverage_80", "bias", "crps_norm",
                      "mae_norm", "n_hours", "n_dropped"}
    assert s["mae_p50"] == pytest.approx(7.0)
    assert s["bias"] == pytest.approx(-7.0)
    assert s["mae_norm"] == pytest.approx(7.0 / 15.0)
    assert s["crps_norm"] == pytest.approx(s["crps"] / 15.0)


def test_folds_do_not_overlap_or_touch_test():
    starts = [pd.Timestamp(a) for a, _ in FOLDS]
    ends = [pd.Timestamp(b) for _, b in FOLDS]
    assert all(a <= b for a, b in zip(starts, ends))
    for prev_end, nxt_start in zip(ends[:-1], starts[1:]):
        assert prev_end < nxt_start
    assert ends[-1] < pd.Timestamp(TEST_PERIOD[0])


def test_period_mask_inclusive_whole_days():
    idx = pd.date_range("2025-04-30 22:00", "2025-08-01 02:00", freq="h")
    m = period_mask(idx, "2025-05-01", "2025-07-31")
    assert idx[m][0] == pd.Timestamp("2025-05-01 00:00")
    assert idx[m][-1] == pd.Timestamp("2025-07-31 23:00")


def test_dm_identical_losses():
    idx = _idx(24 * 30)
    a = pd.Series(np.random.default_rng(0).random(len(idx)), index=idx)
    assert diebold_mariano(a, a.copy()) == (0.0, 1.0)


def test_dm_clearly_better():
    idx = _idx(24 * 60)
    rng = np.random.default_rng(1)
    b = pd.Series(rng.random(len(idx)) + 1.0, index=idx)
    a = b - 0.3 + rng.normal(0, 0.05, len(idx))
    stat, p = diebold_mariano(a, b)
    assert stat < 0 and p < 0.05
    stat2, p2 = diebold_mariano(a, b, h=2)
    assert stat2 < 0 and p2 < 0.05
