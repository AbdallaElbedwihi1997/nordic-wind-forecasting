"""One-shot scoring on the locked test period (used once, in notebook 05).

``score_test(cfg)`` fits the model a run config describes exactly as
``windfc.run`` does for a validation fold (same model factory, features,
recency weights and CQR wrapper), but on delivery days 2022-01-01 …
2026-04-30, and predicts the test delivery days ``evaluate.TEST_PERIOD``
(2026-05-01 … 2026-10-28). Scored hours follow the runner's common evaluation
mask (valid ``actual_total`` and ``BASE_FEATURES + da_total + last_actual``) and
are scored with the frozen ``evaluate.score``.

Features of test days use history up to each day's issue-time cutoff (as in
operation; ``build_features`` runs the leakage check). Training rows are
delivery days before the test period only. Nothing is written to
``experiments/log.csv``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from windfc.cutoff import delivery_day
from windfc.evaluate import QCOLS, TEST_PERIOD, period_mask, pinball, score
from windfc.features import build_features, complete_rows
from windfc.models import recency_weights
from windfc.run import (
    MASK_FEATURES,
    TRAIN_START,
    _frozen_files_unchanged,
    _model_columns,
    default_paths,
    frozen_commit,
    make_model,
    validate_config,
)

TEST_START = pd.Timestamp(TEST_PERIOD[0])
TEST_END = pd.Timestamp(TEST_PERIOD[1])
TRAIN_END = TEST_START - pd.Timedelta(days=1)


def score_test(cfg: dict, data_path=None, check_freeze: bool = True) -> dict:
    """Train on 2022-01-01 … 2026-04-30, score once on the test period.

    Returns ``run_id``, ``model``, ``config``, ``scores`` (the frozen
    ``evaluate.score`` dict, with ``n_dropped`` = test hours outside the common
    mask), ``n_hours``, ``n_dropped``, ``n_train``, ``pinball`` (per-hour loss
    averaged over the 9 quantiles) and ``predictions`` (``actual_total`` + the
    9 quantile columns, indexed by UTC hour).

    ``check_freeze`` refuses to run if ``evaluate.py`` / ``cutoff.py`` differ
    from the commit in the real log's header (the log is only read)."""
    cfg = validate_config(cfg)
    d_data, d_log = default_paths()
    data_path = d_data if data_path is None else Path(data_path)
    if check_freeze:
        commit = frozen_commit(d_log)
        if commit is None or not _frozen_files_unchanged(commit):
            raise RuntimeError("evaluate.py / cutoff.py are not at the frozen commit")

    df = pd.read_parquet(data_path).sort_index()
    df = df[np.asarray(delivery_day(df.index) <= TEST_END)]
    days = pd.date_range(TRAIN_START, TEST_END, freq="D")
    feat_cols = list(dict.fromkeys(cfg["feature_set"] + MASK_FEATURES))
    X_all, _ = build_features(df, days, feat_cols)
    y_all = df["actual_total"].reindex(X_all.index)
    mcols = _model_columns(cfg["feature_set"])
    common = y_all.notna().to_numpy() & X_all[MASK_FEATURES].notna().all(axis=1).to_numpy()

    tr = period_mask(X_all.index, TRAIN_START, TRAIN_END)
    in_test = period_mask(X_all.index, TEST_START, TEST_END)
    te = in_test & common
    Xtr, ytr = complete_rows(X_all.loc[tr, mcols], y_all)
    Xte, yte = X_all.loc[te, mcols], y_all[te]
    if len(Xtr) == 0 or len(Xte) == 0:
        raise ValueError(f"no training ({len(Xtr)}) or scored ({len(Xte)}) rows")
    if (delivery_day(Xtr.index) >= TEST_START).any():
        raise RuntimeError("training rows include test-period delivery days")
    if not ((delivery_day(Xte.index) >= TEST_START) & (delivery_day(Xte.index) <= TEST_END)).all():
        raise RuntimeError("scored rows fall outside the test period")

    w = recency_weights(Xtr.index, cfg["half_life_days"])
    model = make_model(cfg).fit(Xtr, ytr, sample_weight=w)
    p = model.predict(Xte)
    if not p.index.equals(Xte.index) or p[QCOLS].isna().any().any():
        raise ValueError("model predicted NaN or a different index on scored test hours")

    s = score(yte, p)
    if s["n_hours"] != len(Xte):
        raise RuntimeError(f"scored {s['n_hours']} hours, expected {len(Xte)}")
    s["n_dropped"] = int(in_test.sum()) - len(Xte)
    return dict(
        run_id=cfg["run_id"], model=cfg["model"], config=cfg, scores=s,
        n_hours=s["n_hours"], n_dropped=s["n_dropped"], n_train=len(Xtr),
        pinball=pinball(yte, p),
        predictions=p[QCOLS].assign(actual_total=yte)[["actual_total", *QCOLS]],
    )
