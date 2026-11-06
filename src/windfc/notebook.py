"""Leak-guarded helpers shared by the analysis notebooks (01-04).

- ``load_validation`` reads only hours whose Danish local delivery day is before
  the locked test period; test rows are filtered while reading the parquet.
- ``rerun`` re-runs a logged config into a throwaway log (never the real
  ``experiments/log.csv``) and checks that it reproduces the logged mean CRPS.
- ``fold_table`` flattens per-fold metrics of ``run_config`` results.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from windfc.cutoff import delivery_day, delivery_hours
from windfc.evaluate import TEST_PERIOD
from windfc.run import default_paths, frozen_commit, init_log, read_log, run_config

TEST_START = pd.Timestamp(TEST_PERIOD[0])


def load_validation(path=None) -> pd.DataFrame:
    """DK1 hourly frame restricted to delivery days before the test period.

    The bound (first UTC hour of local delivery day 2026-05-01) is pushed into
    the parquet read, so test-period rows are never materialised."""
    path = default_paths()[0] if path is None else Path(path)
    first_test_hour = delivery_hours(TEST_START)[0]
    df = pd.read_parquet(path, filters=[("hour_utc", "<", first_test_hour)]).sort_index()
    if len(df):
        assert delivery_day(df.index).max() < TEST_START, "test-period hours loaded"
    return df


def _same_file(a: Path, b: Path) -> bool:
    if a.resolve() == b.resolve():
        return True
    return a.exists() and b.exists() and a.samefile(b)


def rerun(cfg_path, *, log_path=None, reference_log=None, data_path=None, rtol=1e-6) -> dict:
    """Re-run the config at ``cfg_path`` into a throwaway log and return the result.

    ``reference_log`` (default: the real ``experiments/log.csv``) is only read:
    it supplies the frozen-eval header and the logged ``crps_mean`` that the
    re-run must reproduce. ``log_path`` (default: a fresh temp file) must not
    be the reference log; passing it raises ValueError before any work."""
    import yaml

    reference_log = default_paths()[1] if reference_log is None else Path(reference_log)
    if log_path is None:
        log_path = Path(tempfile.mkdtemp(prefix="windfc_rerun_")) / "log.csv"
    log_path = Path(log_path)
    if _same_file(log_path, reference_log):
        raise ValueError(f"rerun refuses to write to the reference log {reference_log}")
    if log_path.exists():
        raise ValueError(f"rerun needs a fresh log path, {log_path} exists")

    cfg = yaml.safe_load(Path(cfg_path).read_text())
    logged = read_log(reference_log).set_index("run_id")
    if cfg.get("run_id") not in logged.index:
        raise ValueError(f"run_id {cfg.get('run_id')!r} is not in {reference_log}")

    init_log(log_path, frozen_commit(reference_log))
    res = run_config(cfg, data_path=data_path, log_path=log_path)
    want = float(logged.loc[cfg["run_id"], "crps_mean"])
    if not np.isclose(res["crps_mean"], want, rtol=rtol):
        raise AssertionError(f"{cfg['run_id']}: re-run crps_mean {res['crps_mean']} != logged {want}")
    return res


def fold_table(results) -> pd.DataFrame:
    """One row per (run, fold) with the frozen per-fold metrics."""
    rows = [dict(run=r["run_id"], fold=f["fold"], crps=f["crps"], mae_p50=f["mae_p50"],
                 coverage_80=f["coverage_80"], bias=f["bias"], crps_norm=f["crps_norm"],
                 n_hours=f["n_hours"])
            for r in results for f in r["folds"]]
    return pd.DataFrame(rows)
