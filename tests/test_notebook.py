from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from windfc.cutoff import delivery_day
from windfc.notebook import TEST_START, fold_table, load_validation, rerun
from windfc.run import REPO_ROOT, default_paths

PARQUET, REAL_LOG = default_paths()
CFG_001 = REPO_ROOT / "experiments" / "configs" / "001_persistence.yaml"


def _boundary_parquet(tmp_path):
    # 2026-04-30 18:00 .. 2026-05-01 04:00 UTC; local 2026-05-01 starts 2026-04-30 22:00 UTC
    idx = pd.date_range("2026-04-30 18:00", "2026-05-01 04:00", freq="h", name="hour_utc")
    df = pd.DataFrame({"actual_total": np.arange(len(idx), dtype=float)}, index=idx)
    p = tmp_path / "x.parquet"
    df.to_parquet(p)
    return p


def test_load_validation_cuts_at_local_midnight(tmp_path):
    out = load_validation(_boundary_parquet(tmp_path))
    assert out.index.max() == pd.Timestamp("2026-04-30 21:00")
    assert len(out) == 4
    assert (delivery_day(out.index) < TEST_START).all()


@pytest.mark.skipif(not PARQUET.exists(), reason="real dataset missing")
def test_load_validation_real_data_has_no_test_days():
    df = load_validation()
    dd = delivery_day(df.index)
    assert dd.max() == pd.Timestamp("2026-04-30")
    assert (dd < TEST_START).all()


def test_rerun_refuses_real_log():
    before = REAL_LOG.read_bytes()
    with pytest.raises(ValueError, match="refuses"):
        rerun(CFG_001, log_path=REAL_LOG)
    with pytest.raises(ValueError, match="refuses"):
        rerun(CFG_001, log_path=REAL_LOG.parent / ".." / "experiments" / "log.csv")
    assert REAL_LOG.read_bytes() == before


def test_rerun_refuses_existing_log(tmp_path):
    p = tmp_path / "log.csv"
    p.write_text("x")
    with pytest.raises(ValueError, match="fresh"):
        rerun(CFG_001, log_path=p)


@pytest.mark.skipif(not PARQUET.exists(), reason="real dataset missing")
def test_rerun_reproduces_logged_run_without_touching_real_log(tmp_path):
    before = REAL_LOG.read_bytes()
    res = rerun(CFG_001, log_path=tmp_path / "log.csv")
    assert REAL_LOG.read_bytes() == before
    assert (tmp_path / "log.csv").read_text().count("001_persistence") == 1
    ft = fold_table([res])
    assert list(ft.fold) == [1, 2, 3, 4] and (ft.run == "001_persistence").all()
    assert ft.n_hours.sum() == res["n_hours"]
