import datetime as dt

import numpy as np
import pandas as pd
import pytest

from windfc import run as runmod
from windfc.cutoff import delivery_day
from windfc.evaluate import TEST_PERIOD
from windfc.run import LOG_COLUMNS, decide, read_log, run_config

# ---------------------------------------------------------------- decide


BEST = dict(crps_f1=10.0, crps_f2=10.0, crps_f3=10.0, crps_f4=10.0,
            crps_mean=10.0, mae_p50_mean=20.0, coverage_mean=0.80)


def _res(folds, mae=20.0, cov=0.80, model="lgbm"):
    r = {f"crps_f{i + 1}": v for i, v in enumerate(folds)}
    r.update(crps_mean=float(np.mean(folds)), mae_p50_mean=mae, coverage_mean=cov, model=model)
    return r


@pytest.mark.parametrize(
    "res, decision, needle",
    [
        (_res([9.8] * 4), "accepted", ""),                        # all rules hold
        (_res([9.91] * 4), "rejected", "crps"),                   # 0.9% gain only
        (_res([9.0, 9.0, 10.1, 10.1]), "rejected", "folds"),      # 2/4 folds
        (_res([9.8] * 4, mae=20.3), "rejected", "mae"),           # MAE +1.5%
        (_res([9.8] * 4, cov=0.86), "rejected", "coverage"),      # coverage 0.86
    ],
)
def test_decide_accepts_only_when_all_rules_hold(res, decision, needle):
    d, reason = decide(res, BEST)
    assert d == decision
    assert needle in reason.lower()


def test_decide_boundaries_and_multiple_failures():
    # exactly 1% gain, exactly +1% MAE, coverage on the edges -> accepted
    assert decide(_res([9.9] * 4, mae=20.2, cov=0.76), BEST)[0] == "accepted"
    assert decide(_res([9.9] * 4, mae=20.2, cov=0.84), BEST)[0] == "accepted"
    d, reason = decide(_res([9.95] * 4, mae=21.0, cov=0.70), BEST)
    assert d == "rejected"
    for word in ("crps", "mae", "coverage"):
        assert word in reason.lower()


def test_decide_baseline_and_initial_best():
    for m in ("persistence", "vendor_raw", "vendor_capacity"):
        assert decide(_res([1.0] * 4, model=m), BEST)[0] == "baseline"
        assert decide(_res([1.0] * 4, model=m), None)[0] == "baseline"
    assert decide(_res([50.0] * 4, cov=0.80), None) == ("accepted", "initial best")


@pytest.mark.parametrize("cov", [0.693, 0.75, 0.85])
def test_initial_best_requires_coverage(cov):
    d, reason = decide(_res([50.0] * 4, cov=cov), None)
    assert d == "rejected" and "coverage" in reason.lower()
    assert decide(_res([50.0] * 4, cov=0.76), None)[0] == "accepted"
    assert decide(_res([50.0] * 4, cov=0.84), None)[0] == "accepted"


# ---------------------------------------------------------------- runner


@pytest.fixture(scope="module")
def data_path(tmp_path_factory):
    """Synthetic DK1-like hourly frame covering training, folds and part of
    the test period (test-period actuals are absurd to detect any use)."""
    idx = pd.date_range("2021-12-31 23:00", "2026-06-30 21:00", freq="h", name="hour_utc")
    n = len(idx)
    rng = np.random.default_rng(0)
    t = np.arange(n)
    da_on = np.clip(1500 + 900 * np.sin(t / 37.0) + rng.normal(0, 80, n), 0, None)
    da_off = np.clip(700 + 400 * np.cos(t / 53.0) + rng.normal(0, 40, n), 0, None)
    act_on = np.clip(da_on + rng.normal(0, 150, n), 0, None)
    act_off = np.clip(da_off + rng.normal(0, 80, n), 0, None)
    df = pd.DataFrame(
        {
            "actual_on": act_on, "actual_off": act_off, "actual_total": act_on + act_off,
            "da_on": da_on, "da_off": da_off, "da_total": da_on + da_off,
            "cur_on": da_on, "cur_off": da_off,
        },
        index=idx,
    )
    df.iloc[5000:5030] = np.nan  # a source gap
    gap = df.index.get_indexer([pd.Timestamp("2025-06-10 05:00")])[0]
    df.iloc[gap:gap + 7, df.columns.get_loc("actual_total")] = np.nan  # gap inside fold 1
    test = delivery_day(df.index) >= pd.Timestamp(TEST_PERIOD[0])
    df.loc[test, "actual_total"] = 1e9
    p = tmp_path_factory.mktemp("data") / "synthetic.parquet"
    df.to_parquet(p)
    return p


def _cfg(run_id="r1", **kw):
    cfg = dict(
        run_id=run_id,
        hypothesis="sanity floor",
        change="none",
        model="persistence",
        params={},
        feature_set=["last_actual"],
        half_life_days=None,
        cqr_calib_days=None,
    )
    cfg.update(kw)
    return cfg


def test_log_row_appended(data_path, tmp_path):
    log = tmp_path / "log.csv"
    res = run_config(_cfg(), data_path=data_path, log_path=log, check_freeze=False)
    assert log.read_text().splitlines()[0].startswith("# frozen_eval_commit=")
    rows = read_log(log)
    assert list(rows.columns) == LOG_COLUMNS
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["run_id"] == "r1" and row["decision"] == "baseline"
    assert row["model"] == "persistence"
    for i in range(1, 5):
        assert np.isfinite(row[f"crps_f{i}"]) and row[f"crps_f{i}"] > 0
        assert res[f"crps_f{i}"] == pytest.approx(row[f"crps_f{i}"])
    assert row["crps_mean"] == pytest.approx(np.mean([row[f"crps_f{i}"] for i in range(1, 5)]))
    assert row["n_hours"] == sum(f["n_hours"] for f in res["folds"])
    # predictions never touch the test period (whose actuals are 1e9)
    p = res["predictions"]
    assert (delivery_day(p.index) < pd.Timestamp(TEST_PERIOD[0])).all()
    assert p["actual_total"].max() < 1e6
    assert set(p["fold"]) == {1, 2, 3, 4}
    for f in res["folds"]:
        # hours of the fold's delivery days that were not scored (gap -> > 0 somewhere)
        assert f["n_dropped"] >= 0
        assert f["n_dropped"] + f["n_hours"] == f["n_fold_hours"]
    assert sum(f["n_dropped"] for f in res["folds"]) > 0
    assert {"q10", "q50", "q90"} <= set(p.columns)
    assert row["crps_f1"] < 1e6

    # second run: lgbm with weights + CQR; common mask -> same n_hours per fold
    res2 = run_config(
        _cfg("r2", model="lgbm", params={"n_estimators": 5}, feature_set=["da_total_adj", "hour"],
             half_life_days=60, cqr_calib_days=30),
        data_path=data_path, log_path=log, check_freeze=False,
    )
    assert [f["n_hours"] for f in res2["folds"]] == [f["n_hours"] for f in res["folds"]]
    assert res2["decision"] == "accepted" and res2["reason"] == "initial best"
    rows = read_log(log)
    assert list(rows["run_id"]) == ["r1", "r2"]


def test_duplicate_run_id_refused(data_path, tmp_path):
    log = tmp_path / "log.csv"
    run_config(_cfg("dup"), data_path=data_path, log_path=log, check_freeze=False)
    with pytest.raises(ValueError, match="dup"):
        run_config(_cfg("dup"), data_path=data_path, log_path=log, check_freeze=False)
    assert len(read_log(log)) == 1


def test_budget_cap(data_path, tmp_path):
    log = tmp_path / "log.csv"
    run_config(_cfg("seed"), data_path=data_path, log_path=log, check_freeze=False)
    lines = log.read_text().splitlines()
    for i in range(19):
        lines.append(lines[-1].replace("seed", f"fill{i}", 1))
    log.write_text("\n".join(lines) + "\n")
    assert len(read_log(log)) == 20
    with pytest.raises(RuntimeError, match="budget"):
        run_config(_cfg("one_too_many"), data_path=data_path, log_path=log, check_freeze=False)


@pytest.mark.parametrize(
    "extra",
    [
        {"train_end": "2026-05-01"},
        {"train_end": dt.date(2026, 6, 1)},
        {"params": {"note": "fit through 2026-07-15"}},
        {"change": "extend window to 2026-05-02"},
    ],
)
def test_run_refuses_test_period(extra, tmp_path):
    with pytest.raises(ValueError, match="test period"):
        run_config(_cfg(**extra), data_path=tmp_path / "missing.parquet",
                   log_path=tmp_path / "log.csv", check_freeze=False)
    assert not (tmp_path / "log.csv").exists()


def test_test_period_guard_on_folds(data_path, tmp_path, monkeypatch):
    # a fold reaching into the test period (by delivery day) must be refused
    monkeypatch.setattr(runmod, "FOLDS", [("2026-04-01", "2026-05-01")])
    with pytest.raises(ValueError, match="test period"):
        run_config(_cfg(), data_path=data_path, log_path=tmp_path / "log.csv", check_freeze=False)


@pytest.mark.parametrize(
    "bad",
    [
        dict(model="xgboost"),
        dict(model="persistence", feature_set=["da_total"]),  # needs last_actual
        dict(model="vendor_raw", feature_set=["da_total_adj"]),
        dict(feature_set=["last_actual", "nope"]),
        dict(half_life_days=-1),
        dict(cqr_calib_days=0),
        dict(surprise=1),
    ],
)
def test_config_validation(bad, tmp_path):
    with pytest.raises(ValueError):
        run_config(_cfg(**bad), data_path=tmp_path / "missing.parquet",
                   log_path=tmp_path / "log.csv", check_freeze=False)


def test_missing_key_rejected(tmp_path):
    cfg = _cfg()
    del cfg["hypothesis"]
    with pytest.raises(ValueError, match="hypothesis"):
        run_config(cfg, data_path=tmp_path / "x.parquet", log_path=tmp_path / "log.csv",
                   check_freeze=False)


def test_freeze_check(tmp_path):
    cfg = _cfg()
    # no log -> no frozen hash -> refuse
    with pytest.raises(RuntimeError, match="frozen"):
        run_config(cfg, data_path=tmp_path / "x.parquet", log_path=tmp_path / "log.csv")
    # an unknown hash is an error, not a pass
    log = tmp_path / "log2.csv"
    log.write_text("# frozen_eval_commit=" + "0" * 40 + "\n" + ",".join(LOG_COLUMNS) + "\n")
    with pytest.raises(RuntimeError):
        run_config(cfg, data_path=tmp_path / "x.parquet", log_path=log)


def test_freeze_check_detects_diff(tmp_path, monkeypatch):
    calls = []

    def fake_diff(h):
        calls.append(h)
        return False  # files differ

    monkeypatch.setattr(runmod, "_frozen_files_unchanged", fake_diff)
    log = tmp_path / "log.csv"
    log.write_text("# frozen_eval_commit=abc123\n" + ",".join(LOG_COLUMNS) + "\n")
    with pytest.raises(RuntimeError, match="evaluate.py"):
        run_config(_cfg(), data_path=tmp_path / "x.parquet", log_path=log)
    assert calls == ["abc123"]


def test_persistence_config_file_valid():
    import yaml

    from windfc.run import REPO_ROOT, validate_config

    cfg = yaml.safe_load((REPO_ROOT / "experiments/configs/001_persistence.yaml").read_text())
    validate_config(cfg)
    assert cfg["model"] == "persistence" and cfg["feature_set"] == ["last_actual"]
    assert cfg["hypothesis"] == "sanity floor"


def test_default_paths_resolve_against_repo_root():
    import inspect

    from windfc.run import REPO_ROOT, default_paths

    data, log = default_paths()
    assert data == REPO_ROOT / "data" / "dk1_hourly.parquet"
    assert log == REPO_ROOT / "experiments" / "log.csv"
    sig = inspect.signature(run_config)
    assert sig.parameters["data_path"].default is None
    assert sig.parameters["log_path"].default is None


def test_git_hash_dirty_rules():
    from windfc.run import REPO_ROOT, git_hash

    base = git_hash()
    if base.endswith("+dirty") or base == "unknown":
        pytest.skip("working tree already dirty")
    log = REPO_ROOT / "experiments" / "log.csv"
    orig = log.read_bytes()
    probe = REPO_ROOT / "src" / "windfc" / "_dirty_probe_tmp.py"
    try:
        log.write_bytes(orig + b"x\n")  # the runner's own log never marks dirty
        assert not git_hash().endswith("+dirty")
        log.write_bytes(orig)
        probe.write_text("# probe\n")  # untracked file under src/ marks dirty
        assert git_hash().endswith("+dirty")
    finally:
        log.write_bytes(orig)
        probe.unlink(missing_ok=True)
    assert git_hash() == base
