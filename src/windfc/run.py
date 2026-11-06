"""Config-driven experiment runner with a frozen-evaluation, budgeted log.

Usage: ``python -m windfc.run <config.yaml> [--data PATH] [--log PATH]``

For each validation fold the model is refit on delivery days 2022-01-01 …
fold start − 1 day (expanding window), predicts the fold's delivery days and is
scored with ``windfc.evaluate.score`` on a *common evaluation mask*: hours with
a valid ``actual_total`` and valid ``BASE_FEATURES + da_total + last_actual``.
Every model is scored on exactly those hours, so runs are comparable.

The log (``experiments/log.csv``) starts with ``# frozen_eval_commit=<hash>``;
a run aborts if ``evaluate.py`` / ``cutoff.py`` differ from that commit
(including uncommitted changes), if the log already holds ``BUDGET`` runs, or if
its ``run_id`` is already logged. Read the log with ``read_log`` (pandas
``comment="#"``).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from windfc.cutoff import delivery_day
from windfc.evaluate import FOLDS, QCOLS, TEST_PERIOD, period_mask, score
from windfc.features import BASE_FEATURES, build_features, complete_rows
from windfc.models import CQR, LightGBMQR, LinearQR, Persistence, VendorEmpirical, recency_weights

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_FILES = ["src/windfc/evaluate.py", "src/windfc/cutoff.py"]
HEADER_PREFIX = "# frozen_eval_commit="
BUDGET = 20
TRAIN_START = "2022-01-01"
TEST_START = pd.Timestamp(TEST_PERIOD[0])

MODELS = ("persistence", "vendor_raw", "vendor_capacity", "linear_qr", "lgbm")
BASELINES = ("persistence", "vendor_raw", "vendor_capacity")
CONFIG_KEYS = ("run_id", "hypothesis", "change", "model", "params", "feature_set",
               "half_life_days", "cqr_calib_days")
# Columns every scored hour must have, whatever the model uses.
MASK_FEATURES = list(dict.fromkeys(BASE_FEATURES + ["da_total", "last_actual"]))
_REQUIRED_FEATURE = {"persistence": "last_actual", "vendor_raw": "da_total",
                     "vendor_capacity": "da_total_adj"}

LOG_COLUMNS = [
    "run_id", "timestamp", "git_hash", "hypothesis", "change", "model",
    "crps_f1", "crps_f2", "crps_f3", "crps_f4", "crps_mean", "mae_p50_mean",
    "coverage_mean", "bias_mean", "n_hours", "decision", "reason",
]

# Acceptance rule (plan, Global Constraints).
_CRPS_GAIN = 0.99
_MIN_FOLDS = 3
_MAE_SLACK = 1.01
_COV = (0.76, 0.84)
_RTOL = 1e-9  # float slack so exact-boundary values count as meeting the rule

_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


# ---------------------------------------------------------------- config


def _dates_in(obj):
    """Yield every date mentioned in a config (keys and values, recursively)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _dates_in(k)
            yield from _dates_in(v)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            yield from _dates_in(v)
    elif isinstance(obj, (dt.date, dt.datetime, pd.Timestamp)):
        yield pd.Timestamp(obj)
    elif isinstance(obj, str):
        for m in _DATE_RE.finditer(obj):
            try:
                yield pd.Timestamp(int(m[1]), int(m[2]), int(m[3]))
            except ValueError:
                continue


def validate_config(cfg: dict) -> dict:
    """Validate a run config; return a normalized copy. Raises ValueError."""
    if not isinstance(cfg, dict):
        raise ValueError("config must be a mapping")
    late = [d for d in _dates_in(cfg) if d.normalize() >= TEST_START]
    if late:
        raise ValueError(
            f"config references date(s) in the test period (>= {TEST_PERIOD[0]}): "
            f"{sorted({str(d.date()) for d in late})}"
        )
    unknown = sorted(set(cfg) - set(CONFIG_KEYS), key=str)
    if unknown:
        raise ValueError(f"unknown config key(s): {unknown}")
    missing = [k for k in CONFIG_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"missing config key(s): {missing}")

    out = dict(cfg)
    for k in ("run_id", "hypothesis", "change"):
        v = out[k]
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"{k} must be a non-empty string (quote it in YAML)")
        if "#" in v:
            raise ValueError(f"{k} must not contain '#' (the log is read with comment='#')")
        out[k] = " ".join(v.split())
    if out["model"] not in MODELS:
        raise ValueError(f"unknown model {out['model']!r}; expected one of {MODELS}")
    params = out["params"] if out["params"] is not None else {}
    if not isinstance(params, dict):
        raise ValueError("params must be a mapping")
    if params and out["model"] in BASELINES:
        raise ValueError(f"model {out['model']!r} takes no params, got {params}")
    out["params"] = dict(params)

    fs = out["feature_set"]
    if not isinstance(fs, list) or not fs or not all(isinstance(f, str) for f in fs):
        raise ValueError("feature_set must be a non-empty list of feature names")
    if len(set(fs)) != len(fs):
        raise ValueError(f"feature_set has duplicates: {fs}")
    from windfc.features import _KNOWN

    bad = [f for f in fs if f not in _KNOWN]
    if bad:
        raise ValueError(f"unknown feature(s): {bad}")
    req = _REQUIRED_FEATURE.get(out["model"])
    if req and req not in fs:
        raise ValueError(f"model {out['model']!r} requires {req!r} in feature_set")
    out["feature_set"] = list(fs)

    hl = out["half_life_days"]
    if hl is not None and (isinstance(hl, bool) or not isinstance(hl, (int, float))
                           or not np.isfinite(hl) or hl <= 0):
        raise ValueError("half_life_days must be null or a positive number")
    cq = out["cqr_calib_days"]
    if cq is not None and (isinstance(cq, bool) or not isinstance(cq, int) or cq <= 0):
        raise ValueError("cqr_calib_days must be null or a positive integer")
    return out


def make_model(cfg: dict):
    """Fresh (unfitted) model for a validated config."""
    m, params = cfg["model"], cfg["params"]

    def base():
        if m == "persistence":
            return Persistence()
        if m == "vendor_raw":
            return VendorEmpirical("da_total")
        if m == "vendor_capacity":
            return VendorEmpirical("da_total_adj")
        if m == "linear_qr":
            return LinearQR(**params)
        if m == "lgbm":
            return LightGBMQR(**params)
        raise ValueError(f"unknown model {m!r}")

    if cfg["cqr_calib_days"] is not None:
        return CQR(base, calib_days=cfg["cqr_calib_days"])
    return base()


def _model_columns(feature_set):
    cols = []
    for f in feature_set:
        cols += ["level_x_hour_sin", "level_x_hour_cos"] if f == "level_x_hour" else [f]
    return cols


# ---------------------------------------------------------------- decision


def decide(result: dict, best: dict | None) -> tuple[str, str]:
    """Acceptance rule: vs current best, ALL of: mean CRPS ≤ 0.99 × best;
    CRPS better on ≥ 3 of 4 folds; P50 MAE ≤ 1.01 × best; 0.76 ≤ coverage ≤ 0.84.
    Baseline models are logged as "baseline" and never become best."""
    if result["model"] in BASELINES:
        return "baseline", "baseline model; not eligible as current best"
    if best is None:
        return "accepted", "initial best"
    fails = []
    crps, bcrps = float(result["crps_mean"]), float(best["crps_mean"])
    if not crps <= _CRPS_GAIN * bcrps * (1 + _RTOL):
        fails.append(f"crps_mean {crps:.4g} > 0.99 x best {bcrps:.4g} "
                     f"(gain {100 * (1 - crps / bcrps):.2f}% < 1%)")
    n_better = sum(float(result[f"crps_f{i}"]) < float(best[f"crps_f{i}"]) for i in range(1, 5))
    if n_better < _MIN_FOLDS:
        fails.append(f"crps improves on only {n_better}/4 folds (< 3)")
    mae, bmae = float(result["mae_p50_mean"]), float(best["mae_p50_mean"])
    if not mae <= _MAE_SLACK * bmae * (1 + _RTOL):
        fails.append(f"mae_p50_mean {mae:.4g} > 1.01 x best {bmae:.4g} "
                     f"({100 * (mae / bmae - 1):+.2f}%)")
    cov = float(result["coverage_mean"])
    if not (_COV[0] * (1 - _RTOL) <= cov <= _COV[1] * (1 + _RTOL)):
        fails.append(f"coverage_mean {cov:.3f} outside [0.76, 0.84]")
    if fails:
        return "rejected", "; ".join(fails)
    return "accepted", (f"crps_mean {crps:.4g} vs {bcrps:.4g} "
                        f"({100 * (1 - crps / bcrps):.2f}% gain), {n_better}/4 folds")


# ---------------------------------------------------------------- log + git


def _git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True)


def git_hash() -> str:
    r = _git("rev-parse", "HEAD")
    if r.returncode != 0:
        return "unknown"
    h = r.stdout.strip()
    dirty = _git("status", "--porcelain", "--untracked-files=no").stdout.strip()
    return h + ("+dirty" if dirty else "")


def _frozen_files_unchanged(commit: str) -> bool:
    """True if the frozen files equal ``commit`` (working tree incl. uncommitted)."""
    r = _git("diff", "--quiet", commit, "--", *FROZEN_FILES)
    if r.returncode == 0:
        return True
    if r.returncode == 1:
        return False
    raise RuntimeError(f"freeze check failed for commit {commit!r}: {r.stderr.strip()}")


def frozen_commit(log_path) -> str | None:
    p = Path(log_path)
    if not p.exists():
        return None
    first = p.read_text().split("\n", 1)[0].strip()
    if not first.startswith(HEADER_PREFIX):
        raise RuntimeError(f"{p}: first line must be '{HEADER_PREFIX}<hash>'")
    return first[len(HEADER_PREFIX):].strip()


def read_log(log_path) -> pd.DataFrame:
    """Logged runs (data rows only)."""
    rows = pd.read_csv(log_path, comment="#", dtype={"run_id": str, "git_hash": str,
                                                    "decision": str, "reason": str})
    if list(rows.columns) != LOG_COLUMNS:
        raise RuntimeError(f"{log_path}: unexpected columns {list(rows.columns)}")
    return rows


def init_log(log_path, commit: str) -> None:
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as fh:
        fh.write(f"{HEADER_PREFIX}{commit}\n")
        csv.writer(fh).writerow(LOG_COLUMNS)


def current_best(rows: pd.DataFrame) -> dict | None:
    acc = rows[rows["decision"] == "accepted"]
    return None if acc.empty else acc.iloc[-1].to_dict()


def _append_row(log_path, row: dict) -> None:
    with Path(log_path).open("a", newline="") as fh:
        csv.writer(fh).writerow([row[c] for c in LOG_COLUMNS])


# ---------------------------------------------------------------- running


def _assert_before_test(index, what: str) -> None:
    dd = delivery_day(pd.DatetimeIndex(index))
    if len(dd) and (dd >= TEST_START).any():
        raise ValueError(
            f"{what} include delivery days in the test period (>= {TEST_PERIOD[0]}), "
            f"e.g. {dd[dd >= TEST_START][0].date()}"
        )


def run_config(cfg: dict, data_path="data/dk1_hourly.parquet", log_path="experiments/log.csv",
               check_freeze=True) -> dict:
    """Run one experiment config over all folds, log it and return the result."""
    cfg = validate_config(cfg)
    log_path = Path(log_path)

    # --- log guards (before any heavy work)
    commit = frozen_commit(log_path)
    if check_freeze:
        if commit is None:
            raise RuntimeError(f"{log_path}: no frozen_eval_commit header; evaluation is not frozen")
        if not _frozen_files_unchanged(commit):
            raise RuntimeError(
                f"{' / '.join(Path(f).name for f in FROZEN_FILES)} differ from frozen commit "
                f"{commit}; evaluation and cutoff are frozen"
            )
    if commit is None:
        init_log(log_path, "UNFROZEN")
        rows = read_log(log_path)
    else:
        rows = read_log(log_path)
    if len(rows) >= BUDGET:
        raise RuntimeError(f"run budget exhausted: log already has {len(rows)} runs (max {BUDGET})")
    if (rows["run_id"] == cfg["run_id"]).any():
        raise ValueError(f"run_id {cfg['run_id']!r} already logged")

    # --- fold layout and test-period guard (by delivery day)
    folds = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in FOLDS]
    last_day = max(b for _, b in folds)
    days = pd.date_range(TRAIN_START, last_day, freq="D")
    if last_day.normalize() >= TEST_START:
        raise ValueError(f"validation folds reach into the test period (>= {TEST_PERIOD[0]})")

    # --- data and features (history restricted to pre-test delivery days)
    df = pd.read_parquet(data_path).sort_index()
    df = df[np.asarray(delivery_day(df.index) < TEST_START)]
    feat_cols = list(dict.fromkeys(cfg["feature_set"] + MASK_FEATURES))
    X_all, _ = build_features(df, days, feat_cols)
    y_all = df["actual_total"].reindex(X_all.index)
    mcols = _model_columns(cfg["feature_set"])
    common = y_all.notna().to_numpy() & X_all[MASK_FEATURES].notna().all(axis=1).to_numpy()

    fold_res, preds = [], []
    for i, (a, b) in enumerate(folds, start=1):
        tr = period_mask(X_all.index, TRAIN_START, a - pd.Timedelta(days=1))
        te = period_mask(X_all.index, a, b) & common
        Xtr, ytr = complete_rows(X_all.loc[tr, mcols], y_all)
        Xte, yte = X_all.loc[te, mcols], y_all[te]
        _assert_before_test(Xtr.index, f"fold {i} training hours")
        _assert_before_test(Xte.index, f"fold {i} scored hours")
        if len(Xtr) == 0 or len(Xte) == 0:
            raise ValueError(f"fold {i}: no training ({len(Xtr)}) or scored ({len(Xte)}) rows")
        w = recency_weights(Xtr.index, cfg["half_life_days"])
        model = make_model(cfg).fit(Xtr, ytr, sample_weight=w)
        p = model.predict(Xte)
        if not p.index.equals(Xte.index) or p[QCOLS].isna().any().any():
            n_nan = int(p[QCOLS].isna().any(axis=1).sum())
            raise ValueError(f"fold {i}: model predicted NaN on {n_nan} of {len(Xte)} scored hours")
        s = score(yte, p)
        if s["n_hours"] != len(Xte):
            raise RuntimeError(f"fold {i}: scored {s['n_hours']} hours, expected {len(Xte)}")
        fold_res.append(dict(fold=i, start=str(a.date()), end=str(b.date()),
                             n_train=len(Xtr), **s))
        preds.append(p.assign(fold=i, actual_total=yte))

    def mean(k):
        return float(np.mean([f[k] for f in fold_res]))

    result = dict(
        run_id=cfg["run_id"], model=cfg["model"], config=cfg, folds=fold_res,
        **{f"crps_f{f['fold']}": float(f["crps"]) for f in fold_res},
        crps_mean=mean("crps"), mae_p50_mean=mean("mae_p50"),
        coverage_mean=mean("coverage_80"), bias_mean=mean("bias"),
        n_hours=int(sum(f["n_hours"] for f in fold_res)),
        predictions=pd.concat(preds)[["fold", "actual_total", *QCOLS]],
    )
    best = current_best(rows)
    result["decision"], result["reason"] = decide(result, best)

    row = dict(
        run_id=cfg["run_id"],
        timestamp=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        git_hash=git_hash(), hypothesis=cfg["hypothesis"], change=cfg["change"],
        model=cfg["model"],
        **{k: f"{result[k]:.12g}" for k in ("crps_f1", "crps_f2", "crps_f3", "crps_f4",
                                            "crps_mean", "mae_p50_mean", "coverage_mean",
                                            "bias_mean")},
        n_hours=result["n_hours"], decision=result["decision"], reason=result["reason"],
    )
    _append_row(log_path, row)
    result["log_row"] = row
    return result


def main(argv=None) -> int:
    import yaml

    ap = argparse.ArgumentParser(prog="python -m windfc.run", description=__doc__.split("\n")[0])
    ap.add_argument("config", help="experiment config YAML")
    ap.add_argument("--data", default="data/dk1_hourly.parquet")
    ap.add_argument("--log", default="experiments/log.csv")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    res = run_config(cfg, data_path=args.data, log_path=args.log)
    for f in res["folds"]:
        print(f"fold {f['fold']} {f['start']}..{f['end']}: crps={f['crps']:.2f} "
              f"mae_p50={f['mae_p50']:.2f} cov80={f['coverage_80']:.3f} n={f['n_hours']}")
    print(f"{res['run_id']}: crps_mean={res['crps_mean']:.2f} mae_p50_mean={res['mae_p50_mean']:.2f} "
          f"coverage_mean={res['coverage_mean']:.3f} -> {res['decision']} ({res['reason']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
