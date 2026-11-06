# DK1 Day-Ahead Wind Forecasting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a public, audit-ready repo that issues probabilistic day-ahead forecasts of DK1 total wind, separates vendor-coverage gain from modelling skill, and optimizes the model in a logged, budgeted loop.

**Architecture:** A small `windfc` Python package holds all logic (data → cutoff rules → capacity correction → features → models → frozen evaluation → config-driven runner). Notebooks only narrate and plot by calling the package. The optimization loop is a sequence of YAML configs run through `windfc.run`, each appended to `experiments/log.csv`.

**Tech Stack:** Python 3.11+, pandas, numpy, pyarrow, lightgbm, scikit-learn (`QuantileRegressor`), scipy, pyyaml, matplotlib, jupyter/nbconvert, pytest.

**Spec:** `docs/superpowers/specs/2026-11-06-dk1-wind-forecasting-design.md`

## Global Constraints

- Price area DK1 only; target `actual_total` = onshore (`OnshoreWindLt50kW_MWh` + `OnshoreWindGe50kW_MWh`) + offshore (`OffshoreWindLt100MW_MWh` + `OffshoreWindGe100MW_MWh`).
- Issue time for target day D = **D−1 10:00 UTC**; latest usable actual / `ForecastCurrent` hour = **D−1 09:00 UTC**.
- Never use `ForecastIntraday`, `Forecast5Hour`, `Forecast1Hour`, or day-D `ForecastCurrent` as features.
- Quantiles `QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)`; prediction columns `q10 … q90`.
- Train start 2022-01-01. Validation folds: 2025-05-01–2025-07-31, 2025-08-01–2025-10-31, 2025-11-01–2026-01-31, 2026-02-01–2026-04-30 (expanding window). Locked test 2026-05-01–2026-10-28, scored once in notebook 05 only.
- Primary metric = mean pinball loss over 9 quantiles (CRPS approximation), averaged over folds. Guardrails: P50 MAE, P10–P90 coverage in [0.76, 0.84], bias reported.
- Acceptance rule: mean CRPS improves ≥1% vs current best, improves on ≥3 of 4 folds, P50 MAE not worse by >1%, coverage in [0.76, 0.84].
- Optimization budget: max 20 runs. `evaluate.py` and `cutoff.py` frozen before run 1; commit hash in log header.
- Capacity ratio: 30-day window, only hours with `cur_off` > 100 MWh.
- Raw CSVs git-ignored; only `data/dk1_hourly.parquet` committed. README credits Energinet / Energi Data Service.

## Review Focus

1. **Missing hours / NaNs** (gaps in either file, the `UNDEFINED` price-area rows, a day with a partial vendor DA forecast) → rows dropped from scoring with a count reported, never silently imputed as 0. Test in Task 1.
2. **Capacity ratio with no qualifying hours** (calm 30-day window, start of history) → falls back to the most recent valid ratio, else 1.0; never NaN/inf. Test in Task 3.
3. **Quantile crossing after CQR or LightGBM** → outputs sorted per row; `q10 ≤ … ≤ q90` always. Test in Task 5.
4. **Negative or impossible predictions** → all quantiles clipped to ≥ 0. Test in Task 5.
5. **Test-period access before notebook 05** → `evaluate.folds()` never returns test dates and `run.py` refuses any config whose dates overlap 2026-05-01 onward. Test in Task 6.

---

### Task 1: Scaffold + data loading

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/windfc/__init__.py`, `src/windfc/data.py`, `scripts/build_dataset.py`, `scripts/download_data.py`, `tests/test_data.py`, `tests/fixtures/` (tiny CSV snippets in the raw format)

**Interfaces:**
- Produces: `load_raw(settlement_csv: Path, forecasts_csv: Path) -> pd.DataFrame` returning an hourly frame indexed by tz-naive UTC `hour_utc` with float columns `actual_on, actual_off, actual_total, da_on, da_off, da_total, cur_on, cur_off`; `QUANTILES` constant in `windfc/__init__.py`.
- `scripts/build_dataset.py <settlement.csv> <forecasts.csv>` writes `data/dk1_hourly.parquet`.
- `scripts/download_data.py` fetches both datasets from `https://api.energidataservice.dk/dataset/<name>` for DK1, 2022-01-01 onward (not runnable in this workspace; documented for users).

- [ ] **Step 1: Write failing tests** in `tests/test_data.py` using fixture CSVs (`;` sep, `,` decimal):
  - `test_target_is_sum_of_four_wind_columns`: `actual_total == actual_on + actual_off` and `actual_on == Lt50kW + Ge50kW` for fixture rows.
  - `test_only_dk1_and_wind_types_kept`: DK2, `UNDEFINED`, `Solar` rows absent.
  - `test_missing_vendor_hour_is_nan_not_zero`: fixture with one missing Onshore DA row → `da_total` NaN for that hour.
  - `test_index_is_unique_hourly_utc`: index unique, monotonic, named `hour_utc`.
- [ ] **Step 2:** `pytest tests/test_data.py -v` → FAIL (module missing).
- [ ] **Step 3: Implement `load_raw`.** Pivot forecasts on `ForecastType` (`Onshore Wind`→`on`, `Offshore Wind`→`off`) for `ForecastDayAhead` and `ForecastCurrent`; `da_total`/`cur_*` sums use `min_count=2` so a missing component gives NaN. Inner-join on `HourUTC`.
- [ ] **Step 4:** `pytest tests/test_data.py -v` → PASS.
- [ ] **Step 5:** Run `python scripts/build_dataset.py /mnt/user-data/uploads/536528ba-ProductionConsumptionSettlement_1.csv /mnt/user-data/uploads/Forecasts_Hour.csv`; expect ~41,800 rows, 2021-12-31 23:00 → 2026-10-28 23:00, printed NaN counts per column.
- [ ] **Step 6: Commit** `feat: data loading and DK1 hourly dataset`.

### Task 2: Cutoff rules and leakage check

**Files:**
- Create: `src/windfc/cutoff.py`, `tests/test_cutoff.py`

**Interfaces:**
- Produces: `issue_time(target_day: pd.Timestamp) -> pd.Timestamp` (D−1 10:00); `last_known_hour(target_day) -> pd.Timestamp` (D−1 09:00); `check_features(feature_asof: pd.DataFrame, target_index: pd.DatetimeIndex) -> None` raising `LeakageError` if any `feature_asof` timestamp for a row is > `last_known_hour(row's day)`. Feature builders return, alongside features, a same-shaped frame of "as-of" timestamps for every lagged feature (vendor DA columns carry as-of = issue time).
- `FORBIDDEN_COLUMNS = {"ForecastIntraday", "Forecast5Hour", "Forecast1Hour"}`; `check_features` also raises if any feature name contains one of these or `cur_` without a `_lag` suffix.

- [ ] **Step 1: Write failing tests:**
  - `test_issue_time`: `issue_time(2026-03-10) == 2026-03-09 10:00`.
  - `test_last_known_hour_lag_range`: for target hours 00:00 and 23:00 of D the lag to `last_known_hour` is 15 h and 38 h.
  - `test_check_features_rejects_future_asof`: as-of at D−1 10:00 → raises `LeakageError`.
  - `test_check_features_accepts_valid`: as-of at D−1 09:00 → no raise.
  - `test_forbidden_column_names_rejected`: feature named `cur_off` → raises.
- [ ] **Step 2:** `pytest tests/test_cutoff.py -v` → FAIL.
- [ ] **Step 3: Implement** the three functions and `LeakageError(Exception)`.
- [ ] **Step 4:** `pytest tests/test_cutoff.py -v` → PASS.
- [ ] **Step 5: Commit** `feat: issue-time rules and leakage check`.

### Task 3: Capacity correction (step 1)

**Files:**
- Create: `src/windfc/capacity.py`, `tests/test_capacity.py`

**Interfaces:**
- Consumes: frame from Task 1; `last_known_hour` from Task 2.
- Produces: `capacity_ratio(df: pd.DataFrame, target_days: pd.DatetimeIndex, window_days: int = 30, min_cur_off: float = 100.0) -> pd.Series` (index = target day, value = ratio, plus as-of = `last_known_hour`); `apply_capacity_correction(df, ratios) -> pd.DataFrame` adding `da_off_adj`, `da_total_adj` (`da_on` unchanged).

- [ ] **Step 1: Write failing tests:**
  - `test_ratio_uses_only_past_window`: synthetic frame where actual_off doubles from D 00:00 onward → ratio for D equals pre-change value (1.0).
  - `test_ratio_threshold_excludes_low_hours`: hours with `cur_off` ≤ 100 excluded from both sums.
  - `test_ratio_fallback_when_no_qualifying_hours`: all `cur_off` = 50 → ratio = last valid, or 1.0 at start; never NaN/inf.
  - `test_correction_scales_offshore_only`: `da_off_adj == ratio * da_off`, `da_on` unchanged.
- [ ] **Step 2:** `pytest tests/test_capacity.py -v` → FAIL.
- [ ] **Step 3: Implement.** Ratio = Σactual_off / Σcur_off over hours in (`last_known_hour` − 30 days, `last_known_hour`] with `cur_off` > threshold; forward-fill missing, initial fill 1.0.
- [ ] **Step 4:** `pytest tests/test_capacity.py -v` → PASS.
- [ ] **Step 5: Commit** `feat: offshore capacity correction`.

### Task 4: Frozen evaluation

**Files:**
- Create: `src/windfc/evaluate.py`, `tests/test_evaluate.py`

**Interfaces:**
- Produces: `FOLDS: list[tuple[str, str]]` (the four validation folds, inclusive); `TEST_PERIOD = ("2026-05-01", "2026-10-28")`; `pinball(y: pd.Series, preds: pd.DataFrame) -> pd.Series` (per-hour mean over quantiles); `score(y, preds) -> dict` with keys `crps, mae_p50, coverage_80, bias, crps_norm, mae_norm, n_hours, n_dropped`; `diebold_mariano(loss_a: pd.Series, loss_b: pd.Series, h: int = 1) -> tuple[float, float]` (HLN-corrected statistic, p-value) on daily mean losses.

- [ ] **Step 1: Write failing tests:**
  - `test_pinball_hand_value`: y=10, all quantiles predict 8 → loss = mean(τ·2) = 1.0.
  - `test_coverage`: 4 hours, 3 inside [q10,q90] → 0.75.
  - `test_score_drops_nan_rows_and_counts_them`: one NaN y → `n_dropped == 1`.
  - `test_folds_do_not_overlap_or_touch_test`: folds disjoint, ordered, all end < 2026-05-01.
  - `test_dm_identical_losses`: identical series → stat 0, p-value 1.0; clearly better series → p < 0.05.
- [ ] **Step 2:** `pytest tests/test_evaluate.py -v` → FAIL.
- [ ] **Step 3: Implement.** DM: d = loss_a − loss_b on daily means; Newey-West variance with lag h−1; HLN factor `sqrt((n+1−2h+h(h−1)/n)/n)`; two-sided t(n−1) p-value.
- [ ] **Step 4:** `pytest tests/test_evaluate.py -v` → PASS.
- [ ] **Step 5: Commit** `feat: frozen evaluation (metrics, folds, Diebold-Mariano)`.

### Task 5: Features and models

**Files:**
- Create: `src/windfc/features.py`, `src/windfc/models.py`, `tests/test_features.py`, `tests/test_models.py`

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces:
  - `build_features(df, target_days, feature_set: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]` (features, as-of timestamps). Base set: `da_on, da_off_adj, da_total_adj, ramp_1h, ramp_3h, hour, doy_sin, doy_cos, err_mean_7d, err_mae_7d`. Candidate extras: `err_mean_14d, err_mae_14d, err_mean_30d, err_mae_30d, err_by_hour_30d, level_x_hour, off_share`. Calls `check_features` before returning.
  - Common model protocol: `fit(X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None) -> self`; `predict(X) -> pd.DataFrame` with columns `q10…q90`, sorted per row, clipped ≥ 0.
  - Classes: `Persistence`, `VendorEmpirical(base_col: str, window_days=90, n_bins=5)` (`base_col` = `da_total` for baseline 2, `da_total_adj` for baseline 3), `LinearQR(alpha=0.0)`, `LightGBMQR(**params)`, and wrapper `CQR(model, calib_days: int)`.
  - `recency_weights(index, half_life_days: float | None) -> pd.Series` (None → all ones).

- [ ] **Step 1: Write failing tests:**
  - `test_build_features_passes_leakage_check` on the real parquet for one fold.
  - `test_err_mean_7d_uses_only_known_hours`: synthetic error that starts at D−1 10:00 → `err_mean_7d` for D is 0.
  - `test_ramps_within_day`: ramp_1h = da_total_adj(t) − da_total_adj(t−1).
  - `test_predictions_non_crossing_and_non_negative` for each model class on synthetic data.
  - `test_vendor_empirical_median_tracks_vendor`: zero-error history → q50 == base column.
  - `test_cqr_widens_undercovering_intervals`: model with too-narrow intervals → coverage after CQR closer to 0.8.
- [ ] **Step 2:** `pytest tests/test_features.py tests/test_models.py -v` → FAIL.
- [ ] **Step 3: Implement features.** Error stats computed on hourly `da_total − actual_total` restricted to hours ≤ `last_known_hour`, then joined to each target hour of D.
- [ ] **Step 4: Implement models.** LightGBM: one `LGBMRegressor(objective="quantile", alpha=τ)` per quantile. `LinearQR`: sklearn `QuantileRegressor(solver="highs")` per quantile on standardized features. CQR: symmetric conformal adjustment of each lower/upper quantile pair from residuals on the last `calib_days` of training data.
- [ ] **Step 5:** run both test files → PASS.
- [ ] **Step 6: Commit** `feat: leakage-checked features and quantile models`.

### Task 6: Runner, configs, log; freeze

**Files:**
- Create: `src/windfc/run.py`, `experiments/configs/000_baseline.yaml`, `experiments/log.csv` (header only), `tests/test_run.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `python -m windfc.run <config.yaml>` and `run_config(cfg: dict) -> dict`. Config keys: `run_id, hypothesis, change, model (persistence|vendor_raw|vendor_capacity|linear_qr|lgbm), params, feature_set, half_life_days, cqr_calib_days`. For each fold: expanding-window retrain on 2022-01-01 → fold start − 1 day, predict fold, score. Appends one row to `log.csv`: `run_id, timestamp, git_hash, hypothesis, change, crps_f1..f4, crps_mean, mae_p50_mean, coverage_mean, decision, reason`. `decide(result: dict, best: dict) -> tuple[str, str]` implements the acceptance rule verbatim from Global Constraints.

- [ ] **Step 1: Write failing tests:**
  - `test_decide_accepts_only_when_all_rules_hold` (table of 5 cases: pass; 0.9% gain; 2/4 folds; MAE +1.5%; coverage 0.86).
  - `test_run_refuses_test_period`: config with any date ≥ 2026-05-01 → `ValueError`.
  - `test_log_row_appended` using a tiny synthetic dataset.
- [ ] **Step 2:** `pytest tests/test_run.py -v` → FAIL.
- [ ] **Step 3: Implement** `run.py`.
- [ ] **Step 4:** full `pytest -v` → all PASS.
- [ ] **Step 5: Freeze.** Commit `chore: freeze evaluation and cutoff before optimization`; record `git rev-parse HEAD` as `# frozen_eval_commit=<hash>` first line of `log.csv`. `run.py` aborts if `git diff <hash> -- src/windfc/evaluate.py src/windfc/cutoff.py` is non-empty.

### Task 7: Baselines and notebooks 01–03

**Files:**
- Create: `experiments/configs/001_vendor_raw.yaml … 005_lgbm_base.yaml`, `notebooks/01_data_and_vendor_diagnosis.ipynb`, `notebooks/02_baselines_and_capacity_fix.ipynb`, `notebooks/03_probabilistic_model.ipynb`

- [ ] **Step 1:** Run configs: persistence, vendor_raw, vendor_capacity, linear_qr (base features), lgbm (base features, default params). These are runs 1–5 of the 20-run budget; the best of `linear_qr`/`lgbm` becomes "current best".
- [ ] **Step 2:** Notebook 01: data coverage, NaN counts, the offshore coverage finding (quarterly mean and max actual offshore vs `cur_off` and `da_off`; bias by year), one chart reused in README.
- [ ] **Step 3:** Notebook 02: baselines table across folds; capacity ratio over time; gain from step 1 alone.
- [ ] **Step 4:** Notebook 03: base model results, reliability plot (nominal vs empirical quantile coverage), example week fan chart.
- [ ] **Step 5:** Execute all three with `jupyter nbconvert --execute --to notebook --inplace`; verify no errors and outputs present.
- [ ] **Step 6: Commit** `feat: baselines and analysis notebooks 01-03`.

### Task 8: Optimization loop (runs 6–20) and notebook 04

**Files:**
- Create: `experiments/configs/006_*.yaml …` (≤15 more), `notebooks/04_optimization_log.ipynb`

- [ ] **Step 1:** For each run: write hypothesis first in the config, change exactly one thing from the allowed search space (spec §7), run, let `decide` record the decision. Stop at 20 total runs or 5 consecutive rejections.
- [ ] **Step 2:** Notebook 04: log table, CRPS trajectory of accepted runs, short explanation of each accepted change.
- [ ] **Step 3:** Execute notebook; commit `feat: optimization runs and log notebook`.

### Task 9: Final test, README, publish

**Files:**
- Create: `notebooks/05_final_test.ipynb`, `README.md`

- [ ] **Step 1:** Notebook 05: retrain the final config, plus baselines 2 and 3, on 2022-01-01 → 2026-04-30; score once on `TEST_PERIOD`; DM tests vs baseline 2 (total gain) and baseline 3 (skill gain). Execute once; do not re-run after seeing results.
- [ ] **Step 2:** README per spec §10, numbers copied from notebook 05 outputs, limitations section verbatim from spec §10.5, Energinet credit.
- [ ] **Step 3:** Fresh-context review of the whole repo for leakage and claim accuracy (separate reviewer), fix findings.
- [ ] **Step 4:** Commit; push to the user's public GitHub repo once connected (`add_repo` with push access).
