# DK1 Day-Ahead Wind Forecasting — Design (v1)

Date: 2026-11-06 · Status: awaiting review

## 1. Purpose

A public portfolio project supporting an application for Axpo Nordic's *Lead Forecasting* role (Oslo). It must demonstrate, honestly:

- wind power forecasting for a Nordic price area,
- probabilistic forecasting (quantiles) and combining an internal model with a vendor forecast,
- forecast-quality KPIs, root-cause diagnosis of vendor error, and statistically tested improvement,
- disciplined, auditable model optimization.

Success = a repo a forecasting lead can audit in 10 minutes and find no leakage, no cherry-picked test period, and a clearly separated, significance-tested skill gain over the vendor. Size of the gain is secondary to its credibility.

Timeline: v1 ready within days so the application is not delayed. Phase B (weather-based model) comes after applying and is out of scope here.

## 2. Data

Source: Energi Data Service (Energinet), open data, credited in README.

| Dataset | Use | Coverage in hand |
|---|---|---|
| `ProductionConsumptionSettlement` | Actuals: DK1 onshore (`OnshoreWindLt50kW_MWh` + `OnshoreWindGe50kW_MWh`) and offshore (`OffshoreWindLt100MW_MWh` + `OffshoreWindGe100MW_MWh`) | 2021-12-31 23:00 → 2026-10-28 23:00 UTC, hourly |
| `Forecasts_Hour` | Vendor forecasts per technology: `ForecastDayAhead`, `ForecastCurrent` (+ intraday/5h/1h, which are **excluded** — see §4) | 2021-12-31 23:00 → 2026-11-05 22:00 UTC, hourly |

Files are `;`-separated with `,` decimals. Timestamps used: `HourUTC`.

**Target:** DK1 total wind = onshore + offshore actual (MWh per hour).

**Known data finding (to be documented in notebook 01):** from 2024 onward actual DK1 offshore output exceeds the vendor's offshore forecast structurally (max hourly actual ≈1,931 MWh in Sep 2026 vs. vendor forecast capped near ≈1,100–1,140 MWh, even for `ForecastCurrent`). The vendor forecast does not cover newly commissioned offshore capacity. Day-ahead total-wind bias grew from about −12 MWh (2022) to about −450 MWh (2026). Any improvement must therefore be split into a *coverage* part and a *skill* part.

## 3. Forecast setting

- Issue time: **10:00 UTC on day D−1**; horizon: all 24 hours of day D (UTC hours 00–23).
- Information set at issue time:
  - vendor `ForecastDayAhead` for day D (assumed published before 10:00 UTC D−1 — the file has no publication timestamp; stated as a limitation),
  - actuals and vendor `ForecastCurrent` up to and including hour 09:00 UTC on D−1,
  - calendar features.
- Hence effective lag from latest actual to target hour is 15–38 h.

## 4. Leakage controls

- `ForecastIntraday`, `Forecast5Hour`, `Forecast1Hour` and `ForecastCurrent` **for day D** are never features.
- `cutoff.py` exposes `available_at(issue_time)` and a `check_features(df, issue_time)` that raises if any feature value depends on data timestamped after the cutoff. Every model run goes through it; tests cover it.

## 5. Evaluation protocol (fixed before the first optimization run)

**Split**

| Period | Role |
|---|---|
| 2022-01-01 → 2025-04-30 | Initial training |
| 2025-05-01 → 2026-04-30 | Validation: four 3-month folds (May–Jul 25, Aug–Oct 25, Nov 25–Jan 26, Feb–Apr 26), expanding-window retraining on all data before each fold |
| 2026-05-01 → 2026-10-28 | **Locked test.** Scored once, in notebook 05, after optimization ends. Final model retrained on all data before 2026-05-01 |

The test period contains the largest capacity change; it is kept as is, not swapped for an easier window.

**Forecast output:** 9 quantiles, τ ∈ {0.1, 0.2, …, 0.9}.

**Metrics**
- Primary: CRPS approximated by the mean pinball loss over the 9 quantiles, averaged over hours, then over the 4 folds.
- Guardrails: MAE of the P50; empirical coverage of the P10–P90 interval (target 80%, acceptable 76–84%); mean bias (reported, not optimized).
- All metrics also reported normalized by mean actual.

**Baselines** (all evaluated identically)
1. Persistence: actual at the latest available hour (09:00 UTC D−1) carried forward, with quantiles from its trailing 90-day empirical errors (sanity floor).
2. Vendor raw, probabilistic: vendor DA total + empirical quantiles of its trailing 90-day errors, conditioned on 5 bins of forecast level.
3. Vendor + capacity correction (step 1), with the same empirical error quantiles.
4. Linear quantile regression on step-1-corrected features.

**Significance:** Diebold–Mariano test (with Harvey–Leybourne–Newbold small-sample correction) on daily mean pinball loss, comparing the final model against baseline 2 (total gain) and baseline 3 (skill gain). Report statistic and p-value.

## 6. Models

**Step 1 — capacity correction (offshore only)**
At each issue time, compute `r = Σ actual_offshore / Σ ForecastCurrent_offshore` over the previous 30 days, using only hours where `ForecastCurrent_offshore` > 100 MWh. Corrected DA offshore = `r × ForecastDayAhead_offshore`. Onshore is passed through unchanged. Known weakness: lags capacity step-ups by up to the window length; reported, not modelled.

**Step 2 — probabilistic model**
LightGBM, one model per quantile (objective `quantile`), outputs sorted per hour to prevent quantile crossing.

Base features:
- corrected vendor DA: onshore, offshore, total;
- vendor DA total ramps: Δ1h and Δ3h (both directions, within day D's forecast — known at issue time);
- hour-of-day; day-of-year as sin/cos;
- vendor trailing error stats as of cutoff: 7-day mean error and 7-day MAE of the total DA forecast vs actual.

## 7. Optimization loop

- Executed by Claude in-session, max **20 runs**, each defined by a YAML config and logged to `experiments/log.csv` with: run id, timestamp, hypothesis, change, fold-level CRPS/MAE/coverage, decision, reason.
- Allowed changes (search space fixed now):
  - feature add/remove from a predefined candidate list (e.g. 14/30-day error windows, error stats by hour-of-day, forecast-level × hour interaction, offshore share of total);
  - LightGBM hyperparameters within ranges: `num_leaves` 8–64, `learning_rate` 0.02–0.1, `n_estimators` 100–1500, `min_child_samples` 20–200, `feature_fraction` 0.6–1.0;
  - training window: full history vs. exponential recency weighting (half-life 90–730 days);
  - conformalized quantile regression (CQR) calibration on/off, calibration window 30–180 days;
  - model family: LightGBM vs. linear quantile regression.
- Forbidden changes: evaluation code, split, metrics, test data access.
- Acceptance rule: a change is kept only if mean CRPS improves ≥1% vs. the current best **and** it improves on ≥3 of 4 folds **and** guardrails hold (P50 MAE not worse by >1%; coverage within 76–84%).
- `evaluate.py` and `cutoff.py` are committed before run 1; the commit hash is recorded in the log header. Any later edit to them invalidates the loop.

## 8. Repository layout

```
nordic-wind-forecasting/
├── README.md
├── pyproject.toml
├── data/                         # processed DK1 parquet only; raw CSVs git-ignored
├── scripts/download_data.py      # fetches both datasets from the Energi Data Service API
├── src/windfc/
│   ├── data.py                   # load raw CSVs, build hourly DK1 frame + target
│   ├── cutoff.py                 # issue-time rules, availability, leakage check
│   ├── capacity.py               # step 1
│   ├── models.py                 # baselines, linear QR, LightGBM QR, CQR
│   ├── evaluate.py               # metrics, folds, Diebold–Mariano (frozen)
│   └── run.py                    # `python -m windfc.run experiments/configs/<id>.yaml`
├── experiments/configs/, experiments/log.csv
├── notebooks/01_data_and_vendor_diagnosis.ipynb
│             02_baselines_and_capacity_fix.ipynb
│             03_probabilistic_model.ipynb
│             04_optimization_log.ipynb
│             05_final_test.ipynb
├── tests/
└── docs/superpowers/specs/
```

Notebooks import from `src/windfc`; they contain narrative, plots and calls, not core logic. Committed with outputs.

## 9. Testing

pytest, written before the code it covers:
- `check_features` rejects features with post-cutoff timestamps and accepts valid ones;
- pinball loss and coverage match hand-computed values;
- folds and test period do not overlap and respect the expanding window;
- capacity ratio at issue time uses only data ≤ 09:00 UTC D−1;
- quantile outputs are non-crossing.

No tests for plotting or notebook glue.

## 10. README contents

1. One-paragraph summary and headline table: vendor raw → +capacity correction → +model, on the locked test period (CRPS, MAE, coverage, DM p-values).
2. The vendor coverage finding with one chart.
3. Method and leakage controls in brief.
4. Optimization log summary (runs tried, accepted, rejected).
5. Limitations: no weather data; assumed DA publication time; capacity correction lag; single price area.
6. Data credit to Energinet / Energi Data Service.

## 11. Out of scope (v1)

Weather/NWP features, solar, DK2, autonomous agent script, dashboard, imbalance-price cost analysis.

## 12. Dependencies on the user

- Connect GitHub to Claude and create an empty public repo (`nordic-wind-forecasting` suggested) before publishing.
- Review README and be able to explain every accepted change in the optimization log.
