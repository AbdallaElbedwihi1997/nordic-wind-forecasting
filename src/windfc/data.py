"""Load raw Energi Data Service CSVs into one hourly DK1 frame."""
from pathlib import Path

import pandas as pd

_READ_KW = dict(sep=";", decimal=",", encoding="utf-8-sig")

_ON = ["OnshoreWindLt50kW_MWh", "OnshoreWindGe50kW_MWh"]
_OFF = ["OffshoreWindLt100MW_MWh", "OffshoreWindGe100MW_MWh"]
_TYPES = {"Onshore Wind": "on", "Offshore Wind": "off"}

COLUMNS = [
    "actual_on", "actual_off", "actual_total",
    "da_on", "da_off", "da_total", "cur_on", "cur_off",
]


def _load_actuals(settlement_csv: Path) -> pd.DataFrame:
    s = pd.read_csv(settlement_csv, **_READ_KW, parse_dates=["HourUTC"])
    s = s[s["PriceArea"] == "DK1"].set_index("HourUTC")
    out = pd.DataFrame(index=s.index)
    # min_count=1: a component may be legitimately absent; all-missing -> NaN
    out["actual_on"] = s[_ON].sum(axis=1, min_count=1)
    out["actual_off"] = s[_OFF].sum(axis=1, min_count=1)
    out["actual_total"] = out["actual_on"] + out["actual_off"]
    return out


def _load_forecasts(forecasts_csv: Path) -> pd.DataFrame:
    f = pd.read_csv(forecasts_csv, **_READ_KW, parse_dates=["HourUTC"])
    f = f[(f["PriceArea"] == "DK1") & f["ForecastType"].isin(_TYPES)]
    f = f.assign(kind=f["ForecastType"].map(_TYPES))
    wide = f.pivot(
        index="HourUTC", columns="kind",
        values=["ForecastDayAhead", "ForecastCurrent"],
    ).reindex(columns=pd.MultiIndex.from_product(
        [["ForecastDayAhead", "ForecastCurrent"], ["on", "off"]]))
    out = pd.DataFrame(index=wide.index)
    out["da_on"] = wide[("ForecastDayAhead", "on")]
    out["da_off"] = wide[("ForecastDayAhead", "off")]
    out["da_total"] = out[["da_on", "da_off"]].sum(axis=1, min_count=2)
    out["cur_on"] = wide[("ForecastCurrent", "on")]
    out["cur_off"] = wide[("ForecastCurrent", "off")]
    return out


def load_raw(settlement_csv: Path, forecasts_csv: Path) -> pd.DataFrame:
    """Contiguous hourly DK1 frame indexed by tz-naive UTC ``hour_utc``.

    Union of both sources' hours; gaps in either source are NaN.
    """
    df = _load_actuals(Path(settlement_csv)).join(
        _load_forecasts(Path(forecasts_csv)), how="outer")
    # contiguous hourly index; hours missing from a source stay NaN (never 0)
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="h"))
    df = df[COLUMNS].astype("float64")
    df.index.name = "hour_utc"
    return df
