"""Download the raw Energi Data Service datasets for DK1 into data/raw/.

Usage: python scripts/download_data.py [--start 2022-01-01]

Writes ';'-separated, ','-decimal, UTF-8-with-BOM CSVs (the format Energi
Data Service's own export produces), so they feed scripts/build_dataset.py.
"""
import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

API = "https://api.energidataservice.dk/dataset/"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

DATASETS = {
    "ProductionConsumptionSettlement": "ProductionConsumptionSettlement.csv",
    "Forecasts_Hour": "Forecasts_Hour.csv",
}


def fetch(name: str, start: str) -> pd.DataFrame:
    query = urllib.parse.urlencode({
        "start": start,
        "filter": json.dumps({"PriceArea": ["DK1"]}, separators=(",", ":")),
        "sort": "HourUTC asc",
        "limit": 0,  # 0 = no row limit
    })
    with urllib.request.urlopen(f"{API}{name}?{query}", timeout=300) as resp:
        records = json.load(resp)["records"]
    df = pd.DataFrame.from_records(records)
    for col in ("HourUTC", "HourDK"):  # API returns ISO 'T' timestamps
        if col in df:
            df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", default="2022-01-01")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    for name, fname in DATASETS.items():
        df = fetch(name, args.start)
        path = RAW / fname
        df.to_csv(path, sep=";", decimal=",", index=False, encoding="utf-8-sig")
        print(f"{name}: {len(df)} rows -> {path}")


if __name__ == "__main__":
    main()
