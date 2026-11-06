"""Build data/dk1_hourly.parquet from the two raw CSVs.

Usage: python scripts/build_dataset.py <settlement.csv> <forecasts.csv>
"""
import sys
from pathlib import Path

from windfc.data import load_raw

OUT = Path(__file__).resolve().parents[1] / "data" / "dk1_hourly.parquet"


def main(settlement_csv: str, forecasts_csv: str) -> None:
    df = load_raw(Path(settlement_csv), Path(forecasts_csv))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT)
    print(f"wrote {OUT}")
    print(f"rows: {len(df)}")
    print(f"range: {df.index.min()} -> {df.index.max()}")
    print("NaN counts:")
    print(df.isna().sum().to_string())


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
