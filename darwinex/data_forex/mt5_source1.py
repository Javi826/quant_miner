import os
import glob
import pandas as pd

# =============================================================================
# CONFIG — edit this path manually
# =============================================================================
IS_DIR = "/home/javi/projects/quant/quant_miner/darwinex/data_forex/data_fx/04_split/IS/fx_2017-01_2025-01_IS"

# =============================================================================
# READ RANGE
# =============================================================================
def _read_range(path: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns:
        df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df["timestamp"].min(), df["timestamp"].max()

# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    files = sorted(glob.glob(os.path.join(IS_DIR, "*.parquet")))

    if not files:
        print(f"⚠ No parquet files found in {IS_DIR}")
        return

    print(f"\n{'='*60}")
    print(f"  IS — start/end date range ({len(files)} files)")
    print(f"{'='*60}")

    start_dates = []
    end_dates = []

    for path in files:
        key = os.path.splitext(os.path.basename(path))[0]
        start, end = _read_range(path)
        start_dates.append(start)
        end_dates.append(end)
        print(f"  {key}: {start} → {end}")

    print(f"{'='*60}")
    print(f"  Overall start → min: {min(start_dates)} | max: {max(start_dates)}")
    print(f"  Overall end   → min: {min(end_dates)} | max: {max(end_dates)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()