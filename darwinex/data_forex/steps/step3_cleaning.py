import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger("pipeline.step3")

OHLC_COLS = ["open", "high", "low", "close"]
VOLUME_COLS = ["volume"]


def _clean_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    initial_rows = len(df)

    if "timestamp" in df.columns:
        n_dupes = df.duplicated(subset=["timestamp"], keep="first").sum()
        if n_dupes > 0:
            logger.info(f"  🗑 [{symbol}] Dropped {n_dupes} duplicate rows on 'timestamp'")
            df = df.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)

    for col in OHLC_COLS + VOLUME_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    ohlc_present = [c for c in OHLC_COLS if c in df.columns]
    if ohlc_present:
        drop_mask = df[ohlc_present].isna().any(axis=1) | (df[ohlc_present] == 0).any(axis=1)
        n_dropped = drop_mask.sum()
        if n_dropped > 0:
            logger.info(f"  🗑 [{symbol}] Dropped {n_dropped} rows with NaN/zero OHLC")
            df = df[~drop_mask].reset_index(drop=True)

    for col in VOLUME_COLS:
        if col not in df.columns:
            continue
        df[col] = df[col].replace(0, np.nan)
        n_fixed = df[col].isna().sum()
        if n_fixed > 0:
            df[col] = df[col].ffill()
            if df[col].isna().any():
                first_valid = df[col].dropna().iloc[0] if not df[col].dropna().empty else 0.0
                df[col] = df[col].fillna(first_valid)
            logger.info(f"  🔧 [{symbol}] ffill applied to '{col}': {n_fixed} values fixed")

    logger.debug(f"  [{symbol}] Rows: {initial_rows} → {len(df)}")
    return df


def run(config: dict) -> bool:
    input_dir: str = config["raw_dir"]
    output_dir: str = config["clean_dir"]
    timeframe: str = config.get("timeframe", "")
    export_csv: bool = config.get("export_csv", False)
    os.makedirs(output_dir, exist_ok=True)

    selected_symbols = config.get("selected_symbols") or []
    files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".parquet")
        and (not timeframe or f.endswith(f"_{timeframe}.parquet"))
        and (not selected_symbols or any(f.startswith(s) for s in selected_symbols))
    ]) if os.path.exists(input_dir) else []

    if not files:
        logger.warning(f"⚠ No parquet files found in {input_dir}")
        return False

    logger.info(f"🧹 Cleaning — {len(files)} file(s)")

    errors = 0
    for filepath in files:
        filename = os.path.basename(filepath)
        symbol = os.path.splitext(filename)[0].rsplit("_", 1)[0]
        try:
            df = pd.read_parquet(filepath)
        except Exception as e:
            logger.warning(f"  ❌ Could not read {filename}: {e}")
            errors += 1
            continue

        df = _clean_symbol(df, symbol)
        df.to_parquet(os.path.join(output_dir, filename), index=False)
        if export_csv:
            df.to_csv(os.path.join(output_dir, os.path.splitext(filename)[0] + ".csv"), index=False)
        csv_note = " + .csv" if export_csv else ""
        logger.info(f"  💾 [{symbol}] Saved {len(df)} rows → {filename}{csv_note}")

    if errors:
        logger.warning(f"⚠ Cleaning completed with {errors} error(s)")
        return False

    logger.info("✅ Cleaning complete")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _base = os.path.dirname(os.path.abspath(__file__))
    _config = {
        "raw_dir": os.path.join(_base, "..", "data_fx", "01_raw"),
        "clean_dir": os.path.join(_base, "..", "data_fx", "02_clean"),
        "export_csv": False,
    }
    run(_config)