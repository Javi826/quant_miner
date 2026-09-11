import logging
import os
import re
import sys

import pandas as pd

_DATA_FOREX_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _DATA_FOREX_DIR not in sys.path:
    sys.path.insert(0, _DATA_FOREX_DIR)

from mt5_source import fetch_bars, validate_history

logger = logging.getLogger("pipeline.step1")

OUTPUT_COLUMNS = ["open", "high", "low", "close", "volume"]

# =============================================================================
# UTILITIES
# =============================================================================

def sanitize_filename(name: str) -> str:
    return re.sub(r'[^\w\-_\. ]', '_', name).strip()


def _trim_to_range(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date) + pd.Timedelta(days=1)
    return df[(df.index >= start) & (df.index < end)]

# =============================================================================
# SYMBOL PROCESSOR
# =============================================================================

def _process_symbol(symbol: str, config: dict) -> bool:
    start_date = config["start_date"]
    end_date = config["end_date"]
    timeframe = config["timeframe"]
    output_dir = config["raw_dir"]
    export_csv = config.get("export_csv", False)

    logger.info(f"  📥 [{symbol}] Fetching {timeframe} bars {start_date} → {end_date}")
    df = fetch_bars(symbol, timeframe, start_date, end_date)

    if df.empty:
        logger.warning(f"  ❌ [{symbol}] No bars returned. Skipping.")
        return False

    for issue in validate_history(df, symbol, timeframe, start_date):
        logger.warning(f"  ⚠ {issue}")

    df = _trim_to_range(df, start_date, end_date)

    if df.empty:
        logger.warning(f"  ❌ [{symbol}] No bars inside the requested range. Skipping.")
        return False

    df = df[OUTPUT_COLUMNS]
    df.index.name = "timestamp"

    parquet_path = os.path.join(output_dir, sanitize_filename(f"{symbol}_{timeframe}.parquet"))
    df.reset_index().to_parquet(parquet_path, index=False)

    if export_csv:
        csv_path = os.path.join(output_dir, sanitize_filename(f"{symbol}_{timeframe}.csv"))
        df.reset_index().to_csv(csv_path, index=False)

    logger.info(
        f"  💾 [{symbol}] {len(df)} bars "
        f"({df.index.min():%Y-%m-%d} → {df.index.max():%Y-%m-%d}) "
        f"→ {os.path.basename(parquet_path)}"
    )
    return True

# =============================================================================
# RUN
# =============================================================================

def run(config: dict) -> bool:
    selected_symbols = config.get("selected_symbols") or []
    output_dir: str = config["raw_dir"]
    os.makedirs(output_dir, exist_ok=True)

    if not selected_symbols:
        logger.warning("⚠ No symbols selected. Aborting.")
        return False

    if not config.get("end_date"):
        config["end_date"] = pd.Timestamp.today().strftime("%Y-%m-%d")

    logger.info(f"🔁 Extraction [{config['timeframe']}] for {len(selected_symbols)} symbol(s).")

    failed = []
    for i, symbol in enumerate(selected_symbols, start=1):
        logger.info(f"\n[{i}/{len(selected_symbols)}] {symbol}")
        try:
            if not _process_symbol(symbol, config):
                failed.append(symbol)
        except Exception as e:
            logger.warning(f"  ❌ [{symbol}] Failed: {e}")
            failed.append(symbol)

    if failed:
        logger.warning(f"⚠ Extraction completed with {len(failed)} failure(s): {failed}")
        if len(failed) == len(selected_symbols):
            logger.warning("⚠ Every symbol failed — aborting this timeframe.")
            return False

    logger.info("✅ Extraction complete")
    return True

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _base = os.path.dirname(os.path.abspath(__file__))
    _config = {
        "start_date": "2017-01-01",
        "end_date": None,
        "timeframe": "4H",
        "selected_symbols": ["EURGBP"],
        "raw_dir": os.path.join(_base, "..", "data_fx", "01_raw"),
        "export_csv": False,
    }
    run(_config)