import logging
import os
import re

import pandas as pd

from shared.ftp_darwinex import get_asset_files

logger = logging.getLogger("pipeline.step1")

# =============================================================================
# UTILITIES
# =============================================================================

def sanitize_filename(name: str) -> str:
    return re.sub(r'[^\w\-_\. ]', '_', name).strip()


def _ticks_to_timebars(folder_path: str, resample_factor: str) -> pd.DataFrame | None:
    if not os.path.exists(folder_path):
        return None

    files = [
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(".gz")
    ]
    if not files:
        return None

    dfs = []
    for file in files:
        dft = pd.read_csv(
            os.path.join(folder_path, file),
            compression="gzip",
            header=None,
            names=["timestamp", "price", "volume"],
        )
        dfs.append(dft)

    df = pd.concat([d for d in dfs if not d.empty], axis=0)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["price"] = df["price"].astype(float)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.set_index("timestamp").sort_index()

    ticks = df.dropna()
    price = ticks.price.resample(resample_factor)

    bars = pd.DataFrame({
        "open": price.first(),
        "high": price.max(),
        "low": price.min(),
        "close": price.last(),
        "volume": ticks.volume.resample(resample_factor).sum(),
    })

    return bars.dropna()


def _get_sorted_folders(root_dir: str) -> list[str]:
    folders = []
    if not os.path.exists(root_dir):
        return []

    for year in sorted(os.listdir(root_dir)):
        year_path = os.path.join(root_dir, year)
        if os.path.isdir(year_path) and year.isdigit():
            for month in sorted(os.listdir(year_path)):
                month_path = os.path.join(year_path, month)
                if os.path.isdir(month_path) and month.isdigit():
                    folders.append(os.path.join(year, month))
    return folders


def _get_bars(ticks_symbol_dir: str, resample_factor: str) -> pd.DataFrame:
    folders = [os.path.join(ticks_symbol_dir, path) for path in _get_sorted_folders(ticks_symbol_dir)]

    bars_dfs = []
    for folder in folders:
        bars = _ticks_to_timebars(folder, resample_factor)
        if bars is not None:
            bars_dfs.append(bars)

    if not bars_dfs:
        return pd.DataFrame()

    df = pd.concat(bars_dfs, axis=0).sort_index()
    df.index.name = "timestamp"
    return df

# =============================================================================
# SYMBOL PROCESSOR
# =============================================================================

def _process_symbol(symbol: str, config: dict) -> None:
    start_date = config["start_date"]
    end_date = config["end_date"]
    timeframe = config["timeframe"]
    side = config.get("side", "BID")
    ticks_dir = config["ticks_dir"]
    output_dir = config["raw_dir"]
    export_csv = config.get("export_csv", False)

    logger.info(f"  📥 [{symbol}] Downloading ticks ({side}) {start_date} → {end_date}")
    get_asset_files(symbol, start_date, end_date, side, ticks_dir)

    logger.info(f"  📊 [{symbol}] Resampling ticks → {timeframe}")
    df = _get_bars(os.path.join(ticks_dir, symbol), timeframe)

    if df.empty:
        logger.warning(f"  ❌ [{symbol}] No bars generated. Skipping.")
        return

    df = df[(df.index >= start_date) & (df.index <= end_date)]

    parquet_path = os.path.join(output_dir, sanitize_filename(f"{symbol}_{timeframe}.parquet"))
    df.reset_index().to_parquet(parquet_path, index=False)

    if export_csv:
        csv_path = os.path.join(output_dir, sanitize_filename(f"{symbol}_{timeframe}.csv"))
        df.reset_index().to_csv(csv_path, index=False)

    logger.info(f"  💾 [{symbol}] {len(df)} bars → {os.path.basename(parquet_path)}")

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

    for i, symbol in enumerate(selected_symbols, start=1):
        logger.info(f"\n[{i}/{len(selected_symbols)}] {symbol}")
        try:
            _process_symbol(symbol, config)
        except Exception as e:
            logger.warning(f"  ❌ [{symbol}] Failed: {e}")

    return True

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _base = os.path.dirname(os.path.abspath(__file__))
    _config = {
        "start_date": "2025-01-01",
        "end_date": None,
        "timeframe": "4h",
        "selected_symbols": ["EURUSD"],
        "side": "BID",
        "ticks_dir": os.path.join(_base, "data_fx", "00_ticks"),
        "raw_dir": os.path.join(_base, "data_fx", "01_raw"),
        "export_csv": False,
    }
    run(_config)