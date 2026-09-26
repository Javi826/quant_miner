# # data_pipeline/steps/step0_symbol_selection.py
# =============================================================================
# Step 0 — Symbol Selection — selects top N symbols by average volume
# over the last N_LOOKBACK candles of the reference timeframe.
# Only runs when SYMBOL_MODE = "auto" in data_main.py.
# =============================================================================
import logging
import os
import time
import requests
import pandas as pd

from broker_client.broker_config import BASE_URL, PRODUCT_TYPE, API_TIMEOUT
from broker_client.broker_api.api_client import _call_history_candles, get_futures_symbols_from_api

logger = logging.getLogger("pipeline.step0")

# =============================================================================
# CONSTANTS
# =============================================================================
SLEEP_BETWEEN_REQUESTS = 0.06
N_LOOKBACK             = 180     # Number of candles to compute average volume
TIMEFRAME_SYMBOL_SEL   = "1D" # Reference timeframe for volume ranking — always daily


# =============================================================================
# CORE
# =============================================================================

def _fetch_avg_volume(symbol: str) -> float | None:
    """Fetches last N_LOOKBACK candles and returns average volume_quote."""
    try:
        data = _call_history_candles(symbol, TIMEFRAME_SYMBOL_SEL, limit=N_LOOKBACK)
        time.sleep(SLEEP_BETWEEN_REQUESTS)
        if not data:
            return None
        volumes = []
        for row in data:
            try:
                volumes.append(float(row[6]))
            except Exception:
                continue
        return sum(volumes) / len(volumes) if volumes else None
    except Exception as e:
        logger.debug(f"  ⚠ [{symbol}] Error fetching volume: {e}")
        return None

def _get_rwa_symbols() -> set[str]:
    """Returns set of RWA symbols from /contracts endpoint."""
    url = f"{BASE_URL}/api/v2/mix/market/contracts"
    try:
        r    = requests.get(url, params={"productType": PRODUCT_TYPE}, timeout=API_TIMEOUT)
        data = r.json().get("data") or []
        return {item["symbol"] for item in data if item.get("isRwa") == "YES"}
    except Exception as e:
        logger.warning(f"⚠ Could not fetch RWA symbols: {e}")
        return set()
    
def select_symbols(config: dict) -> list[str]:

    symbol_mode      = config.get("symbol_mode", "manual")
    selected_symbols = config.get("selected_symbols", [])
    n_symbols        = config.get("n_symbols_download", 50)
    output_dir: str  = config["raw_dir"]

    if symbol_mode == "manual":
        logger.info(f"📋 Symbol mode: MANUAL — {len(selected_symbols)} symbol(s): {selected_symbols}")
        return selected_symbols

    # AUTO mode
    logger.info(f"🔎 Symbol mode: AUTO — fetching top {n_symbols} symbols by avg volume [{TIMEFRAME_SYMBOL_SEL}]")
    
    all_symbols  = get_futures_symbols_from_api()
    rwa_mode     = config.get("rwa_mode", "crypto_only")
    rwa_symbols  = _get_rwa_symbols()
    if rwa_mode == "crypto_only":
        all_symbols = [s for s in all_symbols if s not in rwa_symbols]
    elif rwa_mode == "rwa_only":
        all_symbols = [s for s in all_symbols if s in rwa_symbols]
    logger.info(f"  RWA filter [{rwa_mode}]: {len(all_symbols)} symbols after filter")

    if not all_symbols:
        logger.warning("⚠ No symbols retrieved from API. Falling back to SELECTED_SYMBOLS.")
        return selected_symbols

    logger.info(f"  Fetching volume for {len(all_symbols)} symbols...")

    volume_data = []
    for i, sym in enumerate(all_symbols, start=1):
        avg_vol = _fetch_avg_volume(sym)
        if avg_vol is not None:
            volume_data.append({"symbol": sym, "avg_volume_usdt": avg_vol})
        if i % 50 == 0:
            logger.info(f"  Progress: {i}/{len(all_symbols)}")

    if not volume_data:
        logger.warning("⚠ Could not fetch volume data. Falling back to SELECTED_SYMBOLS.")
        return selected_symbols

    df          = pd.DataFrame(volume_data).sort_values("avg_volume_usdt", ascending=False)
    top_symbols = df.head(n_symbols)["symbol"].tolist()

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(os.path.dirname(output_dir), "selected_symbols.csv")
    df.head(n_symbols).to_csv(csv_path, index=False)
    logger.info(f"  💾 Symbol selection saved → {os.path.basename(csv_path)}")
    logger.info(f"  ✅ Top {n_symbols} symbols selected by avg volume [{TIMEFRAME_SYMBOL_SEL}]")

    return top_symbols


def run(config: dict) -> bool:
    symbols = select_symbols(config)
    if not symbols:
        logger.warning("⚠ No symbols selected. Aborting.")
        return False
    config["selected_symbols"] = symbols
    return True


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _base = os.path.dirname(os.path.abspath(__file__))
    _config = {
        "symbol_mode":        "auto",
        "selected_symbols":   ["BTCUSDT", "ETHUSDT"],
        "n_symbols_download": 10,
        "raw_dir":            os.path.join(_base, "data", "01_raw"),
    }
    run(_config)