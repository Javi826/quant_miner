# bitget/broker_client/broker_api/api_client.py
# -----------------------------
import logging
import sys
import os
import time

import pandas as pd
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from broker_client.broker_config import BASE_URL, PRODUCT_TYPE, API_TIMEOUT, API_MAX_RETRIES

logger = logging.getLogger("shared.market_data.api_client")


# ---------------- HTTP ----------------

def _http_get(url: str, params: dict | None = None) -> requests.Response:
    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, timeout=API_TIMEOUT)
            if r.status_code in (429, 502, 503, 504) or r.status_code >= 500:
                time.sleep(0.5 * attempt)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException:
            time.sleep(0.5 * attempt)
    raise Exception("Max retries exceeded")


# ---------------- CANDLES ----------------

def _call_history_candles(
    symbol: str,
    granularity: str,
    limit: int = 200,
    startTime: int | None = None,
    endTime: int | None = None,
) -> list:
    url    = f"{BASE_URL}/api/v2/mix/market/history-candles"
    params = {
        "symbol":      symbol,
        "granularity": granularity,
        "limit":       limit,
        "productType": PRODUCT_TYPE,
    }
    if startTime is not None:
        params["startTime"] = str(int(startTime))
    if endTime is not None:
        params["endTime"] = str(int(endTime))
    try:
        r = _http_get(url, params=params)
        j = r.json()
        if isinstance(j, dict) and j.get("code") not in (None, "00000"):
            return []
        data = j.get("data") if isinstance(j, dict) else j
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.error(f"API error (symbol={symbol} start={startTime} end={endTime}): {e}")
        return []


def to_dataframe_from_api(data: list) -> pd.DataFrame:
    if not data:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume_base", "volume_quote"])
    clean = []
    for row in data:
        if not row or len(row) < 7:
            continue
        try:
            clean.append([int(row[0]), row[1], row[2], row[3], row[4], row[5], row[6]])
        except Exception:
            continue
    df = pd.DataFrame(clean, columns=["timestamp", "open", "high", "low", "close", "volume_base", "volume_quote"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


# ---------------- SYMBOLS ----------------

def get_futures_symbols_from_api(product_type: str = PRODUCT_TYPE) -> list[str]:
    url = f"{BASE_URL}/api/v2/mix/market/contracts"
    try:
        r    = _http_get(url, params={"productType": product_type})
        data = r.json().get("data") or []
        symbols = []
        for item in data:
            s = item.get("symbol") or item.get("contract") or item.get("symbolName")
            if s:
                symbols.append(str(s))
        return sorted(set(symbols))
    except Exception as e:
        logger.error(f"Error fetching symbols: {e}")
        return []
