#develop/bitget_tools/bitget_leverage.py
"""
bitget_leverage.py - Set and verify leverage for all futures symbols on a Bitget account.
"""

import sys
import time

sys.path.insert(0, "/home/javi/projects/quant/quant_b/bitget/BOT_trading/execution/brokers")
sys.path.insert(0, "/home/javi/projects/quant/quant_b/bitget/BOT_trading/config/utils")

from bitget_client import BitgetClient
from connect_pass import (
    BITGET_API_KEY_E1,  BITGET_API_SECRET_E1,  BITGET_API_PASS_E1,
    BITGET_API_KEY_00,  BITGET_API_SECRET_00,  BITGET_API_PASS_00,
    BITGET_API_KEY_01,  BITGET_API_SECRET_01,  BITGET_API_PASS_01,
)

sys.path.insert(0, "/home/javi/projects/quant/quant_b/bitget/BOT_trading")
from shared.shared_trading_data.broker_api.api_client import get_futures_symbols_from_api

# =============================================================================
# CONFIGURATION
# =============================================================================

ACCOUNT         = "E1"        # "E1", "00", "01"
LEVERAGE_TARGET = 10
MARGIN_COIN     = "USDT"
MARGIN_MODE     = "crossed"   # "isolated" or "crossed"
PRODUCT_TYPE    = "USDT-FUTURES"
REQUEST_DELAY   = 0.25        # seconds between requests

# =============================================================================
# ACCOUNT CREDENTIALS MAP
# =============================================================================

_CREDENTIALS = {
    "E1": (BITGET_API_KEY_E1, BITGET_API_SECRET_E1, BITGET_API_PASS_E1),
    "00": (BITGET_API_KEY_00, BITGET_API_SECRET_00, BITGET_API_PASS_00),
    "01": (BITGET_API_KEY_01, BITGET_API_SECRET_01, BITGET_API_PASS_01),
}

# =============================================================================
# FUNCTIONS
# =============================================================================

def set_leverage(client: BitgetClient, symbol: str) -> bool:
    """
    Set leverage for a symbol.

    Args:
        client: Authenticated BitgetClient instance
        symbol: Trading symbol (e.g., 'BTCUSDT')

    Returns:
        True if successful, False otherwise
    """
    body = {
        "symbol":      symbol,
        "productType": PRODUCT_TYPE,
        "marginCoin":  MARGIN_COIN,
    }

    if MARGIN_MODE.lower() == "isolated":
        body["longLeverage"]  = str(LEVERAGE_TARGET)
        body["shortLeverage"] = str(LEVERAGE_TARGET)
    else:
        body["leverage"] = str(LEVERAGE_TARGET)

    code, resp = client.send_request("POST", "/api/v2/mix/account/set-leverage", body=body)

    if code == 200 and resp.get("code") == "00000":
        print(f"✅ {symbol}: Leverage set to {LEVERAGE_TARGET}x")
        return True
    else:
        print(f"⚠️ {symbol}: Error setting leverage: {resp}")
        return False


def get_leverage(client: BitgetClient, symbol: str) -> dict | None:
    """
    Retrieve current leverage for a symbol.

    Args:
        client: Authenticated BitgetClient instance
        symbol: Trading symbol

    Returns:
        Account data dict or None on error
    """
    params = {
        "symbol":      symbol,
        "productType": PRODUCT_TYPE,
        "marginCoin":  MARGIN_COIN,
    }

    code, resp = client.send_request("GET", "/api/v2/mix/account/account", params=params)

    if code == 200 and resp.get("code") == "00000":
        data      = resp.get("data", {})
        long_lev  = data.get("longLeverage")
        short_lev = data.get("shortLeverage")
        print(f"🔍 {symbol}: Long={long_lev}x | Short={short_lev}x")
        return data
    else:
        print(f"⚠️ {symbol}: Error fetching leverage: {resp}")
        return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    if ACCOUNT not in _CREDENTIALS:
        print(f"⚠️ Unknown account '{ACCOUNT}'. Valid options: {list(_CREDENTIALS.keys())}")
        return

    api_key, api_secret, api_passphrase = _CREDENTIALS[ACCOUNT]

    client = BitgetClient(
        api_key        = api_key,
        api_secret     = api_secret,
        api_passphrase = api_passphrase,
    )

    print(f"\n📂 Fetching available symbols for account {ACCOUNT}...")
    all_symbols = get_futures_symbols_from_api(PRODUCT_TYPE)

    if not all_symbols:
        print("⚠️ No symbols found.")
        return

    print(f"   {len(all_symbols)} symbols found.")
    print(f"\n🚀 Setting leverage to {LEVERAGE_TARGET}x on {len(all_symbols)} symbols...\n")

    results = {}

    for sym in all_symbols:
        ok = set_leverage(client, sym)
        time.sleep(REQUEST_DELAY)

        if ok:
            data = get_leverage(client, sym)
            if data:
                results[sym] = {
                    "long":  data.get("longLeverage"),
                    "short": data.get("shortLeverage"),
                }
        time.sleep(REQUEST_DELAY)

    print("\n📊 FINAL SUMMARY:\n")
    for sym, vals in results.items():
        print(f"  {sym:<16}: Long={vals['long']}x | Short={vals['short']}x")

    print("\n✅ Process completed.\n")


if __name__ == "__main__":
    main()