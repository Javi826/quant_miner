#develop/bitget_tools/bitget_close_all.py
"""
bitget_close_all.py - Close all open positions for a Bitget account.
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

# =============================================================================
# CONFIGURATION
# =============================================================================

ACCOUNT      = "00"           # "E1", "00", "01"
PRODUCT_TYPE = "USDT-FUTURES"
REQUEST_DELAY = 1.0           # seconds between close requests (rate limit: 1 req/s)

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

def close_all_positions(client: BitgetClient) -> None:
    """
    Fetch all open positions and close them one by one.

    Args:
        client: Authenticated BitgetClient instance
    """
    code, resp = client.send_request(
        "GET",
        "/api/v2/mix/position/all-position",
        params={"productType": PRODUCT_TYPE, "marginCoin": "USDT"}
    )

    if code != 200 or resp.get("code") != "00000":
        print(f"❌ Error fetching positions: {resp}")
        return

    positions = [p for p in resp["data"] if float(p["total"]) > 0]

    if not positions:
        print("ℹ️ No open positions to close.")
        return

    print(f"\n🚀 Closing {len(positions)} open positions...\n")

    for pos in positions:
        body = {
            "symbol":      pos["symbol"],
            "productType": PRODUCT_TYPE,
        }

        close_code, close_resp = client.send_request(
            "POST", "/api/v2/mix/order/close-positions", body=body
        )

        if close_code == 200 and close_resp.get("code") == "00000":
            print(f"💰 FLASH CLOSE executed: {pos['symbol']}")
        else:
            print(f"❌ Failed to close {pos['symbol']}: {close_resp}")

        time.sleep(REQUEST_DELAY)

    print("\n✅ Process completed.\n")


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

    close_all_positions(client)


if __name__ == "__main__":
    main()