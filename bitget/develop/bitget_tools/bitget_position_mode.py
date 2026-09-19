#develop/bitget_tools/bitget_position_mode.py
"""
Z_bitget_position_mode.py - Set position mode for a Bitget account.

Changes position mode (hedge_mode / one_way_mode) for all contracts
of a given product type.
"""

import sys

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

ACCOUNT      = "E1"          # "E1", "00", "01"
TARGET_MODE  = "hedge_mode"  # "hedge_mode" or "one_way_mode"
PRODUCT_TYPE = "USDT-FUTURES"

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

def set_position_mode(client: BitgetClient, product_type: str, mode: str) -> bool:
    """
    Change the position mode for all contracts of a product type.

    Args:
        client:       Authenticated BitgetClient instance
        product_type: Product type (e.g., 'USDT-FUTURES')
        mode:         'hedge_mode' or 'one_way_mode'

    Returns:
        True if successful, False otherwise
    """
    body = {
        "productType": product_type,
        "posMode":     mode,
    }

    code, resp = client.send_request("POST", "/api/v2/mix/account/set-position-mode", body=body)

    if code == 200 and resp.get("code") == "00000":
        print(f"✅ [{product_type}] Position mode changed to '{mode}'.")
        return True
    else:
        print(f"⚠️ [{product_type}] Error changing position mode: {resp}")
        return False


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
    print(f"\n🚀 Setting position mode to '{TARGET_MODE}' for all {PRODUCT_TYPE} contracts...\n")

    set_position_mode(client, PRODUCT_TYPE, TARGET_MODE)

    print("\n✅ Process completed.\n")


if __name__ == "__main__":
    main()