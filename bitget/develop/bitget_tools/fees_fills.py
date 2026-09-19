"""
fills_extractor.py - Extract historical fill data from Bitget for fee analysis.

Block 1: Exchange data extraction via /api/v2/mix/order/fill-history

Fetches both open and close fills within a configurable date range.
Displays a summary with open fees, close fees, and combined total.
"""

import sys
import time
from datetime import datetime, timezone, timedelta

sys.path.insert(0, "/home/javi/projects/quant/quant_b/bitget/BOT_trading/execution/brokers")
sys.path.insert(0, "/home/javi/projects/quant/quant_b/bitget/BOT_trading/config/utils")

from bitget_client import BitgetClient
from connect_pass import BITGET_API_KEY_E1, BITGET_API_SECRET_E1, BITGET_API_PASS_E1

# =============================================================================
# CONFIGURATION
# =============================================================================

ACCOUNT       = "E1"
TARGET_START  = "2026-05-29"  # Filter by fill cTime FROM (YYYY-MM-DD, inclusive)
TARGET_END    = "2026-06-05"  # Filter by fill cTime TO   (YYYY-MM-DD, inclusive)
PRODUCT_TYPE  = "USDT-FUTURES"
PAGE_LIMIT    = 100
REQUEST_DELAY = 0.15          # seconds between requests (rate limit: 10/s)
WEEK_DAYS     = 7             # API max time span per request

# =============================================================================
# HELPERS
# =============================================================================

def _split_into_weekly_windows(start_date: str, end_date: str) -> list[tuple]:
    """Split a date range into weekly windows (API max span is 7 days)."""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt   = datetime.strptime(end_date,   "%Y-%m-%d").replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )
    windows = []
    current = start_dt

    while current <= end_dt:
        window_end = min(
            current + timedelta(days=WEEK_DAYS - 1, hours=23, minutes=59, seconds=59),
            end_dt
        )
        windows.append((int(current.timestamp() * 1000), int(window_end.timestamp() * 1000)))
        current = window_end + timedelta(seconds=1)

    return windows


def _is_open_fill(trade_side: str) -> bool:
    """Check if a fill is an opening fill (hedge mode)."""
    return trade_side == "open"


def _is_close_fill(trade_side: str) -> bool:
    """Check if a fill is a closing fill (hedge mode)."""
    return trade_side == "close"


def _parse_fill(raw: dict) -> dict:
    """Normalize raw API fill into a clean dict."""
    fee_detail = raw.get("feeDetail", [])
    total_fee  = sum(abs(float(f.get("totalFee") or 0)) for f in fee_detail)

    return {
        "trade_id":   raw.get("tradeId"),
        "order_id":   raw.get("orderId"),
        "symbol":     raw.get("symbol"),
        "side":       raw.get("side"),
        "trade_side": raw.get("tradeSide"),
        "price":      float(raw.get("price") or 0),
        "size":       float(raw.get("baseVolume") or 0),
        "quote_vol":  float(raw.get("quoteVolume") or 0),
        "profit":     float(raw.get("profit") or 0),
        "fee":        total_fee,
        "fill_time":  datetime.fromtimestamp(int(raw["cTime"]) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if raw.get("cTime") else None,
    }


# =============================================================================
# EXTRACTION
# =============================================================================

def fetch_fills_for_range(client: BitgetClient, start_date: str, end_date: str) -> tuple[list[dict], list[dict]]:
    """
    Fetch all fills within [start_date, end_date], split into open and close.

    Splits the range into weekly windows (API constraint) and paginates each.

    Args:
        client:     Authenticated BitgetClient instance
        start_date: 'YYYY-MM-DD' (inclusive)
        end_date:   'YYYY-MM-DD' (inclusive)

    Returns:
        Tuple of (open_fills, close_fills)
    """
    windows     = _split_into_weekly_windows(start_date, end_date)
    open_fills  = []
    close_fills = []
    total_pages = 0

    print(f"\n{'=' * 60}")
    print(f"  Account  : {ACCOUNT}")
    print(f"  Range    : {start_date} -> {end_date} (inclusive)")
    print(f"  Windows  : {len(windows)} weekly API calls")
    print(f"{'=' * 60}")

    for w_idx, (start_ms, end_ms) in enumerate(windows, 1):
        w_start = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        w_end   = datetime.fromtimestamp(end_ms   / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        print(f"\n  Window {w_idx}/{len(windows)}: {w_start} -> {w_end}")

        id_less_than = None
        page         = 0

        while True:
            page        += 1
            total_pages += 1

            params = {
                "productType": PRODUCT_TYPE,
                "startTime":   str(start_ms),
                "endTime":     str(end_ms),
                "limit":       str(PAGE_LIMIT),
            }
            if id_less_than:
                params["idLessThan"] = id_less_than

            code, resp = client.send_request("GET", "/api/v2/mix/order/fill-history", params=params)

            if code != 200 or resp.get("code") != "00000":
                print(f"    [ERROR] Page {page}: HTTP {code} - {resp}")
                break

            data     = resp.get("data", {})
            raw_list = data.get("fillList", [])
            end_id   = data.get("endId")

            w_open  = [_parse_fill(r) for r in raw_list if _is_open_fill(r.get("tradeSide", ""))]
            w_close = [_parse_fill(r) for r in raw_list if _is_close_fill(r.get("tradeSide", ""))]
            open_fills.extend(w_open)
            close_fills.extend(w_close)

            print(f"    Page {page:>2} - {len(raw_list):>3} total | {len(w_open):>3} open | {len(w_close):>3} close")

            if len(raw_list) < PAGE_LIMIT or not end_id:
                break

            id_less_than = end_id
            time.sleep(REQUEST_DELAY)

    print(f"\n  Total API pages : {total_pages}")
    print(f"  Open fills      : {len(open_fills)}")
    print(f"  Close fills     : {len(close_fills)}")

    return open_fills, close_fills


# =============================================================================
# DISPLAY
# =============================================================================

def print_fills_summary(open_fills: list[dict], close_fills: list[dict]) -> None:
    """Print fee summary split by open/close and combined total."""
    open_fee    = sum(f["fee"]       for f in open_fills)
    close_fee   = sum(f["fee"]       for f in close_fills)
    open_quote  = sum(f["quote_vol"] for f in open_fills)
    close_quote = sum(f["quote_vol"] for f in close_fills)
    total_profit= sum(f["profit"]    for f in close_fills)

    print(f"\n{'=' * 60}")
    print(f"  FEE SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'':30} {'Fills':>6} {'Notional':>12} {'Fee':>10}")
    print(f"  {'─' * 58}")
    print(f"  {'Open fills':<30} {len(open_fills):>6} {open_quote:>12.2f} {open_fee:>10.4f}")
    print(f"  {'Close fills':<30} {len(close_fills):>6} {close_quote:>12.2f} {close_fee:>10.4f}")
    print(f"  {'─' * 58}")
    print(f"  {'TOTAL fees':<30} {len(open_fills)+len(close_fills):>6} {open_quote+close_quote:>12.2f} {open_fee+close_fee:>10.4f}")
    print(f"  {'─' * 58}")
    print(f"  {'Net profit (close fills)':<30} {'':>6} {'':>12} {total_profit:>+10.4f}")
    print(f"{'=' * 60}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    client = BitgetClient(
        api_key        = BITGET_API_KEY_E1,
        api_secret     = BITGET_API_SECRET_E1,
        api_passphrase = BITGET_API_PASS_E1,
    )

    open_fills, close_fills = fetch_fills_for_range(client, TARGET_START, TARGET_END)
    print_fills_summary(open_fills, close_fills)


if __name__ == "__main__":
    main()