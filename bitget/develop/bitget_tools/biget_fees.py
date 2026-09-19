#develop/bitget_tools/bitget_fees.py
"""
fees_extractor.py - Extract historical position data from Bitget for fee analysis.

Block 1: Exchange data extraction via /api/v2/mix/position/history-position

Filters by open_time (position open date), not close date.
The API query window is extended backwards to capture positions opened before
the target range but closed within it.
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

ACCOUNT        = "E1"
TARGET_START   = "2026-06-01"  # Filter by open_time FROM (YYYY-MM-DD, inclusive)
TARGET_END     = "2026-06-08"  # Filter by open_time TO   (YYYY-MM-DD, inclusive)
QUERY_LOOKBACK = 30            # Extra days to extend API query backwards
PRODUCT_TYPE   = "USDT-FUTURES"
PAGE_LIMIT     = 100
REQUEST_DELAY  = 0.15          # seconds between requests (rate limit: 20/s)

# =============================================================================
# HELPERS
# =============================================================================

def _range_to_timestamps(start_date: str, end_date: str):
    """
    Convert date strings to API query timestamps.

    Query window is extended QUERY_LOOKBACK days before start_date to ensure
    positions opened before the target range but closed within it are retrieved,
    then filtered locally by open_time.

    Returns:
        (query_start_ms, query_end_ms, filter_start_dt, filter_end_dt)
    """
    filter_start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    filter_end   = datetime.strptime(end_date,   "%Y-%m-%d").replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )
    query_start  = filter_start - timedelta(days=QUERY_LOOKBACK)
    query_end    = filter_end

    return (
        int(query_start.timestamp() * 1000),
        int(query_end.timestamp()   * 1000),
        filter_start,
        filter_end,
    )


def _parse_position(raw: dict) -> dict:
    """Normalize raw API position into a clean dict."""
    return {
        "position_id":     raw.get("positionId"),
        "symbol":          raw.get("symbol"),
        "hold_side":       raw.get("holdSide"),
        "open_avg_price":  float(raw.get("openAvgPrice") or 0),
        "close_avg_price": float(raw.get("closeAvgPrice") or 0),
        "open_total_pos":  float(raw.get("openTotalPos") or 0),
        "close_total_pos": float(raw.get("closeTotalPos") or 0),
        "pnl":             float(raw.get("pnl") or 0),
        "net_profit":      float(raw.get("netProfit") or 0),
        "open_fee":        float(raw.get("openFee") or 0),
        "close_fee":       float(raw.get("closeFee") or 0),
        "total_fee":       float(raw.get("openFee") or 0) + float(raw.get("closeFee") or 0),
        "total_funding":   float(raw.get("totalFunding") or 0),
        "margin_mode":     raw.get("marginMode"),
        "open_time":       datetime.fromtimestamp(int(raw["ctime"]) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if raw.get("ctime") else None,
        "close_time":      datetime.fromtimestamp(int(raw["utime"]) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if raw.get("utime") else None,
    }


# =============================================================================
# EXTRACTION
# =============================================================================

def fetch_positions_for_range(client: BitgetClient, start_date: str, end_date: str) -> list[dict]:
    """
    Fetch closed positions filtered by open_time within [start_date, end_date].

    The API query window is extended backwards by QUERY_LOOKBACK days to capture
    positions that opened before start_date. Local filtering by open_time is applied
    after retrieval.

    Args:
        client:     Authenticated BitgetClient instance
        start_date: Filter start date 'YYYY-MM-DD' (inclusive, based on open_time)
        end_date:   Filter end date   'YYYY-MM-DD' (inclusive, based on open_time)

    Returns:
        List of normalized position dicts filtered by open_time
    """
    query_start_ms, query_end_ms, filter_start, filter_end = _range_to_timestamps(start_date, end_date)

    all_positions = []
    id_less_than  = None
    page          = 0

    print(f"\n{'=' * 60}")
    print(f"  Account  : {ACCOUNT}")
    print(f"  Filter   : open_time {start_date} -> {end_date} (inclusive)")
    print(f"  API query: {(filter_start - timedelta(days=QUERY_LOOKBACK)).strftime('%Y-%m-%d')} -> {end_date} (extended window)")
    print(f"{'=' * 60}")

    while True:
        page += 1
        params = {
            "productType": PRODUCT_TYPE,
            "startTime":   str(query_start_ms),
            "endTime":     str(query_end_ms),
            "limit":       str(PAGE_LIMIT),
        }
        if id_less_than:
            params["idLessThan"] = id_less_than

        code, resp = client.send_request("GET", "/api/v2/mix/position/history-position", params=params)

        if code != 200 or resp.get("code") != "00000":
            print(f"  [ERROR] Page {page}: HTTP {code} - {resp}")
            break

        data     = resp.get("data", {})
        raw_list = data.get("list", [])
        end_id   = data.get("endId")

        print(f"  Page {page:>2} - {len(raw_list):>3} records from API")

        for raw in raw_list:
            all_positions.append(_parse_position(raw))

        if len(raw_list) < PAGE_LIMIT or not end_id:
            break

        id_less_than = end_id
        time.sleep(REQUEST_DELAY)

    # Filter locally by open_time
    filtered = [
        p for p in all_positions
        if p["open_time"] and filter_start
        <= datetime.strptime(p["open_time"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        <= filter_end
    ]

    print(f"\n  Total fetched : {len(all_positions)}")
    print(f"  After filter  : {len(filtered)} (open_time within {start_date} -> {end_date})")

    return filtered


# =============================================================================
# DISPLAY
# =============================================================================

def print_positions_summary(positions: list[dict]) -> None:
    """Print a summary of extracted positions."""
    if not positions:
        print("\n  No positions found for the selected period.\n")
        return

    total_pnl       = sum(p["pnl"]          for p in positions)
    total_net       = sum(p["net_profit"]    for p in positions)
    total_open_fee  = sum(p["open_fee"]      for p in positions)
    total_close_fee = sum(p["close_fee"]     for p in positions)
    total_fee       = sum(p["total_fee"]     for p in positions)
    total_funding   = sum(p["total_funding"] for p in positions)

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY - {len(positions)} positions")
    print(f"{'=' * 60}")
    print(f"  {'PnL (gross)':<24}: {total_pnl:>+10.4f} USDT")
    print(f"  {'Net profit':<24}: {total_net:>+10.4f} USDT")
    print(f"  {'Open fees':<24}: {total_open_fee:>10.4f} USDT")
    print(f"  {'Close fees':<24}: {total_close_fee:>10.4f} USDT")
    print(f"  {'Total fees':<24}: {total_fee:>10.4f} USDT")
    print(f"  {'Total funding':<24}: {total_funding:>10.4f} USDT")
    print(f"{'=' * 60}")




# =============================================================================
# BLOCK 2: RECONCILIATION WITH POSTGRESQL
# =============================================================================

def reconcile_with_postgres(positions: list[dict]) -> None:
    """
    Compare global totals between exchange (history-position) and PostgreSQL
    for the configured date range.

    Exchange source : history-position filtered by open_time
    PostgreSQL source: trades table filtered by open_at

    Args:
        positions: List of exchange positions from fetch_positions_for_range()
    """
    import psycopg2
    import psycopg2.extras

    sys.path.insert(0, "/home/javi/projects/quant/quant_b/bitget/BOT_trading")
    from config.settings import POSTGRES_CONFIG

    print(f"\n{'=' * 60}")
    print(f"  BLOCK 2 — RECONCILIATION: Exchange vs PostgreSQL")
    print(f"  Period : {TARGET_START} -> {TARGET_END}")
    print(f"{'=' * 60}")

    # -------------------------------------------------------------------------
    # Exchange totals (already computed from positions)
    # -------------------------------------------------------------------------
    ex_positions  = len(positions)
    ex_pnl_gross  = sum(p["pnl"]          for p in positions)
    ex_net_profit = sum(p["net_profit"]    for p in positions)
    ex_open_fee   = sum(p["open_fee"]      for p in positions)
    ex_close_fee  = sum(p["close_fee"]     for p in positions)
    ex_total_fee  = sum(p["total_fee"]     for p in positions)
    ex_funding    = sum(p["total_funding"] for p in positions)

    # -------------------------------------------------------------------------
    # PostgreSQL totals
    # -------------------------------------------------------------------------
    try:
        conn   = psycopg2.connect(**POSTGRES_CONFIG)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cursor.execute("""
            SELECT
                COUNT(*)                  AS num_trades,
                COALESCE(SUM(profit), 0)  AS total_profit,
                COALESCE(SUM(fee),    0)  AS total_fee
            FROM trades
            WHERE account  = %s
              AND open_at >= %s
              AND open_at <= %s
        """, (ACCOUNT, TARGET_START, TARGET_END + " 23:59:59"))

        row = dict(cursor.fetchone())
        cursor.close()
        conn.close()

        pg_trades      = int(row["num_trades"])
        pg_net_profit  = float(row["total_profit"])
        pg_total_fee   = float(row["total_fee"])

    except Exception as e:
        print(f"  [ERROR] Could not fetch PostgreSQL data: {e}")
        return

    # -------------------------------------------------------------------------
    # Comparison table
    # -------------------------------------------------------------------------
    ex_total_fee_abs = abs(ex_total_fee)
    pg_total_fee_abs = abs(pg_total_fee)
    fee_diff         = pg_total_fee_abs - ex_total_fee_abs
    profit_diff      = pg_net_profit    - ex_net_profit

    print(f"\n  {'Metric':<30} {'Exchange':>12} {'PostgreSQL':>12} {'Diff':>10}")
    print(f"  {'─' * 68}")
    print(f"  {'Records':<30} {ex_positions:>12} {pg_trades:>12} {pg_trades - ex_positions:>+10}")
    print(f"  {'─' * 68}")
    print(f"  {'PnL gross':<30} {ex_pnl_gross:>+12.2f} {'N/A':>12} {'':>10}")
    print(f"  {'Net profit':<30} {ex_net_profit:>+12.2f} {pg_net_profit:>+12.2f} {profit_diff:>+10.2f}")
    print(f"  {'─' * 68}")
    print(f"  {'Open fees':<30} {abs(ex_open_fee):>12.2f} {'N/A':>12} {'':>10}")
    print(f"  {'Close fees':<30} {abs(ex_close_fee):>12.2f} {'N/A':>12} {'':>10}")
    print(f"  {'Total fees':<30} {ex_total_fee_abs:>12.2f} {pg_total_fee_abs:>12.2f} {fee_diff:>+10.2f}")
    print(f"  {'Funding':<30} {ex_funding:>12.2f} {'N/A':>12} {'':>10}")
    print(f"  {'─' * 68}")
    print(f"  {'Net profit + funding (ex)':<30} {ex_net_profit + ex_funding:>+12.2f} {'':>12} {'':>10}")
    print(f"{'=' * 60}\n")


# =============================================================================
# BLOCK 3: FEE RATE ANALYSIS
# =============================================================================

FEE_RATE_ALTERNATIVE = 0.04  # % — change this to simulate a different fee tier

def analyze_fee_rate(positions: list[dict]) -> None:
    """
    Calculate the real fee rate from exchange data and simulate an alternative tier.

    Notional is computed from history-position fields:
        open_notional  = open_avg_price  * open_total_pos
        close_notional = close_avg_price * close_total_pos

    Args:
        positions: List of exchange positions from fetch_positions_for_range()
    """
    open_notional  = sum(p["open_avg_price"]  * p["open_total_pos"]  for p in positions)
    close_notional = sum(p["close_avg_price"] * p["close_total_pos"] for p in positions)
    total_notional = open_notional + close_notional

    total_open_fee  = abs(sum(p["open_fee"]  for p in positions))
    total_close_fee = abs(sum(p["close_fee"] for p in positions))
    total_fee       = total_open_fee + total_close_fee

    # Real fee rate (open and close separately, then combined)
    real_rate_open  = (total_open_fee  / open_notional  * 100) if open_notional  > 0 else 0
    real_rate_close = (total_close_fee / close_notional * 100) if close_notional > 0 else 0
    real_rate_total = (total_fee       / total_notional * 100) if total_notional > 0 else 0

    # Alternative fee simulation
    alt_open_fee  = open_notional  * FEE_RATE_ALTERNATIVE / 100
    alt_close_fee = close_notional * FEE_RATE_ALTERNATIVE / 100
    alt_total_fee = alt_open_fee + alt_close_fee
    saving        = total_fee - alt_total_fee

    print(f"\n{'=' * 60}")
    print(f"  BLOCK 3 — FEE RATE ANALYSIS")
    print(f"  Period : {TARGET_START} -> {TARGET_END}")
    print(f"{'=' * 60}")
    print(f"\n  {'Notional':<30}")
    print(f"  {'─' * 50}")
    print(f"  {'Open notional':<30}: {open_notional:>12.2f} USDT")
    print(f"  {'Close notional':<30}: {close_notional:>12.2f} USDT")
    print(f"  {'Total notional':<30}: {total_notional:>12.2f} USDT")
    print(f"\n  {'Real fee rate (from exchange)':<30}")
    print(f"  {'─' * 50}")
    print(f"  {'Open fee rate':<30}: {real_rate_open:>11.4f} %")
    print(f"  {'Close fee rate':<30}: {real_rate_close:>11.4f} %")
    print(f"  {'Combined fee rate':<30}: {real_rate_total:>11.4f} %")
    print(f"\n  {'Fee simulation':<30} {'Current':>10} {'Alt {:.2f}%'.format(FEE_RATE_ALTERNATIVE):>12} {'Saving':>10}")
    print(f"  {'─' * 50}")
    print(f"  {'Open fees':<30} {total_open_fee:>10.2f} {alt_open_fee:>12.2f} {total_open_fee - alt_open_fee:>+10.2f}")
    print(f"  {'Close fees':<30} {total_close_fee:>10.2f} {alt_close_fee:>12.2f} {total_close_fee - alt_close_fee:>+10.2f}")
    print(f"  {'Total fees':<30} {total_fee:>10.2f} {alt_total_fee:>12.2f} {saving:>+10.2f}")
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

    positions = fetch_positions_for_range(client, TARGET_START, TARGET_END)
    print_positions_summary(positions)
    reconcile_with_postgres(positions)
    analyze_fee_rate(positions)


if __name__ == "__main__":
    main()