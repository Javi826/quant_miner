import os
import sys
from collections import Counter

import pandas as pd

_BITGET = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _BITGET not in sys.path:
    sys.path.insert(0, _BITGET)

from broker_client.broker_api.api_client import _call_history_candles, to_dataframe_from_api

# =============================================================================
# CONFIGURATION
# =============================================================================
SYMBOLS = ["BTCUSDT"]

N_BARS       = 8
H1_LIMIT     = 200
DIFF_TOL_PCT = 0.01

# Each entry: granularity -> (bar_size, offset from 00:00 UTC)
SOURCES = {
    "6Hutc":  (pd.Timedelta(hours=6),  pd.Timedelta(hours=0)),
    "6H":     (pd.Timedelta(hours=6),  pd.Timedelta(hours=4)),
    "12Hutc": (pd.Timedelta(hours=12), pd.Timedelta(hours=0)),
    "12H":    (pd.Timedelta(hours=12), pd.Timedelta(hours=4)),
}

H1_SIZE    = pd.Timedelta(hours=1)
PRICE_COLS = ["open", "high", "low", "close"]
PRICES_W   = 43

# =============================================================================
# DATA
# =============================================================================

def fetch(symbol: str, granularity: str, limit: int) -> pd.DataFrame:
    df = to_dataframe_from_api(_call_history_candles(symbol, granularity, limit=limit))
    if df.empty:
        return df
    df[PRICE_COLS] = df[PRICE_COLS].astype(float)
    return df.set_index("timestamp")[PRICE_COLS]


def rebuild_from_1h(df_1h: pd.DataFrame, bar_size: pd.Timedelta, offset: pd.Timedelta, now: pd.Timestamp) -> pd.DataFrame:
    h1_per_bar = int(bar_size / H1_SIZE)
    closed     = df_1h[df_1h.index + H1_SIZE <= now]
    grouped    = closed.resample(bar_size, origin="epoch", offset=offset, label="left", closed="left")
    bars       = grouped.agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    counts     = grouped["close"].count()
    return bars[counts == h1_per_bar]

# =============================================================================
# CHECKS
# =============================================================================

def diff_pct(row: pd.Series, ref: pd.Series) -> float:
    return float(((row - ref).abs() / ref).max() * 100)


def bar_status(ts: pd.Timestamp, bar_size: pd.Timedelta, row: pd.Series, ref: pd.Series | None, now: pd.Timestamp) -> str:
    if ts + bar_size > now:
        return "OPEN"
    if row["open"] == row["high"] == row["low"] == row["close"]:
        return "FLAT"
    if ref is None:
        return "NO_1H"
    return "DIFF" if diff_pct(row, ref) > DIFF_TOL_PCT else "ok"


def fmt_prices(row: pd.Series) -> str:
    return " ".join(f"{row[c]:>10.6g}" for c in PRICE_COLS)


def check_source(symbol: str, source: str, bar_size: pd.Timedelta, offset: pd.Timedelta, df_1h: pd.DataFrame, now: pd.Timestamp) -> Counter:
    native  = fetch(symbol, source, N_BARS)
    rebuilt = rebuild_from_1h(df_1h, bar_size, offset, now)
    counts  = Counter()

    print(f"\n  {symbol} | {source}")
    if native.empty:
        print("  no data (empty response or API error)")
        counts["NO_DATA"] += 1
        return counts

    print(f"  {'BAR_OPEN_UTC':<16} {'STATUS':<6} {'NATIVE  O / H / L / C':<{PRICES_W}}   {'FROM 1H  O / H / L / C':<{PRICES_W}} {'DIFF%':>7}")
    for ts, row in native.iterrows():
        ref    = rebuilt.loc[ts] if ts in rebuilt.index else None
        status = bar_status(ts, bar_size, row, ref, now)
        counts[status] += 1

        ref_str  = fmt_prices(ref) if ref is not None else f"{'-':<{PRICES_W}}"
        diff_str = f"{diff_pct(row, ref):>7.3f}" if ref is not None else f"{'-':>7}"
        print(f"  {ts:%Y-%m-%d %H:%M} {status:<6} {fmt_prices(row)}   {ref_str} {diff_str}")

    return counts

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    now     = pd.Timestamp.now(tz="UTC")
    summary = {source: Counter() for source in SOURCES}

    print(f"\n{'=' * 130}")
    print(f"  CANDLE SOURCE CHECK — now {now:%Y-%m-%d %H:%M:%S} UTC | tolerance {DIFF_TOL_PCT}%")
    print(f"{'=' * 130}")

    for symbol in SYMBOLS:
        df_1h = fetch(symbol, "1H", H1_LIMIT)
        if df_1h.empty:
            print(f"\n  {symbol} | 1H: no data, skipped")
            continue
        for source, (bar_size, offset) in SOURCES.items():
            summary[source] += check_source(symbol, source, bar_size, offset, df_1h, now)

    print(f"\n{'=' * 130}")
    print("  SUMMARY")
    print(f"{'=' * 130}")
    for source, counts in summary.items():
        detail = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"  {source:<6}: {detail}")
    print(f"{'=' * 130}\n")