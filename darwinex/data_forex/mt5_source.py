import logging
import sys
import os
from datetime import datetime, timedelta

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from broker_client.broker_api.mt5_client import (
    get_client,
    mt5_timeframe,
    sync_server_offset,
    drop_forming_bar,
)
from broker_client.broker_config import TIMEFRAME_MINUTES

logger = logging.getLogger("pipeline.mt5_source")

# =============================================================================
# CONFIG
# =============================================================================

CHUNK_MONTHS_BY_TIMEFRAME = {
    "1m" : 1,
    "5m" : 3,
    "15m": 12,
}
CHUNK_MONTHS_DEFAULT = 12

# Forex trades ~120h/week, 52 weeks/year.
TRADING_HOURS_PER_YEAR = 6240
DENSITY_WARN_RATIO     = 0.90
OUTPUT_COLUMNS = ["open", "high", "low", "close", "volume"]

# =============================================================================
# CONNECTION
# =============================================================================
def _validate_timeframe(timeframe: str) -> str:
    """Raises on any label outside TIMEFRAME_MINUTES."""
    if timeframe not in TIMEFRAME_MINUTES:
        raise ValueError(
            f"Unsupported timeframe: {timeframe!r}. "
            f"Expected one of {list(TIMEFRAME_MINUTES)}"
        )
    return timeframe

# =============================================================================
# FETCH
# =============================================================================
def _fetch_chunk(client, symbol: str, tf_constant: int, start: datetime, end: datetime) -> pd.DataFrame:
    """Single copy_rates_range() call, normalised to an OHLCV frame."""
    rates = client.copy_rates_range(symbol, tf_constant, start, end)

    if rates is None or len(rates) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["timestamp"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={"tick_volume": "volume"})
    return df.set_index("timestamp")[OUTPUT_COLUMNS]


def fetch_bars(symbol: str, timeframe: str, start_date: str, end_date: str | None = None) -> pd.DataFrame:

    timeframe = _validate_timeframe(timeframe)
    client    = get_client()

    if not client.symbol_select(symbol, True):
        logger.warning(f"  ⚠ [{symbol}] symbol_select() failed — not available on this account")
        return pd.DataFrame()

    tf_constant  = mt5_timeframe(timeframe)
    chunk_months = CHUNK_MONTHS_BY_TIMEFRAME.get(timeframe, CHUNK_MONTHS_DEFAULT)

    start = pd.to_datetime(start_date).to_pydatetime()
    end   = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).to_pydatetime() \
            if end_date else datetime.now() + timedelta(days=2)

    frames = []
    cursor = start
    while cursor < end:
        chunk_end = min((pd.Timestamp(cursor) + pd.DateOffset(months=chunk_months)).to_pydatetime(), end)
        chunk     = _fetch_chunk(client, symbol, tf_constant, cursor, chunk_end)
        if chunk.empty:
            logger.debug(f"    [{symbol}] {cursor:%Y-%m} → {chunk_end:%Y-%m}: no bars")
        else:
            frames.append(chunk)
        cursor = chunk_end

    if not frames:
        logger.warning(f"  ⚠ [{symbol}] No {timeframe} bars returned for {start:%Y-%m-%d} → {end:%Y-%m-%d}")
        return pd.DataFrame()

    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="first")].sort_index()

    sync_server_offset([symbol])
    return drop_forming_bar(df, timeframe)


# =============================================================================
# VALIDATION
# =============================================================================
def expected_bars_per_year(timeframe: str) -> int:
    """Theoretical bar count for a full trading year."""
    return int(TRADING_HOURS_PER_YEAR * 60 / TIMEFRAME_MINUTES[_validate_timeframe(timeframe)])


def validate_history(df: pd.DataFrame, symbol: str, timeframe: str, requested_start: str) -> list[str]:

    issues: list[str] = []

    if df.empty:
        issues.append(f"[{symbol} {timeframe}] empty history")
        return issues

    timeframe = _validate_timeframe(timeframe)
    requested = pd.to_datetime(requested_start)
    first_bar = df.index.min()

    # Left-side truncation: history starts well after what was asked for.
    if first_bar > requested + pd.Timedelta(days=30):
        issues.append(
            f"[{symbol} {timeframe}] history starts {first_bar:%Y-%m-%d}, "
            f"requested {requested:%Y-%m-%d} — check 'Max bars in chart' is Unlimited"
        )

    # Thin years: skip the first and last, which are legitimately partial.
    expected   = expected_bars_per_year(timeframe)
    per_year   = df.groupby(df.index.year).size()
    full_years = per_year.iloc[1:-1] if len(per_year) > 2 else per_year.iloc[0:0]

    for year, count in full_years.items():
        if count < expected * DENSITY_WARN_RATIO:
            pct = count / expected * 100
            issues.append(f"[{symbol} {timeframe}] {year}: {count} bars ({pct:.0f}% of expected {expected})")

    return issues


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    _symbol, _timeframe, _start = "EURGBP", "15m", "2000-01-01"
    _df = fetch_bars(_symbol, _timeframe, _start)

    print(f"\n{_symbol} {_timeframe}: {len(_df)} bars | {_df.index.min()} → {_df.index.max()}")
    print(_df.head())

    for _issue in validate_history(_df, _symbol, _timeframe, _start):
        print(f"  ⚠ {_issue}")