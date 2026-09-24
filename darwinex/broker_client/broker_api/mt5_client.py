# darwinex/broker_client/broker_api/mt5_client.py
import logging
import sys
import os
import time

import pandas as pd
from mt5linux import MetaTrader5

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from broker_client.broker_config import (
    MT5_HOST,
    MT5_PORT,
    MT5_LOGIN,
    MT5_PASSWORD,
    MT5_SERVER,
    TIMEFRAME_MINUTES,
    SERVER_OFFSET_STEP_SECONDS,
    SERVER_OFFSET_DRIFT_WARN_SECONDS,
    BAR_CLOSE_BUFFER_SECONDS,
)

logger = logging.getLogger("shared.mt5_client")

_client                 = None
_server_offset_seconds  = None


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────────────────────────────────────
def get_client() -> MetaTrader5:
    """Returns a connected MT5 client, reusing it across calls."""
    global _client

    if _client is not None:
        return _client

    client = MetaTrader5(host=MT5_HOST, port=MT5_PORT)

    if not client.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        raise ConnectionError(f"MT5 initialize() failed: {client.last_error()}")
    if not client.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        raise ConnectionError(f"MT5 login() failed: {client.last_error()}")

    info = client.account_info()
    logger.info(f"MT5 connected | account={info.login} | balance={info.balance} {info.currency}")
    _client = client
    return _client


def mt5_timeframe(label: str) -> int:
    """Resolves a canonical timeframe label to the terminal's constant."""
    if label not in TIMEFRAME_MINUTES:
        raise ValueError(f"Unsupported timeframe: {label!r}. Expected one of {list(TIMEFRAME_MINUTES)}")

    client    = get_client()
    constants = {
        "1m" : client.TIMEFRAME_M1,
        "5m" : client.TIMEFRAME_M5,
        "15m": client.TIMEFRAME_M15,
        "30m": client.TIMEFRAME_M30,
        "1H" : client.TIMEFRAME_H1,
        "4H" : client.TIMEFRAME_H4,
        "1D" : client.TIMEFRAME_D1,
    }
    return constants[label]


# ─────────────────────────────────────────────────────────────────────────────
# SERVER CLOCK
# ─────────────────────────────────────────────────────────────────────────────
def sync_server_offset(symbols: list) -> int | None:

    global _server_offset_seconds

    client        = get_client()
    local_utc     = time.time()
    freshest_tick = None

    for symbol in symbols:
        tick = client.symbol_info_tick(symbol)

        if tick is None:
            continue

        tick_time = int(tick.time)

        if freshest_tick is None or tick_time > freshest_tick:
            freshest_tick = tick_time

    if freshest_tick is None:
        logger.warning("No tick available on any symbol | keeping previous server offset")
        return _server_offset_seconds

    raw_offset     = freshest_tick - local_utc
    snapped_offset = int(round(raw_offset / SERVER_OFFSET_STEP_SECONDS)) * SERVER_OFFSET_STEP_SECONDS
    drift          = abs(raw_offset - snapped_offset)

    if drift > SERVER_OFFSET_DRIFT_WARN_SECONDS:
        logger.warning(
            f"Tick feed appears stale | raw_offset={raw_offset:.0f}s "
            f"snapped={snapped_offset}s drift={drift:.0f}s"
        )

    if _server_offset_seconds is None:
        logger.info(f"Server clock offset established | {snapped_offset / 3600:+.0f}h")
    elif snapped_offset != _server_offset_seconds:
        logger.warning(
            f"Server clock offset changed | "
            f"{_server_offset_seconds / 3600:+.0f}h -> {snapped_offset / 3600:+.0f}h "
            f"(DST shift or broker change)"
        )

    _server_offset_seconds = snapped_offset

    return _server_offset_seconds


def get_server_time() -> float:

    if _server_offset_seconds is None:
        raise RuntimeError("Server clock not synced | call sync_server_offset() first")

    return time.time() + _server_offset_seconds


# ─────────────────────────────────────────────────────────────────────────────
# BARS
# ─────────────────────────────────────────────────────────────────────────────
def drop_forming_bar(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:

    if df.empty:
        return df

    bar_seconds  = TIMEFRAME_MINUTES[timeframe] * 60
    last_open_ts = int(df.index[-1].timestamp())
    last_end_ts  = last_open_ts + bar_seconds

    if get_server_time() < last_end_ts + BAR_CLOSE_BUFFER_SECONDS:
        return df.iloc[:-1]

    return df


def get_rates(symbol: str, timeframe: str = "4H", n: int = 500) -> pd.DataFrame:
    """Returns the last n CLOSED bars for a symbol, most recent last."""
    client = get_client()
    rates  = client.copy_rates_from_pos(symbol, mt5_timeframe(timeframe), 0, n + 1)
    df     = pd.DataFrame(rates)

    if df.empty:
        return df

    df["time"] = pd.to_datetime(df["time"], unit="s")
    df         = df.set_index("time")

    return drop_forming_bar(df, timeframe)


def get_last_closed_bar_time(symbol: str, timeframe: str):
    """Bar-open timestamp of the most recent closed bar, or None if unavailable."""
    df = get_rates(symbol, timeframe=timeframe, n=1)

    if df.empty:
        return None

    return int(df.index[-1].timestamp())


def count_closed_bars_since(symbol: str, timeframe: str, from_ts: int, n: int = 500) -> int | None:

    df = get_rates(symbol, timeframe=timeframe, n=n)

    if df.empty:
        return None

    bar_seconds  = TIMEFRAME_MINUTES[timeframe] * 60
    bar_open_ts  = df.index.as_unit("s").asi8
    closed_after = bar_open_ts + bar_seconds > from_ts

    return int(closed_after.sum())