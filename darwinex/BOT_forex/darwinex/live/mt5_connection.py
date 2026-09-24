#quant_miner/darwinex/BOT_forex/darwinex/live/mt5_connection.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import time
import pandas as pd
from loguru import logger
from broker_client.broker_api.mt5_client import get_client, mt5_timeframe
from broker_client.broker_config import TIMEFRAME_MINUTES

MAX_COMMENT_LEN     = 15
RETRY_DELAY_SECONDS = 15

__all__ = [
    "get_client",
    "get_bars", "last_closed_bar_time", "get_closed_bars", "count_closed_bars_since",
    "get_open_positions", "send_order", "run", "close_position", "resume",
]

def _truncate_comment(comment):
    """MT5 rejects order comments longer than 31 characters."""
    return str(comment)[:MAX_COMMENT_LEN]

def fmt_bar(ts) -> str:
    """Bar-open timestamp (broker time) as a readable string, for logs."""
    return "None" if ts is None else time.strftime("%Y-%m-%d %H:%M", time.gmtime(int(ts)))

# ─────────────────────────────────────────────────────────────────────────────
# BARS
# ─────────────────────────────────────────────────────────────────────────────
# Bars are indexed by their open time in seconds (broker time), as MT5 gives it.
# No datetime conversion, so no dependency on the pandas version.

def get_bars(symbol: str, timeframe: str, n: int) -> pd.DataFrame:
    """
    Last n bars exactly as MT5 returns them, oldest first.
    The last row is the bar still forming: it changes with every tick.
    """
    client = get_client()
    rates  = client.copy_rates_from_pos(symbol, mt5_timeframe(timeframe), 0, n)

    if rates is None or len(rates) == 0:
        return pd.DataFrame()

    df         = pd.DataFrame(rates)
    df["time"] = df["time"].astype("int64")
    return df.set_index("time")


def last_closed_bar_time(symbol: str, timeframe: str) -> int | None:
    """
    Open time of the last closed bar: the second-to-last row, because the last
    one is forming. If the new bar has no tick yet, the last row is actually
    closed; it is simply picked up one probe later, never too early.
    """
    df = get_bars(symbol, timeframe, 2)

    if len(df) < 2:
        return None

    return int(df.index[-2])


def get_closed_bars(symbol: str, timeframe: str, until: int, n: int) -> pd.DataFrame:
    """
    Last n closed bars up to and including `until`.
    `until` is the bar that triggered the cycle, already known to be closed, so
    the forming bar is never included, even for a symbol that has not received
    its first tick in the new bar yet.
    """
    df = get_bars(symbol, timeframe, n + 2)

    if df.empty:
        return df

    return df[df.index <= until].tail(n)


def count_closed_bars_since(symbol: str, timeframe: str, entry_ts: int,
                            until: int, n: int) -> int | None:
    """
    Closed bars from the entry bar (included) up to `until` (included).
    Exact up to n; callers only compare it against a threshold below n.
    """
    df = get_closed_bars(symbol, timeframe, until, n)

    if df.empty:
        return None

    bar_seconds = TIMEFRAME_MINUTES[timeframe] * 60
    bar_close   = df.index.to_numpy() + bar_seconds

    return int((bar_close > entry_ts).sum())

# ─────────────────────────────────────────────────────────────────────────────
# POSITIONS
# ─────────────────────────────────────────────────────────────────────────────
def get_open_positions(magic: int | None = None) -> list:
    """Open positions, optionally filtered by magic."""
    client    = get_client()
    positions = list(client.positions_get() or [])

    if magic is None:
        return positions

    return [p for p in positions if p.magic == magic]

# ─────────────────────────────────────────────────────────────────────────────
# ORDER EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def _build_deal_request(symbol, lot, is_buy, pct_tp, pct_sl, comment, magic):

    client = get_client()
    tick   = client.symbol_info_tick(symbol)

    if tick is None:
        return None

    filling_type = 0

    if is_buy:
        price      = tick.ask
        tp_price   = price * (1 + pct_tp / 100)
        sl_price   = price * (1 - pct_sl / 100)
        order_type = client.ORDER_TYPE_BUY
    else:
        price      = tick.bid
        tp_price   = price * (1 - pct_tp / 100)
        sl_price   = price * (1 + pct_sl / 100)
        order_type = client.ORDER_TYPE_SELL

    return {
        "action"      : client.TRADE_ACTION_DEAL,
        "symbol"      : symbol,
        "volume"      : lot,
        "type"        : order_type,
        "price"       : price,
        "deviation"   : 10,
        "tp"          : tp_price,
        "sl"          : sl_price,
        "magic"       : magic,
        "comment"     : comment,
        "type_filling": filling_type,
        "type_time"   : client.ORDER_TIME_GTC,
    }


def send_order(symbol, lot, buy, sell, pct_tp=2.0, pct_sl=2.0, comment="", magic=0):

    if not (buy or sell):
        return None

    client            = get_client()
    is_buy            = bool(buy)
    label             = "OPEN LONG" if is_buy else "OPEN SHORT"
    comment           = _truncate_comment(comment)
    retryable_retcode = int(client.TRADE_RETCODE_MARKET_CLOSED)
    retcode_done      = int(client.TRADE_RETCODE_DONE)

    for attempt in (1, 2):
        try:
            request = _build_deal_request(symbol, lot, is_buy, pct_tp, pct_sl, comment, magic)

            if request is None:
                logger.error(
                    f"{label} {symbol} | magic={magic} | no tick available to price "
                    f"the order | signal dropped"
                )
                return None

            result = client.order_send(request)

            if result is None:
                logger.error(f"{label} {symbol} | order_send() returned None: {client.last_error()}")
                return None

            retcode    = int(getattr(result, "retcode", -1))
            broker_msg = getattr(result, "comment", "") or ""

            if retcode == retcode_done:
                logger.info(
                    f"{label} {symbol} | ticket={result.order} magic={magic} | "
                    f"price={request['price']:.5f} tp={request['tp']:.5f} sl={request['sl']:.5f} | "
                    f"{broker_msg}"
                )
                return result

            if attempt == 1 and retcode == retryable_retcode:
                logger.warning(
                    f"{label} {symbol} | magic={magic} | rejected retcode={retcode} "
                    f"({broker_msg}) | retrying once in {RETRY_DELAY_SECONDS}s"
                )
                time.sleep(RETRY_DELAY_SECONDS)
                continue

            logger.error(
                f"{label} {symbol} | magic={magic} | rejected retcode={retcode} "
                f"({broker_msg}) | signal dropped"
            )
            return None

        except Exception as e:
            logger.error(
                f"{label} {symbol} | magic={magic} | unexpected failure on attempt "
                f"{attempt}: {e} | signal dropped"
            )
            return None

    return None


def run(symbol, buy, sell, lot, pct_tp=2.0, pct_sl=2.0, comment="", magic=0):

    logger.info(f"{symbol} | BUY={buy} SELL={sell}")

    if buy:
        send_order(symbol, lot, True, False, pct_tp=pct_tp, pct_sl=pct_sl, comment=comment, magic=magic)

    if sell:
        send_order(symbol, lot, False, True, pct_tp=pct_tp, pct_sl=pct_sl, comment=comment, magic=magic)


def resume():
    """Returns a DataFrame with open positions. Diagnostics only."""
    client    = get_client()
    columns   = ["ticket", "position", "symbol", "volume", "magic", "profit", "price", "tp", "sl"]
    positions = client.positions_get() or []
    rows      = [
        [p.ticket, p.type, p.symbol, p.volume, p.magic, p.profit, p.price_open, p.tp, p.sl]
        for p in positions
    ]
    return pd.DataFrame(rows, columns=columns)

# ─────────────────────────────────────────────────────────────────────────────
# POSITION CLOSING
# ─────────────────────────────────────────────────────────────────────────────
def _build_close_request(position, comment):

    client = get_client()
    tick   = client.symbol_info_tick(position.symbol)

    if tick is None:
        return None

    is_long = int(position.type) == int(client.ORDER_TYPE_BUY)

    return {
        "action"      : client.TRADE_ACTION_DEAL,
        "symbol"      : position.symbol,
        "volume"      : position.volume,
        "type"        : client.ORDER_TYPE_SELL if is_long else client.ORDER_TYPE_BUY,
        "position"    : position.ticket,
        "price"       : tick.bid if is_long else tick.ask,
        "deviation"   : 10,
        "magic"       : position.magic,
        "comment"     : comment,
        "type_filling": 0,
        "type_time"   : client.ORDER_TIME_GTC,
    }


def close_position(position, comment="") -> bool:
    """Closes a single position by ticket with an opposite market order."""
    client            = get_client()
    label             = "CLOSE LONG" if int(position.type) == int(client.ORDER_TYPE_BUY) else "CLOSE SHORT"
    comment           = _truncate_comment(comment)
    retryable_retcode = int(client.TRADE_RETCODE_MARKET_CLOSED)
    retcode_done      = int(client.TRADE_RETCODE_DONE)

    for attempt in (1, 2):
        try:
            request = _build_close_request(position, comment)

            if request is None:
                logger.error(
                    f"{label} {position.symbol} | ticket={position.ticket} | no tick "
                    f"available to price the close | position left open"
                )
                return False

            result = client.order_send(request)

            if result is None:
                logger.error(
                    f"{label} {position.symbol} | ticket={position.ticket} | "
                    f"order_send() returned None: {client.last_error()}"
                )
                return False

            retcode    = int(getattr(result, "retcode", -1))
            broker_msg = getattr(result, "comment", "") or ""

            if retcode == retcode_done:
                logger.info(
                    f"{label} {position.symbol} | ticket={position.ticket} "
                    f"magic={position.magic} | volume={position.volume} "
                    f"profit={position.profit} | {broker_msg}"
                )
                return True

            if attempt == 1 and retcode == retryable_retcode:
                logger.warning(
                    f"{label} {position.symbol} | ticket={position.ticket} | rejected "
                    f"retcode={retcode} ({broker_msg}) | retrying once in {RETRY_DELAY_SECONDS}s"
                )
                time.sleep(RETRY_DELAY_SECONDS)
                continue

            logger.error(
                f"{label} {position.symbol} | ticket={position.ticket} | rejected "
                f"retcode={retcode} ({broker_msg}) | position left open"
            )
            return False

        except Exception as e:
            logger.error(
                f"{label} {position.symbol} | ticket={position.ticket} | unexpected "
                f"failure on attempt {attempt}: {e} | position left open"
            )
            return False

    return False