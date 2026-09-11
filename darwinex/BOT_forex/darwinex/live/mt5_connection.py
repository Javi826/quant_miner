#quant_d/darwinex/BOT_forex/darwinex/live/mt5_connection.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import time
from loguru import logger
from broker_client.broker_api.mt5_client import (
    get_client,
    get_rates,
    get_last_closed_bar_time,
    sync_server_offset,
    get_server_time,
)

MAX_COMMENT_LEN     = 15
RETRY_DELAY_SECONDS = 15

__all__ = [
    "get_client", "get_rates", "get_last_closed_bar_time",
    "sync_server_offset", "get_server_time",
    "send_order", "resume", "run",
]

def _truncate_comment(comment):
    """MT5 rejects order comments longer than 31 characters."""
    return str(comment)[:MAX_COMMENT_LEN]

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


def resume():
    """Returns a DataFrame with open positions. Diagnostics only."""
    import pandas as pd

    client    = get_client()
    columns   = ["ticket", "position", "symbol", "volume", "magic", "profit", "price", "tp", "sl"]
    positions = client.positions_get() or []
    rows      = [
        [p.ticket, p.type, p.symbol, p.volume, p.magic, p.profit, p.price_open, p.tp, p.sl]
        for p in positions
    ]
    return pd.DataFrame(rows, columns=columns)

def run(symbol, buy, sell, lot, pct_tp=2.0, pct_sl=2.0, comment="", magic=0):

    logger.info(f"{symbol} | BUY={buy} SELL={sell}")

    if buy:
        send_order(symbol, lot, True, False, pct_tp=pct_tp, pct_sl=pct_sl, comment=comment, magic=magic)

    if sell:
        send_order(symbol, lot, False, True, pct_tp=pct_tp, pct_sl=pct_sl, comment=comment, magic=magic)