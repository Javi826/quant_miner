#quant_miner/darwinex/BOT_forex/darwinex/live/strategies.py (forex)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from loguru import logger
from signals.signal_builder import build_signal_fn
from live.mt5_connection import get_closed_bars, fmt_bar
from live.strategies_config import STRATEGIES

SIGNAL_BARS = 500


def get_active_strategies():
    return [s for s in STRATEGIES if s.get("active", True)]


def get_signal(strategy, bar_time: int) -> list:
    """
    Evaluates the strategy on every one of its symbols using the closed bars up
    to and including bar_time (the bar that triggered the cycle).
    Returns one {"symbol", "buy", "sell"} per symbol.
    """
    timeframe = strategy["timeframe"]
    direction = strategy["direction"]
    signal_fn = build_signal_fn(strategy["specs"], direction)

    results = []

    for symbol in strategy["symbols"]:
        try:
            df = get_closed_bars(symbol, timeframe, until=bar_time, n=SIGNAL_BARS)

            if df.empty or int(df.index[-1]) != bar_time:
                last = None if df.empty else int(df.index[-1])
                logger.warning(
                    f"{strategy['id']} | {symbol} | bar {fmt_bar(bar_time)} not available "
                    f"(last={fmt_bar(last)}) | signal dropped"
                )
                results.append({"symbol": symbol, "buy": False, "sell": False})
                continue

            arr = {
                "open"  : df["open"].values,
                "high"  : df["high"].values,
                "low"   : df["low"].values,
                "close" : df["close"].values,
                "volume": df["tick_volume"].values,
            }

            signals     = signal_fn(arr, live_trading=True)
            last_signal = int(signals[-1])

            buy  = direction == "long" and last_signal == 1
            sell = direction == "short" and last_signal == -1

            results.append({"symbol": symbol, "buy": buy, "sell": sell})
            logger.info(
                f"{strategy['id']} | {symbol} | bar {fmt_bar(bar_time)} | "
                f"bars={len(df)} | BUY={buy} SELL={sell}"
            )

        except Exception as e:
            logger.error(f"Error in get_signal({strategy['id']}, {symbol}): {e}")
            results.append({"symbol": symbol, "buy": False, "sell": False})

    return results