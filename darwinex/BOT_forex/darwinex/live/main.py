#quant_miner/darwinex/BOT_forex/darwinex/live/main.py (forex)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "core")))
import time
from loguru import logger
from broker_client.broker_api.mt5_client import get_client
from broker_client.broker_config import BAR_CLOSE_BUFFER_SECONDS
from live.mt5_connection import (
    run,
    get_open_positions,
    close_position,
    last_closed_bar_time,
    count_closed_bars_since,
    fmt_bar,
)
from live.strategies import get_active_strategies, get_signal

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
LOG_FILE = os.path.join(os.path.dirname(__file__), "../../logs/live.log")
logger.add(LOG_FILE, rotation="1 week", retention="1 month", level="INFO")

LOOP_SLEEP_SECONDS = 5.0

# ALLOW_PYRAMIDING = False  (backtester NO_PYRAMID)
#   A strategy opens its symbols together, as one batch, on the same bar, and
#   does not look for signals again until every position of the batch is
#   closed (TP/SL, or time exit of the whole batch after sell_after_ncandles).
# ALLOW_PYRAMIDING = True   (backtester PYRAMID)
#   A strategy looks for signals on every bar and opens them regardless of the
#   positions it already has. Each position has its own time exit, counted
#   from its own entry.
# sell_after_ncandles = 0 in a strategy disables its time exit: TP/SL only.
ALLOW_PYRAMIDING = False

# Open positions allowed per strategy (magic), in both modes. New signals
# beyond the limit are dropped, in symbol order, like the backtester's cash cap.
MAX_POSITIONS = 20

# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────────────────────────────────────
def connect() -> bool:
    try:
        client = get_client()
    except ConnectionError as e:
        logger.error(f"Conexión falló: {e}")
        return False

    info = client.account_info()
    logger.success(f"Conectado | Cuenta: {info.login} | Balance: {info.balance} {info.currency}")
    return True

# ─────────────────────────────────────────────────────────────────────────────
# BAR DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def get_symbols_by_timeframe() -> dict:
    """Maps every active timeframe to the symbols traded on it."""
    symbols_by_timeframe = {}
    for strategy in get_active_strategies():
        symbols = symbols_by_timeframe.setdefault(strategy["timeframe"], set())
        symbols.update(strategy["symbols"])
    return {tf: sorted(symbols) for tf, symbols in symbols_by_timeframe.items()}


def get_last_closed_bar(timeframe: str, symbols: list) -> int | None:

    latest = None

    for symbol in symbols:
        try:
            bar_time = last_closed_bar_time(symbol, timeframe)
        except Exception as e:
            logger.error(f"Could not read bars | {symbol} | {timeframe} | {e}")
            continue

        #logger.debug(f"Last closed bar | {symbol} | {timeframe} | {fmt_bar(bar_time)}")

        if bar_time is not None and (latest is None or bar_time > latest):
            latest = bar_time

    return latest


def detect_closed_bars(symbols_by_timeframe: dict, last_bar: dict) -> dict:

    closed = {}

    for timeframe, symbols in symbols_by_timeframe.items():
        bar_time = get_last_closed_bar(timeframe, symbols)

        if bar_time is None:
            continue

        known = last_bar.get(timeframe)

        if known is None:
            last_bar[timeframe] = bar_time
            logger.info(f"Reference bar | {timeframe:<3} | {fmt_bar(bar_time)}")
            continue

        if bar_time > known:
            last_bar[timeframe] = bar_time
            closed[timeframe]   = bar_time
            logger.info(f"Vela cerrada | {timeframe:<3} | {fmt_bar(bar_time)}")

    return closed

# ─────────────────────────────────────────────────────────────────────────────
# EXITS
# ─────────────────────────────────────────────────────────────────────────────
def bars_since_entry(strategy: dict, position, bar_time: int) -> int | None:
    """Closed bars from the position's entry bar up to bar_time, both included."""
    return count_closed_bars_since(
        position.symbol, strategy["timeframe"], int(position.time), bar_time,
        strategy["sell_after_ncandles"] + 5,
    )


def close_expired_batch(strategy: dict, bar_time: int, positions: list) -> bool:

    strategy_id = strategy["id"]
    max_candles = strategy.get("sell_after_ncandles", 0)

    if not max_candles:
        logger.info(f"Holding | {strategy_id} | {len(positions)} pos at bar close | no time exit")
        return False

    # Every position of the batch was opened on the same bar: count from the oldest.
    entry   = min(positions, key=lambda p: p.time)
    elapsed = bars_since_entry(strategy, entry, bar_time)

    if elapsed is None:
        logger.error(f"Could not count bars | {strategy_id} | {entry.symbol} | time exit deferred")
        return False

    if elapsed < max_candles:
        logger.info(
            f"Holding | {strategy_id} | candle {elapsed}/{max_candles} | "
            f"{len(positions)} pos at bar close"
        )
        return False

    open_now = get_open_positions(strategy["magic"])
    logger.info(
        f"Time exit reached | {strategy_id} | candle {elapsed}/{max_candles} | "
        f"closing {len(open_now)} pos"
    )

    closed_all = True
    for position in open_now:
        if not close_position(position, comment=strategy_id):
            closed_all = False

    if not closed_all:
        logger.error(f"Time exit incomplete | {strategy_id} | stays busy until next bar")

    return closed_all


def close_expired_positions(strategy: dict, bar_time: int) -> None:
    """Pyramiding: time exit of each position, counted from its own entry."""
    strategy_id = strategy["id"]
    max_candles = strategy.get("sell_after_ncandles", 0)

    if not max_candles:
        return

    for position in get_open_positions(strategy["magic"]):
        elapsed = bars_since_entry(strategy, position, bar_time)

        if elapsed is None:
            logger.error(
                f"Could not count bars | {strategy_id} | {position.symbol} | "
                f"ticket={position.ticket} | time exit deferred"
            )
            continue

        if elapsed < max_candles:
            logger.debug(
                f"Holding | {strategy_id} | {position.symbol} | ticket={position.ticket} | "
                f"candle {elapsed}/{max_candles}"
            )
            continue

        logger.info(
            f"Time exit reached | {strategy_id} | {position.symbol} | "
            f"ticket={position.ticket} | candle {elapsed}/{max_candles}"
        )
        close_position(position, comment=strategy_id)


def run_exits(strategy: dict, bar_time: int, positions_at_close: list) -> bool:
    """Applies the time exits of the mode. Returns True if the strategy may look for signals."""
    if ALLOW_PYRAMIDING:
        close_expired_positions(strategy, bar_time)
        return True

    if not positions_at_close:
        return True

    if close_expired_batch(strategy, bar_time, positions_at_close):
        return True

    logger.info(f"Skip signals | {strategy['id']} | batch still open")
    return False

# ─────────────────────────────────────────────────────────────────────────────
# ENTRIES
# ─────────────────────────────────────────────────────────────────────────────
def open_signals(strategy: dict, bar_time: int) -> None:

    strategy_id = strategy["id"]
    signals     = get_signal(strategy, bar_time)
    fired       = sorted((s for s in signals if s["buy"] or s["sell"]), key=lambda s: s["symbol"])
    open_count  = len(get_open_positions(strategy["magic"]))
    free_slots  = max(0, MAX_POSITIONS - open_count)

    logger.info(
        f"Signals | {strategy_id} | {fmt_bar(bar_time)} | {len(fired)}/{len(signals)} symbols | "
        f"open {open_count}/{MAX_POSITIONS}"
    )

    if len(fired) > free_slots:
        dropped = [s["symbol"] for s in fired[free_slots:]]
        logger.warning(f"Max positions | {strategy_id} | dropped {dropped}")
        fired = fired[:free_slots]

    for sig in fired:
        run(
            symbol  = sig["symbol"],
            buy     = sig["buy"],
            sell    = sig["sell"],
            lot     = strategy["lot"],
            pct_tp  = strategy["tp_pct"],
            pct_sl  = strategy["sl_pct"],
            comment = strategy_id,
            magic   = strategy["magic"],
        )

# ─────────────────────────────────────────────────────────────────────────────
# BAR CLOSE CYCLE
# ─────────────────────────────────────────────────────────────────────────────
def positions_by_magic(strategies: list) -> dict:

    by_magic = {}
    for position in get_open_positions():
        by_magic.setdefault(position.magic, []).append(position)

    return {s["magic"]: by_magic.get(s["magic"], []) for s in strategies}


def process_bar_close(closed: dict) -> None:

    strategies = [s for s in get_active_strategies() if s["timeframe"] in closed]

    if not strategies:
        return

    positions_at_close = positions_by_magic(strategies)

    for strategy in strategies:
        logger.info(
            f"At bar close | {strategy['id']} | {len(positions_at_close[strategy['magic']])} pos"
        )

    logger.info(f"Waiting {BAR_CLOSE_BUFFER_SECONDS}s before exits and signals")
    time.sleep(BAR_CLOSE_BUFFER_SECONDS)

    for strategy in strategies:
        bar_time = closed[strategy["timeframe"]]

        try:
            if run_exits(strategy, bar_time, positions_at_close[strategy["magic"]]):
                open_signals(strategy, bar_time)

        except Exception as e:
            logger.error(f"Strategy aborted, remaining strategies continue | {strategy['id']} | {e}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    logger.info("═" * 60)
    logger.info("Arrancando quant_d live trading")
    logger.info(f"ALLOW_PYRAMIDING={ALLOW_PYRAMIDING} | MAX_POSITIONS={MAX_POSITIONS}")
    logger.info("═" * 60)

    connected = False
    for attempt in range(1, 6):
        logger.info(f"Intento de conexión {attempt}/5...")
        connected = connect()
        if connected:
            break
        time.sleep(10)

    if not connected:
        logger.error("No se pudo conectar con MT5. Abortando.")
        return

    symbols_by_timeframe = get_symbols_by_timeframe()
    logger.info(f"Timeframes y símbolos activos: {symbols_by_timeframe}")

    last_bar = {}
    detect_closed_bars(symbols_by_timeframe, last_bar)

    logger.info("Sistema listo. Entrando en loop principal...")

    try:
        while True:
            try:
                closed = detect_closed_bars(symbols_by_timeframe, last_bar)
                if closed:
                    process_bar_close(closed)
            except Exception as e:
                logger.error(f"Cycle aborted | {e}")

            time.sleep(LOOP_SLEEP_SECONDS)

    except KeyboardInterrupt:
        logger.info("Detenido por el usuario.")


if __name__ == "__main__":
    main()