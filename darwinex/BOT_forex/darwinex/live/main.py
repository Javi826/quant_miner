#quant_miner/darwinex/BOT_forex/darwinex/live/main.py (forex)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "core")))
import time
from loguru import logger
from broker_client.broker_api.mt5_client import get_client,get_last_closed_bar_time,sync_server_offset,count_closed_bars_since
from broker_client.broker_config import TIMEFRAME_MINUTES
from live.mt5_connection import run, has_open_position, get_open_positions, close_position
from live.strategies import get_active_strategies, get_signal, get_strategy_by_magic

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
LOG_FILE = os.path.join(os.path.dirname(__file__), "../../logs/live.log")
logger.add(LOG_FILE, rotation="1 week", retention="1 month", level="INFO")

LOOP_SLEEP_SECONDS   = 5.0
STALE_BAR_FACTOR     = 3
SIGNAL_DELAY_SECONDS = 15.0
ALLOW_PYRAMIDING     = False
ENABLE_TIME_EXITS    = True

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
# BAR CLOCK
# ─────────────────────────────────────────────────────────────────────────────
def get_symbols_by_timeframe() -> dict:
    """Maps every active timeframe to the symbols traded on it."""
    symbols_by_timeframe = {}
    for strategy in get_active_strategies():
        symbols = symbols_by_timeframe.setdefault(strategy["timeframe"], set())
        symbols.update(strategy["symbols"])
    return {tf: sorted(symbols) for tf, symbols in symbols_by_timeframe.items()}

def get_all_symbols(symbols_by_timeframe: dict) -> list:
    """Flat, deduplicated symbol list used to sync the server clock."""
    all_symbols = set()
    for symbols in symbols_by_timeframe.values():
        all_symbols.update(symbols)
    return sorted(all_symbols)

def init_bar_clock(symbols_by_timeframe: dict) -> dict:

    last_bar_time = {}
    for timeframe, symbols in symbols_by_timeframe.items():
        for symbol in symbols:
            bar_time = get_last_closed_bar_time(symbol, timeframe)
            last_bar_time[(symbol, timeframe)] = bar_time
            logger.info(f"Last closed bar at startup | {symbol} | {timeframe:<3} | {bar_time}")
    return last_bar_time


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def execute_strategy(strategy: dict, timeframe: str, new_symbols: list,
                     last_bar_time: dict, processed: dict) -> None:

    if not ALLOW_PYRAMIDING and has_open_position(strategy["magic"]):
        logger.info(f"Pyramiding blocked | {strategy['id']} | position already open")
        return

    expected_bar_times = {
        symbol: last_bar_time[(symbol, timeframe)]
        for symbol in new_symbols
        if symbol in strategy["symbols"]
    }

    if not expected_bar_times:
        return

    signals = get_signal(strategy, expected_bar_times)

    for sig in signals:
        symbol = sig["symbol"]

        if not (sig["buy"] or sig["sell"]):
            continue

        key      = (strategy["id"], symbol)
        bar_time = expected_bar_times[symbol]

        if processed.get(key) == bar_time:
            logger.info(f"Already traded this bar | {strategy['id']} | {symbol}")
            continue

        processed[key] = bar_time
        run(
            symbol  = symbol,
            buy     = sig["buy"],
            sell    = sig["sell"],
            lot     = strategy["lot"],
            pct_tp  = strategy["tp_pct"],
            pct_sl  = strategy["sl_pct"],
            comment = strategy["id"],
            magic   = strategy["magic"],
        )


def run_strategies(timeframe: str, new_symbols: list, last_bar_time: dict, processed: dict) -> None:

    strategies = [
        s for s in get_active_strategies()
        if s["timeframe"] == timeframe and set(s["symbols"]) & set(new_symbols)
    ]

    for strategy in strategies:
        logger.info(f"▶ Ejecutando: {strategy['id']}")
        try:
            execute_strategy(strategy, timeframe, new_symbols, last_bar_time, processed)
        except Exception as e:
            logger.error(
                f"Strategy aborted, remaining strategies continue | "
                f"{strategy['id']} | {timeframe} | {e}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# TIME-BASED EXITS
# ─────────────────────────────────────────────────────────────────────────────
def close_expired_position(position) -> None:

    strategy = get_strategy_by_magic(position.magic)

    if strategy is None:
        logger.warning(
            f"Open position with unknown magic | ticket={position.ticket} "
            f"magic={position.magic} | time exit not applied"
        )
        return

    max_candles = strategy.get("sell_after_ncandles", 0)

    if not max_candles:
        return

    elapsed = count_closed_bars_since(
        position.symbol, strategy["timeframe"], int(position.time)
    )

    if elapsed is None:
        logger.error(
            f"Could not count bars since entry | {strategy['id']} | "
            f"{position.symbol} | ticket={position.ticket} | time exit deferred"
        )
        return

    if elapsed < max_candles:
        logger.info(
            f"Holding | {strategy['id']} | {position.symbol} | "
            f"ticket={position.ticket} | candle {elapsed}/{max_candles}"
        )
        return

    logger.info(
        f"Time exit reached | {strategy['id']} | {position.symbol} | "
        f"ticket={position.ticket} | candle {elapsed}/{max_candles}"
    )
    close_position(position, comment=strategy["id"])


def manage_time_exits() -> None:
    """Closes positions that have reached sell_after_ncandles, before new entries."""
    if not ENABLE_TIME_EXITS:
        return

    try:
        positions = get_open_positions()
    except Exception as e:
        logger.error(f"Could not read open positions | time exits skipped | {e}")
        return

    for position in positions:
        try:
            close_expired_position(position)
        except Exception as e:
            logger.error(
                f"Time exit aborted, remaining positions continue | "
                f"ticket={position.ticket} magic={position.magic} | {e}"
            )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    logger.info("═" * 60)
    logger.info("Arrancando quant_d live trading")
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
    all_symbols          = get_all_symbols(symbols_by_timeframe)
    logger.info(f"Timeframes y símbolos activos: {symbols_by_timeframe}")

    if sync_server_offset(all_symbols) is None:
        logger.error("No se pudo establecer el reloj del servidor. Abortando.")
        return

    last_bar_time = init_bar_clock(symbols_by_timeframe)
    last_change   = {key: time.monotonic() for key in last_bar_time}
    processed     = {}
    first_loop    = True

    manage_time_exits()

    logger.info("Sistema listo. Entrando en loop principal...")

    try:
        while True:
            sync_server_offset(all_symbols)

            for timeframe, symbols in symbols_by_timeframe.items():
                new_symbols = []

                for symbol in symbols:
                    key = (symbol, timeframe)
                    try:
                        bar_time = get_last_closed_bar_time(symbol, timeframe)
                    except Exception as e:
                        logger.error(f"Could not probe bars | {symbol} | {timeframe} | {e}")
                        continue

                    if bar_time is None:
                        continue

                    known = last_bar_time.get(key)
                    if known is not None and bar_time <= known:
                        stale_after = TIMEFRAME_MINUTES[timeframe] * 60 * STALE_BAR_FACTOR
                        if time.monotonic() - last_change[key] > stale_after:
                            logger.warning(
                                f"No new bar for over {stale_after}s | {symbol} | {timeframe} "
                                f"(market closed or feed down)"
                            )
                            last_change[key] = time.monotonic()
                        continue

                    last_bar_time[key] = bar_time
                    last_change[key]   = time.monotonic()
                    if not first_loop:
                        new_symbols.append(symbol)
                        logger.info(f"Vela cerrada | {symbol} | {timeframe} | {bar_time}")

                if new_symbols:
                    logger.info(f"Waiting {SIGNAL_DELAY_SECONDS:.0f}s before firing signals | {timeframe}")
                    time.sleep(SIGNAL_DELAY_SECONDS)
                    manage_time_exits()
                    try:
                        run_strategies(timeframe, new_symbols, last_bar_time, processed)
                    except Exception as e:
                        logger.error(f"Error ejecutando estrategias {timeframe}: {e}")

            first_loop = False
            time.sleep(LOOP_SLEEP_SECONDS)

    except KeyboardInterrupt:
        logger.info("Detenido por el usuario.")


if __name__ == "__main__":
    main()