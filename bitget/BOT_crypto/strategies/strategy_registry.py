#BOT_crypto/strategies/strategy_registry.py (crypto)

import sys
import os
import logging

logger = logging.getLogger('BOT_crypto.strategies.registry')

#===========================================================================
# PATH SETUP
#===========================================================================

current_dir = os.path.dirname(os.path.abspath(__file__))  # strategies/
bot_dir     = os.path.dirname(current_dir)                  # BOT_crypto/
bitget_dir  = os.path.dirname(bot_dir)                      # bitget/

if bitget_dir not in sys.path:
    sys.path.insert(0, bitget_dir)

#===========================================================================
# IMPORTS
#===========================================================================

from market_data import fetch_ohlcv_data, normalize_live_ohlcv, df_to_arrays_live
from signals.signal_builder import build_signal_fn


#===========================================================================
# MAIN SIGNAL DETECTION FUNCTION
#===========================================================================

def detect_signals_for_strategy(
    strat: dict,
    final_symbols: list,
    exchange,
) -> list:

    strategy_id = strat['id']
    timeframe   = strat['timeframe']

    logger.info(f"Processing strategy: {strategy_id}")
    logger.info("-" * 48)

    if not final_symbols:
        logger.error(f"No symbols to process for {strategy_id}")
        return []

    ohlcv_data  = fetch_ohlcv_data(final_symbols, timeframe)
    all_signals = []

    for symbol, df in ohlcv_data.items():
        if df is None or df.empty:
            continue

        df_norm = normalize_live_ohlcv(df)
        arr     = df_to_arrays_live(df_norm)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"[CANDLE_CHECK] {symbol} {timeframe} — last 50 candles:\n"
                f"{df_norm[['open','high','low','close','volume_quote']].tail(50).to_string()}"
            )

        try:
            if 'specs' not in strat:
                logger.warning(
                    f"WAR-Strategy '{strategy_id}' has no 'specs' (rule-engine format). Skipping."
                )
                continue

            signal_fn = build_signal_fn(strat['specs'], strat['direction'])
            signals   = signal_fn(arr, live_trading=True)

            if signals is None or len(signals) == 0:
                continue

            if signals[-1] != 0:
                last_row = df_norm.iloc[-1]
                all_signals.append({
                    'symbol':    symbol,
                    'timestamp': last_row.name if 'timestamp' not in df_norm.columns else last_row['timestamp'],
                    'close':     float(arr['close'][-1]),
                })

        except Exception as e:
            logger.error(f"Error detecting signals for {symbol} ({strategy_id}): {e}")
            continue

    logger.debug(f"Signals detected {strategy_id}: {len(all_signals)}")

    return all_signals


# ==============================================================
# IMPLEMENTED STRATEGIES - kept for import compatibility
# ==============================================================

IMPLEMENTED_STRATEGIES = set()