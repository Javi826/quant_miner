#BOT_crypto/state/candle_tracker.py

from typing import Dict

from execution.order_manager import close_position
from state.state_manager import save_state_local

import logging
logger = logging.getLogger('BOT_crypto.execution.candle_tracker')


# ==========================================================================
# CANDLE COUNTING
# ==========================================================================
def increment_strategy_candles(strat_id: str,
                               strategy_candles: Dict,
                               open_positions: Dict,
                               account_number: str,  # ← AÑADIR
                               state_file: str) -> None:

    if strat_id not in strategy_candles:
        strategy_candles[strat_id] = 0
    
    strategy_candles[strat_id] += 1
    try:
        save_state_local(open_positions, strategy_candles, account_number, state_file)
    except Exception as e:
        logger.error(f"CRITICAL ERROR saving state after incrementing candles")
        logger.error(f"Strategy: {strat_id}, Candles: {strategy_candles[strat_id]}")
        logger.error(f"Error: {e}")
        raise


def reset_strategy_candles(strat_id: str,
                           strategy_candles: Dict,
                           open_positions: Dict,
                           account_number: str,  # ← AÑADIR
                           state_file: str) -> None:

    strategy_candles[strat_id] = 0
    try:
        save_state_local(open_positions, strategy_candles, account_number, state_file)
    except Exception as e:
        logger.error(f"CRITICAL ERROR saving state after resetting candles")
        logger.error(f"Strategy: {strat_id}")
        logger.error(f"Error: {e}")
        raise


def check_candles_timeout_for_strategy(strat_id: str,
                                       sell_after_ncandles: int,
                                       open_positions: Dict,
                                       strategy_candles: Dict,
                                       account_number: str,
                                       state_file: str,
                                       send_request_func,
                                       bot_state=None) -> None:

    candles_elapsed = strategy_candles.get(strat_id, 0)

    if candles_elapsed < sell_after_ncandles:
        return

    if strat_id not in open_positions or not open_positions[strat_id]:
        return

    positions = open_positions[strat_id][:]

    if not positions:
        return

    logger.info(f"TIMEOUT REACHED for {strat_id}")
    logger.info(f"Candles: {candles_elapsed}/{sell_after_ncandles}")
    logger.info(f"Closing {len(positions)} positions...")

    positions_failed = []
    for pos in positions:
        position_data = {
            'opened_at': pos['opened_at'],
            'strategy_id': strat_id,
            'usdt_amount': pos.get('usdt_amount', 0),
            'entry_price': pos['entry_price'],
            'tp': pos.get('tp'),
            'sl': pos.get('sl'),
        }

        if close_position(
            pos['symbol'],
            pos['size'],
            pos['direction'],
            send_request_func,
            reason="TIMEOUT",
            position_data=position_data,
            bot_state=bot_state
        ):
            open_positions[strat_id].remove(pos)
        else:
            positions_failed.append(pos['symbol'])

    # Always reset candle counter to avoid infinite retry loop
    strategy_candles[strat_id] = 0

    try:
        save_state_local(open_positions, strategy_candles, account_number, state_file)
    except Exception as e:
        logger.error(f"CRITICAL ERROR saving state after timeout close")
        logger.error(f"Strategy: {strat_id}")
        logger.error(f"Error: {e}")
        raise

    if positions_failed:
        logger.warning(f"WAR-Timeout close failed for: {positions_failed} — kept in local state")