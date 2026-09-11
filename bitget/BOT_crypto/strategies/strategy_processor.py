#BOT_crypto/strategies/strategy_processor.py (crypto)
import time
import logging
from typing import Dict, List, Callable, Any
logger = logging.getLogger('BOT_crypto.strategies.processor')
from state import reset_strategy_candles
from .strategy_registry import detect_signals_for_strategy



class StrategyProcessor:
    def __init__(
        self,
        send_request_func: Callable,
        get_balance_func:  Callable,
        hour_zone,
        account_number:    str,
        state_file:        str,
    ):
        self.send_request   = send_request_func
        self.get_balance    = get_balance_func
        self.hour_zone      = hour_zone
        self.account_number = account_number
        self.state_file     = state_file
        self.operative      = None
        self._detect_real_signals = detect_signals_for_strategy
        logger.info("StrategyProcessor initialized")
    def detect_signals(
        self,
        strat:        Dict[str, Any],
        final_symbols: List[str],
        exchange,
    ) -> List[Dict]:
        strat_id = strat['id']
        signals  = self._detect_real_signals(strat, final_symbols, None)
        logger.info(f"Signals detected {strat_id}: {len(signals)}")
        return signals

    def execute_signals(
        self,
        strat:            Dict[str, Any],
        signals:          List[Dict],
        open_positions:   Dict,
        strategy_candles: Dict,
        order_amount:     float,
    ) -> None:

        strat_id = strat['id']

        if not signals:
            return

        reset_strategy_candles(
            strat_id,
            strategy_candles,
            open_positions,
            self.account_number,
            self.state_file
        )

        for sig in signals:
            self.operative.place_order(
                symbol       = sig['symbol'],
                direction    = strat['direction'],
                usdt_amount  = order_amount,
                tp_pct       = strat['tp_pct'],
                sl_pct       = strat['sl_pct'],
                strategy_id  = strat_id,
                signal_close = sig.get('close', 0),
            )
            time.sleep(0.05)