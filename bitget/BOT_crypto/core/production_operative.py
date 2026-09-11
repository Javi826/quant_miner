#BOT_crypto/core/production_operative.py


import logging
from typing import Dict, List, Tuple

from state.state_manager import load_state, save_state_local, sync_broker
from state import increment_strategy_candles
from execution import check_all_tp_sl, check_tp_sl_for_strategy, place_order, add_position, get_fills_for_order, get_usdt_balance_ws
import time
from decimal import Decimal
from config.settings import HOUR_ZONE
logger = logging.getLogger('BOT_crypto.core.production_operative')


class ProductionOperative:


    def __init__(self, account_number: str, state_file: str,
                 send_request_func, bot_state):
        self.account_number    = account_number
        self.state_file        = state_file
        self.send_request_func = send_request_func
        self.bot_state         = bot_state
        self.strategies:       List[Dict] = []
        self.open_positions:   Dict       = {}
        self.strategy_candles: Dict       = {}
    
    def attach(self, open_positions: Dict, strategy_candles: Dict, 
               strategies: List[Dict]) -> None:
        self.open_positions   = open_positions
        self.strategy_candles = strategy_candles
        self.strategies       = strategies
        logger.debug("[PRODUCTION] Shared references attached")

    def load_state(self) -> Tuple[Dict, Dict]:
        """Load state from PostgreSQL (primary) or JSON (fallback)."""
        return load_state(self.account_number, self.state_file)

    def save_state(self) -> None:
        """Persist state to PostgreSQL + JSON."""
        save_state_local(
            self.open_positions,
            self.strategy_candles,
            self.account_number,
            self.state_file
        )

    def sync_broker(self) -> None:
        """Reconcile local state with broker positions."""
        sync_broker(
            self.open_positions,
            self.strategy_candles,
            self.account_number,
            self.state_file
        )

    def monitor_exits(self) -> None:
        """Monitor open positions for TP/SL exits via broker."""
        check_all_tp_sl(
            self.strategies,
            self.open_positions,
            self.strategy_candles,
            self.account_number,
            self.state_file,
            self.send_request_func,
            check_tp_sl_for_strategy,
            bot_state=self.bot_state
        )

    def increment_candles(self, strat_id: str) -> None:

        increment_strategy_candles(
            strat_id,
            self.strategy_candles,
            self.open_positions,
            self.account_number,
            self.state_file
        )
        
    def place_order(self, symbol: str, direction: str, usdt_amount: float,
                tp_pct: float, sl_pct: float, strategy_id: str,
                signal_close: float = 0) -> None:

        usdt_balance = get_usdt_balance_ws()
        if usdt_balance < usdt_amount:
            logger.warning(f"WAR-Insufficient balance ({usdt_balance:.2f} USDT) for {symbol}")
            return
    
        order_result = place_order(
            symbol=symbol,
            direction=direction,
            usdt_amount=usdt_amount,
            send_request_func=self.send_request_func
        )
    
        if order_result is None:
            logger.error(f"Error-Failed to place order for {symbol}")
            return
    
        resp_order = order_result.get('resp_order')
        if resp_order is None:
            logger.error(f"Error-Invalid order result for {symbol}")
            return
    
        data     = resp_order.get('data', {}) if isinstance(resp_order, dict) else {}
        order_id = data.get('orderId')
    
        if order_id:
            filled_size, entry_price_from_fills, _, _ = get_fills_for_order(
                order_id=order_id,
                symbol=symbol,
                send_request_func=self.send_request_func
            )
            time.sleep(0.05)

            if filled_size is None or filled_size == 0:
                size        = Decimal(str(data.get('size', data.get('filledQty', data.get('baseVolume', 0)))))
                entry_price = Decimal(str(data.get('price', signal_close)))
            else:
                size        = filled_size
                entry_price = entry_price_from_fills if entry_price_from_fills is not None else Decimal(str(signal_close))

            add_position(
                strat_id=strategy_id,
                symbol=symbol,
                size=size,
                entry_price=entry_price,
                direction=direction,
                tp_pct=tp_pct,
                sl_pct=sl_pct,
                order_id=order_id,
                open_positions=self.open_positions,
                strategy_candles=self.strategy_candles,
                account_number=self.account_number,
                state_file=self.state_file,
                hour_zone=HOUR_ZONE,
                usdt_amount=usdt_amount
            )
            logger.info(
                f"{direction.upper()} {symbol}  ${float(entry_price):.4f} | "
                f"${usdt_amount:.2f}"
            )
        else:
            logger.warning(f"WAR-Order executed but no orderId for {symbol}")