#core/demo_operative.py

import os
import json
import time
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import logging

from execution.order_manager import get_current_price

logger = logging.getLogger('BOT_crypto.demo_operative')
from config.settings import HOUR_ZONE, COMMISSION_PCT

class DemoOperative:

    
    def __init__(self, account_number: str, ws_manager, excel_path: str, 
                 strategy_configs: List[Dict]):

        self.account_number = account_number
        self.ws_manager = ws_manager
        self.excel_path = excel_path
        self.strategy_configs = {s['id']: s for s in strategy_configs}
        
        # JSON state file path
        state_dir = os.path.dirname(excel_path)
        self.json_path = os.path.join(state_dir, f'demo_state_{account_number}.json')
        
        # References (injected by orchestrator)
        self.open_positions: Optional[Dict] = None
        self.strategy_candles: Optional[Dict] = None
        self.state_file: Optional[str] = None  # Ignored in demo mode
        
        logger.info(f"[DEMO] Demo mode initialized for account {account_number}")
        logger.info(f"[DEMO] Persistence: JSON (state) + Excel (trades)")
        logger.info(f"[DEMO] JSON path: {self.json_path}")
        logger.info(f"[DEMO] Excel path: {excel_path}")
        logger.info(f"[DEMO] NO PostgreSQL writes")
        logger.info(f"[DEMO] NO broker API calls")
    
    
    def place_simulated_order(self, symbol: str, direction: str,
                         usdt_amount: float, tp_pct: float,
                         sl_pct: float, strategy_id: str) -> Optional[Dict]:
        try:
            current_price = get_current_price(symbol, max_cache_age=0.5)
            if not current_price:
                logger.warning(f"[DEMO] No price for {symbol}")
                return None
    
            entry_price = float(current_price)
    
            if direction.lower() == 'long':
                tp_price = entry_price * (1 + tp_pct / 100)
                sl_price = entry_price * (1 - sl_pct / 100)
            else:
                tp_price = entry_price * (1 - tp_pct / 100)
                sl_price = entry_price * (1 + sl_pct / 100)
    
            size = usdt_amount / entry_price
    
            position = {
                'symbol':               symbol,
                'size':                 size,
                'entry_price':          entry_price,
                'direction':            direction.lower(),
                'tp':                   tp_price,
                'sl':                   sl_price,
                'order_id':             f"demo_{int(time.time() * 1000000)}",
                'opened_at':            datetime.now(HOUR_ZONE),
                'usdt_amount':          usdt_amount,
                'tp_pct':               tp_pct,
                'sl_pct':               sl_pct,
                'simulated':            True
            }
    
            if strategy_id not in self.open_positions:
                self.open_positions[strategy_id] = []
    
            self.open_positions[strategy_id].append(position)
            self._save_state_json()
    
            logger.info(
                f"[DEMO] ENTRY {direction.upper()} {symbol} @ ${entry_price:.4f} | "
                f"${usdt_amount:.2f} | TP: ${tp_price:.4f} | SL: ${sl_price:.4f}"
            )
    
            return position
    
        except Exception as e:
            logger.error(f"[DEMO] Error placing simulated order {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    
    def increment_candles(self, strategy_id: str) -> None:
        if strategy_id not in self.strategy_candles:
            self.strategy_candles[strategy_id] = 0
        self.strategy_candles[strategy_id] += 1
        self._save_state_json()
    
    def monitor_exits(self) -> None:
        if not self.open_positions:
            return
        
        positions_to_close = []
        
        for strategy_id, positions in self.open_positions.items():
            if not positions:
                continue
            
            for position in positions:
                if not position.get('simulated', False):
                    continue
                
                symbol = position['symbol']
                
                try:
                    current_price = get_current_price(symbol, max_cache_age=0.5)
                except (TimeoutError, RuntimeError):
                    continue
                except Exception as e:
                    logger.error(f"[DEMO] Error getting price {symbol}: {e}")
                    continue
                
                try:
                    current_price_float = float(current_price)
                    direction = position['direction']
                    
                    tp_hit = (direction == 'long' and current_price_float >= position['tp']) or \
                             (direction == 'short' and current_price_float <= position['tp'])
                    
                    if tp_hit:
                        positions_to_close.append((strategy_id, position, 'TP', current_price_float))
                        continue
                    
                    sl_hit = (direction == 'long' and current_price_float <= position['sl']) or \
                             (direction == 'short' and current_price_float >= position['sl'])
                    
                    if sl_hit:
                        positions_to_close.append((strategy_id, position, 'SL', current_price_float))
                        continue
                    
                    if strategy_id in self.strategy_configs:
                        max_candles = self.strategy_configs[strategy_id].get('sell_after_ncandles', 50)
                        if self.strategy_candles.get(strategy_id, 0) >= max_candles:
                            positions_to_close.append((strategy_id, position, 'TIMEOUT', current_price_float))
                
                except Exception as e:
                    logger.error(f"[DEMO] Error processing {symbol}: {e}")
        
        for strategy_id, position, reason, close_price in positions_to_close:
            symbol   = position['symbol']
            now_time = datetime.now(HOUR_ZONE).strftime('%H:%M')
            logger.info(f"{reason} for {symbol} ({strategy_id}) at {now_time}")
            self._close_simulated_position(strategy_id, position, reason, close_price)
    
    
    def _close_simulated_position(self, strategy_id: str, position: Dict, 
                                  reason: str, close_price: float) -> None:
        try:
            entry_price = position['entry_price']
            direction   = position['direction']
            usdt_amount = position['usdt_amount']
            symbol      = position['symbol']
    
            # Calculate profit
            if direction == 'long':
                profit_pct = ((close_price - entry_price) / entry_price) * 100
            else:
                profit_pct = ((entry_price - close_price) / entry_price) * 100
    
            profit_usd = usdt_amount * (profit_pct / 100)
    
            # Apply commission (open + close)
            commission  = usdt_amount * (COMMISSION_PCT / 100) * 2
            profit_usd -= commission
            profit_pct -= (COMMISSION_PCT * 2)
    
            size = position.get('size', usdt_amount / entry_price)
    
            # Parse entry time
            entry_time = position['opened_at']
            if isinstance(entry_time, str):
                entry_time = datetime.strptime(entry_time, '%Y-%m-%d %H:%M:%S')
                entry_time = entry_time.replace(tzinfo=HOUR_ZONE)
    
            close_time = datetime.now(HOUR_ZONE)
    
            # Log to Excel ONLY (NO PostgreSQL)
            self._log_to_excel(
                strategy=strategy_id,
                symbol=symbol,
                direction=direction.upper(),
                entry_price=entry_price,
                close_price=close_price,
                profit=profit_usd,
                profit_pct=profit_pct,
                reason=reason,
                entry_time=entry_time,
                close_time=close_time,
                usdt_amount=usdt_amount,
                size=size,
                tp_target=position['tp'],
                sl_target=position['sl'],
                commission=commission,
            )
    
            logger.info(
                f"[DEMO] EXIT {symbol} | {reason} | "
                f"${profit_usd:.2f} ({profit_pct:+.2f}%) | "
                f"Fee: ${commission:.4f} | Strategy: {strategy_id}"
            )
    
            # Remove position by symbol
            if strategy_id in self.open_positions:
                count_before = len(self.open_positions[strategy_id])
                self.open_positions[strategy_id] = [
                    p for p in self.open_positions[strategy_id]
                    if p['symbol'] != symbol
                ]
                count_after = len(self.open_positions[strategy_id])
    
                if count_after == count_before:
                    logger.error(
                        f"[DEMO] CRITICAL: Failed to remove {symbol} from {strategy_id}! "
                        f"Positions before: {count_before}, after: {count_after}"
                    )
                    raise RuntimeError(f"Failed to remove position {symbol} from state")
                else:
                    logger.debug(
                        f"[DEMO] Removed {symbol} from {strategy_id} "
                        f"({count_before} → {count_after} positions)"
                    )
    
            self._save_state_json()
    
        except Exception as e:
            logger.error(f"[DEMO] Error closing position {position.get('symbol')}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    
    def _log_to_excel(self, strategy: str, symbol: str, direction: str,
                     entry_price: float, close_price: float, profit: float,
                     profit_pct: float, reason: str, entry_time: datetime,
                     close_time: datetime, usdt_amount: float, size: float,
                     tp_target: float, sl_target: float, commission: float = 0.0) -> None:
        try:
            duration_days = (close_time - entry_time).total_seconds() / (3600 * 24)
    
            trade_data = {
                'OPEN_AT':              entry_time.strftime('%Y-%m-%d %H:%M:%S'),
                'CLOSE_AT':             close_time.strftime('%Y-%m-%d %H:%M:%S'),
                'DURATION_DAYS':        round(duration_days, 4),
                'STRATEGY':             strategy,
                'SYMBOL':               symbol,
                'DIRECTION':            direction,
                'USDT_AMOUNT':          round(usdt_amount, 2),
                'SIZE':                 round(size, 6),
                'PRICE_ENTRY':          round(entry_price, 6),
                'PRICE_CLOSE':          round(close_price, 6),
                'PROFIT':               round(profit, 2),
                'FEE':                  round(commission, 4),
                'PROFIT_PCT':           round(profit_pct, 2),
                'REASON_OUT':           reason,
                'ORDER_PRICE_CLOSE':    None,
                'ORDER_TS_CLOSE':       None,
                'EXEC_TS_CLOSE':        None,
                'TP_TARGET':            round(tp_target, 6),
                'SL_TARGET':            round(sl_target, 6),
                'SIMULATED':            'YES'
            }
    
            if os.path.exists(self.excel_path):
                df = pd.read_excel(self.excel_path, engine='openpyxl')
                df = pd.concat([df, pd.DataFrame([trade_data])], ignore_index=True)
            else:
                df = pd.DataFrame([trade_data])
    
            df.to_excel(self.excel_path, index=False, engine='openpyxl')
    
        except Exception as e:
            logger.error(f"[DEMO] Error logging to Excel: {e}")
            import traceback
            traceback.print_exc()
    
    
    def _save_state_json(self) -> None:

        try:
            if self.open_positions is None or self.strategy_candles is None:
                return
            
            # Serialize datetime objects to ISO format
            serializable_positions = {}
            for strategy_id, positions in self.open_positions.items():
                serializable_positions[strategy_id] = []
                for pos in positions:
                    pos_copy = pos.copy()
                    if isinstance(pos_copy.get('opened_at'), datetime):
                        pos_copy['opened_at'] = pos_copy['opened_at'].isoformat()
                    serializable_positions[strategy_id].append(pos_copy)
            
            state_data = {
                'open_positions': serializable_positions,  # ← CAMBIAR AQUÍ
                'strategy_candles': self.strategy_candles,
                'account': self.account_number,
                'last_updated': datetime.now(HOUR_ZONE).isoformat()
            }
            
            # Write to JSON
            with open(self.json_path, 'w') as f:
                json.dump(state_data, f, indent=2)
            
        except Exception as e:
            logger.error(f"[DEMO] Error saving JSON state: {e}")
            import traceback
            traceback.print_exc()
    
    
    def get_open_positions_count(self) -> int:
        """Get count of open simulated positions."""
        if not self.open_positions:
            return 0
        
        count = 0
        for positions in self.open_positions.values():
            count += len([p for p in positions if p.get('simulated', False)])
        
        return count
    
    
    def get_positions_by_strategy(self, strategy_id: str) -> List[Dict]:
        """Get positions for strategy."""
        if not self.open_positions or strategy_id not in self.open_positions:
            return []
        
        return [p for p in self.open_positions[strategy_id] if p.get('simulated', False)]
    
    def attach(self, open_positions: Dict, strategy_candles: Dict,
           strategies: List[Dict]) -> None:
        self.open_positions   = open_positions
        self.strategy_candles = strategy_candles
        self.strategy_configs = {s['id']: s for s in strategies}
        logger.debug("[DEMO] Shared references attached")
        
    def load_state(self) -> tuple:

        if not os.path.exists(self.json_path):
            logger.warning(f"[DEMO] No state file found, returning empty state")
            return {}, {}
    
        with open(self.json_path, 'r') as f:
            data = json.load(f)
    
        positions = {}
        for strategy_id, pos_list in data.get('open_positions', {}).items():
            positions[strategy_id] = []
            for pos in pos_list:
                pos_copy = pos.copy()
                if isinstance(pos_copy.get('opened_at'), str):
                    pos_copy['opened_at'] = datetime.fromisoformat(pos_copy['opened_at'])
                positions[strategy_id].append(pos_copy)
    
        total = sum(len(p) for p in positions.values())
        logger.info(f"[DEMO] Loaded {total} positions from {self.json_path}")
    
        return positions, data.get('strategy_candles', {})
    
    def save_state(self) -> None:
        self._save_state_json()
        
    def place_order(self, symbol: str, direction: str, usdt_amount: float,
                    tp_pct: float, sl_pct: float, strategy_id: str,
                    signal_close: float = 0) -> None:
        self.place_simulated_order(
            symbol=symbol,
            direction=direction,
            usdt_amount=usdt_amount,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            strategy_id=strategy_id
        )
        
    def sync_broker(self) -> None:
        """No-op in demo mode — no broker to sync with."""
        logger.debug("[DEMO] Skipping broker sync (demo mode)")
