#BOT_crypto/state/state_manager.py
"""
state/state_manager.py State Manager - Handles bot state persistence and broker synchronization.

This module is responsible for:
- Loading bot state from JSON file
- Saving bot state to JSON file (PRIMARY + dual-write to PostgreSQL)
- Synchronizing local state with broker positions

This module imports from trade_logger for logging removed positions,
but trade_logger does not import back, avoiding circular dependencies.
"""

import os
import json
import copy
import traceback
from datetime import datetime
from decimal import Decimal
from typing import Dict, Tuple

from market_data import get_ws_manager
from config.settings import POSTGRES_CONFIG


import logging
logger = logging.getLogger('BOT_crypto.execution.state_manager')


# ==========================================================================
# POSTGRESQL CONFIGURATION (DUAL-WRITE)
# ==========================================================================
from config.settings import ACCOUNTS

POSTGRES_ENABLED = True  # default, overridden at runtime by configure_postgres()

def configure_postgres(account_number: str) -> None:
    global POSTGRES_ENABLED
    POSTGRES_ENABLED = ACCOUNTS.get(account_number, {}).get('postgresql_enabled', True)
    
IS_DEMO = False  # default, overridden at runtime by configure_demo()

def configure_demo(account_number: str) -> None:
    global IS_DEMO
    IS_DEMO = ACCOUNTS.get(account_number, {}).get('type') == 'demo'
# ==========================================================================
# STATE PERSISTENCE
# ==========================================================================
class BotState:
    def __init__(self):
        self.closed_total_profit = 0.0
        
def load_state(account_number: str, state_file: str) -> Tuple[Dict, Dict]:
    """..."""
        
    OPEN_POSITIONS = {}
    STRATEGY_CANDLES = {}
    
    logger.info(f"Loading state for account {account_number}...")
    
    
    # ======================================================================
    # PRIMARY: Try PostgreSQL first
    # ======================================================================
    if POSTGRES_ENABLED:
        try:
            import psycopg2
            
            conn = psycopg2.connect(**POSTGRES_CONFIG)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT state_data FROM bot_state WHERE account = %s",
                (account_number,)
            )
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result:
                # Parse JSON from PostgreSQL
                data = result[0]
                
                STRATEGY_CANDLES = data.get('strategy_candles', {})
                positions_data = data.get('positions', {})
                
                # Reconstruct positions with Decimal types
                for strat_id, positions in positions_data.items():
                    OPEN_POSITIONS[strat_id] = []
                    for pos in positions:
                        OPEN_POSITIONS[strat_id].append({
                            'symbol': pos.get('symbol'),
                            'size': Decimal(pos.get('size')),
                            'entry_price': Decimal(pos.get('entry_price')),
                            'direction': pos.get('direction'),
                            'tp': Decimal(pos.get('tp')),
                            'sl': Decimal(pos.get('sl')),
                            'order_id': pos.get('order_id'),
                            'opened_at': datetime.fromisoformat(pos.get('opened_at')),
                            'usdt_amount': float(pos.get('usdt_amount', 0)),
                        })
                
                total_positions = sum(len(p) for p in OPEN_POSITIONS.values())
                
                logger.info(f"State loaded from PostgreSQL: {total_positions} positions")
                logger.info(f"{'-' * 48}")
                
                # Display summary
                for strat_id, positions in OPEN_POSITIONS.items():
                    if positions:
                        candles = STRATEGY_CANDLES.get(strat_id, 0)
                        logger.debug(f"{strat_id:<24}: {len(positions):>2} positions | Candles: {candles:>2}")
                
                return OPEN_POSITIONS, STRATEGY_CANDLES
            else:
                logger.warning(f"No state found in PostgreSQL for account {account_number}")
                # Fall through to JSON fallback
                
        except Exception as e:
            logger.error(f"Error-PostgreSQL load failed: {e}")
            logger.info("Falling back to JSON...")
            # Fall through to JSON fallback
    
    # ======================================================================
    # FALLBACK: Load from JSON
    # ======================================================================
    logger.info(f"Loading from JSON fallback: {state_file}")
    
    if not os.path.exists(state_file):
        logger.info(f"No previous state file found")
        return OPEN_POSITIONS, STRATEGY_CANDLES
    
    try:
        with open(state_file, 'r') as f:
            data = json.load(f)
        
        STRATEGY_CANDLES = data.get('strategy_candles', {})
        positions_data = data.get('positions', {})
        
        # Reconstruct positions with Decimal types
        for strat_id, positions in positions_data.items():
            OPEN_POSITIONS[strat_id] = []
            for pos in positions:
                OPEN_POSITIONS[strat_id].append({
                    'symbol': pos.get('symbol'),
                    'size': Decimal(pos.get('size')),
                    'entry_price': Decimal(pos.get('entry_price')),
                    'direction': pos.get('direction'),
                    'tp': Decimal(pos.get('tp')),
                    'sl': Decimal(pos.get('sl')),
                    'order_id': pos.get('order_id'),
                    'opened_at': datetime.fromisoformat(pos.get('opened_at')),
                    'usdt_amount': float(pos.get('usdt_amount', 0)),
                })
        
        total_positions = sum(len(p) for p in OPEN_POSITIONS.values())
        
        logger.debug(f"State loaded from JSON: {total_positions} positions")
        logger.debug(f"{'-' * 48}")
        
        # Display summary
        for strat_id, positions in OPEN_POSITIONS.items():
            if positions:
                candles = STRATEGY_CANDLES.get(strat_id, 0)
                logger.debug(f"{strat_id:<24}: {len(positions):>2} positions | Candles: {candles:>2}")
        
        return OPEN_POSITIONS, STRATEGY_CANDLES
        
    except Exception as e:
        logger.error(f"✗ Error loading state from JSON: {e}")
        traceback.print_exc()
        return OPEN_POSITIONS, STRATEGY_CANDLES


def save_state_local(
    open_positions: Dict,
    strategy_candles: Dict,
    account_number: str,
    state_file: str
) -> None:
    
    if IS_DEMO:
        return
    
    try:
        # Deep copy to avoid modifying original
        positions_copy = copy.deepcopy(open_positions)
        strategy_candles_copy = copy.deepcopy(strategy_candles)

        # Convert Decimal and datetime to serializable types
        serializable_positions = {}
        for strat_id, positions in positions_copy.items():
            serializable_positions[strat_id] = []
            for pos in positions:
                serializable_positions[strat_id].append({
                    'symbol': pos['symbol'],
                    'size': str(pos['size']),
                    'entry_price': str(pos['entry_price']),
                    'direction': pos['direction'],
                    'tp': str(pos['tp']),
                    'sl': str(pos['sl']),
                    'order_id': pos['order_id'],
                    'opened_at': pos['opened_at'].isoformat(),
                    'usdt_amount': float(pos.get('usdt_amount', 0)),
                })

        state_data = {
            'positions': serializable_positions,
            'strategy_candles': strategy_candles_copy
        }

        # ======================================================================
        # DUAL-WRITE: JSON (PRIMARY) + PostgreSQL (BACKUP)
        # ======================================================================
        
        json_success = False
        postgres_success = False
        
        # Write to JSON (primary)
        try:
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            json_success = True
        except Exception as e:
            logger.error(f"✗ JSON state save failed: {e}")
        
        # Write to PostgreSQL (dual-write backup)
        if POSTGRES_ENABLED:
            try:
                import psycopg2
                from psycopg2 import sql
                
                # Use account_number parameter directly (no extraction needed)
                account = account_number
                
                # Connect and upsert
                conn = psycopg2.connect(**POSTGRES_CONFIG)
                cursor = conn.cursor()
                
                upsert_query = sql.SQL("""
                    INSERT INTO bot_state (account, state_data, updated_at)
                    VALUES (%(account)s, %(state_data)s, CURRENT_TIMESTAMP)
                    ON CONFLICT (account)
                    DO UPDATE SET
                        state_data = EXCLUDED.state_data,
                        updated_at = CURRENT_TIMESTAMP
                """)
                
                cursor.execute(upsert_query, {
                    'account': account,
                    'state_data': json.dumps(state_data)
                })
                
                conn.commit()
                cursor.close()
                conn.close()
                
                postgres_success = True
            except Exception as e:
                logger.error(f"Error-PostgreSQL state save failed: {e}")
        
        # Log dual-write status (only if verbose)
        status_indicators = []
        if postgres_success:
            status_indicators.append("PG✓")
        if json_success:
            status_indicators.append("JSON✓")
        
        if status_indicators:
            total_positions = sum(len(p) for p in open_positions.values())
            logger.debug(f"[{' '.join(status_indicators)}] State saved: {total_positions} positions")
            
    except Exception as e:
        logger.error(f"Error-saving state: {e}")
        traceback.print_exc()

# ==========================================================================
# BROKER SYNCHRONIZATION (READ-ONLY MONITORING)
# ==========================================================================
def sync_broker(open_positions: Dict, 
                strategy_candles: Dict,
                account_number: str,
                state_file: str) -> None:

    import time
    from alerts.telegram_notifier import send_sync_alert
    from execution.trade_logger import log_sync_discrepancy

    
    if not get_ws_manager():
        raise RuntimeError("WebSocket manager not init.")
    
    ws = get_ws_manager()
    
    # Refresh WebSocket position data
    ws.refresh_positions()
    
    # Give extra time for snapshot to arrive (thread synchronization)
    time.sleep(1.5)
    
    # Count total local positions
    total_local = sum(len(positions) for positions in open_positions.values())
    
    # CASE 1: Both empty → Nothing to sync
    if total_local == 0 and len(ws.positions) == 0:
        logger.info("[SYNC] No positions to sync")
        return
    
    # CASE 2: Local has positions but WS empty → Timing issue, skip
    if total_local > 0 and len(ws.positions) == 0:
        logger.error(
            f"[SYNC] Local has {total_local} position(s) but WebSocket empty "
            f"- SKIPPING (snapshot timing issue)"
        )
        return
    
    # CASE 3: WS has data → Proceed with aggregation and comparison
    
    # ==========================================================================
    # STEP 1: Aggregate local positions by symbol + direction
    # ==========================================================================
    local_by_symbol = {}
    
    for strat_id, positions in open_positions.items():
        for pos in positions:
            symbol = pos['symbol']
            direction = pos['direction'].lower()
            size = float(pos['size'])
            
            if symbol not in local_by_symbol:
                local_by_symbol[symbol] = {
                    'long_size': 0.0,
                    'long_strategies': [],
                    'short_size': 0.0,
                    'short_strategies': []
                }
            
            if direction == 'long':
                local_by_symbol[symbol]['long_size'] += size
                local_by_symbol[symbol]['long_strategies'].append(strat_id)
            else:  # short
                local_by_symbol[symbol]['short_size'] += size
                local_by_symbol[symbol]['short_strategies'].append(strat_id)
    
    # DEBUG: Show aggregated local positions
    logger.debug(f"[SYNC DEBUG] Local aggregation: {len(local_by_symbol)} symbols")
    for sym, data in local_by_symbol.items():
        logger.debug(f"[SYNC DEBUG]   {sym}: LONG={data['long_size']:.6f}, SHORT={data['short_size']:.6f}")
    
    # ==========================================================================
    # STEP 2: Compare each symbol with broker (HEDGE MODE SAFE + PRECISION)
    # ==========================================================================
    total_issues = 0
    
    for symbol, local_data in local_by_symbol.items():
        try:
            local_long = local_data['long_size']
            local_short = local_data['short_size']
            
            # Get broker positions for this symbol (hedge mode safe)
            broker_positions = ws.get_positions_by_symbol(symbol)
            
            broker_long = 0.0
            broker_short = 0.0
            
            if broker_positions['long']:
                broker_long = float(broker_positions['long'].get('total', 0))
            
            if broker_positions['short']:
                broker_short = float(broker_positions['short'].get('total', 0))
            
            # DEBUG: Show broker positions
            logger.debug(f"[SYNC DEBUG] Broker {symbol}: LONG={broker_long:.6f}, SHORT={broker_short:.6f}")
            
            # ======================================================================
            # GET CONTRACT PRECISION (volumePlace from WebSocket cache)
            # ======================================================================
            tolerance = 0.0001  # Default fallback
            
            try:
                contract = ws.get_contract(symbol)
                if contract and 'volumePlace' in contract:
                    volume_place = int(contract['volumePlace'])
                    # Tolerance = 1 unit at the precision level
                    # Example: volumePlace=2 → tolerance=0.01
                    #          volumePlace=4 → tolerance=0.0001
                    tolerance = 10 ** (-volume_place)
            except Exception as e:
                logger.debug(f"[SYNC] Could not get contract precision for {symbol}: {e}")
            
            logger.debug(f"[SYNC DEBUG] {symbol} tolerance: {tolerance}")
            
            # ======================================================================
            # CHECK LONG POSITIONS
            # ======================================================================
            if local_long > 0:
                diff_long = abs(broker_long - local_long)
                logger.debug(f"[SYNC DEBUG] {symbol} LONG diff: {diff_long:.8f} (tolerance: {tolerance})")
                
                if broker_long == 0:
                    # NOT_IN_BROKER: Local has LONG but broker doesn't
                    logger.warning(
                        f"[SYNC] Position NOT FOUND in broker: {symbol} LONG "
                        f"| Local: {local_long} | Broker: 0.0 "
                        f"| Strategies: {', '.join(local_data['long_strategies'])}"
                    )
                    
                    # Telegram alert
                    send_sync_alert(
                        account=account_number,
                        symbol=symbol,
                        issue_type="NOT_IN_BROKER_LONG",
                        local_size=local_long,
                        broker_size=0.0,
                        strategies=local_data['long_strategies']
                    )
                    #from execution.trade_logger import log_sync_discrepancy
                    # PostgreSQL log
                    log_sync_discrepancy(
                        account=account_number,
                        symbol=symbol,
                        issue_type="not_in_broker_long",
                        local_size=local_long,
                        broker_size=0.0,
                        strategies=local_data['long_strategies']
                    )
                    
                    total_issues += 1
                    
                elif diff_long >= tolerance:
                    # SIZE_MISMATCH: Both have LONG but different sizes
                    logger.warning(
                        f"[SYNC] SIZE MISMATCH: {symbol} LONG "
                        f"| Local: {local_long} | Broker: {broker_long} "
                        f"| Diff: {diff_long:.8f} (tolerance: {tolerance}) "
                        f"| Strategies: {', '.join(local_data['long_strategies'])}"
                    )
                    
                    # Telegram alert
                    send_sync_alert(
                        account=account_number,
                        symbol=symbol,
                        issue_type="SIZE_MISMATCH_LONG",
                        local_size=local_long,
                        broker_size=broker_long,
                        strategies=local_data['long_strategies']
                    )
                    
                    # PostgreSQL log
                    log_sync_discrepancy(
                        account=account_number,
                        symbol=symbol,
                        issue_type="size_mismatch_long",
                        local_size=local_long,
                        broker_size=broker_long,
                        strategies=local_data['long_strategies']
                    )
                    
                    total_issues += 1
            
            # ======================================================================
            # CHECK SHORT POSITIONS
            # ======================================================================
            if local_short > 0:
                diff_short = abs(broker_short - local_short)
                logger.debug(f"[SYNC DEBUG] {symbol} SHORT diff: {diff_short:.8f} (tolerance: {tolerance})")
                
                if broker_short == 0:
                    # NOT_IN_BROKER: Local has SHORT but broker doesn't
                    logger.warning(
                        f"[SYNC] Position NOT FOUND in broker: {symbol} SHORT "
                        f"| Local: {local_short} | Broker: 0.0 "
                        f"| Strategies: {', '.join(local_data['short_strategies'])}"
                    )
                    
                    # Telegram alert
                    send_sync_alert(
                        account=account_number,
                        symbol=symbol,
                        issue_type="NOT_IN_BROKER_SHORT",
                        local_size=local_short,
                        broker_size=0.0,
                        strategies=local_data['short_strategies']
                    )
                    
                    # PostgreSQL log
                    log_sync_discrepancy(
                        account=account_number,
                        symbol=symbol,
                        issue_type="not_in_broker_short",
                        local_size=local_short,
                        broker_size=0.0,
                        strategies=local_data['short_strategies']
                    )
                    
                    total_issues += 1
                    
                elif diff_short >= tolerance:
                    # SIZE_MISMATCH: Both have SHORT but different sizes
                    logger.warning(
                        f"[SYNC] SIZE MISMATCH: {symbol} SHORT "
                        f"| Local: {local_short} | Broker: {broker_short} "
                        f"| Diff: {diff_short:.8f} (tolerance: {tolerance}) "
                        f"| Strategies: {', '.join(local_data['short_strategies'])}"
                    )
                    
                    # Telegram alert
                    send_sync_alert(
                        account=account_number,
                        symbol=symbol,
                        issue_type="SIZE_MISMATCH_SHORT",
                        local_size=local_short,
                        broker_size=broker_short,
                        strategies=local_data['short_strategies']
                    )
                    
                    # PostgreSQL log
                    log_sync_discrepancy(
                        account=account_number,
                        symbol=symbol,
                        issue_type="size_mismatch_short",
                        local_size=local_short,
                        broker_size=broker_short,
                        strategies=local_data['short_strategies']
                    )
                    
                    total_issues += 1
            
        except Exception as e:
            logger.error(f"[SYNC] Error checking {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    if total_issues > 0:
        logger.warning(
            f"[SYNC] Broker sync completed: {total_issues} issue(s) detected "
            f"- MANUAL REVIEW REQUIRED"
        )
    else:
        logger.info(f"[SYNC] Broker sync completed: All positions match")