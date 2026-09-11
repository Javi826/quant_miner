#BOT_crypto/execution/trade_logger.py

import os
import pandas as pd
import traceback
from datetime import datetime
from decimal import Decimal
from typing import Optional
from config.settings import POSTGRES_CONFIG
import logging
logger = logging.getLogger('BOT_crypto.execution.trade_logger')
from config.settings import HOUR_ZONE


# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# Excel configuration
TRADES_LOG_PATH = None

# PostgreSQL configuration (dual-write)
from config.settings import ACCOUNTS

POSTGRES_ENABLED = True  # default, overridden at runtime by configure_postgres()

def configure_postgres(account_number: str) -> None:
    global POSTGRES_ENABLED
    POSTGRES_ENABLED = ACCOUNTS.get(account_number, {}).get('postgresql_enabled', True)

# =============================================================================
# CONFIGURATION FUNCTIONS
# =============================================================================

def configure_log_path(trades_log_path: str) -> None:
    """
    Configure path for trade logging Excel file.
    
    Args:
        trades_log_path: Path to Excel file for trade logging
    """
    global TRADES_LOG_PATH
    TRADES_LOG_PATH = trades_log_path


# =============================================================================
# POSTGRESQL HOOK (NEW - NON-BLOCKING)
# =============================================================================

def _write_to_postgresql(opened_at_dt, closed_at, delta_days, strategy_id, symbol, 
                        direction, usdt_amount, size_val, entry_price, close_price,
                        profit, fee, profit_pct, reason,
                        order_price_close, order_ts_close, exec_ts_close,
                        tp_target, sl_target):  # ← AÑADIR AQUÍ
    """
    Write trade to PostgreSQL (dual-write hook).
    
    This function is called after all calculations are done.
    Errors are logged but don't affect Excel writing.
    """
    if not POSTGRES_ENABLED:
        return
    logger.debug(f"[POSTGRES] tp_target={tp_target}, sl_target={sl_target}, strategy={strategy_id}")
    
    try:
        import psycopg2
        from psycopg2 import sql
        
        # Extract account from TRADES_LOG_PATH
        account = 'unknown'
        if TRADES_LOG_PATH:
            filename = os.path.basename(TRADES_LOG_PATH)
            if filename.startswith('bot_trades_') and filename.endswith('.xlsx'):
                account = filename.replace('bot_trades_', '').replace('.xlsx', '')
        
        # Connect and insert
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        cursor = conn.cursor()
        
        insert_query = sql.SQL("""
            INSERT INTO trades (
                account, open_at, close_at, duration_days, strategy, symbol, direction,
                usdt_amount, size, price_entry, price_close, profit, fee,
                profit_pct, reason_out,
                order_price_close, order_ts_close, exec_ts_close,
                tp_target, sl_target  -- ← AÑADIR AQUÍ
            ) VALUES (
                %(account)s, %(open_at)s, %(close_at)s, %(duration_days)s, %(strategy)s,
                %(symbol)s, %(direction)s, %(usdt_amount)s, %(size)s,
                %(price_entry)s, %(price_close)s, %(profit)s, %(fee)s,
                %(profit_pct)s, %(reason_out)s,
                %(order_price_close)s, %(order_ts_close)s, %(exec_ts_close)s,
                %(tp_target)s, %(sl_target)s  -- ← AÑADIR AQUÍ
            )
        """)
        
        cursor.execute(insert_query, {
            'account': account,
            'open_at': opened_at_dt.replace(tzinfo=None),
            'close_at': closed_at.replace(tzinfo=None),
            'duration_days': round(delta_days, 4),
            'strategy': strategy_id,
            'symbol': symbol,
            'direction': direction.upper(),
            'usdt_amount': round(usdt_amount, 2),
            'size': round(size_val, 6) if size_val else None,
            'price_entry': round(entry_price, 6),
            'price_close': round(close_price, 6),
            'profit': round(profit, 2),
            'fee': round(fee, 4),
            'profit_pct': round(profit_pct, 1),
            'reason_out': reason,
            'order_price_close': order_price_close,
            'order_ts_close': order_ts_close,
            'exec_ts_close': exec_ts_close,
            'tp_target': tp_target,  # ← AÑADIR AQUÍ
            'sl_target': sl_target
        })
        
        conn.commit()
        logger.debug(f"[DEBUG POSTGRES] ✓ Trade inserted - tp_target={tp_target}, sl_target={sl_target}")
        
        cursor.close()
        conn.close()
        

    except Exception as e:
        logger.error(f"Error-PostgreSQL write FAILED: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# ORIGINAL CODE - UNCHANGED
# =============================================================================

def log_closed_position(opened_at,
                       strategy_id: str,
                       symbol: str,
                       direction: str,
                       usdt_amount: float,
                       entry_price: Decimal,
                       close_price: Decimal,
                       reason: str,
                       size: Decimal,
                       profit_from_api: Optional[Decimal] = None,
                       fee_from_api: Optional[Decimal] = None,
                       bot_state=None,
                       order_price_close: Optional[float] = None,
                       order_ts_close: Optional[float] = None,
                       exec_ts_close: Optional[float] = None,
                       tp_target: Optional[float] = None,
                       sl_target: Optional[float] = None
                       ) -> None: 
    """
    Log a closed position to Excel file.
    
    This function:
    1. Calculates profit and fees
    2. Formats trade data
    3. Appends to Excel file
    4. Updates bot state
    
    Args:
        opened_at: Position open timestamp (datetime or string)
        strategy_id: Strategy ID that opened the position
        symbol: Trading symbol (e.g., 'BTCUSDT')
        direction: Position direction ('long' or 'short')
        usdt_amount: USDT amount invested
        entry_price: Entry price
        close_price: Close price
        reason: Close reason ('TP', 'SL', 'TIMEOUT', 'OUT_OF_MARGIN', etc.)
        size: Position size
        profit_from_api: Profit from API (if available)
        fee_from_api: Fee from API (if available)
        bot_state: Bot state object for profit tracking
    """
    if TRADES_LOG_PATH is None:
        logger.warning("WAR-TRADES_LOG_PATH not configured. Trade not logged.")
        return
    
    try:
        # Convert to float
        entry_price = float(entry_price)
        close_price = float(close_price)
        usdt_amount = float(usdt_amount)

        # Extract size value
        size_val = None
        if size is not None:
            try:
                size_val = float(size)
            except Exception:
                size_val = None

        # Calculate USDT amount if not provided
        if usdt_amount == 0 and size_val is not None:
            usdt_amount = size_val * entry_price

        # Calculate profit
        if profit_from_api is not None:
            # Use API profit
            profit_gross = float(profit_from_api)
            fee = float(fee_from_api) if fee_from_api is not None else 0
            fee = 2 * fee  # Double fee (open + close)
            profit = profit_gross - fee
            
            if usdt_amount > 0:
                profit_pct = (profit / usdt_amount) * 100
            else:
                profit_pct = 0
        else:
            # Calculate from prices
            if size_val is not None:
                if direction.lower() == 'long':
                    profit     = (close_price - entry_price) * size_val
                    profit_pct = ((close_price - entry_price) / entry_price) * 100
                else:
                    profit     = (entry_price - close_price) * size_val
                    profit_pct = ((entry_price - close_price) / entry_price) * 100
            else:
                if direction.lower() == 'long':
                    profit     = (close_price - entry_price) * (usdt_amount / entry_price)
                    profit_pct = ((close_price - entry_price) / entry_price) * 100
                else:
                    profit     = (entry_price - close_price) * (usdt_amount / entry_price)
                    profit_pct = ((entry_price - close_price) / entry_price) * 100
            
            fee = 0

        closed_at = datetime.now(HOUR_ZONE)

        # Convert opened_at to datetime if string
        if isinstance(opened_at, str):
            opened_at_dt = datetime.strptime(opened_at, '%Y-%m-%d %H:%M:%S')
        else:
            opened_at_dt = opened_at

        # Ensure both are timezone-aware in UTC
        if opened_at_dt.tzinfo is None:
            opened_at_dt = opened_at_dt.replace(tzinfo=HOUR_ZONE)
        
        if closed_at.tzinfo is None:
            closed_at = closed_at.replace(tzinfo=HOUR_ZONE)

        # Calculate duration
        delta_days = (closed_at - opened_at_dt).total_seconds() / (3600 * 24)


        new_record = {
            'OPEN_AT': opened_at_dt.strftime('%Y-%m-%d %H:%M:%S'),
            'CLOSE_AT': closed_at.strftime('%Y-%m-%d %H:%M:%S'),
            'DURATION_DAYS': round(delta_days, 4),
            'STRATEGY': strategy_id,
            'SYMBOL': symbol,
            'DIRECTION': direction.upper(),
            'USDT_AMOUNT': round(usdt_amount, 2),
            'SIZE': round(size_val, 6) if size_val else 0,
            'PRICE_ENTRY': round(entry_price, 6),
            'PRICE_CLOSE': round(close_price, 6),
            'PROFIT': round(profit, 2),
            'FEE': round(fee, 4) if profit_from_api is not None else 0,
            'PROFIT_PCT': round(profit_pct, 1),
            'REASON_OUT': reason,
            'ORDER_PRICE_CLOSE': order_price_close,
            'ORDER_TS_CLOSE': order_ts_close,
            'EXEC_TS_CLOSE': exec_ts_close,
            'TP_TARGET': tp_target,  # ← AÑADIR
            'SL_TARGET': sl_target   # ← AÑADIR
        }

        # Append to Excel
        if os.path.exists(TRADES_LOG_PATH):
            df = pd.read_excel(TRADES_LOG_PATH)
            df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
        else:
            df = pd.DataFrame([new_record])

        df.to_excel(TRADES_LOG_PATH, index=False, engine='openpyxl')
        
        # Update bot state
        if bot_state is not None:
            bot_state.closed_total_profit += profit

        # =====================================================================
        # POSTGRESQL HOOK (NEW - added before logging)
        # =====================================================================
        _write_to_postgresql(
            opened_at_dt, closed_at, delta_days, strategy_id, symbol,
            direction, usdt_amount, size_val, entry_price, close_price,
            profit, fee, profit_pct, reason,
            order_price_close, order_ts_close, exec_ts_close,
            tp_target, sl_target
        )

        
        logger.info(f"Logged: {symbol} | Profit: {profit:.2f} $ ({profit_pct:+.2f}%)")

    except Exception as e:
        logger.error(f"Error-logging to Excel: {e}")
        traceback.print_exc()

# =============================================================================
# SYNC DISCREPANCY LOGGING (NEW)
# =============================================================================

def log_sync_discrepancy(account: str,
                        symbol: str,
                        issue_type: str,
                        local_size: float,
                        broker_size: float,
                        strategies: list) -> None:
    """
    Log a broker synchronization discrepancy to PostgreSQL.
    
    Args:
        account: Account identifier ('00', 'E1', '01')
        symbol: Trading symbol (e.g., 'BTCUSDT')
        issue_type: Type of issue ('not_in_broker', 'size_mismatch', 'not_in_local')
        local_size: Size in local state
        broker_size: Size in broker
        strategies: List of strategy IDs involved
    """
    if not POSTGRES_ENABLED:
        return
    
    try:
        import psycopg2
        from psycopg2 import sql
        
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        cursor = conn.cursor()
        
        insert_query = sql.SQL("""
            INSERT INTO sync_discrepancies 
            (timestamp, account, symbol, issue_type, local_size, broker_size, strategies)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """)
        
        cursor.execute(insert_query, [
            datetime.now(HOUR_ZONE),
            account,
            symbol,
            issue_type,
            local_size,
            broker_size,
            strategies
        ])
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.debug(f"[POSTGRES] ✓ Sync discrepancy logged: {symbol} - {issue_type}")
        
    except Exception as e:
        logger.error(f"Error-PostgreSQL sync discrepancy write FAILED: {e}")
        import traceback
        traceback.print_exc()