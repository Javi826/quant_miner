#BOT_crypto/execution/position_tracker.py
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional
import logging
from state.state_manager import save_state_local
logger = logging.getLogger('BOT_crypto.execution.position_tracker')

from execution.order_manager import get_current_price, close_position

# ==========================================================================
# TP/SL CALCULATIONS
# ==========================================================================
def calculate_tp_sl_prices(entry_price: Decimal, 
                          direction: str, 
                          tp_pct: float, 
                          sl_pct: float) -> tuple[Decimal, Decimal]:

    entry = Decimal(str(entry_price))
    tp_decimal = Decimal(str(tp_pct)) / Decimal('100')
    sl_decimal = Decimal(str(sl_pct)) / Decimal('100')
    
    if direction.lower() == 'long':
        tp_price = entry * (Decimal('1') + tp_decimal)
        sl_price = entry * (Decimal('1') - sl_decimal)
    else:  # short
        tp_price = entry * (Decimal('1') - tp_decimal)
        sl_price = entry * (Decimal('1') + sl_decimal)
    
    return tp_price, sl_price


def calculate_pnl(direction: str, 
                 entry_price: Decimal, 
                 current_price: Decimal, 
                 size: Decimal) -> float:

    entry_float   = float(entry_price)
    current_float = float(current_price)
    size_float    = float(size)
    
    if direction.lower() == 'long':
        pnl = (current_float - entry_float) * size_float
    else:  # short
        pnl = (entry_float - current_float) * size_float
    
    return pnl


# ==========================================================================
# POSITION MANAGEMENT
# ==========================================================================
def add_position(
    strat_id: str, 
    symbol: str, 
    size: Decimal, 
    entry_price: Decimal,
    direction: str, 
    tp_pct: float, 
    sl_pct: float, 
    order_id: str,
    open_positions: Dict, 
    strategy_candles: Dict,
    account_number: str,  
    state_file: str,
    hour_zone, 
    usdt_amount: float = 0,
) -> None:
    """
    Add a new position to tracking system.
    
    This function:
    1. Creates position dictionary with TP/SL prices
    2. Adds to open_positions tracking
    3. Saves state to file
    
    Args:
        strat_id: Strategy ID
        symbol: Trading symbol
        size: Position size
        entry_price: Entry price
        direction: 'long' or 'short'
        tp_pct: Take profit percentage
        sl_pct: Stop loss percentage
        order_id: Order ID
        open_positions: Dictionary of open positions
        strategy_candles: Dictionary of candle counters
        state_file: Path to state file
        hour_zone: Timezone for timestamps
        usdt_amount: USDT amount invested

    """
    
    
    if strat_id not in open_positions:
        open_positions[strat_id] = []

    tp_price, sl_price = calculate_tp_sl_prices(entry_price, direction, tp_pct, sl_pct)
    
    position = {
        'symbol': symbol,
        'size': size,
        'entry_price': entry_price,
        'direction': direction,
        'tp': tp_price,
        'sl': sl_price,
        'order_id': order_id,
        'opened_at': datetime.now(hour_zone),
        'usdt_amount': usdt_amount,
    }
    
    open_positions[strat_id].append(position)
    try:
        save_state_local(open_positions, strategy_candles, account_number, state_file)
    except Exception as e:
        logger.error(f"CRITICAL ERROR saving state after opening position")
        logger.error(f"Strategy: {strat_id}, Symbol: {symbol}, Direction: {direction}")
        logger.error(f"Position was opened in broker but NOT saved to state!")
        logger.error(f"Error: {e}")
        raise


# ==========================================================================
# TP/SL MONITORING
# ==========================================================================
def check_tp_sl_for_strategy(strat_id: str, 
                             strat_config: Dict,
                             open_positions: Dict, 
                             strategy_candles: Dict,
                             account_number: str,  # ← AÑADIR AQUÍ
                             state_file: str, 
                             send_request_func,
                             pnl_accumulator: Optional[Dict] = None,
                             bot_state=None) -> None:
    """
    Check TP/SL for all positions of a strategy via WebSocket.
    
    This function:
    1. Iterates through all positions for the strategy
    2. Gets current price via WebSocket
    3. Checks if TP or SL is hit
    4. Closes position if hit
    5. Updates state
    
    Args:
        strat_id: Strategy ID
        strat_config: Strategy configuration dictionary
        open_positions: Dictionary of open positions
        strategy_candles: Dictionary of candle counters
        state_file: Path to state file
        send_request_func: Function to send REST requests
        pnl_accumulator: Dictionary to accumulate PnL
        bot_state: Bot state for profit tracking
    """
    from state.state_manager import save_state_local
    
    if strat_id not in open_positions or not open_positions[strat_id]:
        return
    
    positions = open_positions[strat_id][:]
    positions_to_remove = []
    
    sell_after_ncandles = strat_config.get('sell_after_ncandles') if strat_config else None
    
    for i, pos in enumerate(positions):
        try:
            symbol = pos['symbol']
            
            # Get current price via WebSocket
            try:
                current_price = get_current_price(symbol, max_cache_age=0.5)
            except (TimeoutError, RuntimeError) as e:
                logger.info(f"No price for {symbol}: {e}")
                continue
            
            if current_price is None:
                continue
            
            direction   = pos['direction']
            tp_price    = pos['tp']
            sl_price    = pos['sl']
            entry_price = pos['entry_price']
            
            current_price = Decimal(str(current_price))
            
            # Calculate PnL if accumulator provided
            if pnl_accumulator is not None:
                pnl = calculate_pnl(direction, entry_price, current_price, pos['size'])
                pnl_accumulator['total'] += pnl
                   
            # Check TP/SL hits
            hit_tp = current_price >= tp_price if direction.lower() == 'long' else current_price <= tp_price
            hit_sl = current_price <= sl_price if direction.lower() == 'long' else current_price >= sl_price
            
            if hit_tp:
                position_data = {
                    'opened_at': pos['opened_at'],
                    'strategy_id': strat_id,
                    'usdt_amount': pos.get('usdt_amount', 0),
                    'entry_price': pos['entry_price'],
                    'tp': pos.get('tp'),  
                    'sl': pos.get('sl'),          
                }
                if close_position(symbol, pos['size'], direction, send_request_func, 
                                reason="TP", position_data=position_data, bot_state=bot_state):
                    positions_to_remove.append(i)
                    
            elif hit_sl:
                position_data = {
                    'opened_at': pos['opened_at'],
                    'strategy_id': strat_id,
                    'usdt_amount': pos.get('usdt_amount', 0),
                    'entry_price': pos['entry_price'],
                    'tp': pos.get('tp'),   
                    'sl': pos.get('sl'),        
                }
                if close_position(symbol, pos['size'], direction, send_request_func, 
                                reason="SL", position_data=position_data, bot_state=bot_state):
                    positions_to_remove.append(i)
                    
        except Exception as e:
            logger.error(f"CRITICAL ERROR processing position {symbol} in strategy {strat_id}")
            logger.error(f"Position index: {i}, Direction: {pos.get('direction')}")
            logger.error(f"Error: {e}")
            raise  # Bot stops with full context
    
    
    # Remove closed positions
    if positions_to_remove:
        for i in reversed(positions_to_remove):
            if i < len(open_positions[strat_id]):
                open_positions[strat_id].pop(i)
        try:
            save_state_local(open_positions, strategy_candles, account_number, state_file)
        except Exception as e:
            logger.error(f"CRITICAL ERROR saving state after closing positions")
            logger.error(f"Strategy: {strat_id}, Positions removed: {len(positions_to_remove)}")
            logger.error(f"Error: {e}")
            raise


def check_all_tp_sl(
    strategies: List[Dict], 
    open_positions: Dict,
    strategy_candles: Dict,
    account_number: str,  # ← AÑADIR
    state_file: str,
    send_request_func, 
    check_tp_sl_for_strategy_func,
    bot_state=None
) -> Dict:
    """
    Check TP/SL for all strategies (simplified for dashboard).
    
    This function iterates through all active strategies and checks
    TP/SL for their positions. It's designed to be called periodically
    from the main bot loop.
    
    Args:
        strategies: List of strategy configurations
        open_positions: Dictionary of open positions
        strategy_candles: Dictionary of candle counters
        state_file: Path to state file
        send_request_func: Function to send REST requests
        check_tp_sl_for_strategy_func: Function to check TP/SL
        bot_state: Bot state for profit tracking
    
    Returns:
        Dictionary with total unrealized PnL
    """
    
    # PnL accumulator
    pnl_accumulator = {'total': 0.0}
    
    # Process all strategies
    for strat in strategies:
        strat_id = strat['id']
        positions = open_positions.get(strat_id, [])
        
        if positions:
            try:
                strat_pnl_acc = {'total': 0.0}
                check_tp_sl_for_strategy_func(
                    strat_id, 
                    strat, 
                    open_positions, 
                    strategy_candles,
                    account_number,
                    state_file, 
                    send_request_func, 
                    strat_pnl_acc, 
                    bot_state
                )
                pnl_accumulator['total'] += strat_pnl_acc['total']
            except Exception as e:
               logger.error(f"CRITICAL ERROR checking TP/SL for strategy {strat_id}")
               logger.error(f"Account: {account_number}, Positions: {len(positions)}")
               logger.error(f"Error: {e}")
               raise  # Bot stops but with full context logged
    
    return pnl_accumulator