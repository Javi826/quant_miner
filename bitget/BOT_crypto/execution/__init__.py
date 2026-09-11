"""
Execution module - Order placement and position management.

This module handles all order execution logic including:
- Market order placement
- Position closing
- Position tracking and TP/SL monitoring
- Trade logging to Excel

Module organization eliminates circular dependencies by separating
logging functionality into its own module.
"""

"""
Execution module - Order management and position tracking.
"""

# Order management
from .order_manager import (
    place_order,
    close_position,
    get_current_price,
    configure_paths,
    get_usdt_balance_ws,
    get_fills_for_order
)

# Position tracking
from .position_tracker import (
    add_position,
    check_tp_sl_for_strategy,
    check_all_tp_sl,
    calculate_tp_sl_prices
)

# Trade logging
from .trade_logger import (
    log_closed_position
)

# ⭐ AÑADIR ESTA SECCIÓN:
# Broker clients
from .brokers import BitgetClient

__all__ = [
    # Order management
    'place_order',
    'close_position',
    'get_current_price',
    'configure_paths',
    'get_usdt_balance_ws',
    'get_fills_for_order',
    
    # Position tracking
    'add_position',
    'check_tp_sl_for_strategy',
    'check_all_tp_sl',
    'calculate_tp_sl_prices',
    
    # Trade logging
    'log_closed_position',
    
    # ⭐ AÑADIR ESTA LÍNEA:
    # Broker clients
    'BitgetClient',
]
