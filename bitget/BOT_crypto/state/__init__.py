"""
State module - Handles bot state persistence and synchronization.
This module manages:
- Loading and saving bot state to JSON
- Synchronizing with broker
- Candle counting for strategies
- Timeout checking
- Runtime bot state tracking (BotState)
"""
from .state_manager import (
    BotState,
    load_state,
    save_state_local,
    sync_broker
)
from .candle_tracker import (
    increment_strategy_candles,
    reset_strategy_candles,
    check_candles_timeout_for_strategy
)

__all__ = [
    # Runtime state
    'BotState',
    
    # State management
    'load_state',
    'save_state_local',
    'sync_broker',
    
    # Candle tracking
    'increment_strategy_candles',
    'reset_strategy_candles',
    'check_candles_timeout_for_strategy',
]