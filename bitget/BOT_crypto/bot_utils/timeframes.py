from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict

# ==========================================================================
# TIMEFRAME CALCULATIONS
# ==========================================================================
def calculate_next_candle_time(timeframe: str = '4H', hour_zone=None) -> datetime:

    now = datetime.now(hour_zone)
    
    # Parse timeframe
    if timeframe.endswith('Hutc'):
        hours = int(timeframe[:-4])
        minutes = hours * 60
    elif timeframe.endswith('H'):
        hours = int(timeframe[:-1])
        minutes = hours * 60
    elif timeframe.endswith('m'):
        minutes = int(timeframe[:-1])
    elif timeframe.endswith('Dutc'):
        days = int(timeframe[:-4])
        minutes = days * 24 * 60
    else:
        raise ValueError(
            "Invalid timeframe. Use 'm', 'H', 'Hutc', or 'Dutc'. "
            "Examples: '15m', '4H', '6Hutc', '1Dutc'"
        )
    
    # Calculate next candle time
    total_minutes      = now.hour * 60 + now.minute
    next_total_minutes = ((total_minutes // minutes) + 1) * minutes
    delta_minutes      = next_total_minutes - total_minutes
    
    next_candle = now + timedelta(
        minutes=delta_minutes,
        seconds=-now.second,
        microseconds=-now.microsecond
    )
    
    next_candle = next_candle + timedelta(seconds=20)
    
    return next_candle


# ==========================================================================
# STRATEGY GROUPING
# ==========================================================================
def group_strategies_by_timeframe(strategies: List[Dict]) -> Dict[str, List]:

    grouped = defaultdict(list)
    for strat in strategies:
        grouped[strat['timeframe']].append(strat)
    return grouped


def get_unique_timeframes(strategies: List[Dict]) -> List[str]:

    return sorted(list(set(s['timeframe'] for s in strategies)))
