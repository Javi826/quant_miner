#BOT_crypto/bot_utils/timeframes.py
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict

import re
from datetime import  timezone

from config.settings import CANDLE_GRID_OFFSET_HOURS, CANDLE_CLOSE_BUFFER

# ==========================================================================
# TIMEFRAME CALCULATIONS
# ==========================================================================
GRID_OFFSET  = timedelta(hours=CANDLE_GRID_OFFSET_HOURS)
CLOSE_BUFFER = timedelta(seconds=CANDLE_CLOSE_BUFFER)

_EPOCH             = datetime(1970, 1, 1, tzinfo=timezone.utc)
_TIMEFRAME_PATTERN = re.compile(r"^(\d+)([mHD])$")
_UNIT_TO_DELTA     = {
    "m": timedelta(minutes=1),
    "H": timedelta(hours=1),
    "D": timedelta(days=1),
}


def timeframe_to_timedelta(timeframe: str) -> timedelta:
    match = _TIMEFRAME_PATTERN.match(timeframe)
    if not match:
        raise ValueError(
            f"Invalid timeframe '{timeframe}'. Expected <n><m|H|D>, e.g. '15m', '4H', '1D'"
        )
    value, unit = int(match.group(1)), match.group(2)
    return value * _UNIT_TO_DELTA[unit]


def calculate_next_candle_time(timeframe: str = '4H', hour_zone=timezone.utc) -> datetime:
    duration   = timeframe_to_timedelta(timeframe)
    now        = datetime.now(timezone.utc)
    elapsed    = (now - _EPOCH + GRID_OFFSET) % duration
    next_close = now - elapsed + duration + CLOSE_BUFFER
    return next_close.astimezone(hour_zone)


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
