#core/utils/ohlcv_utils.py
import pandas as pd
import numpy as np
from setup.config_core import settings

import logging
logger = logging.getLogger("shared.utils.ohlcv_utils")

BARS_PER_DAY = {
    '15m' : 96,
    '30m' : 48,
    '1H'  : 24,
    '4H'  : 6,
    '6H'  : 4,
    '12H' : 2,
    '1D'  : 1,
}

def get_bars_per_day(timeframe: str) -> int:
    if timeframe not in BARS_PER_DAY:
        raise ValueError(f"Timeframe not in mapping: {timeframe}")
    return BARS_PER_DAY[timeframe]

def get_bars_per_year(timeframe: str) -> int:
    return settings.DAYS_PER_YEAR * get_bars_per_day(timeframe)

def prepare_ohlcv_arrays(ohlcv_data):
    ohlcv_arr = {}
    for sym, df in ohlcv_data.items():
        ohlcv_arr[sym] = {
            'ts': df.index.values.astype('datetime64[ns]'),
            'open': df['open'].to_numpy(dtype=np.float64),
            'high': df['high'].to_numpy(dtype=np.float64),
            'low': df['low'].to_numpy(dtype=np.float64),
            'close': df['close'].to_numpy(dtype=np.float64),
            'volume': df['volume'].to_numpy(dtype=np.float64),
            'low_time': (pd.to_datetime(df['low_time']).to_numpy(dtype='datetime64[ns]')),
            'high_time': (pd.to_datetime(df['high_time']).to_numpy(dtype='datetime64[ns]'))          
        }
        
    return ohlcv_arr
