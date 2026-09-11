#core/utils/ohlcv_utils.py
import pandas as pd
import numpy as np
from setup.config_core import settings

import logging
logger = logging.getLogger("shared.utils.ohlcv_utils")

def get_bars_per_year(timeframe: str) -> int:
    mapping = {
        '15m'    : settings.DAYS_PER_YEAR * 96,
        '30m'    : settings.DAYS_PER_YEAR * 48,
        '1H'     : settings.DAYS_PER_YEAR * 24,
        '4H'     : settings.DAYS_PER_YEAR * 6,
        '6Hutc'  : settings.DAYS_PER_YEAR * 4,
        '12Hutc' : settings.DAYS_PER_YEAR * 2,
        '1Dutc'  : settings.DAYS_PER_YEAR,
    }
    if timeframe not in mapping:
        raise ValueError(f"Timeframe not in mapping: {timeframe}")
    return mapping[timeframe]

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
