#BOT_crypto/market_data/data_utils.py

import os
import sys
import logging
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "broker_client", "broker_api")))
from api_client import _call_history_candles, to_dataframe_from_api

from pandas.api.types import is_datetime64_any_dtype
from config.settings import API_LIMIT_DATA

logger = logging.getLogger('BOT_crypto.market_data.data_utils')


def normalize_live_ohlcv(df: pd.DataFrame) -> pd.DataFrame:

    logger.debug(f"Normalizing OHLCV DataFrame with {len(df)} rows")
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df.index = pd.to_datetime(df['timestamp'])
        else:
            df.index = pd.to_datetime(df.index)
    
    # Convert OHLCV columns to numeric
    for col in ['open', 'high', 'low', 'close', 'volume_base', 'volume_quote']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def df_to_arrays_live(df: pd.DataFrame) -> dict:

    logger.debug(f"Converting DataFrame to arrays ({len(df)} rows)")
    
    if not is_datetime64_any_dtype(df.index):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    
    arrays = {
        'ts': df.index.to_numpy(dtype='datetime64[ns]'),
        'open': df['open'].to_numpy(dtype=np.float64),
        'high': df['high'].to_numpy(dtype=np.float64),
        'low': df['low'].to_numpy(dtype=np.float64),
        'close': df['close'].to_numpy(dtype=np.float64),
        'volume_quote': (
            df['volume_quote'].to_numpy(dtype=np.float64)
            if 'volume_quote' in df
            else np.zeros(len(df))
        )
    }
    
    return arrays

def load_final_symbols(
    all_symbols: list,
    live_symbols: list,
    strategy: str = "_",
    timeframe: str = "4H",
) -> list:

    if not live_symbols:
        error_msg = f"No symbols defined in strategy '{strategy}' timeframe '{timeframe}'"
        logger.error(error_msg)
        raise ValueError(error_msg)

    final_symbols = set(all_symbols) & set(live_symbols)

    if not final_symbols:
        logger.warning(
            f"No symbols match between strategy and exchange for {strategy} {timeframe}. "
            f"Strategy has {len(live_symbols)} symbols, but none are available on exchange."
        )

    logger.debug(f"Loaded {len(final_symbols)} symbols for {strategy} {timeframe}")
    return sorted(final_symbols)
        
def fetch_ohlcv_data(symbols: list, timeframe: str) -> dict:

    logger.debug(f"Fetching OHLCV data for {len(symbols)} symbols ({timeframe})")
    
    ohlcv_data = {}
    
    for sym in symbols:
        try:
            recent_candles = _call_history_candles(
                symbol=sym,
                granularity=timeframe,
                limit=API_LIMIT_DATA
            )
            df = to_dataframe_from_api(recent_candles)
            ohlcv_data[sym] = df
            
        except Exception as e:
            logger.error(f"Error-Failed to fetch OHLCV for {sym}: {e}")
            ohlcv_data[sym] = None
    
    logger.debug(f"Successfully fetched data for {len(ohlcv_data)} symbols")
    return ohlcv_data