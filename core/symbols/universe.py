# core/symbols/universe.py
"""
Fecha de referencia: 2026-09-04
N  | Símbolos incluidos                                                                         | Mínimo común
---|--------------------------------------------------------------------------------------------|--------------
2  | BNBUSDT, BTCUSDT                                                                           | 2019-07-10
3  | BNBUSDT, BTCUSDT, ETHUSDT                                                                  | 2019-08-10
4  | BNBUSDT, BTCUSDT, ETHUSDT, XRPUSDT                                                         | 2019-08-28
5  | BNBUSDT, BTCUSDT, ETHUSDT, XRPUSDT, BCHUSDT                                                | 2019-09-24
8  | BNBUSDT, BTCUSDT, ETHUSDT, XRPUSDT, BCHUSDT, LINKUSDT, ADAUSDT, UNIUSDT                    | 2020-09-22
10 | BNBUSDT, BTCUSDT, ETHUSDT, XRPUSDT, BCHUSDT, LINKUSDT, ADAUSDT, UNIUSDT, XLMUSDT, DOGEUSDT | 2021-05-21
"""
import os
import sys
import logging
import pandas as pd
from setup.config_backtest import MIN_PRICE
logger = logging.getLogger("BOT_batch.pipeline.universe")

# =============================================================================
# UNIVERSE DATA REQUIREMENTS
# =============================================================================
# =============================================================================
# MIN_START_DATE_IS     = "2022-01-01"
# MIN_START_DATE_OOS    = "2024-01-10"
# MIN_START_DATE_MERGED = "2022-01-01"
# =============================================================================

MIN_START_DATE_IS     = "2017-01-02"
MIN_START_DATE_OOS    = "2024-01-24"
MIN_START_DATE_MERGED = "2018-01-02"

MIN_START_DATE_BY_DATASET = {
    "IS":     MIN_START_DATE_IS,
    "OOS":    MIN_START_DATE_OOS,
    "MERGED": MIN_START_DATE_MERGED,
}
END_DATE_DAYS  = 10


# =============================================================================
# BUILD UNIVERSE — single entry point: validate every symbol across every
# =============================================================================
def build_universe(
    data_folder_is:          str,
    symbols_by_timeframe:    dict,
    dataset:                  str,
    end_date_tolerance_days: int          = END_DATE_DAYS,
) -> dict:
    if dataset not in MIN_START_DATE_BY_DATASET:
        raise ValueError(f"Unknown dataset={dataset!r}; expected one of {list(MIN_START_DATE_BY_DATASET)}.")

    min_start_date = MIN_START_DATE_BY_DATASET[dataset]
    cutoff_ts      = pd.Timestamp(min_start_date)
    failures       = []
    ohlcv_loaded = {timeframe: {} for timeframe in symbols_by_timeframe}

    for timeframe, symbols in symbols_by_timeframe.items():
        for sym in symbols:
            file_path = os.path.join(data_folder_is, f"{sym}_{timeframe}.parquet")

            if not os.path.exists(file_path):
                failures.append(f"{sym} ({timeframe}): file missing")
                continue

            df = pd.read_parquet(file_path)

            if df.empty:
                failures.append(f"{sym} ({timeframe}): no data")
                continue

            if df.index[0] > cutoff_ts:
                failures.append(f"{sym} ({timeframe}): starts at {df.index[0]}, requires {min_start_date}")
                continue

            if MIN_PRICE is not None and df['close'].iloc[-1] <= MIN_PRICE:
                failures.append(f"{sym} ({timeframe}): last close {df['close'].iloc[-1]} <= min_price {MIN_PRICE}")
                continue

            ohlcv_loaded[timeframe][sym] = df

    # -------------------------------------------------------------------
    # END-DATE ALIGNMENT — within each timeframe, every symbol must end
    # -------------------------------------------------------------------
    for timeframe, ohlcv_data in ohlcv_loaded.items():
        if not ohlcv_data:
            continue
        end_dates = {sym: df.index[-1] for sym, df in ohlcv_data.items()}
        latest_end = max(end_dates.values())
        for sym, end_date in end_dates.items():
            lag_days = (latest_end - end_date).days
            if lag_days > end_date_tolerance_days:
                failures.append(
                    f"{sym} ({timeframe}): ends at {end_date}, {lag_days}d behind latest ({latest_end})"
                )

    if failures:
        logger.error(f"🚫 Universe validation FAILED — {len(failures)} symbol(s):")
        for failure in failures:
            logger.error(f"  {failure}")
        sys.exit(1)

    counts_str = " | ".join(f"{timeframe}: {len(symbols)} symbol(s)" for timeframe, symbols in ohlcv_loaded.items())
    logger.debug(f"✅ Universe validation OK [{dataset}] — {counts_str}")

    ohlcv_by_timeframe = {}
    for timeframe, ohlcv_data in ohlcv_loaded.items():
        ohlcv_cut = {}
        for sym, df in ohlcv_data.items():
            start_before = df.index[0]
            end_date     = df.index[-1]
            df_cut = df[df.index >= cutoff_ts]
            ohlcv_cut[sym] = df_cut
            start_after = df_cut.index[0] if len(df_cut) else None
            logger.debug(f"  [{timeframe}] {sym:<12} start_before: {str(start_before):<19}  start_after: {str(start_after):<19}  end: {end_date}")
        ohlcv_by_timeframe[timeframe] = ohlcv_cut
        logger.debug(f"IS pool ({len(ohlcv_cut):>3}) [{timeframe}]: {list(ohlcv_cut.keys())}")

    return ohlcv_by_timeframe
