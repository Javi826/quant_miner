#core/utils/batch_metrics.py
import logging
import numpy as np
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
from setup.config_core import settings
logger = logging.getLogger("BOT_batch.utils.batch_metrics")
# =============================================================================
# METRICS CONFIG
# =============================================================================
SHARPE_ABS_CAP = 50.0  # annualized Sharpe values beyond this are treated as invalid/degenerate

# =============================================================================
# TRADING-DAY CALENDAR HELPERS
# =============================================================================
def _to_trading_days(days: np.ndarray) -> np.ndarray:

    return np.busday_offset(days, 0, roll="preceding", weekmask=settings.WEEKMASK)


def _trading_days_between(start_day, end_day):

    return np.busday_count(start_day, end_day, weekmask=settings.WEEKMASK)


_TRADING_DAY_TABLE_PAD = 366    # extra calendar days cached on each side of the requested range
_TRADING_DAY_TABLES: dict = {}  # weekmask key -> (first calendar day, trading day per calendar day, business-day index)


def _trading_day_table(first_day: int, last_day: int) -> tuple:

    weekmask = settings.WEEKMASK
    key   = weekmask if isinstance(weekmask, str) else tuple(np.asarray(weekmask).ravel().tolist())
    table = _TRADING_DAY_TABLES.get(key)
    if table is not None:
        table_first = table[0]
        table_last  = table_first + table[1].shape[0] - 1
        if table_first <= first_day and last_day <= table_last:
            return table
        first_day, last_day = min(first_day, table_first), max(last_day, table_last)
    first_day -= _TRADING_DAY_TABLE_PAD
    last_day  += _TRADING_DAY_TABLE_PAD
    calendar_days = np.arange(first_day, last_day + 1, dtype=np.int64).astype("datetime64[D]")
    trading_days  = _to_trading_days(calendar_days)
    trading_index = _trading_days_between(trading_days[0], trading_days)
    table = (first_day, trading_days, trading_index)
    _TRADING_DAY_TABLES[key] = table
    return table

# =============================================================================
# R_SQUARED
# =============================================================================
def _r_squared_linear_trend(y: np.ndarray) -> float:

    n = len(y)
    x = np.arange(n, dtype=np.float64)
    x_mean = x.mean()
    y_mean = y.mean()
    x_dev  = x - x_mean
    y_dev  = y - y_mean

    ss_xx = np.dot(x_dev, x_dev)
    if ss_xx == 0.0:
        # Single point (n == 1): the fitted line passes through it exactly.
        return 1.0

    b = np.dot(x_dev, y_dev) / ss_xx
    a = y_mean - b * x_mean
    y_pred = a + b * x

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum(y_dev ** 2)

    if ss_tot == 0.0:
        return 1.0 if ss_res < 1e-12 else 0.0

    return float(1.0 - ss_res / ss_tot)

def sharpe_from_daily_values(daily_values: np.ndarray) -> float:

    n          = daily_values.size
    daily_mean = np.add.reduce(daily_values, axis=None) / n
    dev        = daily_values - daily_mean
    np.multiply(dev, dev, out=dev)
    daily_std  = np.sqrt(np.add.reduce(dev, axis=None) / n)
    sharpe = (round(float(daily_mean / daily_std * np.sqrt(settings.DAYS_PER_YEAR)), 3)
              if daily_std > 0 else np.nan)
    if sharpe is not None and np.isfinite(sharpe) and abs(sharpe) > SHARPE_ABS_CAP:
        sharpe = np.nan
    return sharpe

# =============================================================================
# SKEW / KURTOSIS
# =============================================================================
def skew_kurtosis_from_daily_values(daily_values: np.ndarray) -> tuple:

    deviations = daily_values - daily_values.mean()
    m2 = np.mean(deviations ** 2)
    m3 = np.mean(deviations ** 3)
    m4 = np.mean(deviations ** 4)
    return float(m3 / (m2 ** 1.5)), float(m4 / (m2 ** 2))

def daily_values_from_sell_days(sell_days_ns: np.ndarray, profits: np.ndarray) -> tuple:

    sell_days = sell_days_ns.astype("datetime64[D]").view(np.int64)
    first_day = int(sell_days.min())
    last_day  = int(sell_days.max())
    if first_day == np.iinfo(np.int64).min:
        raise ValueError("Cannot compute a business day count with a NaT (not-a-time) date")
    table_first, trading_days, trading_index = _trading_day_table(first_day, last_day)
    start_pos    = first_day - table_first
    day_offset   = trading_index.take(sell_days - table_first)
    day_offset  -= trading_index[start_pos]
    daily_values = np.bincount(day_offset, weights=profits)
    return daily_values, daily_values.shape[0], trading_days[start_pos]

def equity_from_daily_values(daily_values: np.ndarray, capital: float) -> tuple:

    eq       = capital + np.cumsum(daily_values)
    cm       = np.maximum.accumulate(eq)
    max_dd   = ((eq - cm) / cm * 100).min()
    net_gain = (eq[-1] - capital) / capital * 100
    return eq, max_dd, net_gain

# =============================================================================
# COMPUTE METRICS
# =============================================================================

def compute_metrics(
    trade_log: pd.DataFrame,
    capital: float,
    name: str = "Equity",
    include_weekly: bool = True,
    include_skew_kurtosis: bool = True,
    include_r2: bool = True,
) -> dict:
    tl            = trade_log
    profits       = tl["profit"].values
    win_rate      = round((profits > 0).mean() * 100, 1)
    gains         = profits[profits > 0].sum()
    losses        = -profits[profits < 0].sum()
    pf            = round(float(gains / losses), 3) if losses > 0 else np.inf

    daily_values, n_days, start_day = daily_values_from_sell_days(
        tl["sell_time"].values, profits,
    )
    date_index    = pd.bdate_range(start=start_day, periods=n_days, freq=CustomBusinessDay(weekmask=settings.WEEKMASK))
    eq, max_dd, net_gain = equity_from_daily_values(daily_values, capital)
    profit_abs    = round(float(eq[-1] - capital), 2)
    calmar        = round(float(net_gain / abs(max_dd)), 3) if max_dd < 0 else np.nan
    if include_weekly:
        eq_series     = pd.Series(eq, index=date_index)
        weekly        = eq_series.resample("W").last().pct_change().dropna()
        weekly_pct    = (weekly > 0).mean() * 100
        running_max_w   = weekly.add(1).cumprod().cummax()
        equity_w        = weekly.add(1).cumprod()
        is_underwater   = equity_w < running_max_w
        recovery_weeks  = 0
        max_weeks_to_recovery = 0
        for underwater in is_underwater:
            if underwater:
                recovery_weeks += 1
                max_weeks_to_recovery = max(max_weeks_to_recovery, recovery_weeks)
            else:
                recovery_weeks = 0
    else:
        weekly_pct            = np.nan
        max_weeks_to_recovery = 0

    sharpe = sharpe_from_daily_values(daily_values)

    if include_skew_kurtosis and n_days > 2:
        skew_daily, kurt_daily = skew_kurtosis_from_daily_values(daily_values)
    else:
        skew_daily = np.nan
        kurt_daily = np.nan
    if "buy_time" in tl.columns and "sell_time" in tl.columns:
        duration_d = round(float(
            (tl["sell_time"] - tl["buy_time"]).dt.total_seconds().mean() / 86400
        ), 2)
    else:
        duration_d = np.nan
    if include_r2:
        r2 = round(_r_squared_linear_trend(eq), 3)
    else:
        r2 = np.nan
    return {
        "Curve":         name,
        "Net_Gain_pct":  round(float(net_gain), 2),
        "Max_DD_pct":    round(float(max_dd), 2),
        "Win_Rate":      win_rate,
        "R_Squared":     r2,
        "Profit_Factor": pf,
        "Calmar":        calmar,
        "Profit_abs":    profit_abs,
        "Sharpe":        sharpe,
        "Skew":          skew_daily,
        "Kurtosis":      kurt_daily,
        "N_days":        n_days,
        "Duration_d":    duration_d,
        "Weekly_pct":    round(float(weekly_pct), 2),
        "Max_Weeks_to_Recovery": int(max_weeks_to_recovery),
    }