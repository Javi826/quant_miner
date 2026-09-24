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
    daily_std = daily_values.std()
    sharpe = (round(float(daily_values.mean() / daily_std * np.sqrt(settings.DAYS_PER_YEAR)), 3)
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

    sell_days  = sell_days_ns.astype("datetime64[D]")
    sell_days  = _to_trading_days(sell_days)
    start_day  = sell_days.min()
    end_day    = sell_days.max()
    n_days     = int(np.busday_count(start_day, end_day, weekmask=settings.WEEKMASK)) + 1
    day_offset = np.busday_count(start_day, sell_days, weekmask=settings.WEEKMASK)
    daily_values = np.bincount(day_offset, weights=profits, minlength=n_days)
    return daily_values, n_days, start_day

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
    eq            = capital + np.cumsum(daily_values)
    cm            = np.maximum.accumulate(eq)
    max_dd        = ((eq - cm) / cm * 100).min()
    net_gain      = (eq[-1] - capital) / capital * 100
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
