#develop/indicators/candidate_pool.py
import os
import sys
import itertools
import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from numpy.lib.stride_tricks import sliding_window_view
from numba import njit

# =============================================================================
# CONFIG — periods & thresholds grids, one place for all candidates.
# =============================================================================

# --- A: existing --------------------------------------------------------------
RSI_PERIODS                    = [7,14,21]
RSI_THRESHOLDS                 = [30,40,60,70]

ADX_PERIODS                    = [7,14,21]
ADX_THRESHOLDS                 = [10,20,30]

MA_DIST_PERIODS                = [20,50,100]
MA_DIST_THRESHOLDS             = [-1.0,-0.8,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.8,1.0]

MOMENTUM_PERIODS               = [10,20,30]
MOMENTUM_THRESHOLDS            = [-1.0,-0.8,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.8,1.0]

ATR_REGIME_PERIODS             = [7,14,21]
ATR_REGIME_SMA_PERIODS         = [30,60]
ATR_REGIME_THRESHOLDS          = [0.5,0.7,1.0,1.4,2.0]

HISTVOL_REGIME_PERIODS         = [20,30]
HISTVOL_REGIME_SMA_PERIODS     = [40,60]
HISTVOL_REGIME_THRESHOLDS      = [0.5,0.7,1.0,1.4,2.0]

# --- B: range / reversion -----------------------------------------------------
BB_PCTB_PERIODS                = [10,20,40]
BB_PCTB_THRESHOLDS             = [0.2,0.4,0.6,0.7,0.8,0.9]

DONCHIAN_POS_PERIODS           = [7,14,21]
DONCHIAN_POS_THRESHOLDS        = [0.2,0.4,0.6,0.8,0.9]

CLOSE_PCT_RANK_PERIODS           = [100]
CLOSE_PCT_RANK_THRESHOLDS        = [20.0, 50.0, 80.0]

PIVOT_DIST_PERIODS               = [10,20,40]
PIVOT_DIST_THRESHOLDS            = [-1.0, -0.3, 0.3, 1.0]

# --- C: trend -----------------------------------------------------------------
VORTEX_PERIODS                 = [7,14,21]
VORTEX_THRESHOLDS              = [0.7,0.9,1.1,1.3]

HURST_PERIODS                  = [30,60]
HURST_THRESHOLDS               = [0.4,0.5,0.6]

ICHIMOKU_TENKAN_PERIODS             = [9,18]
ICHIMOKU_KIJUN_PERIODS              = [13,26]
ICHIMOKU_SENKOU_B_PERIODS           = [27,52]
ICHIMOKU_TENKAN_KIJUN_THRESHOLDS    = [-0.5,0.5]
ICHIMOKU_CLOUD_THICKNESS_THRESHOLDS = [0.2,0.4,0.6,0.8,1.0]
ICHIMOKU_PRICE_VS_CLOUD_THRESHOLDS  = [-1.0,-0.5,0.5,1.0]

TII_PERIODS                      = [15,30,60]
TREND_INTENSITY_INDEX_THRESHOLDS = [30.0, 50.0, 70.0]

# --- D: momentum / acceleration -----------------------------------------------
MACD_HIST_FAST_PERIODS         = [12,24]
MACD_HIST_SLOW_PERIODS         = [26,52]
MACD_HIST_SIGNAL_PERIODS       = [9]
MACD_HIST_THRESHOLDS           = [0.2, 0.4]

ROC_SPREAD_FAST_PERIODS        = [5,10]
ROC_SPREAD_SLOW_PERIODS        = [20,40]
ROC_SPREAD_THRESHOLDS          = [0.4, 0.6, 0.8]

TSI_LONG_PERIODS               = [25,50]
TSI_SHORT_PERIODS              = [13,26]
TSI_THRESHOLDS                 = [-40.0, -20.0, 20.0, 40.0]

ACCELERATION_THRESHOLDS        = [0.4,0.6,0.8, 1.0]

PPO_FAST_PERIODS                = [12,22]
PPO_SLOW_PERIODS                = [26,52]
PPO_THRESHOLDS                  = [-1.0, -0.3, 0.3, 1.0]

RVI_PERIODS                     = [10,20,40]
RVI_THRESHOLDS                  = [-0.3, -0.1, 0.1, 0.3]

# --- E: volatility ------------------------------------------------------------
BB_BANDWIDTH_PERIODS            = [15,30,60]
BB_BANDWIDTH_THRESHOLDS         = [0.01,0.02,0.03,0.04]

CHOPPINESS_PERIODS              = [7,14,21]
CHOPPINESS_THRESHOLDS           = [38.0, 62.0]

PARKINSON_RATIO_PERIODS         = [10,20,40]
PARKINSON_RATIO_THRESHOLDS      = [0.8, 1.2]

VOL_OF_VOL_PERIODS              = [10,20]
VOL_OF_VOL_SMA_PERIODS          = [30,60]
VOL_OF_VOL_THRESHOLDS           = [0.3,0.4,0.5,0.6]

RET_SKEW_PERIODS                = [30,60]
RET_SKEW_THRESHOLDS             = [-1.0, -0.5, 0.5, 1.0]

DONCHIAN_WIDTH_PERIODS          = [10,20,40]
DONCHIAN_WIDTH_THRESHOLDS       = [0.01,0.02,0.03,0.04]

ULCER_PERIODS                   = [7,14,21]
ULCER_INDEX_THRESHOLDS          = [1.0,2.0,3.0, 4.0]

BB_PCTB_SLOPE_PERIODS            = [5]
BB_PCTB_SLOPE_THRESHOLDS         = [-0.2, -0.05, 0.05, 0.2]

# --- F: microstructure --------------------------------------------------------
CLOSE_POS_IN_BAR_THRESHOLDS     = [0.1, 0.2, 0.8, 0.9]

RANGE_EXPANSION_PERIODS         = [10,20,40]
RANGE_EXPANSION_THRESHOLDS      = [0.5, 0.75, 1.5, 2.0]

INSIDE_OUTSIDE_PERIODS            = [20,40]
INSIDE_OUTSIDE_RATIO_THRESHOLDS   = [10.0, 25.0, 40.0]

OPEN_CLOSE_MOMENTUM_THRESHOLDS    = [-0.5, -0.2, 0.2, 0.5]

# --- G: cycles / statistics ---------------------------------------------------
ENTROPY_PERIODS                   = [30,60]
RETURN_ENTROPY_THRESHOLDS         = [1.5, 2.0, 2.5]

KALMAN_SLOPE_THRESHOLDS           = [-0.3, -0.1, 0.1, 0.3]

# --- H: calendar / seasonality ------------------------------------------------
DAY_SLOT_SLOTS                    = [0,1,2,3,4,5]
DAY_SLOT_THRESHOLDS               = [0.5]

VOL_DESEASON_DAYS                 = [20,60]
VOL_DESEASON_THRESHOLDS           = [0.5,0.7,1.0,1.4,2.0]

# --- I: price levels / profile ------------------------------------------------
ROUND_LEVEL_GRIDS                 = [50,100]          # grid size in pips
ROUND_LEVEL_PHASE_THRESHOLDS      = [0.05,0.1,0.2]

TPO_DENSITY_PERIODS               = [50,100,200]
TPO_DENSITY_THRESHOLDS            = [0.05,0.15,0.3]

# --- J: swing structure -------------------------------------------------------
SWING_ATR_MULTS                   = [2,3,5]           # zigzag reversal, in ATR units
SWING_LEG_RATIO_THRESHOLDS        = [0.25,0.5,1.0,1.5]

# =============================================================================
# SHAPE CONSTANTS — not exposed as grids, they define a formula's internal
# =============================================================================
ATR_N                = 14   # shared ATR normalizer used across most indicators
BB_N                 = 20   # underlying BB window used inside e_bb_pctb_slope

BB_K                 = 2.0

ENTROPY_BINS         = 6
KALMAN_PROCESS_VAR   = 1e-5
KALMAN_MEASURE_VAR   = 1e-2

SLOT_HOURS           = 4    # hours per day slot (4H bars -> 6 slots per day)

_HERE     = os.path.abspath(os.path.dirname(__file__))
_DARWINEX = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _DARWINEX not in sys.path:
    sys.path.insert(0, _DARWINEX)


# =============================================================================
# PRIMITIVES — unchanged formulas, still take explicit window args.
# =============================================================================
def _sma(close: np.ndarray, window: int) -> np.ndarray:
    n   = len(close)
    out = np.full(n, np.nan)
    if window > n:
        return out
    csum = np.cumsum(close, dtype=np.float64)
    out[window - 1] = csum[window - 1] / window
    out[window:]    = (csum[window:] - csum[:n - window]) / window
    return out


def _rolling_mean_skipnan(values: np.ndarray, window: int) -> np.ndarray:
    out = pd.Series(values, dtype=np.float64).rolling(window=window, min_periods=1).mean().to_numpy(copy=True)
    out[:window - 1] = np.nan
    return out


def _rsi(close: np.ndarray, window: int) -> np.ndarray:
    n     = len(close)
    out   = np.full(n, np.nan)
    alpha = 1.0 / window

    diff  = np.diff(close)
    up    = np.where(diff > 0, diff, 0.0)
    dn    = np.where(diff < 0, -diff, 0.0)
    emaup = pd.Series(np.concatenate(([0.0], up))).ewm(alpha=alpha, adjust=False).mean().to_numpy()[1:]
    emadn = pd.Series(np.concatenate(([0.0], dn))).ewm(alpha=alpha, adjust=False).mean().to_numpy()[1:]

    safe_emadn = np.where(emadn == 0.0, 1.0, emadn)
    rsi_vals   = np.where(emadn == 0.0, 100.0, 100.0 - (100.0 / (1.0 + emaup / safe_emadn)))

    idx  = np.arange(1, n)
    mask = idx >= (window - 1)
    out[idx[mask]] = rsi_vals[mask]
    return out


def _wilder_sum_smooth(seed: float, x: np.ndarray, window: int) -> np.ndarray:
    if len(x) == 0:
        return np.array([], dtype=np.float64)
    r  = 1.0 - 1.0 / window
    b  = [1.0]
    a  = [1.0, -r]
    zi = sp_signal.lfiltic(b, a, [seed], [0.0])
    y, _ = sp_signal.lfilter(b, a, x, zi=zi)
    return y


def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    n        = len(close)
    adx_full = np.zeros(n)

    k = n - (window - 1)
    if k <= window + 1:
        return adx_full

    diff_dm = np.zeros(n)
    pos     = np.zeros(n)
    neg     = np.zeros(n)

    prev_close = close[:-1]
    pdm = np.maximum(high[1:], prev_close)
    pdn = np.minimum(low[1:], prev_close)
    diff_dm[1:] = pdm - pdn

    diff_up   = high[1:] - high[:-1]
    diff_down = low[:-1] - low[1:]
    pos[1:] = np.where((diff_up > diff_down) & (diff_up > 0), diff_up, 0.0)
    neg[1:] = np.where((diff_down > diff_up) & (diff_down > 0), diff_down, 0.0)

    trs_s = float(diff_dm[1:window + 1].sum())
    dip_s = float(pos[1:window + 1].sum())
    din_s = float(neg[1:window + 1].sum())

    trs = np.zeros(k)
    dip = np.zeros(k)
    din = np.zeros(k)
    trs[0], dip[0], din[0] = trs_s, dip_s, din_s

    x_trs = diff_dm[window + 1:window + k - 1]
    x_dip = pos[window + 1:window + k - 1]
    x_din = neg[window + 1:window + k - 1]
    trs[1:k - 1] = _wilder_sum_smooth(trs_s, x_trs, window)
    dip[1:k - 1] = _wilder_sum_smooth(dip_s, x_dip, window)
    din[1:k - 1] = _wilder_sum_smooth(din_s, x_din, window)

    with np.errstate(divide="ignore", invalid="ignore"):
        di_p = np.where(trs != 0, 100.0 * dip / trs, 0.0)
        di_n = np.where(trs != 0, 100.0 * din / trs, 0.0)
    denom = di_p + di_n
    with np.errstate(divide="ignore", invalid="ignore"):
        dx = np.where(denom != 0, 100.0 * np.abs(di_p - di_n) / denom, 0.0)

    adx_smooth = np.zeros(k)
    seed_adx   = float(dx[:window].sum() / window)
    adx_smooth[window] = seed_adx

    tail = dx[window:k - 1]
    if len(tail) > 0:
        alpha    = 1.0 / window
        virtual  = np.concatenate(([seed_adx], tail))
        ema      = pd.Series(virtual).ewm(alpha=alpha, adjust=False).mean().to_numpy()
        adx_smooth[window + 1:k] = ema[1:]

    prefix = window - 1
    adx_full[prefix:prefix + k] = adx_smooth
    return adx_full


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    n   = len(close)
    out = np.empty(n)
    out[0] = high[0] - low[0]
    prev_close = close[:-1]
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - prev_close)
    tr3 = np.abs(low[1:] - prev_close)
    out[1:] = np.maximum(np.maximum(tr1, tr2), tr3)
    return out


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    tr  = _true_range(high, low, close)
    n   = len(tr)
    out = np.full(n, np.nan)

    if window > n:
        return out

    seed = float(tr[:window].sum() / window)
    out[window - 1] = seed

    tail = tr[window:]
    if len(tail) > 0:
        alpha   = 1.0 / window
        virtual = np.concatenate(([seed], tail))
        ema     = pd.Series(virtual).ewm(alpha=alpha, adjust=False).mean().to_numpy()
        out[window:] = ema[1:]

    return out


def _historical_volatility(close: np.ndarray, window: int) -> np.ndarray:
    n           = len(close)
    log_returns = np.full(n, np.nan)
    log_returns[1:] = np.log(close[1:] / close[:-1])

    out = pd.Series(log_returns).rolling(window=window).std(ddof=0).to_numpy(copy=True)
    out[:window] = np.nan
    return out


def _s(x: np.ndarray) -> pd.Series:
    return pd.Series(x, dtype=np.float64)


def _roll_max(x: np.ndarray, n: int) -> np.ndarray:
    return _s(x).rolling(n).max().to_numpy()


def _roll_min(x: np.ndarray, n: int) -> np.ndarray:
    return _s(x).rolling(n).min().to_numpy()


def _roll_mean(x: np.ndarray, n: int) -> np.ndarray:
    return _s(x).rolling(n).mean().to_numpy()


def _roll_std(x: np.ndarray, n: int) -> np.ndarray:
    return _s(x).rolling(n).std(ddof=0).to_numpy()


def _roll_sum(x: np.ndarray, n: int) -> np.ndarray:
    return _s(x).rolling(n).sum().to_numpy()


def _ema(x: np.ndarray, n: int) -> np.ndarray:
    return _s(x).ewm(span=n, adjust=False).mean().to_numpy()


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Division yielding NaN (not zero) wherever the denominator vanishes."""
    b = np.asarray(b, dtype=np.float64)
    out = np.full(len(b), np.nan)
    ok = np.isfinite(b) & (np.abs(b) > 1e-15)
    out[ok] = np.asarray(a, dtype=np.float64)[ok] / b[ok]
    return out


def _log_returns(close: np.ndarray) -> np.ndarray:
    out = np.full(len(close), np.nan)
    out[1:] = np.log(close[1:] / close[:-1])
    return out


def _atr_pct(arr: dict, window: int = ATR_N) -> np.ndarray:
    atr = _atr(arr["high"], arr["low"], arr["close"], window)
    return _safe_div(atr, arr["close"])


def _atr_pct_horizon(arr: dict, horizon: int, window: int = ATR_N) -> np.ndarray:
    return _atr_pct(arr, window) * np.sqrt(float(horizon))

def _rolling_pct_rank(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out

    windows = sliding_window_view(x, n)
    last    = windows[:, -1]

    less  = (windows < last[:, None]).sum(axis=1)
    equal = (windows == last[:, None]).sum(axis=1)
    rank  = less + (equal + 1) / 2.0
    pct   = rank / n

    out[n - 1:] = pct
    out[n - 1:][np.isnan(last)] = np.nan
    return out

@njit(cache=True)
def _shannon_entropy(x: np.ndarray, bins: int) -> float:
    finite = x[np.isfinite(x)]
    n = len(finite)
    if n < bins:
        return np.nan

    x_min = finite.min()
    x_max = finite.max()
    if x_max == x_min:
        return np.nan

    width  = (x_max - x_min) / bins
    counts = np.zeros(bins)
    for i in range(n):
        idx = int((finite[i] - x_min) / width)
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1.0

    total = counts.sum()
    ent = 0.0
    for i in range(bins):
        p = counts[i] / total
        if p > 0.0:
            ent -= p * np.log(p)
    return ent


def _kalman_slope_filter(close: np.ndarray, q: float, r: float) -> np.ndarray:
    m = len(close)
    out = np.full(m, np.nan)

    p = np.eye(2) * 1.0
    f_mat = np.array([[1.0, 1.0], [0.0, 1.0]])
    h_vec = np.array([1.0, 0.0])
    q_mat = np.eye(2) * q

    state = np.array([close[0], 0.0])
    for i in range(m):
        state = f_mat @ state
        p = f_mat @ p @ f_mat.T + q_mat

        y = close[i] - h_vec @ state
        s = h_vec @ p @ h_vec.T + r
        k_gain = (p @ h_vec) / s
        state = state + k_gain * y
        p = p - np.outer(k_gain, h_vec) @ p

        out[i] = state[1]
    return out


# -----------------------------------------------------------------------------
# Primitives for blocks H, I, J. All causal: the value at t uses bars <= t only.
# -----------------------------------------------------------------------------
def _day_slot(ts: np.ndarray) -> np.ndarray:
    """Slot of the day (hour // SLOT_HOURS) from the bar open timestamp.

    Only for bars of exactly SLOT_HOURS hours, it checks that:
      - every bar opens at minute 0;
      - bar open hours never take both remainders 0 and SLOT_HOURS - 1 (mod SLOT_HOURS).
        That happens when a DST change moves the opens across a slot boundary, so the
        same bar would get different slots in summer and winter. Server time, UTC and
        Madrid time are all safe with a GMT+2/+3 server; a fixed-offset clock is not.
    """
    ts     = np.asarray(ts).astype("datetime64[ns]")
    in_day = ts - ts.astype("datetime64[D]")
    hour   = (in_day // np.timedelta64(1, "h")).astype(np.int64)
    minute = (in_day // np.timedelta64(1, "m")).astype(np.int64) % 60

    if len(ts) > 1:
        step = float(np.median(np.diff(ts.view(np.int64))))
        if step == SLOT_HOURS * 3_600_000_000_000:
            if (minute != 0).any():
                raise ValueError("day_slot: some bars do not open at minute 0")
            rem = hour % SLOT_HOURS
            if (rem == 0).any() and (rem == SLOT_HOURS - 1).any():
                raise ValueError("day_slot: bar opens cross a slot boundary across DST changes; "
                                 "timestamps are not in a DST-consistent clock")
    return hour // SLOT_HOURS


def _same_slot_median(x: np.ndarray, slot: np.ndarray, n_prev: int) -> np.ndarray:
    """Median of x over the n_prev previous bars of the same slot (current bar excluded)."""
    x   = np.asarray(x, dtype=np.float64)
    out = np.full(len(x), np.nan)
    for s in np.unique(slot):
        idx = np.flatnonzero(slot == s)
        med = _s(x[idx]).shift(1).rolling(n_prev).median().to_numpy()
        out[idx] = med
    return out


def _price_decimals(close: np.ndarray) -> int:
    """Number of decimals in the quotes (max over the series; CSVs drop trailing zeros).

    Static metadata of the symbol, not a statistic of the price dynamics.
    The tolerance is relative so that prices stored as float32 upstream still match.
    """
    x = np.asarray(close, dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0.0)]
    if len(x) == 0:
        raise ValueError("price_decimals: no valid prices")
    for d in range(0, 9):
        scaled = x * (10.0 ** d)
        tol    = 1e-4 + 5e-7 * np.abs(scaled)
        if np.all(np.abs(scaled - np.round(scaled)) < tol):
            return d
    raise ValueError("price_decimals: could not determine quote precision")


def _pip_size(close: np.ndarray) -> float:
    """Pip for fractional quoting: 5 decimals -> 0.0001, 3 decimals (JPY) -> 0.01."""
    return 10.0 ** -(_price_decimals(close) - 1)


@njit(cache=True)
def _tpo_density(high, low, close, n):
    """Fraction of bars t-n..t-1 whose [low, high] contains close[t]."""
    m   = len(close)
    out = np.full(m, np.nan)
    for t in range(n, m):
        c = close[t]
        if c != c:
            continue
        cnt   = 0
        valid = True
        for i in range(t - n, t):
            h = high[i]
            l = low[i]
            if h != h or l != l:
                valid = False
                break
            if l <= c and c <= h:
                cnt += 1
        if valid:
            out[t] = cnt / n
    return out


@njit(cache=True)
def _zigzag_confirmed(high, low, atr, k):
    """Zigzag on high/low with reversal threshold k * atr[t]. Causal.

    Rule: while in an up leg with extreme E (max high since the leg started), the high
    is confirmed as a pivot at the first bar t where E - (min low after E, up to t)
    >= k * atr[t]. The down leg then starts at that min low. Symmetric for down legs.
    Several reversals can be confirmed in the same bar.

    A pivot is registered at the bar where it is confirmed, never at the bar of the
    extreme itself. A bar that makes a new extreme is not tested against that same
    extreme (the order of high and low inside the bar is unknown).

    For every bar t, after processing bar t:
      last_idx[t] : bar index of the extreme of the last confirmed pivot (-1 if none yet)
      last_px[t]  : its price
      prev_px[t]  : price of the pivot before it (NaN if none yet)
    """
    m        = len(high)
    last_idx = np.full(m, -1, dtype=np.int64)
    last_px  = np.full(m, np.nan)
    prev_px  = np.full(m, np.nan)

    direction = 0            # 0: no pivot yet, 1: up leg, -1: down leg
    hi   = np.nan            # running extremes before the first pivot
    lo   = np.nan
    hi_i = -1
    lo_i = -1
    e_i  = -1                # bar of the current leg extreme
    cur_idx  = -1
    cur_px   = np.nan
    cur_prev = np.nan

    for t in range(m):
        h = high[t]
        l = low[t]
        a = atr[t]
        if h == h and l == l and a == a and a > 0.0:
            thr = k * a
            if direction == 0:
                if hi != hi or h > hi:
                    hi = h
                    hi_i = t
                if lo != lo or l < lo:
                    lo = l
                    lo_i = t
                if hi_i != lo_i and hi - lo >= thr:
                    if lo_i < hi_i:
                        cur_px = lo
                        cur_idx = lo_i
                        direction = 1
                        e_i = hi_i
                    else:
                        cur_px = hi
                        cur_idx = hi_i
                        direction = -1
                        e_i = lo_i
            elif direction == 1:
                if h > high[e_i]:
                    e_i = t
            else:
                if l < low[e_i]:
                    e_i = t

            # reversals from the current leg extreme (bars after it, up to t)
            while direction != 0 and e_i < t:
                j = -1
                if direction == 1:
                    for i in range(e_i + 1, t + 1):
                        if low[i] == low[i] and (j < 0 or low[i] < low[j]):
                            j = i
                    if j < 0 or high[e_i] - low[j] < thr:
                        break
                    cur_prev = cur_px
                    cur_px = high[e_i]
                    cur_idx = e_i
                    direction = -1
                else:
                    for i in range(e_i + 1, t + 1):
                        if high[i] == high[i] and (j < 0 or high[i] > high[j]):
                            j = i
                    if j < 0 or high[j] - low[e_i] < thr:
                        break
                    cur_prev = cur_px
                    cur_px = low[e_i]
                    cur_idx = e_i
                    direction = 1
                e_i = j
        last_idx[t] = cur_idx
        last_px[t] = cur_px
        prev_px[t] = cur_prev
    return last_idx, last_px, prev_px


def _zigzag(arr: dict, k: float):
    atr = _atr(arr["high"], arr["low"], arr["close"], ATR_N)
    return _zigzag_confirmed(np.asarray(arr["high"], dtype=np.float64),
                             np.asarray(arr["low"], dtype=np.float64),
                             np.asarray(atr, dtype=np.float64), float(k))


# =============================================================================
# BLOCK A — existing
# =============================================================================
def a_rsi(arr, ctx, params):
    return _rsi(arr["close"], params["period"])


def a_adx(arr, ctx, params):
    out = _adx(arr["high"], arr["low"], arr["close"], params["period"])
    out = np.asarray(out, dtype=np.float64).copy()
    out[out == 0.0] = np.nan
    return out


def a_ma_dist(arr, ctx, params):
    """Distance from the moving average, in ATR units."""
    dist = _safe_div(arr["close"], _sma(arr["close"], params["period"])) - 1.0
    return _safe_div(dist, _atr_pct(arr))


def a_momentum(arr, ctx, params):
    """N-candle return, in horizon-scaled ATR units."""
    period = params["period"]
    close = arr["close"]
    out = np.full(len(close), np.nan)
    out[period:] = close[period:] / close[:-period] - 1.0
    return _safe_div(out, _atr_pct_horizon(arr, period))


def a_atr_regime(arr, ctx, params):
    atr = _atr(arr["high"], arr["low"], arr["close"], params["period"])
    return _safe_div(atr, _rolling_mean_skipnan(atr, params["sma_period"]))


def a_histvol_regime(arr, ctx, params):
    hv = _historical_volatility(arr["close"], params["period"])
    return _safe_div(hv, _rolling_mean_skipnan(hv, params["sma_period"]))


# =============================================================================
# BLOCK B — range position / mean reversion
# =============================================================================
def b_bb_pctb(arr, ctx, params):
    period = params["period"]
    close = arr["close"]
    mid, sd = _roll_mean(close, period), _roll_std(close, period)
    return _safe_div(close - (mid - BB_K * sd), 2.0 * BB_K * sd)

def b_donchian_pos(arr, ctx, params):
    period = params["period"]
    hh, ll = _roll_max(arr["high"], period), _roll_min(arr["low"], period)
    return _safe_div(arr["close"] - ll, hh - ll)


def b_close_pct_rank(arr, ctx, params):
    return _rolling_pct_rank(arr["close"], params["period"]) * 100.0


def b_pivot_dist(arr, ctx, params):
    n = params["period"]
    hh, ll, c = _roll_max(arr["high"], n), _roll_min(arr["low"], n), arr["close"]
    pivot = (hh + ll + c) / 3.0
    atr = _atr(arr["high"], arr["low"], arr["close"], ATR_N)
    return _safe_div(c - pivot, atr)


# =============================================================================
# BLOCK C — trend / directionality
# =============================================================================
def c_vortex(arr, ctx, params):
    n = params["period"]
    high, low, close = arr["high"], arr["low"], arr["close"]
    m = len(close)
    vmp, vmm = np.full(m, np.nan), np.full(m, np.nan)
    vmp[1:] = np.abs(high[1:] - low[:-1])
    vmm[1:] = np.abs(low[1:] - high[:-1])
    tr_sum = _roll_sum(_true_range(high, low, close), n)
    return _safe_div(_roll_sum(vmp, n) - _roll_sum(vmm, n), tr_sum)


def c_hurst(arr, ctx, params):
    n = params["period"]
    r  = _log_returns(arr["close"])
    r2 = r + np.concatenate(([np.nan], r[:-1]))
    v1 = _roll_std(r,  n) ** 2
    v2 = _roll_std(r2, n) ** 2
    ratio = _safe_div(v2, v1)
    out = np.full(len(r), np.nan)
    ok = np.isfinite(ratio) & (ratio > 0)
    out[ok] = 0.5 * np.log2(ratio[ok])
    return out


def c_ichimoku_tenkan_kijun(arr, ctx, params):
    tenkan_n, kijun_n = params["tenkan"], params["kijun"]
    tenkan = (_roll_max(arr["high"], tenkan_n) + _roll_min(arr["low"], tenkan_n)) / 2.0
    kijun = (_roll_max(arr["high"], kijun_n) + _roll_min(arr["low"], kijun_n)) / 2.0
    return _safe_div(tenkan - kijun, _atr(arr["high"], arr["low"], arr["close"], ATR_N))


def c_ichimoku_cloud_thickness(arr, ctx, params):
    tenkan_n, kijun_n, senkou_b_n = params["tenkan"], params["kijun"], params["senkou_b"]
    tenkan = (_roll_max(arr["high"], tenkan_n) + _roll_min(arr["low"], tenkan_n)) / 2.0
    kijun = (_roll_max(arr["high"], kijun_n) + _roll_min(arr["low"], kijun_n)) / 2.0
    span_a = (tenkan + kijun) / 2.0
    span_b = (_roll_max(arr["high"], senkou_b_n) + _roll_min(arr["low"], senkou_b_n)) / 2.0
    return _safe_div(np.abs(span_a - span_b), _atr(arr["high"], arr["low"], arr["close"], ATR_N))


def c_ichimoku_price_vs_cloud(arr, ctx, params):
    tenkan_n, kijun_n, senkou_b_n = params["tenkan"], params["kijun"], params["senkou_b"]
    tenkan = (_roll_max(arr["high"], tenkan_n) + _roll_min(arr["low"], tenkan_n)) / 2.0
    kijun = (_roll_max(arr["high"], kijun_n) + _roll_min(arr["low"], kijun_n)) / 2.0
    span_a = (tenkan + kijun) / 2.0
    span_b = (_roll_max(arr["high"], senkou_b_n) + _roll_min(arr["low"], senkou_b_n)) / 2.0
    cloud_mid = (span_a + span_b) / 2.0
    return _safe_div(arr["close"] - cloud_mid, _atr(arr["high"], arr["low"], arr["close"], ATR_N))


def c_trend_intensity_index(arr, ctx, params):
    n = params["period"]
    close = arr["close"]
    sma = _sma(close, n)
    above = (close > sma).astype(np.float64)
    above[np.isnan(sma)] = np.nan
    return _roll_mean(above, n) * 100.0


# =============================================================================
# BLOCK D — advanced momentum / acceleration
# =============================================================================
def d_macd_hist(arr, ctx, params):
    close = arr["close"]
    macd  = _ema(close, params["fast"]) - _ema(close, params["slow"])
    hist  = macd - _ema(macd, params["signal"])
    return _safe_div(hist, _atr(arr["high"], arr["low"], arr["close"], ATR_N))


def d_roc_spread(arr, ctx, params):
    fast_n, slow_n = params["fast"], params["slow"]
    close = arr["close"]
    m = len(close)
    fast, slow = np.full(m, np.nan), np.full(m, np.nan)
    fast[fast_n:] = close[fast_n:] / close[:-fast_n] - 1.0
    slow[slow_n:] = close[slow_n:] / close[:-slow_n] - 1.0
    # The slow leg dominates the spread's dispersion, so it sets the scale.
    return _safe_div(fast - slow, _atr_pct_horizon(arr, slow_n))


def d_tsi(arr, ctx, params):
    long_n, short_n = params["long"], params["short"]
    close = arr["close"]
    mom = np.full(len(close), np.nan)
    mom[1:] = np.diff(close)
    num = _ema(_ema(np.nan_to_num(mom, nan=0.0), long_n), short_n)
    den = _ema(_ema(np.abs(np.nan_to_num(mom, nan=0.0)), long_n), short_n)
    out = 100.0 * _safe_div(num, den)
    out[:long_n + short_n] = np.nan
    return out


def d_acceleration(arr, ctx, params):
    close = arr["close"]
    out = np.full(len(close), np.nan)
    out[2:] = close[2:] - 2.0 * close[1:-1] + close[:-2]
    return _safe_div(out, _atr(arr["high"], arr["low"], arr["close"], ATR_N))


def d_ppo(arr, ctx, params):
    close = arr["close"]
    fast, slow = _ema(close, params["fast"]), _ema(close, params["slow"])
    return 100.0 * _safe_div(fast - slow, slow)

def d_rvi(arr, ctx, params):
    n = params["period"]
    o, h, l, c = arr["open"], arr["high"], arr["low"], arr["close"]
    num = _roll_mean(c - o, n)
    den = _roll_mean(h - l, n)
    return _safe_div(num, den)


# =============================================================================
# BLOCK E — volatility / regime
# =============================================================================
def e_bb_bandwidth(arr, ctx, params):
    period = params["period"]
    close = arr["close"]
    mid, sd = _roll_mean(close, period), _roll_std(close, period)
    return _safe_div(2.0 * BB_K * sd, mid)


def e_choppiness(arr, ctx, params):
    n = params["period"]
    high, low, close = arr["high"], arr["low"], arr["close"]
    tr_sum = _roll_sum(_true_range(high, low, close), n)
    rng    = _roll_max(high, n) - _roll_min(low, n)
    ratio  = _safe_div(tr_sum, rng)
    out = np.full(len(close), np.nan)
    ok = np.isfinite(ratio) & (ratio > 0)
    out[ok] = 100.0 * np.log10(ratio[ok]) / np.log10(n)
    return out


def e_parkinson_ratio(arr, ctx, params):
    """Intraday efficiency: range vol (Parkinson) vs close-to-close vol."""
    period = params["period"]
    hl = np.log(_safe_div(arr["high"], arr["low"]))
    park = np.sqrt(_roll_mean(hl ** 2, period) / (4.0 * np.log(2.0)))
    cc   = _roll_std(_log_returns(arr["close"]), period)
    return _safe_div(park, cc)


def e_vol_of_vol(arr, ctx, params):
    hv = _historical_volatility(arr["close"], params["period"])
    return _safe_div(_roll_std(hv, params["sma_period"]), _rolling_mean_skipnan(hv, params["sma_period"]))


def e_ret_skew(arr, ctx, params):
    return _s(_log_returns(arr["close"])).rolling(params["period"]).skew().to_numpy()


def e_donchian_width(arr, ctx, params):
    n = params["period"]
    hh, ll = _roll_max(arr["high"], n), _roll_min(arr["low"], n)
    return _safe_div(hh - ll, arr["close"])

def e_ulcer_index(arr, ctx, params):
    n = params["period"]
    close = arr["close"]
    rollmax = _roll_max(close, n)
    drawdown = 100.0 * _safe_div(close - rollmax, rollmax)
    return np.sqrt(_roll_mean(drawdown ** 2, n))

def e_bb_pctb_slope(arr, ctx, params):
    n = params["period"]
    close = arr["close"]
    mid, sd = _roll_mean(close, BB_N), _roll_std(close, BB_N)
    pctb = _safe_div(close - (mid - BB_K * sd), 2.0 * BB_K * sd)
    slope = np.full(len(pctb), np.nan)
    slope[n:] = pctb[n:] - pctb[:-n]
    return slope


# =============================================================================
# BLOCK F — bar microstructure
# =============================================================================
def f_close_pos_in_bar(arr, ctx, params):
    return _safe_div(arr["close"] - arr["low"], arr["high"] - arr["low"])


def f_range_expansion(arr, ctx, params):
    n = params["period"]
    rng = arr["high"] - arr["low"]
    return _safe_div(rng, _roll_mean(rng, n))


def f_inside_outside_ratio(arr, ctx, params):
    h, l = arr["high"], arr["low"]
    prev_h = np.concatenate(([np.nan], h[:-1]))
    prev_l = np.concatenate(([np.nan], l[:-1]))
    inside = ((h <= prev_h) & (l >= prev_l)).astype(np.float64)
    return _roll_mean(inside, params["period"]) * 100.0


def f_open_close_momentum(arr, ctx, params):
    atr = _atr(arr["high"], arr["low"], arr["close"], ATR_N)
    return _safe_div(arr["close"] - arr["open"], atr)


# =============================================================================
# BLOCK G — cycles / statistics
# =============================================================================
def g_return_entropy(arr, ctx, params):
    ret = _log_returns(arr["close"])
    return _s(ret).rolling(params["period"]).apply(
        _shannon_entropy, raw=True, engine="numba", engine_kwargs={"nopython": True, "cache": True},
        args=(ENTROPY_BINS,),
    ).to_numpy()

def g_kalman_slope(arr, ctx, params):
    slope = _kalman_slope_filter(arr["close"], KALMAN_PROCESS_VAR, KALMAN_MEASURE_VAR)
    atr = _atr(arr["high"], arr["low"], arr["close"], ATR_N)
    return _safe_div(slope, atr)


# =============================================================================
# BLOCK H: calendar / seasonality
# =============================================================================
def h_day_slot(arr, ctx, params):
    """1.0 if the bar opens in the given slot of the day (hour // SLOT_HOURS), else 0.0."""
    return (_day_slot(arr["ts"]) == params["slot"]).astype(np.float64)


def h_vol_deseason(arr, ctx, params):
    """Bar range / median range of the same slot over the previous `days` occurrences.

    With 4H bars each slot occurs once per day, so `days` previous occurrences = `days` days.
    The current bar is excluded from its own baseline.
    """
    rng  = np.asarray(arr["high"], dtype=np.float64) - np.asarray(arr["low"], dtype=np.float64)
    base = _same_slot_median(rng, _day_slot(arr["ts"]), params["days"])
    return _safe_div(rng, base)


# =============================================================================
# BLOCK I: price levels / profile
# =============================================================================
def i_round_level_phase(arr, ctx, params):
    """Distance from close to the nearest multiple of `grid` pips, as a fraction of the grid.

    Range [0, 0.5]: 0 = on a round level, 0.5 = halfway between two. Not ATR-normalized,
    so it carries no volatility information. The pip is inferred from the quote decimals.
    """
    close   = np.asarray(arr["close"], dtype=np.float64)
    grid_px = params["grid"] * _pip_size(close)
    pos     = close / grid_px
    frac    = pos - np.floor(pos)
    return np.minimum(frac, 1.0 - frac)


def i_tpo_density(arr, ctx, params):
    """Fraction of the previous `period` bars whose [low, high] contains close[t]."""
    return _tpo_density(np.asarray(arr["high"], dtype=np.float64),
                        np.asarray(arr["low"], dtype=np.float64),
                        np.asarray(arr["close"], dtype=np.float64),
                        int(params["period"]))


# =============================================================================
# BLOCK J: swing structure (zigzag with ATR reversal, confirmed pivots only)
# =============================================================================
def j_swing_leg_ratio(arr, ctx, params):
    """|close - last pivot| / |last pivot - previous pivot|: current move vs previous leg."""
    _, last_px, prev_px = _zigzag(arr, params["k"])
    return _safe_div(np.abs(np.asarray(arr["close"], dtype=np.float64) - last_px), np.abs(last_px - prev_px))


# =============================================================================
# REGISTRY — single source of truth for research AND production.
# =============================================================================
GROUP_NAMES = {
    "A": "Existing", "B": "Range/Reversion", "C": "Trend", "D": "Momentum/Accel",
    "E": "Volatility", "F": "Microstructure", "G": "Cycles/Stats", "H": "Calendar",
    "I": "Levels/Profile", "J": "Swing structure", "Z": "NOISE CONTROL",
}

CANDIDATE_REGISTRY = {
    # --- A ---
    "rsi":                      {"group": "A", "fn": a_rsi, "directional": False,
                                 "params_grid": {"period": RSI_PERIODS}, "thresholds": RSI_THRESHOLDS},
    "adx":                      {"group": "A", "fn": a_adx, "directional": False,
                                 "params_grid": {"period": ADX_PERIODS}, "thresholds": ADX_THRESHOLDS},
    "ma_dist":                  {"group": "A", "fn": a_ma_dist, "directional": True,
                                 "params_grid": {"period": MA_DIST_PERIODS}, "thresholds": MA_DIST_THRESHOLDS},
    "momentum":                 {"group": "A", "fn": a_momentum, "directional": True,
                                 "params_grid": {"period": MOMENTUM_PERIODS}, "thresholds": MOMENTUM_THRESHOLDS},
    "atr_regime":               {"group": "A", "fn": a_atr_regime, "directional": False,
                                 "params_grid": {"period": ATR_REGIME_PERIODS, "sma_period": ATR_REGIME_SMA_PERIODS},
                                 "thresholds": ATR_REGIME_THRESHOLDS},
    "histvol_regime":           {"group": "A", "fn": a_histvol_regime, "directional": False,
                                 "params_grid": {"period": HISTVOL_REGIME_PERIODS,
                                                 "sma_period": HISTVOL_REGIME_SMA_PERIODS},
                                 "thresholds": HISTVOL_REGIME_THRESHOLDS},

    # --- B ---
    "bb_pctb":                  {"group": "B", "fn": b_bb_pctb, "directional": False,
                                 "params_grid": {"period": BB_PCTB_PERIODS}, "thresholds": BB_PCTB_THRESHOLDS},
    "donchian_pos":             {"group": "B", "fn": b_donchian_pos, "directional": False,
                                 "params_grid": {"period": DONCHIAN_POS_PERIODS},
                                 "thresholds": DONCHIAN_POS_THRESHOLDS},
    "close_pct_rank":           {"group": "B", "fn": b_close_pct_rank, "directional": False,
                                 "params_grid": {"period": CLOSE_PCT_RANK_PERIODS},
                                 "thresholds": CLOSE_PCT_RANK_THRESHOLDS},
    "pivot_dist":               {"group": "B", "fn": b_pivot_dist, "directional": False,
                                 "params_grid": {"period": PIVOT_DIST_PERIODS}, "thresholds": PIVOT_DIST_THRESHOLDS},

    # --- C ---
    "vortex":                   {"group": "C", "fn": c_vortex, "directional": True,
                                 "params_grid": {"period": VORTEX_PERIODS}, "thresholds": VORTEX_THRESHOLDS},
    "hurst":                    {"group": "C", "fn": c_hurst, "directional": False,
                                 "params_grid": {"period": HURST_PERIODS}, "thresholds": HURST_THRESHOLDS},
    "ichimoku_tenkan_kijun":    {"group": "C", "fn": c_ichimoku_tenkan_kijun, "directional": True,
                                 "params_grid": {"tenkan": ICHIMOKU_TENKAN_PERIODS, "kijun": ICHIMOKU_KIJUN_PERIODS},
                                 "thresholds": ICHIMOKU_TENKAN_KIJUN_THRESHOLDS},
    "ichimoku_cloud_thickness": {"group": "C", "fn": c_ichimoku_cloud_thickness, "directional": False,
                                 "params_grid": {"tenkan": ICHIMOKU_TENKAN_PERIODS, "kijun": ICHIMOKU_KIJUN_PERIODS,
                                                 "senkou_b": ICHIMOKU_SENKOU_B_PERIODS},
                                 "thresholds": ICHIMOKU_CLOUD_THICKNESS_THRESHOLDS},
    "ichimoku_price_vs_cloud":  {"group": "C", "fn": c_ichimoku_price_vs_cloud, "directional": True,
                                 "params_grid": {"tenkan": ICHIMOKU_TENKAN_PERIODS, "kijun": ICHIMOKU_KIJUN_PERIODS,
                                                 "senkou_b": ICHIMOKU_SENKOU_B_PERIODS},
                                 "thresholds": ICHIMOKU_PRICE_VS_CLOUD_THRESHOLDS},
    "trend_intensity_index":    {"group": "C", "fn": c_trend_intensity_index, "directional": False,
                                 "params_grid": {"period": TII_PERIODS},
                                 "thresholds": TREND_INTENSITY_INDEX_THRESHOLDS},

    # --- D ---
    "macd_hist":                {"group": "D", "fn": d_macd_hist, "directional": True,
                                 "params_grid": {"fast": MACD_HIST_FAST_PERIODS, "slow": MACD_HIST_SLOW_PERIODS,
                                                 "signal": MACD_HIST_SIGNAL_PERIODS},
                                 "thresholds": MACD_HIST_THRESHOLDS},
    "roc_spread":               {"group": "D", "fn": d_roc_spread, "directional": True,
                                 "params_grid": {"fast": ROC_SPREAD_FAST_PERIODS, "slow": ROC_SPREAD_SLOW_PERIODS},
                                 "thresholds": ROC_SPREAD_THRESHOLDS},
    "tsi":                      {"group": "D", "fn": d_tsi, "directional": True,
                                 "params_grid": {"long": TSI_LONG_PERIODS, "short": TSI_SHORT_PERIODS},
                                 "thresholds": TSI_THRESHOLDS},
    "acceleration":             {"group": "D", "fn": d_acceleration, "directional": True,
                                 "params_grid": {}, "thresholds": ACCELERATION_THRESHOLDS},
    "ppo":                      {"group": "D", "fn": d_ppo, "directional": True,
                                 "params_grid": {"fast": PPO_FAST_PERIODS, "slow": PPO_SLOW_PERIODS},
                                 "thresholds": PPO_THRESHOLDS},
    "rvi":                      {"group": "D", "fn": d_rvi, "directional": False,
                                 "params_grid": {"period": RVI_PERIODS}, "thresholds": RVI_THRESHOLDS},

    # --- E ---
    "bb_bandwidth":             {"group": "E", "fn": e_bb_bandwidth, "directional": False,
                                 "params_grid": {"period": BB_BANDWIDTH_PERIODS},
                                 "thresholds": BB_BANDWIDTH_THRESHOLDS},
    "choppiness":               {"group": "E", "fn": e_choppiness, "directional": False,
                                 "params_grid": {"period": CHOPPINESS_PERIODS}, "thresholds": CHOPPINESS_THRESHOLDS},
    "parkinson_ratio":          {"group": "E", "fn": e_parkinson_ratio, "directional": False,
                                 "params_grid": {"period": PARKINSON_RATIO_PERIODS},
                                 "thresholds": PARKINSON_RATIO_THRESHOLDS},
    "vol_of_vol":               {"group": "E", "fn": e_vol_of_vol, "directional": False,
                                 "params_grid": {"period": VOL_OF_VOL_PERIODS, "sma_period": VOL_OF_VOL_SMA_PERIODS},
                                 "thresholds": VOL_OF_VOL_THRESHOLDS},
    "ret_skew":                 {"group": "E", "fn": e_ret_skew, "directional": False,
                                 "params_grid": {"period": RET_SKEW_PERIODS}, "thresholds": RET_SKEW_THRESHOLDS},
    "donchian_width":           {"group": "E", "fn": e_donchian_width, "directional": False,
                                 "params_grid": {"period": DONCHIAN_WIDTH_PERIODS},
                                 "thresholds": DONCHIAN_WIDTH_THRESHOLDS},
    "ulcer_index":              {"group": "E", "fn": e_ulcer_index, "directional": False,
                                 "params_grid": {"period": ULCER_PERIODS}, "thresholds": ULCER_INDEX_THRESHOLDS},
    "bb_pctb_slope":            {"group": "E", "fn": e_bb_pctb_slope, "directional": False,
                                 "params_grid": {"period": BB_PCTB_SLOPE_PERIODS},
                                 "thresholds": BB_PCTB_SLOPE_THRESHOLDS},

    # --- F ---
    "close_pos_in_bar":         {"group": "F", "fn": f_close_pos_in_bar, "directional": False,
                                 "params_grid": {}, "thresholds": CLOSE_POS_IN_BAR_THRESHOLDS},
    "range_expansion":          {"group": "F", "fn": f_range_expansion, "directional": False,
                                 "params_grid": {"period": RANGE_EXPANSION_PERIODS},
                                 "thresholds": RANGE_EXPANSION_THRESHOLDS},
    "inside_outside_ratio":     {"group": "F", "fn": f_inside_outside_ratio, "directional": False,
                                 "params_grid": {"period": INSIDE_OUTSIDE_PERIODS},
                                 "thresholds": INSIDE_OUTSIDE_RATIO_THRESHOLDS},
    "open_close_momentum":      {"group": "F", "fn": f_open_close_momentum, "directional": True,
                                 "params_grid": {}, "thresholds": OPEN_CLOSE_MOMENTUM_THRESHOLDS},

    # --- G ---
    "return_entropy":           {"group": "G", "fn": g_return_entropy, "directional": False,
                                 "params_grid": {"period": ENTROPY_PERIODS}, "thresholds": RETURN_ENTROPY_THRESHOLDS},
    "kalman_slope":             {"group": "G", "fn": g_kalman_slope, "directional": True,
                                 "params_grid": {}, "thresholds": KALMAN_SLOPE_THRESHOLDS},

    # --- H ---
    "day_slot":                 {"group": "H", "fn": h_day_slot, "directional": False,
                                 "params_grid": {"slot": DAY_SLOT_SLOTS},
                                 "thresholds": DAY_SLOT_THRESHOLDS, "ops": (">",)},
    "vol_deseason":             {"group": "H", "fn": h_vol_deseason, "directional": False,
                                 "params_grid": {"days": VOL_DESEASON_DAYS}, "thresholds": VOL_DESEASON_THRESHOLDS},

    # --- I ---
    "round_level_phase":        {"group": "I", "fn": i_round_level_phase, "directional": False,
                                 "params_grid": {"grid": ROUND_LEVEL_GRIDS},
                                 "thresholds": ROUND_LEVEL_PHASE_THRESHOLDS},
    "tpo_density":              {"group": "I", "fn": i_tpo_density, "directional": False,
                                 "params_grid": {"period": TPO_DENSITY_PERIODS}, "thresholds": TPO_DENSITY_THRESHOLDS},

    # --- J ---
    "swing_leg_ratio":          {"group": "J", "fn": j_swing_leg_ratio, "directional": False,
                                 "params_grid": {"k": SWING_ATR_MULTS}, "thresholds": SWING_LEG_RATIO_THRESHOLDS},
}
# =============================================================================
# SPEC GENERATOR — expands params_grid x thresholds x ops into flat specs.
# =============================================================================
def build_flat_instances() -> list:

    instances = []
    for name, meta in CANDIDATE_REGISTRY.items():
        grid = meta["params_grid"]
        param_names = list(grid.keys())
        param_combos = list(itertools.product(*grid.values())) if param_names else [()]
        for combo in param_combos:
            instances.append({"indicator": name, "params": dict(zip(param_names, combo))})
    return instances

def instance_key(indicator: str, params: dict) -> str:
    """Unique string id for one (indicator, params) value series."""
    if not params:
        return indicator
    parts = "_".join(f"{k}{v}" for k, v in sorted(params.items()))
    return f"{indicator}__{parts}"

def build_flat_specs() -> list:
    specs = []
    for inst in build_flat_instances():
        name, params = inst["indicator"], inst["params"]
        meta = CANDIDATE_REGISTRY[name]
        key = instance_key(name, params)
        ops = meta.get("ops", (">", "<"))
        for th in meta["thresholds"]:
            for op in ops:
                specs.append({
                    "indicator": name,
                    "params": params,
                    "key": key,
                    "op": op,
                    "threshold": float(th),
                })
    return specs

def describe_spec(spec: dict) -> str:
    parts = []
    for k, v in spec["params"].items():
        key = "" if k == "period" else (k[:-len("period")].rstrip("_") if k.endswith("period") else k)
        parts.append(f"{key}{v}" if key else f"{v}")
    param_str = "_".join(parts)
    prefix = f"{spec['indicator']}_{param_str}" if param_str else spec["indicator"]
    return f"{prefix}{spec['op']}{spec['threshold']:g}"

# =============================================================================
# RULE COMBINATION LOGIC — shared by rule_generator.py (backtesting) and
# =============================================================================
def implied_side(spec: dict) -> str | None:
    """'long'/'short' if this spec only makes directional sense that way; None otherwise."""
    if not CANDIDATE_REGISTRY[spec["indicator"]]["directional"]:
        return None
    op, threshold = spec["op"], spec["threshold"]
    if op == ">" and threshold > 0:
        return "long"
    if op == "<" and threshold < 0:
        return "short"
    return None

def _combo_has_side_conflict(members: tuple, specs: list) -> bool:
    sides = {implied_side(specs[i]) for i in members}
    sides.discard(None)
    return len(sides) > 1

def generate_valid_combos(specs: list, depth: int, indices: list = None) -> list:
    candidate_indices = indices if indices is not None else range(len(specs))

    by_indicator = {}
    for i in candidate_indices:
        by_indicator.setdefault(specs[i]["indicator"], []).append(i)

    names  = list(by_indicator.keys())
    combos = []
    for name_combo in itertools.combinations(names, depth):
        pools = [by_indicator[n] for n in name_combo]
        for members in itertools.product(*pools):
            members = tuple(sorted(members))
            if _combo_has_side_conflict(members, specs):
                continue
            combos.append(members)
    return combos