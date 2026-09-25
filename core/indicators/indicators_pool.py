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
RSI_PDS               = [7,14,21]
RSI_THS               = [30,40,60,70]

ADX_PDS               = [7,14,21]
ADX_THS               = [10,20,30]

MA_DIST_PDS           = [20,50,100]
MA_DIST_THS           = [-1.0,-0.8,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.8,1.0]

MOMENTUM_PDS          = [10,20,30]
MOMENTUM_THS          = [-1.0,-0.8,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.8,1.0]

ATR_REGIME_PDS        = [7,14,21]
ATR_REGIME_SMA_PDS    = [30,60]
ATR_REGIME_THS        = [0.5,0.7,1.0,1.4,2.0]

HISTVOL_REGIME_PDS    = [20,30]
HISTVOL_REGIME_SMA_PDS= [40,60]
HISTVOL_REGIME_THS    = [0.5,0.7,1.0,1.4,2.0]

# --- B: range / reversion -----------------------------------------------------
BB_PCTB_PDS           = [10,20,40]
BB_PCTB_THS           = [0.2,0.4,0.6,0.7,0.8,0.9]

DONCHIAN_POS_PDS      = [7,14,21]
DONCHIAN_POS_THS      = [0.2,0.4,0.6,0.8,0.9]

CLOSE_PCT_RANK_PDS    = [100]
CLOSE_PCT_RANK_THS    = [20.0, 50.0, 80.0]

PIVOT_DIST_PDS        = [10,20,40]
PIVOT_DIST_THS        = [-1.0, -0.3, 0.3, 1.0]

# --- C: trend -----------------------------------------------------------------
VORTEX_PDS            = [7,14,21]
VORTEX_THS            = [-0.5,-0.3,0.3,0.5]

HURST_PDS             = [30,60]
HURST_THS             = [0.4,0.5,0.6]

ICHIMOKU_TENKAN_PDS   = [9]
ICHIMOKU_K_PDS        = [26]
ICHIMOKU_SENKOU_B_PDS = [52]
ICHIMOKU_TENKAN_K_THS = [-0.5,0.5]
ICHIMOKU_CL_THICK_THS = [0.2,0.4,0.6,0.8,1.0]
ICHIMOKU_PR_VS_CL_THS = [-1.0,-0.5,0.5,1.0]

TII_PDS                = [15,30,50]
TREND_INTEN_INDEX_THS  = [30.0, 50.0, 70.0]

# --- D: momentum / acceleration -----------------------------------------------
MACD_HIST_FAST_PDS    = [6,12]
MACD_HIST_SLOW_PDS    = [26,52]
MACD_HIST_SIGNAL_PDS  = [9]
MACD_HIST_THS         = [-0.4,-0.2,0.2,0.4]

ROC_SPREAD_FAST_PDS   = [5,10]
ROC_SPREAD_SLOW_PDS   = [20,40]
ROC_SPREAD_THS        = [-0.8,-0.6,-0.4,0.4,0.6,0.8]

TSI_LONG_PDS          = [25,50]
TSI_SHORT_PDS         = [13,26]
TSI_THS               = [-40.0, -20.0, 20.0, 40.0]

ACCELERATION_THS      = [-1.0,-0.8,-0.6,-0.4,0.4,0.6,0.8,1.0]

PPO_FAST_PDS          = [6,12]
PPO_SLOW_PDS          = [26,52]
PPO_THS               = [-1.0, -0.3, 0.3, 1.0]

RVI_PDS               = [10,20,40]
RVI_THS               = [-0.3, -0.1, 0.1, 0.3]

# --- E: volatility ------------------------------------------------------------
BB_BANDWIDTH_PDS      = [15,30,60]
BB_BANDWIDTH_THS      = [0.01,0.02,0.03,0.04]

CHOPPINESS_PDS        = [7,14,21]
CHOPPINESS_THS        = [38.0, 62.0]

PARKINSON_RATIO_PDS   = [10,20,40]
PARKINSON_RATIO_THS   = [0.8, 1.2]

VOL_OF_VOL_PDS        = [10,20]
VOL_OF_VOL_SMA_PDS    = [30,60]
VOL_OF_VOL_THS        = [0.3,0.4,0.5,0.6]

RET_SKEW_PDS          = [30,60]
RET_SKEW_THS          = [-1.0, -0.5, 0.5, 1.0]

DONCHIAN_WIDTH_PDS    = [10,20,40]
DONCHIAN_WIDTH_THS    = [0.01,0.02,0.03,0.04,0.05]

ULCER_PDS             = [7,14,21]
ULCER_INDEX_THS       = [1.0,2.0,3.0, 4.0]

BB_PCTB_SLOPE_PDS     = [5]
BB_PCTB_SLOPE_THS     = [-0.2, -0.05, 0.05, 0.2]

# --- F: microstructure --------------------------------------------------------
CLOSE_POS_IN_BAR_THS  = [0.1, 0.2, 0.8, 0.9]

RANGE_EXPANSION_PDS   = [10,20,40]
RANGE_EXPANSION_THS   = [0.5, 0.75, 1.5, 2.0]

INSIDE_OUTSIDE_PDS       = [20,40]
INSIDE_OUTSIDE_RATIO_THS = [10.0, 25.0, 40.0]

OPEN_CLOSE_MOMENTUM_THS  = [-0.5, -0.2, 0.2, 0.5]

# --- G: cycles / statistics ---------------------------------------------------
ENTROPY_PDS           = [30,60]
RETURN_ENTROPY_THS    = [1.55,1.75,1.9,2.05]

KALMAN_SLOPE_THS      = [-0.3, -0.1, 0.1, 0.3]

# --- H: calendar / seasonality ------------------------------------------------
DAY_SLOT_SLOTS        = [0,1,2,3,4,5]
DAY_SLOT_THS          = [0.5]

VOL_DESEASON_DAYS     = [10,16]
VOL_DESEASON_THS      = [0.5,0.7,1.0,1.4,2.0]

TPO_DENSITY_PDS       = [50,75,99]
TPO_DENSITY_THS       = [0.05,0.15,0.3]

# =============================================================================
# SHAPE CONSTANTS — not exposed as grids, they define a formula's internal
# =============================================================================
ATR_N                = 14   # shared ATR normalizer used across most indicators
BB_N                 = 20   # underlying BB window used inside e_bb_pctb_slope

BB_K                 = 2.0

ENTROPY_BINS         = 10
KALMAN_PROCESS_VAR   = 1e-5
KALMAN_MEASURE_VAR   = 1e-2

SLOT_HOURS           = 4    #MIN HOURS

MAX_LOOKBACK         = 100  # max candles (the current one included) any indicator value may use

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


@njit(cache=True)
def _fir(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """y[t] = sum of w[k] * x[t-k], k = 0..len(w)-1 (w[0]: weight of the newest value).
    NaN for the first len(w)-1 values and wherever a value of the window is NaN."""
    n   = len(x)
    win = len(w)
    out = np.full(n, np.nan)
    for t in range(win - 1, n):
        acc = 0.0
        for k in range(win):
            acc += w[k] * x[t - k]
        out[t] = acc
    return out


def _ewm_win(x: np.ndarray, alpha: float, win: int) -> np.ndarray:
    """EWM (adjust=False recursion) truncated to its last `win` values, weights renormalized.
    The value at t uses x[t-win+1..t] only; NaN if any of them is NaN."""
    w = (1.0 - alpha) ** np.arange(win, dtype=np.float64)       # weight of x[t-k], newest first
    return _fir(np.ascontiguousarray(x, dtype=np.float64), w / w.sum())


def _rsi(close: np.ndarray, window: int) -> np.ndarray:
    n     = len(close)
    out   = np.full(n, np.nan)
    alpha = 1.0 / window

    diff  = np.diff(close)
    up    = np.where(diff > 0, diff, 0.0)
    dn    = np.where(diff < 0, -diff, 0.0)
    emaup = _ewm_win(up, alpha, MAX_LOOKBACK - 1)                 # 99 changes = 100 closes
    emadn = _ewm_win(dn, alpha, MAX_LOOKBACK - 1)

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
    n = len(close)

    diff_dm = np.full(n, np.nan)       # index 0 has no previous candle
    pos     = np.full(n, np.nan)
    neg     = np.full(n, np.nan)

    prev_close = close[:-1]
    pdm = np.maximum(high[1:], prev_close)
    pdn = np.minimum(low[1:], prev_close)
    diff_dm[1:] = pdm - pdn

    diff_up   = high[1:] - high[:-1]
    diff_down = low[:-1] - low[1:]
    pos[1:] = np.where((diff_up > diff_down) & (diff_up > 0), diff_up, 0.0)
    neg[1:] = np.where((diff_down > diff_up) & (diff_down > 0), diff_down, 0.0)

    # Wilder smoothing truncated: TR and DM over 50 changes (51 candles), then DX over 50 values -> 100 candles
    half  = MAX_LOOKBACK // 2
    alpha = 1.0 / window
    trs = _ewm_win(diff_dm, alpha, half)
    dip = _ewm_win(pos, alpha, half)
    din = _ewm_win(neg, alpha, half)

    with np.errstate(divide="ignore", invalid="ignore"):
        di_p = np.where(trs != 0, 100.0 * dip / trs, 0.0)
        di_n = np.where(trs != 0, 100.0 * din / trs, 0.0)
    denom = di_p + di_n
    with np.errstate(divide="ignore", invalid="ignore"):
        dx = np.where(denom != 0, 100.0 * np.abs(di_p - di_n) / denom, 0.0)

    return _ewm_win(dx, alpha, half)


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


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int,
         win: int = MAX_LOOKBACK - 1) -> np.ndarray:
    """Wilder ATR truncated to its last `win` true ranges (win + 1 candles; default 100)."""
    tr    = _true_range(high, low, close)
    tr[0] = np.nan                     # no previous close
    return _ewm_win(tr, 1.0 / window, win)


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


def _ema(x: np.ndarray, n: int, win: int) -> np.ndarray:
    """EMA (span n) truncated to its last `win` values."""
    return _ewm_win(x, 2.0 / (n + 1.0), win)


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

@njit(cache=True)
def _kalman_slope_core(close: np.ndarray, q: float, r: float) -> np.ndarray:
    m   = len(close)
    out = np.empty(m)
    if m == 0:
        return out

    # state = [level, slope], covariance p = [[p00, p01], [p10, p11]]
    s0  = close[0]
    s1  = 0.0
    p00 = 1.0
    p01 = 0.0
    p10 = 0.0
    p11 = 1.0

    for i in range(m):
        # predict: state = F @ state, p = F @ p @ F.T + Q, with F = [[1, 1], [0, 1]]
        s0  = s0 + s1
        a00 = p00 + p10
        a01 = p01 + p11
        p00 = (a00 + a01) + q
        p01 = a01
        p10 = p10 + p11
        p11 = p11 + q

        # update, with H = [1, 0]
        y  = close[i] - s0
        s  = p00 + r
        k0 = p00 / s
        k1 = p10 / s
        s0 = s0 + k0 * y
        s1 = s1 + k1 * y

        n00 = p00 - k0 * p00
        n01 = p01 - k0 * p01
        n10 = p10 - k1 * p00
        n11 = p11 - k1 * p01
        p00 = n00
        p01 = n01
        p10 = n10
        p11 = n11

        out[i] = s1
    return out
def _kalman_slope_weights(q: float, r: float, win: int) -> np.ndarray:
    """Weights of the last slope of _kalman_slope_core run on `win` closes (index 0: newest close).
    The filter is linear in the closes (start state included) and its gains do not depend on them,
    so its last slope is sum of w[k] * close[t-k]: one impulse per position gives w."""
    w = np.empty(win)
    e = np.zeros(win)
    for k in range(win):
        e[:] = 0.0
        e[win - 1 - k] = 1.0
        w[k] = _kalman_slope_core(e, q, r)[-1]
    return w


def _kalman_slope_filter(close: np.ndarray, q: float, r: float, win: int = MAX_LOOKBACK) -> np.ndarray:
    """Slope of the Kalman filter restarted at every candle on its last `win` closes (NaN before)."""
    w = _kalman_slope_weights(float(q), float(r), int(win))
    return _fir(np.ascontiguousarray(close, dtype=np.float64), w)

# -----------------------------------------------------------------------------
# Primitives for blocks H, I. All causal: the value at t uses bars <= t only.
# -----------------------------------------------------------------------------
def _bar_step_ns(ts: np.ndarray):

    if len(ts) < 2:
        return 0, 0
    gaps = np.diff(ts.view(np.int64))
    gaps = gaps[gaps > 0]
    if len(gaps) == 0:
        return 0, 0
    vals, counts = np.unique(gaps, return_counts=True)
    i = int(np.argmax(counts))
    return int(vals[i]), int(counts[i])


def _day_slot(ts: np.ndarray) -> np.ndarray:

    ts     = np.asarray(ts).astype("datetime64[ns]")
    in_day = ts - ts.astype("datetime64[D]")
    hour   = (in_day // np.timedelta64(1, "h")).astype(np.int64)
    minute = (in_day // np.timedelta64(1, "m")).astype(np.int64) % 60

    hour_ns    = 3_600_000_000_000
    step, n    = _bar_step_ns(ts)
    slot_hours = SLOT_HOURS
    if step > SLOT_HOURS * hour_ns and n >= 2:
        slot_hours = min(step, 24 * hour_ns) // hour_ns
        if step < 24 * hour_ns and (step % hour_ns != 0 or 24 % slot_hours != 0):
            raise ValueError("day_slot: bars above SLOT_HOURS must be a whole number of hours dividing 24")

    if step == slot_hours * hour_ns:
        if (minute != 0).any():
            raise ValueError("day_slot: some bars do not open at minute 0")
        rem = hour % slot_hours
        if (rem == 0).any() and (rem == slot_hours - 1).any():
            raise ValueError("day_slot: bar opens cross a slot boundary across DST changes; "
                             "timestamps are not in a DST-consistent clock")
    return hour // slot_hours


@njit(cache=True)
def _same_slot_median_pos(x, slot, n_pos):

    m   = len(x)
    out = np.full(m, np.nan)
    buf = np.empty(n_pos)
    for t in range(n_pos, m):
        s     = slot[t]
        k     = 0
        valid = True
        for j in range(t - n_pos, t):
            if slot[j] == s:
                v = x[j]
                if v != v:
                    valid = False
                    break
                buf[k] = v
                k += 1
        if valid and k > 0:
            out[t] = np.median(buf[:k])
    return out

def _same_slot_median(x: np.ndarray, slot: np.ndarray, n_prev: int) -> np.ndarray:

    x      = np.ascontiguousarray(x, dtype=np.float64)
    slot   = np.ascontiguousarray(slot, dtype=np.int64)
    n_slot = int(slot.max()) + 1 if len(slot) else 1
    return _same_slot_median_pos(x, slot, int(n_prev) * n_slot)

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
    # ATR over 100 - sma_period candles, then its mean over sma_period values -> 100 candles
    atr = _atr(arr["high"], arr["low"], arr["close"], params["period"], win=MAX_LOOKBACK - params["sma_period"])
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


def c_ichimoku_tenkan_K(arr, ctx, params):
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
    # EMAs over 70 closes, signal over 31 MACD values -> 100 closes
    macd  = _ema(close, params["fast"], 70) - _ema(close, params["slow"], 70)
    hist  = macd - _ema(macd, params["signal"], MAX_LOOKBACK + 1 - 70)
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
    # long EMA over 65 changes (66 closes), short EMA over 35 of its values -> 100 closes
    num = _ema(_ema(mom, long_n, 65), short_n, MAX_LOOKBACK - 65)
    den = _ema(_ema(np.abs(mom), long_n, 65), short_n, MAX_LOOKBACK - 65)
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
    fast, slow = _ema(close, params["fast"], MAX_LOOKBACK), _ema(close, params["slow"], MAX_LOOKBACK)
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

    rng  = np.asarray(arr["high"], dtype=np.float64) - np.asarray(arr["low"], dtype=np.float64)
    base = _same_slot_median(rng, _day_slot(arr["ts"]), params["days"])
    return _safe_div(rng, base)


# =============================================================================
# BLOCK I: price levels / profile
# =============================================================================
def i_tpo_density(arr, ctx, params):
    """Fraction of the previous `period` bars whose [low, high] contains close[t]."""
    return _tpo_density(np.asarray(arr["high"], dtype=np.float64),
                        np.asarray(arr["low"], dtype=np.float64),
                        np.asarray(arr["close"], dtype=np.float64),
                        int(params["period"]))


# =============================================================================
# REGISTRY — single source of truth for research AND production.
# =============================================================================
GROUP_NAMES = {
    "A": "Existing", "B": "Range/Reversion", "C": "Trend", "D": "Momentum/Accel",
    "E": "Volatility", "F": "Microstructure", "G": "Cycles/Stats", "H": "Calendar",
    "I": "Levels/Profile",
}

CANDIDATE_REGISTRY = {
    # --- A ---
    "rsi":                      {"group": "A", "fn": a_rsi, "role": "signal",
                                 "params_grid": {"period": RSI_PDS}, "thresholds": RSI_THS},
    "adx":                      {"group": "A", "fn": a_adx, "role": "filter",
                                 "params_grid": {"period": ADX_PDS}, "thresholds": ADX_THS},
    "ma_dist":                  {"group": "A", "fn": a_ma_dist, "role": "signal",
                                 "params_grid": {"period": MA_DIST_PDS}, "thresholds": MA_DIST_THS},
    "momentum":                 {"group": "A", "fn": a_momentum, "role": "signal",
                                 "params_grid": {"period": MOMENTUM_PDS}, "thresholds": MOMENTUM_THS},
    "atr_regime":               {"group": "A", "fn": a_atr_regime, "role": "filter",
                                 "params_grid": {"period": ATR_REGIME_PDS, "sma_period": ATR_REGIME_SMA_PDS},
                                 "thresholds": ATR_REGIME_THS},
    "histvol_regime":           {"group": "A", "fn": a_histvol_regime, "role": "filter",
                                 "params_grid": {"period": HISTVOL_REGIME_PDS,
                                                 "sma_period": HISTVOL_REGIME_SMA_PDS},
                                 "thresholds": HISTVOL_REGIME_THS},

    # --- B ---
    "bb_pctb":                  {"group": "B", "fn": b_bb_pctb, "role": "signal",
                                 "params_grid": {"period": BB_PCTB_PDS}, "thresholds": BB_PCTB_THS},
    "donchian_pos":             {"group": "B", "fn": b_donchian_pos, "role": "signal",
                                 "params_grid": {"period": DONCHIAN_POS_PDS},
                                 "thresholds": DONCHIAN_POS_THS},
    "close_pct_rank":           {"group": "B", "fn": b_close_pct_rank, "role": "signal",
                                 "params_grid": {"period": CLOSE_PCT_RANK_PDS},
                                 "thresholds": CLOSE_PCT_RANK_THS},
    "pivot_dist":               {"group": "B", "fn": b_pivot_dist, "role": "signal",
                                 "params_grid": {"period": PIVOT_DIST_PDS}, "thresholds": PIVOT_DIST_THS},

    # --- C ---
    "vortex":                   {"group": "C", "fn": c_vortex, "role": "signal",
                                 "params_grid": {"period": VORTEX_PDS}, "thresholds": VORTEX_THS},
    "hurst":                    {"group": "C", "fn": c_hurst, "role": "filter",
                                 "params_grid": {"period": HURST_PDS}, "thresholds": HURST_THS},
    "ichimoku_tenkan_K":    {"group": "C", "fn": c_ichimoku_tenkan_K, "role": "signal",
                                 "params_grid": {"tenkan": ICHIMOKU_TENKAN_PDS, "kijun": ICHIMOKU_K_PDS},
                                 "thresholds": ICHIMOKU_TENKAN_K_THS},
    "ichimoku_cloud_thickness": {"group": "C", "fn": c_ichimoku_cloud_thickness, "role": "filter",
                                 "params_grid": {"tenkan": ICHIMOKU_TENKAN_PDS, "kijun": ICHIMOKU_K_PDS,
                                                 "senkou_b": ICHIMOKU_SENKOU_B_PDS},
                                 "thresholds": ICHIMOKU_CL_THICK_THS},
    "ichimoku_price_vs_cloud":  {"group": "C", "fn": c_ichimoku_price_vs_cloud, "role": "signal",
                                 "params_grid": {"tenkan": ICHIMOKU_TENKAN_PDS, "kijun": ICHIMOKU_K_PDS,
                                                 "senkou_b": ICHIMOKU_SENKOU_B_PDS},
                                 "thresholds": ICHIMOKU_PR_VS_CL_THS},
    "trend_intensity_index":    {"group": "C", "fn": c_trend_intensity_index, "role": "signal",
                                 "params_grid": {"period": TII_PDS},
                                 "thresholds": TREND_INTEN_INDEX_THS},

    # --- D ---
    "macd_hist":                {"group": "D", "fn": d_macd_hist, "role": "signal",
                                 "params_grid": {"fast": MACD_HIST_FAST_PDS, "slow": MACD_HIST_SLOW_PDS,
                                                 "signal": MACD_HIST_SIGNAL_PDS},
                                 "thresholds": MACD_HIST_THS},
    "roc_spread":               {"group": "D", "fn": d_roc_spread, "role": "signal",
                                 "params_grid": {"fast": ROC_SPREAD_FAST_PDS, "slow": ROC_SPREAD_SLOW_PDS},
                                 "thresholds": ROC_SPREAD_THS},
    "tsi":                      {"group": "D", "fn": d_tsi, "role": "signal",
                                 "params_grid": {"long": TSI_LONG_PDS, "short": TSI_SHORT_PDS},
                                 "thresholds": TSI_THS},
    "acceleration":             {"group": "D", "fn": d_acceleration, "role": "signal",
                                 "params_grid": {}, "thresholds": ACCELERATION_THS},
    "ppo":                      {"group": "D", "fn": d_ppo, "role": "signal",
                                 "params_grid": {"fast": PPO_FAST_PDS, "slow": PPO_SLOW_PDS},
                                 "thresholds": PPO_THS},
    "rvi":                      {"group": "D", "fn": d_rvi, "role": "signal",
                                 "params_grid": {"period": RVI_PDS}, "thresholds": RVI_THS},

    # --- E ---
    "bb_bandwidth":             {"group": "E", "fn": e_bb_bandwidth, "role": "filter",
                                 "params_grid": {"period": BB_BANDWIDTH_PDS},
                                 "thresholds": BB_BANDWIDTH_THS},
    "choppiness":               {"group": "E", "fn": e_choppiness, "role": "filter",
                                 "params_grid": {"period": CHOPPINESS_PDS}, "thresholds": CHOPPINESS_THS},
    "parkinson_ratio":          {"group": "E", "fn": e_parkinson_ratio, "role": "filter",
                                 "params_grid": {"period": PARKINSON_RATIO_PDS},
                                 "thresholds": PARKINSON_RATIO_THS},
    "vol_of_vol":               {"group": "E", "fn": e_vol_of_vol, "role": "filter",
                                 "params_grid": {"period": VOL_OF_VOL_PDS, "sma_period": VOL_OF_VOL_SMA_PDS},
                                 "thresholds": VOL_OF_VOL_THS},
    "ret_skew":                 {"group": "E", "fn": e_ret_skew, "role": "filter",
                                 "params_grid": {"period": RET_SKEW_PDS}, "thresholds": RET_SKEW_THS},
    "donchian_width":           {"group": "E", "fn": e_donchian_width, "role": "filter",
                                 "params_grid": {"period": DONCHIAN_WIDTH_PDS},
                                 "thresholds": DONCHIAN_WIDTH_THS},
    "ulcer_index":              {"group": "E", "fn": e_ulcer_index, "role": "filter",
                                 "params_grid": {"period": ULCER_PDS}, "thresholds": ULCER_INDEX_THS},
    "bb_pctb_slope":            {"group": "E", "fn": e_bb_pctb_slope, "role": "signal",
                                 "params_grid": {"period": BB_PCTB_SLOPE_PDS},
                                 "thresholds": BB_PCTB_SLOPE_THS},

    # --- F ---
    "close_pos_in_bar":         {"group": "F", "fn": f_close_pos_in_bar, "role": "signal",
                                 "params_grid": {}, "thresholds": CLOSE_POS_IN_BAR_THS},
    "range_expansion":          {"group": "F", "fn": f_range_expansion, "role": "filter",
                                 "params_grid": {"period": RANGE_EXPANSION_PDS},
                                 "thresholds": RANGE_EXPANSION_THS},
    "inside_outside_ratio":     {"group": "F", "fn": f_inside_outside_ratio, "role": "filter",
                                 "params_grid": {"period": INSIDE_OUTSIDE_PDS},
                                 "thresholds": INSIDE_OUTSIDE_RATIO_THS},
    "open_close_momentum":      {"group": "F", "fn": f_open_close_momentum, "role": "signal",
                                 "params_grid": {}, "thresholds": OPEN_CLOSE_MOMENTUM_THS},

    # --- G ---
    "return_entropy":           {"group": "G", "fn": g_return_entropy, "role": "filter",
                                 "params_grid": {"period": ENTROPY_PDS}, "thresholds": RETURN_ENTROPY_THS},
    "kalman_slope":             {"group": "G", "fn": g_kalman_slope, "role": "signal",
                                 "params_grid": {}, "thresholds": KALMAN_SLOPE_THS},

# =============================================================================
#     # --- H ---
#     "day_slot":                 {"group": "H", "fn": h_day_slot, "role": "filter",
#                                  "params_grid": {"slot": DAY_SLOT_SLOTS},
#                                  "thresholds": DAY_SLOT_THS},
#     "vol_deseason":             {"group": "H", "fn": h_vol_deseason, "role": "filter",
#                                  "params_grid": {"days": VOL_DESEASON_DAYS}, "thresholds": VOL_DESEASON_THS},
# 
# =============================================================================
    # --- I ---
    "tpo_density":              {"group": "I", "fn": i_tpo_density, "role": "filter",
                                 "params_grid": {"period": TPO_DENSITY_PDS}, "thresholds": TPO_DENSITY_THS},
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