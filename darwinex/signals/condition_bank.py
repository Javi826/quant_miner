#signals/condition_bank.py
import numpy as np
import pandas as pd
from scipy import signal as sp_signal

# =============================================================================
# CONFIG
# =============================================================================

RSI_PERIODS                = [7,14,21]  
RSI_THRESHOLDS             = [30,40,50,60,70]  

ADX_PERIODS                = [7,14,21]  
ADX_THRESHOLDS             = [20,25,30,35]   
 
MA_PERIODS                 = [20,50,100]
 
MOMENTUM_PERIODS           = [5,10,15,20]   

HISTVOL_BASE_PERIODS       = [10,20,30]     
HISTVOL_REGIME_SMA_PERIODS = [20,30,40,50]     

ATR_BASE_PERIODS           = [7,14,21]        
ATR_REGIME_SMA_PERIODS     = [10,20,30,40,50]


#------------------------------------------------------------------------------

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

    diff = np.diff(close)
    up   = np.where(diff > 0, diff, 0.0)
    dn   = np.where(diff < 0, -diff, 0.0)

    # Recurrence starts from a virtual zero seed (emaup_1 = alpha*up_1, not up_1
    # itself), matching the original loop's initialization at i=1.
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

def _ema(values: np.ndarray, period: int) -> np.ndarray:
    return pd.Series(values, dtype=np.float64).ewm(span=period, adjust=False).mean().to_numpy()

from numpy.lib.stride_tricks import sliding_window_view
 
# ---- Ubicación 2: reemplaza las dos funciones _rolling_max_shifted / _rolling_min_shifted ----
 
def _rolling_max_shifted(values: np.ndarray, window: int) -> np.ndarray:
    n   = len(values)
    out = np.full(n, np.nan)
    if window >= n:
        return out
    windows      = sliding_window_view(values[:-1], window)
    out[window:] = windows.max(axis=1)
    return out
  
def _rolling_min_shifted(values: np.ndarray, window: int) -> np.ndarray:
    n   = len(values)
    out = np.full(n, np.nan)
    if window >= n:
        return out
    windows      = sliding_window_view(values[:-1], window)
    out[window:] = windows.min(axis=1)
    return out

# =============================================================================
# INDICATOR REGISTRY
# =============================================================================

def _build_specs_threshold(entry):
    specs = []
    for period in entry["periods"]:
        for th in entry["thresholds"]:
            for op in entry["ops"]:
                specs.append({"type": entry["type"], "period": period, "op": op, "value": th})
    return specs


def _build_specs_own_value(entry):
    specs = []
    for value in entry["periods"]:
        for op in entry["ops"]:
            specs.append({"type": entry["type"], "op": op, "value": value})
    return specs


def _build_specs_two_periods(entry):
    specs = []
    for period in entry["periods"]:
        for sma_period in entry["sma_periods"]:
            for op in entry["ops"]:
                specs.append({"type": entry["type"], "period": period, "sma_period": sma_period, "op": op})
    return specs

INDICATOR_REGISTRY = [
    {
        "type": "rsi",
        "identity_keys": ["period"],
        "has_threshold": True,
        "periods": RSI_PERIODS,
        "thresholds": RSI_THRESHOLDS,
        "ops": [">", "<"],
        "build_specs": _build_specs_threshold,
        "evaluate": lambda bank, spec: (
            bank._get_cached("rsi", spec["period"], lambda b: _rsi(b.close, spec["period"]))
            > spec["value"]
            if spec["op"] == ">"
            else bank._get_cached("rsi", spec["period"], lambda b: _rsi(b.close, spec["period"]))
            < spec["value"]
        ),
        "describe": lambda spec: f"RSI{spec['period']}{spec['op']}{spec['value']}",
    },
    {
        "type": "adx",
        "identity_keys": ["period"],
        "has_threshold": True,
        "periods": ADX_PERIODS,
        "thresholds": ADX_THRESHOLDS,
        "ops": [">"],
        "build_specs": _build_specs_threshold,
        "evaluate": lambda bank, spec: (
            bank._get_cached("adx", spec["period"], lambda b: _adx(b.high, b.low, b.close, spec["period"]))
            > spec["value"]
        ),
        "describe": lambda spec: f"ADX{spec['period']}{spec['op']}{spec['value']}",
    },
    {
        "type": "ma",
        "identity_keys": ["value"],
        "has_threshold": False,
        "periods": MA_PERIODS,
        "ops": [">", "<"],
        "build_specs": _build_specs_own_value,
        "evaluate": lambda bank, spec: (
            bank.close
            > bank._get_cached("ma", spec["value"], lambda b: _sma(b.close, spec["value"]))
            if spec["op"] == ">"
            else bank.close
            < bank._get_cached("ma", spec["value"], lambda b: _sma(b.close, spec["value"]))
        ),
        "describe": lambda spec: f"CLOSE{spec['op']}MA{spec['value']}",
    },
    {
        "type": "momentum",
        "identity_keys": ["value"],
        "has_threshold": False,
        "periods": MOMENTUM_PERIODS,
        "ops": [">", "<"],
        "build_cache": None,
        "build_specs": _build_specs_own_value,
        "evaluate": lambda bank, spec: bank._momentum_mask(spec["op"], spec["value"]),
        "describe": lambda spec: f"CLOSE{spec['op']}CLOSE[-{spec['value']}]",
    },
    {
        "type": "atr_regime",
        "identity_keys": ["period", "sma_period"],
        "has_threshold": False,
        "periods": ATR_BASE_PERIODS,
        "sma_periods": ATR_REGIME_SMA_PERIODS,
        "ops": [">", "<"],
        "build_specs": _build_specs_two_periods,
        "evaluate": lambda bank, spec: (
            bank._get_cached(
                "atr_regime", (spec["period"], spec["sma_period"]),
                lambda b: (_atr(b.high, b.low, b.close, spec["period"]),
                           _rolling_mean_skipnan(_atr(b.high, b.low, b.close, spec["period"]), spec["sma_period"]))
            )[0]
            > bank._get_cached(
                "atr_regime", (spec["period"], spec["sma_period"]),
                lambda b: (_atr(b.high, b.low, b.close, spec["period"]),
                           _rolling_mean_skipnan(_atr(b.high, b.low, b.close, spec["period"]), spec["sma_period"]))
            )[1]
            if spec["op"] == ">"
            else bank._get_cached(
                "atr_regime", (spec["period"], spec["sma_period"]),
                lambda b: (_atr(b.high, b.low, b.close, spec["period"]),
                           _rolling_mean_skipnan(_atr(b.high, b.low, b.close, spec["period"]), spec["sma_period"]))
            )[0]
            < bank._get_cached(
                "atr_regime", (spec["period"], spec["sma_period"]),
                lambda b: (_atr(b.high, b.low, b.close, spec["period"]),
                           _rolling_mean_skipnan(_atr(b.high, b.low, b.close, spec["period"]), spec["sma_period"]))
            )[1]
        ),
        "describe": lambda spec: f"ATR{spec['period']}{spec['op']}SMA_ATR{spec['sma_period']}",
    },
    {
        "type": "histvol_regime",
        "identity_keys": ["period", "sma_period"],
        "has_threshold": False,
        "periods": HISTVOL_BASE_PERIODS,
        "sma_periods": HISTVOL_REGIME_SMA_PERIODS,
        "ops": [">", "<"],
        "build_specs": _build_specs_two_periods,
        "evaluate": lambda bank, spec: (
            bank._get_cached(
                "histvol_regime", (spec["period"], spec["sma_period"]),
                lambda b: (_historical_volatility(b.close, spec["period"]),
                           _rolling_mean_skipnan(_historical_volatility(b.close, spec["period"]), spec["sma_period"]))
            )[0]
            > bank._get_cached(
                "histvol_regime", (spec["period"], spec["sma_period"]),
                lambda b: (_historical_volatility(b.close, spec["period"]),
                           _rolling_mean_skipnan(_historical_volatility(b.close, spec["period"]), spec["sma_period"]))
            )[1]
            if spec["op"] == ">"
            else bank._get_cached(
                "histvol_regime", (spec["period"], spec["sma_period"]),
                lambda b: (_historical_volatility(b.close, spec["period"]),
                           _rolling_mean_skipnan(_historical_volatility(b.close, spec["period"]), spec["sma_period"]))
            )[0]
            < bank._get_cached(
                "histvol_regime", (spec["period"], spec["sma_period"]),
                lambda b: (_historical_volatility(b.close, spec["period"]),
                           _rolling_mean_skipnan(_historical_volatility(b.close, spec["period"]), spec["sma_period"]))
            )[1]
        ),
        "describe": lambda spec: f"HISTVOL{spec['period']}{spec['op']}SMA_HISTVOL{spec['sma_period']}",
    },
]

class ConditionBank:

    _REGISTRY_BY_TYPE = {entry["type"]: entry for entry in INDICATOR_REGISTRY}

    def __init__(self, arr: dict):

        self.open   = np.array(arr["open"],  dtype=np.float64, copy=True)
        self.high   = np.array(arr["high"],  dtype=np.float64, copy=True)
        self.low    = np.array(arr["low"],   dtype=np.float64, copy=True)
        self.close  = np.array(arr["close"], dtype=np.float64, copy=True)
        self.n      = len(self.close)
        self._cache = {}

    def _get_cached(self, spec_type: str, key, compute_fn):
        cache_key = (spec_type, key)
        if cache_key not in self._cache:
            self._cache[cache_key] = compute_fn(self)
        return self._cache[cache_key]

    def _momentum_mask(self, op: str, nbar: int) -> np.ndarray:
        result = np.zeros(self.n, dtype=bool)
        if op == ">":
            result[nbar:] = self.close[nbar:] > self.close[:-nbar]
        else:
            result[nbar:] = self.close[nbar:] < self.close[:-nbar]
        return result

    def build_condition_specs(self) -> list:
        specs = []
        for entry in INDICATOR_REGISTRY:
            specs.extend(entry["build_specs"](entry))
        return specs

    def evaluate(self, spec: dict) -> np.ndarray:
        entry = self._REGISTRY_BY_TYPE.get(spec["type"])
        if entry is None:
            raise ValueError(f"Unknown condition type: {spec['type']}")
        return entry["evaluate"](self, spec)

    def describe(self, spec: dict) -> str:
        entry = self._REGISTRY_BY_TYPE.get(spec["type"])
        if entry is None:
            raise ValueError(f"Unknown condition type: {spec['type']}")
        return entry["describe"](spec)