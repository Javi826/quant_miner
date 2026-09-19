#research/calibrations/calibration_BLOCK_mv.py
"""
Block-size candidate for multiverse.py's MCPT bootstrap — M3
(Politis & White 2004) applied to log_ret_close squared
(volatility clustering proxy), per symbol, across multiple timeframes.
"""
import os
import sys
import logging
import numpy as np
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "core")))

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger("BOT_batch.multiverse_block_size")

from symbols.universe import build_universe
from setup.config_paths import DATA_FOLDER_BY_DATASET
# =============================================================================
# CONFIGURATION
# =============================================================================
DATASET    = "MERGED"   # "IS", "OOS" or "MERGED"
TIMEFRAMES = ["1H", "4H"]

SYMBOLS_BY_TIMEFRAME = {
    "1H":     ["ADAUSDT","AVAXUSDT","BCHUSDT","BNBUSDT","DOGEUSDT","LINKUSDT","NEARUSDT","SOLUSDT","UNIUSDT","XRPUSDT"],
    "4H":     ["ADAUSDT","AVAXUSDT","BCHUSDT","BNBUSDT","DOGEUSDT","LINKUSDT","NEARUSDT","SOLUSDT","UNIUSDT","XRPUSDT"],
    "6Hutc":  ["ADAUSDT","AVAXUSDT","BCHUSDT","BNBUSDT","DOGEUSDT","LINKUSDT","NEARUSDT","SOLUSDT","UNIUSDT","XRPUSDT"],
    "12Hutc": ["ADAUSDT","AVAXUSDT","BCHUSDT","BNBUSDT","DOGEUSDT","LINKUSDT","NEARUSDT","SOLUSDT","UNIUSDT","XRPUSDT"],
}

PW_C_SIGNIF = 2.0  # multiplier of the sqrt(log10(n)/n) significance band

TIMEFRAME_HOURS = {
    "1H": 1,
    "4H": 4,
    "6Hutc": 6,
    "12Hutc": 12,
}
# =============================================================================
# AUTOCOVARIANCE / AUTOCORRELATION — single series, FFT-based, biased (n)
# =============================================================================
def _autocovariance_1d(series: np.ndarray, max_lag: int) -> np.ndarray:
    n_obs = series.shape[0]
    max_lag = min(max_lag, n_obs - 1)
    centered = series - series.mean()

    fft_len = 1 << int(np.ceil(np.log2(2 * n_obs)))
    spectrum = np.fft.rfft(centered, n=fft_len)
    acov_full = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_len)
    return acov_full[: max_lag + 1] / n_obs


def _acf_from_acov(acov: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        acf = acov / acov[0]
    return np.nan_to_num(acf, nan=0.0)

# =============================================================================
# M3 — POLITIS & WHITE (2004) AUTOMATIC BLOCK-LENGTH SELECTION
# =============================================================================
def _flat_top_window(lags: np.ndarray, bandwidth: float) -> np.ndarray:
    s = lags / bandwidth
    return np.where(s <= 0.5, 1.0, np.where(s <= 1.0, 2.0 * (1.0 - s), 0.0))


def method_politis_white(series: np.ndarray) -> float:
    n_obs = series.shape[0]

    k_n = int(max(5, np.ceil(np.sqrt(np.log10(n_obs)))))
    m_max = int(np.ceil(np.sqrt(n_obs)) + k_n)
    b_max = int(np.ceil(min(3.0 * np.sqrt(n_obs), n_obs / 3.0)))
    lag_max = min(m_max + k_n, n_obs - 1)

    acov = _autocovariance_1d(series, lag_max)
    acf = _acf_from_acov(acov)
    band = PW_C_SIGNIF * np.sqrt(np.log10(n_obs) / n_obs)

    signif = np.abs(acf[1:]) >= band
    m_hat = lag_max
    for window_start in range(len(signif) - k_n + 1):
        if not signif[window_start: window_start + k_n].any():
            m_hat = window_start
            break

    bandwidth = float(np.clip(2 * m_hat, 1, m_max))
    lags = np.arange(1, lag_max + 1, dtype=np.float64)
    weights = _flat_top_window(lags, bandwidth)

    g_hat = acov[0] + 2.0 * np.sum(weights * acov[1:])
    g_big = 2.0 * np.sum(weights * lags * acov[1:])

    if g_hat <= 0 or not np.isfinite(g_big):
        return float("nan")

    d_mbb = (4.0 / 3.0) * g_hat ** 2
    b_opt = np.cbrt(2.0 * g_big ** 2 / d_mbb) * np.cbrt(n_obs)
    return float(np.clip(b_opt, 1.0, b_max))

# =============================================================================
# PER-TIMEFRAME DIAGNOSTIC
# =============================================================================
def compute_block_size_range(ohlcv_is: dict) -> dict:
    b_opt_values = []
    for symbol, df in ohlcv_is.items():
        close = df["close"].to_numpy(dtype=np.float64)
        log_ret = np.diff(np.log(close))
        log_ret_sq = log_ret ** 2

        b_opt = method_politis_white(log_ret_sq)
        if np.isfinite(b_opt):
            b_opt_values.append(b_opt)

    if not b_opt_values:
        return {"p50_candles": float("nan"), "p90_candles": float("nan")}

    return {
        "p50_candles": float(np.percentile(b_opt_values, 50)),
        "p90_candles": float(np.percentile(b_opt_values, 90)),
    }


def candles_to_days(n_candles: float, timeframe_hours: int) -> float:
    return n_candles * timeframe_hours / 24.0

# =============================================================================
# MAIN — block-size range per timeframe
# =============================================================================
if __name__ == "__main__":
    logger.info(f"DATASET: {DATASET} ── {os.path.basename(DATA_FOLDER_BY_DATASET[DATASET])}")
    ohlcv_data_by_timeframe = build_universe(
        DATA_FOLDER_BY_DATASET[DATASET], SYMBOLS_BY_TIMEFRAME,
        dataset=DATASET,
    )
    results = []
    for timeframe in TIMEFRAMES:
        block_range = compute_block_size_range(ohlcv_data_by_timeframe[timeframe])
        tf_hours = TIMEFRAME_HOURS[timeframe]

        p50_days = candles_to_days(block_range["p50_candles"], tf_hours)
        p90_days = candles_to_days(block_range["p90_candles"], tf_hours)

        results.append((timeframe, block_range["p50_candles"], block_range["p90_candles"], p50_days, p90_days))

    logger.info(f"\n{'TIMEFRAME':<12}{'CANDLES (p50-p90)':<22}{'DAYS (p50-p90)':<20}")
    logger.info("-" * 54)
    for timeframe, p50_c, p90_c, p50_d, p90_d in results:
        candles_str = f"{p50_c:.0f} - {p90_c:.0f}"
        days_str = f"{p50_d:.1f} - {p90_d:.1f}"
        logger.info(f"{timeframe:<12}{candles_str:<22}{days_str:<20}")