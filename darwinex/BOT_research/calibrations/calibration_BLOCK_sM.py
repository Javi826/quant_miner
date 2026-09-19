#research/calibrations/calibration_BLOCK_sM.py

import os
import sys
import time
import logging
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "core")))

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger("BOT_batch.main_BLOCK")
logger.setLevel(logging.INFO)
logging.getLogger("BOT_batch.pipeline.backtest_runner").setLevel(logging.INFO)
logging.getLogger("BOT_batch.pipeline.stepM").setLevel(logging.WARNING)
logging.getLogger("joblib").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

from symbols.universe import build_universe
from setup.config_paths import DATA_FOLDER_BY_DATASET
from rule_mining.rule_generator import MAX_DEPTH as RULE_MAX_DEPTH
from rule_mining.rule_runner import _build_rule_dicts
from utils.ohlcv_utils import prepare_ohlcv_arrays
from setup.config_backtest import ORDER_AMOUNT
from pipeline import backtest_runner as backtest_module
from pipeline.signal_cleaning import pipe_signal_cleaning_jaccard

# =============================================================================
# UNIVERSE / SEARCH SPACE CONFIGURATION — mirror main_COMP.py
# =============================================================================
DATASET = "IS"   # "IS", "OOS" or "MERGED"
N_JOBS  = -1

TIMEFRAMES = ["1H", "4H"]

SYMBOLS_BY_TIMEFRAME = {
    "1H":     ["USDCAD","CHFJPY"],
    "4H":     ["USDJPY","CHFJPY"],
}

PARAM_GRID = {
    "SELL_AFTER": [50],
    "TP_PCT":     [0.35,0.5,0.7],
    "SL_PCT":     [0.35,0.5,0.7],
}

# =============================================================================
# BLOCK-SIZE ANALYSIS CONFIGURATION
# =============================================================================
BLOCK_SIZE_GRID = [10,20,50]

RANDOM_SEED = 42

DIAG_MAX_COLUMNS = 500_000    # columns used by the block-length diagnostic

MIN_ACTIVE_DAYS_FRACTION = 0.05

# M3 — Politis & White automatic selection
PW_C_SIGNIF    = 2.0     # multiplier of the sqrt(log10(n)/n) significance band
PW_QUANTILES   = [50, 75, 90, 95]
PW_RECOMMEND_Q = 90      

ACF_CHUNK_COLUMNS = 250         # used by _autocovariance_by_column

# =============================================================================
# SHARED HELPERS
# =============================================================================
def _subsample_columns(matrix_arr: np.ndarray, max_columns: int, seed: int) -> tuple:
    """Fixed-seed column subsample so every method looks at a comparable slice."""
    n_cols = matrix_arr.shape[1]
    if n_cols <= max_columns:
        return matrix_arr, np.arange(n_cols)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n_cols, size=max_columns, replace=False))
    return np.ascontiguousarray(matrix_arr[:, idx]), idx


def _filter_active_columns(matrix_arr: np.ndarray, min_fraction: float) -> tuple:

    n_obs = matrix_arr.shape[0]
    active_days = np.count_nonzero(matrix_arr, axis=0)
    active_fraction = active_days / n_obs
    keep_mask = active_fraction >= min_fraction
    kept_idx = np.flatnonzero(keep_mask)
    return np.ascontiguousarray(matrix_arr[:, kept_idx]), kept_idx, active_fraction


def _log_sparsity_diagnostics(matrix_arr: np.ndarray, timeframe: str) -> None:
    """Report how much of the matrix is zero-P&L days, before any filtering."""
    n_obs, n_cols = matrix_arr.shape
    active_days = np.count_nonzero(matrix_arr, axis=0)
    active_fraction = active_days / n_obs
    overall_zero_fraction = 1.0 - (active_days.sum() / (n_obs * n_cols))

    _header(f"SPARSITY DIAGNOSTIC ── {timeframe}")
    logger.info(f"  n_obs={n_obs}  n_cols={n_cols}")
    logger.info(f"  overall zero-day fraction : {overall_zero_fraction:.4f}")
    logger.info(
        f"  per-column active fraction : "
        f"p10={np.percentile(active_fraction, 10):.4f}  "
        f"p50={np.percentile(active_fraction, 50):.4f}  "
        f"p90={np.percentile(active_fraction, 90):.4f}"
    )
    logger.info(
        f"  columns below MIN_ACTIVE_DAYS_FRACTION ({MIN_ACTIVE_DAYS_FRACTION}) : "
        f"{int(np.sum(active_fraction < MIN_ACTIVE_DAYS_FRACTION))}/{n_cols}"
    )


def _sharpe_influence_function(values: np.ndarray) -> tuple:

    x64 = values.astype(np.float64, copy=False)
    mu  = x64.mean(axis=0)
    sd  = x64.std(axis=0, ddof=1)

    valid = sd > 0
    centered = x64 - mu[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        standardized = centered / sd[None, :]
        sharpe       = mu / sd
    psi = standardized - 0.5 * sharpe[None, :] * (standardized * standardized - 1.0)
    return psi, valid


def _autocovariance_by_column(values: np.ndarray, max_lag: int, chunk_columns: int = ACF_CHUNK_COLUMNS) -> np.ndarray:
    """FFT autocovariance, biased estimator (divide by n), lags 0..max_lag."""
    n_obs, n_cols = values.shape
    max_lag = min(max_lag, n_obs - 1)
    fft_len = 1 << int(np.ceil(np.log2(2 * n_obs)))

    acov = np.empty((max_lag + 1, n_cols), dtype=np.float64)
    for start in range(0, n_cols, chunk_columns):
        end   = min(start + chunk_columns, n_cols)
        chunk = values[:, start:end].astype(np.float64, copy=True)
        chunk -= chunk.mean(axis=0)[None, :]
        spectrum = np.fft.rfft(chunk, n=fft_len, axis=0)
        acf_full = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_len, axis=0)
        acov[:, start:end] = acf_full[: max_lag + 1] / n_obs
    return acov


def _flat_top_lag_window(lags: np.ndarray, bandwidth: np.ndarray) -> np.ndarray:
    """Politis & White flat-top window: 1 on [0,1/2], 2(1-s) on [1/2,1], 0 beyond."""
    with np.errstate(divide="ignore", invalid="ignore"):
        s = lags[:, None] / bandwidth[None, :]
    weights = np.where(s <= 0.5, 1.0, np.where(s <= 1.0, 2.0 * (1.0 - s), 0.0))
    return np.where(np.isfinite(weights), weights, 0.0)


def _header(title: str) -> None:
    logger.info(f"\n{'=' * 78}")
    logger.info(f"  {title}")
    logger.info(f"{'=' * 78}")


# =============================================================================
# POLITIS & WHITE (2004) AUTOMATIC BLOCK-LENGTH SELECTION
# =============================================================================
def method_politis_white(matrix_arr: np.ndarray, timeframe: str) -> dict:

    _header(f"POLITIS & WHITE AUTOMATIC SELECTION ── {timeframe}")

    psi, valid = _sharpe_influence_function(matrix_arr)
    psi = np.ascontiguousarray(psi[:, valid])
    n_obs, n_cols = psi.shape

    k_n     = int(max(5, np.ceil(np.sqrt(np.log10(n_obs)))))
    m_max   = int(np.ceil(np.sqrt(n_obs)) + k_n)
    b_max   = int(np.ceil(min(3.0 * np.sqrt(n_obs), n_obs / 3.0)))
    lag_max = min(m_max + k_n, n_obs - 1)

    acov = _autocovariance_by_column(psi, max_lag=lag_max)
    with np.errstate(divide="ignore", invalid="ignore"):
        acf = acov / acov[0][None, :]
    acf = np.nan_to_num(acf, nan=0.0)

    band       = PW_C_SIGNIF * np.sqrt(np.log10(n_obs) / n_obs)
    signif     = (np.abs(acf[1:]) >= band).astype(np.int32)   # signif[i] <-> lag (i+1)
    cumulative = np.vstack([np.zeros((1, n_cols), dtype=np.int32), np.cumsum(signif, axis=0)])

    n_windows = signif.shape[0] - k_n + 1
    m_hat     = np.full(n_cols, lag_max, dtype=np.int64)
    resolved  = np.zeros(n_cols, dtype=bool)
    for window_start in range(max(n_windows, 0)):
        clean = (cumulative[window_start + k_n] - cumulative[window_start]) == 0
        newly = clean & (~resolved)
        if newly.any():

            m_hat[newly] = window_start + 1
            resolved |= newly
        if resolved.all():
            break

    bandwidth = np.clip(2 * m_hat, 1, m_max).astype(np.float64)
    lags      = np.arange(1, lag_max + 1, dtype=np.float64)
    weights   = _flat_top_lag_window(lags, bandwidth)

    g_hat = acov[0] + 2.0 * (weights * acov[1:]).sum(axis=0)
    g_big = 2.0 * (weights * lags[:, None] * acov[1:]).sum(axis=0)

    # D_CB = (4/3) g_hat(0)^2 — the MBB/circular-bootstrap constant. Unlike
    # D_SB, this is NOT affected by the Patton/Politis/White (2009) correction.
    usable = (g_hat > 0) & np.isfinite(g_big)
    d_mbb  = (4.0 / 3.0) * g_hat[usable] ** 2
    b_opt  = np.cbrt(2.0 * g_big[usable] ** 2 / d_mbb) * np.cbrt(n_obs)
    b_opt  = np.clip(b_opt, 1.0, b_max)

    logger.info(f"  n_obs={n_obs}  n_cols={n_cols}  K_N={k_n}  M_max={m_max}  b_max={b_max}")
    logger.info(f"  usable columns        : {int(usable.sum())}/{n_cols}  (positive spectral density at zero)")
    logger.info(f"  bandwidth M           : p50={np.percentile(bandwidth, 50):.0f}  p90={np.percentile(bandwidth, 90):.0f}")
    logger.info(f"\n  {'QUANTILE':<12}{'b_opt':<10}")
    logger.info(f"  {'-' * 22}")
    quantile_values = {}
    for q in PW_QUANTILES:
        value = float(np.percentile(b_opt, q))
        quantile_values[q] = value
        logger.info(f"  p{q:<11}{value:<10.2f}")
    logger.info(f"  mean                 : {b_opt.mean():.2f}    max: {b_opt.max():.2f}")

    b_recommended = int(np.ceil(quantile_values[PW_RECOMMEND_Q]))
    logger.info(f"\n  --> b_politis_white (p{PW_RECOMMEND_Q}) : {b_recommended}")
    return {"b_recommended": b_recommended, "quantiles": quantile_values, "b_opt": b_opt}


# =============================================================================
# RESULT REPORT
# =============================================================================
def print_result(result: dict, timeframe: str) -> None:
    _header(f"RESULT ── {timeframe}")

    final_choice = result["b_recommended"]

    logger.info(f"  {'METHOD':<34}{'b':<10}")
    logger.info(f"  {'-' * 44}")
    logger.info(f"  {f'Politis-White (p{PW_RECOMMEND_Q})':<34}{final_choice:<10}")

    candidate = min([b for b in BLOCK_SIZE_GRID if b >= final_choice], default=None)
    if candidate is None:
        candidate = max(BLOCK_SIZE_GRID)
        logger.warning(
            f"  final_choice ({final_choice}) exceeds the largest grid candidate "
            f"({max(BLOCK_SIZE_GRID)}) — clamping down; consider widening BLOCK_SIZE_GRID."
        )

    logger.info(f"\n  ==> WHITE_BLOCK_SIZE = {candidate}")
    logger.info(f"{'=' * 78}\n")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    start = time.time()

    logger.info(f"\n{'=' * 78}")
    logger.info(f"  WHITE_BLOCK_SIZE ESTIMATION")
    logger.info(f"{'=' * 78}")
    logger.info(f"  DATASET           : {DATASET} ── {os.path.basename(DATA_FOLDER_BY_DATASET[DATASET])}")
    logger.info(f"  TIMEFRAMES        : {TIMEFRAMES}")
    logger.info(f"  SYMBOLS_BY_TF     : {{tf: len(s) for tf, s in SYMBOLS_BY_TIMEFRAME.items()}}")
    logger.info(f"  PARAM_GRID        : {PARAM_GRID}")
    logger.info(f"  BLOCK_SIZE_GRID   : {BLOCK_SIZE_GRID}")
    logger.info(f"{'=' * 78}")

    ohlcv_data_by_timeframe = build_universe(
        DATA_FOLDER_BY_DATASET[DATASET], SYMBOLS_BY_TIMEFRAME,
        dataset=DATASET,
    )
    for timeframe in TIMEFRAMES:
        ohlcv_is  = ohlcv_data_by_timeframe[timeframe]
        ohlcv_arr = prepare_ohlcv_arrays(ohlcv_is)
        rules = _build_rule_dicts(ohlcv_is, timeframe, timeframe, RULE_MAX_DEPTH)

        rules = pipe_signal_cleaning_jaccard(
            rules     = rules,
            ohlcv_arr = ohlcv_arr,
            timeframe = timeframe,
        )

        original_n_jobs = backtest_module.BACKTEST_N_JOBS
        backtest_module.BACKTEST_N_JOBS = N_JOBS
        try:
            _raw_results, _n_combos, matrix_arr, col_names = backtest_module.pipe_backtesting(
                rules        = rules,
                ohlcv_arr    = ohlcv_arr,
                param_grid   = PARAM_GRID,
                order_amount = ORDER_AMOUNT,
                timeframe    = timeframe,
            )
        finally:
            backtest_module.BACKTEST_N_JOBS = original_n_jobs

        if matrix_arr is None or matrix_arr.shape[1] < 2:
            logger.warning(f"  {timeframe} — matrix has fewer than 2 usable columns, skipping")
            continue

        logger.info(f"\n  MATRIX ── {timeframe} ── {matrix_arr.shape[0]} days x {matrix_arr.shape[1]} columns")

        _log_sparsity_diagnostics(matrix_arr, timeframe)

        active_matrix, active_idx, _ = _filter_active_columns(matrix_arr, MIN_ACTIVE_DAYS_FRACTION)
        logger.info(
            f"  ACTIVITY FILTER ── {timeframe} ── kept {active_matrix.shape[1]}/{matrix_arr.shape[1]} "
            f"columns (>= {MIN_ACTIVE_DAYS_FRACTION:.0%} active days)"
        )

        if active_matrix.shape[1] < 2:
            logger.warning(f"  {timeframe} — fewer than 2 columns survive the activity filter, skipping")
            continue

        diag_matrix, _ = _subsample_columns(active_matrix, DIAG_MAX_COLUMNS, RANDOM_SEED)

        result = method_politis_white(diag_matrix, timeframe)
        print_result(result, timeframe)

    elapsed = int(time.time() - start)
    logger.info(f"🏁 TOTAL — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")