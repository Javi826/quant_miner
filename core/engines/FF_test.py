# core/engines/FF_test.py
import os
import time
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pipeline.stepM import compute_deviation_matrix, WHITE_N_BOOTSTRAP
from pipeline.stepM import WHITE_BLOCK_SIZE, RANDOM_SEED, CROSS_SECTIONAL_PERCENTILES
logger = logging.getLogger("BOT_batch.pipeline.FF_test")

# =============================================================================
# CONFIG — every knob that affects the null is inherited from stepM.py so the
# =============================================================================
FF_N_BOOTSTRAP = WHITE_N_BOOTSTRAP
FF_BLOCK_SIZE  = WHITE_BLOCK_SIZE
FF_RANDOM_SEED = RANDOM_SEED
FF_PERCENTILES = CROSS_SECTIONAL_PERCENTILES

# =============================================================================
# MEMORY-CHUNKING CONFIG — percentile phase processes replicas in batches
# =============================================================================
BOOTSTRAP_CHUNK_SIZE = 500

# =============================================================================
# PERCENTILE THREADING CONFIG — replica rows are reduced independently, so the
# =============================================================================
FF_PERCENTILE_N_THREADS   = 0
FF_PERCENTILE_MAX_THREADS = 32


def _format_thousands(value: int) -> str:
    return format(value, ",").replace(",", ".")


# =============================================================================
# THREADED ROW-WISE PERCENTILES — one cross-section per bootstrap replica.
# =============================================================================
def _resolve_percentile_threads(n_rows: int, n_threads: int = FF_PERCENTILE_N_THREADS) -> int:
    if n_threads <= 0:
        n_threads = min(FF_PERCENTILE_MAX_THREADS, os.cpu_count() or 1)
    return max(1, min(n_threads, n_rows))


def _percentile_row_range(
    deviations_batch: np.ndarray,
    percentiles: np.ndarray,
    out: np.ndarray,
    row_start: int,
    row_end: int,
) -> int:

    n_empty = 0
    for r in range(row_start, row_end):
        row    = deviations_batch[r]
        finite = row[np.isfinite(row)]
        if finite.size == 0:
            out[r] = np.nan
            n_empty += 1
        else:
            out[r] = np.percentile(finite, percentiles, overwrite_input=True)
    return n_empty


def _percentiles_rows_threaded(
    deviations_batch: np.ndarray,
    percentiles: np.ndarray,
    n_threads: int = FF_PERCENTILE_N_THREADS,
) -> np.ndarray:

    n_rows = deviations_batch.shape[0]
    n_pct  = percentiles.shape[0]

    out_dtype = np.percentile(np.zeros(2, dtype=deviations_batch.dtype), percentiles).dtype
    out       = np.empty((n_rows, n_pct), dtype=out_dtype)

    n_threads = _resolve_percentile_threads(n_rows, n_threads)
    if n_threads == 1:
        n_empty = _percentile_row_range(deviations_batch, percentiles, out, 0, n_rows)
    else:
        bounds = np.linspace(0, n_rows, n_threads + 1).astype(np.int64)
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            futures = [
                pool.submit(_percentile_row_range, deviations_batch, percentiles, out, int(lo), int(hi))
                for lo, hi in zip(bounds[:-1], bounds[1:]) if hi > lo
            ]
            n_empty = sum(future.result() for future in futures)

    if n_empty:
        logger.warning(f"FF BOOTSTRAP ── {n_empty} all-non-finite replica row(s) encountered")

    return out


def _cross_sectional_percentiles(
    studentized_deviations: np.ndarray,
    real_percentiles: np.ndarray,
    percentiles: np.ndarray,
    chunk_size: int,
) -> tuple:

    n_bootstrap = studentized_deviations.shape[0]

    percentile_sum   = np.zeros(percentiles.shape[0], dtype=np.float64)
    below_actual_cnt = np.zeros(percentiles.shape[0], dtype=np.int64)
    n_valid_runs     = 0

    for start in range(0, n_bootstrap, chunk_size):
        end   = min(start + chunk_size, n_bootstrap)
        batch = studentized_deviations[start:end]

        batch_percentiles = _percentiles_rows_threaded(batch, percentiles)  # (n_runs, n_pct)

        percentile_sum   += batch_percentiles.sum(axis=0)
        below_actual_cnt += (batch_percentiles < real_percentiles[None, :]).sum(axis=0)
        n_valid_runs     += end - start

    avg_sim_percentiles = percentile_sum / n_valid_runs
    pct_below_actual    = 100.0 * below_actual_cnt / n_valid_runs
    return avg_sim_percentiles, pct_below_actual


# =============================================================================
# REPORT
# =============================================================================
def _log_ff_report(
    percentiles: np.ndarray,
    real_percentiles: np.ndarray,
    sim_percentiles: np.ndarray,
    pct_below_actual: np.ndarray,
    n_ge_percentile: np.ndarray,
    n_cols_built: int,
    n_dropped: int,
    n_bootstrap: int,
    block_size: int,
    timeframe: str,
) -> None:
    logger.debug(f"\n{'─' * 85}")
    logger.debug(f"  FAMA-FRENCH (2010) JOINT BLOCK BOOTSTRAP — z-stat percentiles ── {timeframe}")
    logger.debug(f"{'─' * 85}")
    logger.debug(
        f"  columns (rule × combo) : {_format_thousands(n_cols_built)}   "
        f"dropped (degenerate) : {_format_thousands(n_dropped)}   "
        f"bootstrap runs : {_format_thousands(n_bootstrap)}   "
        f"block size : {block_size}"
    )
    logger.debug(f"{'─' * 85}")
    logger.debug(f"  {'Pct':>5} │ {'N≥Pct':>9} │ {'Sim':>8} {'Real':>8} {'%<Real':>8}")
    for i, pct in enumerate(percentiles):
        logger.debug(
            f"  {pct:>5.0f} │ {_format_thousands(int(n_ge_percentile[i])):>9} │ "
            f"{sim_percentiles[i]:>8.2f} {real_percentiles[i]:>8.2f} {pct_below_actual[i]:>7.2f}%"
        )
    logger.debug(f"{'─' * 85}")
    logger.debug("  Statistic : z = Sharpe / sigma_hat — the exact StepM test statistic")
    logger.debug("  Sim       : average z at that percentile across bootstrap replicas (pure luck)")
    logger.debug("  Real      : actual z at that percentile in the real data")
    logger.debug("  %<Real    : n of bootstrap replicas that are < Real at that percentile")
    logger.debug(f"{'─' * 85}")
    logger.debug(
        "  Real far below Sim in the left tail and/or far above in the right tail signals "
        "genuine skill (positive or negative) beyond what pure luck would produce."
    )
    logger.debug(f"{'─' * 85}\n")


# =============================================================================
# PIPE FF BOOTSTRAP — orchestration. The null comes from stepM; this layer
# only turns it into a cross-sectional percentile table.
# =============================================================================
def pipe_FF_test(
    matrix_arr: np.ndarray,
    col_names: np.ndarray = None,
    n_bootstrap: int = FF_N_BOOTSTRAP,
    percentiles: np.ndarray = FF_PERCENTILES,
    chunk_size: int = BOOTSTRAP_CHUNK_SIZE,
    seed: int = FF_RANDOM_SEED,
    block_size: int = FF_BLOCK_SIZE,
    enabled: bool = True,
    timeframe: str = "",
    n_sample_replicas: int = 0,
    bootstrap_result: dict = None,
) -> dict:
    """Cross-sectional percentile diagnostic on StepM's own null.

    WARNING: `compute_deviation_matrix` compacts degenerate columns in place,
    so `matrix_arr` may be reordered and truncated by this call. Pass a copy if
    the caller needs the original layout afterwards, or reuse a single
    `bootstrap_result` across both this diagnostic and `pipe_stepm`.

    Pass `bootstrap_result` to skip the resampling entirely and reuse the dict
    returned by `compute_deviation_matrix`. That is the cheapest path when the
    StepM pipeline runs on the same matrix in the same session, and it makes
    the two views bit-identical rather than merely equivalent.
    """
    if not enabled:
        logger.info(f"FF BOOTSTRAP ── {timeframe} ── disabled, skipping")
        return None

    if bootstrap_result is None:
        if matrix_arr is None or matrix_arr.shape[1] < 2:
            logger.warning(f"FF BOOTSTRAP ── {timeframe} ── insufficient columns — skipping")
            return None
        if matrix_arr.shape[0] < 2:
            raise ValueError(
                f"FF BOOTSTRAP ── {timeframe} ── n_obs ({matrix_arr.shape[0]}) must be >= 2 "
                f"to compute a sample variance."
            )
        if block_size > matrix_arr.shape[0]:
            raise ValueError(
                f"FF BOOTSTRAP ── {timeframe} ── block_size ({block_size}) exceeds "
                f"n_obs ({matrix_arr.shape[0]}); cannot form a single block."
            )

    start = time.time()

    # ---- Phase A: the null. Delegated wholesale to stepM ------------------
    if bootstrap_result is None:
        n_cols_built = matrix_arr.shape[1]
        if col_names is None:
            col_names = np.arange(n_cols_built)

        bootstrap_result = compute_deviation_matrix(
            matrix_arr, list(col_names), n_bootstrap=n_bootstrap,
            block_size=block_size, seed=seed, progress_label=timeframe,
        )
    else:
        logger.info(f"FF BOOTSTRAP ── {timeframe} ── reusing precomputed StepM null")
        n_cols_built = int(bootstrap_result["kept_columns"].shape[0])

    studentized_deviations = bootstrap_result["studentized_deviations"]
    z_stat                 = bootstrap_result["z_stat"]
    real_sharpe            = bootstrap_result["real_sharpe"]
    sigma_hat              = bootstrap_result["sigma_hat"]
    kept_columns            = bootstrap_result["kept_columns"]

    n_kept     = int(kept_columns.shape[0])
    n_dropped  = n_cols_built - n_kept
    n_replicas = int(studentized_deviations.shape[0])

    logger.info(
        f"FF BOOTSTRAP ── {timeframe} ── {_format_thousands(n_dropped)} degenerate "
        f"columns dropped ── {_format_thousands(n_kept)} columns remain"
    )

    # ---- Sample of raw null replicas, kept only for plotting purposes ------
    sim_z_sample = studentized_deviations[:n_sample_replicas].copy() if n_sample_replicas > 0 else None

    # ---- Phase B: real cross-section --------------------------------------
    real_percentiles = np.percentile(z_stat, percentiles)

    sorted_z_asc    = np.sort(z_stat)
    n_ge_percentile = sorted_z_asc.shape[0] - np.searchsorted(sorted_z_asc, real_percentiles, side="left")

    # ---- Phase C: null cross-sections, replica-chunked ---------------------
    sim_percentiles, pct_below_actual = _cross_sectional_percentiles(
        studentized_deviations, real_percentiles, percentiles, chunk_size,
    )

    _log_ff_report(
        percentiles, real_percentiles, sim_percentiles, pct_below_actual, n_ge_percentile,
        n_cols_built, n_dropped, n_replicas, block_size, timeframe,
    )

    result = {
        "percentiles":      percentiles,
        "real_percentiles": real_percentiles,
        "sim_percentiles":  sim_percentiles,
        "pct_below_actual": pct_below_actual,
        "real_z_stat":      z_stat,
        "real_sharpe":      real_sharpe,
        "sigma_hat":        sigma_hat,
        "sim_z_sample":     sim_z_sample,
        "kept_columns":     kept_columns,
        "n_cols_built":     n_cols_built,
        "n_dropped":        n_dropped,
        "n_bootstrap":      n_replicas,
        "block_size":       block_size,
        "timeframe":        timeframe,
        "bootstrap_result": bootstrap_result,
    }

    elapsed = int(time.time() - start)
    #logger.debug(f"FF BOOTSTRAP ── {timeframe} ── elapsed {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")

    return result
