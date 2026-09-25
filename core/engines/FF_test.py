# core/engines/FF_test.py
import time
import logging
import numpy as np
from pipeline.stepM_is import compute_bootstrap_null, WHITE_N_BOOTSTRAP
from pipeline.stepM_is import WHITE_BLOCK_SIZE, RANDOM_SEED, CROSS_SECTIONAL_PERCENTILES
logger = logging.getLogger("BOT_batch.pipeline.FF_test")

# =============================================================================
# CONFIG — every knob that affects the null is inherited from stepM.py so the
# =============================================================================
FF_N_BOOTSTRAP    = WHITE_N_BOOTSTRAP
FF_BLOCK_SIZE     = WHITE_BLOCK_SIZE
FF_RANDOM_SEED    = RANDOM_SEED
FF_MIN_PERCENTILE = 90      # lowest percentile of the table: the top-M kept per replica is ~(100 - it)% of the columns
FF_PERCENTILES    = CROSS_SECTIONAL_PERCENTILES[CROSS_SECTIONAL_PERCENTILES >= FF_MIN_PERCENTILE]


def _format_thousands(value: int) -> str:
    return format(value, ",").replace(",", ".")


def _prefix(timeframe: str) -> str:
    return f"{'FF BOOTSTRAP':<15}{timeframe}"

# =============================================================================
# PERCENTILES FROM THE TOP-M: stepM v3 keeps, per replica, only the M largest
# studentized deviations (descending). A percentile p of the full cross-section
# only needs the order statistics above it, so it is read exactly from the top-M
# as long as M covers (100 - p)% of the columns. Same linear interpolation as
# np.percentile on the full row.
# =============================================================================
def _percentile_ranks(pct: float, n_cols: int) -> tuple:
    """Descending top-M ranks of the two order statistics np.percentile interpolates, and the weight."""
    idx          = (pct / 100.0) * (n_cols - 1.0)
    idx_below    = int(np.floor(idx))
    weight_above = idx - idx_below
    rank_bottom  = n_cols - 1 - idx_below          # ascending order statistic -> descending top-M rank
    rank_top     = max(rank_bottom - 1, 0)         # pct=100: weight_above is 0, the upper operand is unused
    return rank_bottom, rank_top, weight_above


def _topm_size_for(percentiles: np.ndarray, n_cols: int) -> int:
    rank_bottom, _, _ = _percentile_ranks(float(np.min(percentiles)), n_cols)
    return rank_bottom + 1


def _null_percentiles(topm_vals: np.ndarray, n_cols: int, percentiles: np.ndarray) -> np.ndarray:
    """(n_bootstrap, n_pct): np.percentile(full_row, percentiles) for every replica, from the top-M."""
    out = np.empty((topm_vals.shape[0], percentiles.shape[0]), dtype=np.float64)
    for i, pct in enumerate(percentiles):
        rank_bottom, rank_top, weight_above = _percentile_ranks(float(pct), n_cols)
        a    = topm_vals[:, rank_bottom].astype(np.float64)
        b    = topm_vals[:, rank_top].astype(np.float64)
        diff = b - a
        out[:, i] = np.where(weight_above >= 0.5, b - diff * (1.0 - weight_above), a + diff * weight_above)
    return out

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
    seed: int = FF_RANDOM_SEED,
    block_size: int = FF_BLOCK_SIZE,
    enabled: bool = True,
    timeframe: str = "",
) -> dict:
    """Cross-sectional percentile diagnostic on StepM's own null (same seed, block and replicas).

    WARNING: `compute_bootstrap_null` compacts non-finite Sharpe columns in place,
    so `matrix_arr` may be reordered and truncated by this call. Pass a copy if
    the caller needs the original layout afterwards.

    Cost: the top-M kept per replica covers (100 - min(percentiles))% of the columns,
    so lowering the grid below FF_MIN_PERCENTILE grows RAM and VRAM accordingly.
    """
    if not enabled:
        logger.info(f"FF BOOTSTRAP ── {timeframe} ── disabled, skipping")
        return None

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

    start       = time.time()
    percentiles = np.asarray(percentiles, dtype=np.float64)

    # ---- Phase A: the null. Delegated wholesale to stepM ------------------
    n_cols_built = matrix_arr.shape[1]
    if col_names is None:
        col_names = np.arange(n_cols_built)

    null = compute_bootstrap_null(
        matrix_arr, list(col_names), n_bootstrap=n_bootstrap, block_size=block_size, seed=seed,
        topm_size=_topm_size_for(percentiles, n_cols_built),
        progress_label=f"FF {timeframe}", desc=f"{'FF BST':<15} {timeframe}",
    )

    n_kept     = null.n_kept
    n_dropped  = n_cols_built - n_kept
    n_replicas = null.n_bootstrap

    logger.info(
        f"{_prefix(timeframe)}: {_format_thousands(n_dropped)} degenerate "
        f"columns dropped ── {_format_thousands(n_kept)} columns remain"
    )
    if n_kept < 2:
        logger.warning(f"{_prefix(timeframe)}: fewer than 2 columns with a valid bootstrap sigma ── skipping")
        return None

    null.ensure_topm(_topm_size_for(percentiles, n_kept))   # no-op: the first pass already holds enough ranks

    # ---- Phase B: real cross-section --------------------------------------
    z_stat           = null.z_stat
    real_percentiles = np.percentile(z_stat, percentiles)

    sorted_z_asc    = np.sort(z_stat)
    n_ge_percentile = sorted_z_asc.shape[0] - np.searchsorted(sorted_z_asc, real_percentiles, side="left")

    # ---- Phase C: null cross-sections, read from the top-M ------------------
    null_pct         = _null_percentiles(null.topm_vals, n_kept, percentiles)   # (n_replicas, n_pct)
    sim_percentiles  = null_pct.mean(axis=0)
    pct_below_actual = 100.0 * (null_pct < real_percentiles[None, :]).mean(axis=0)

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
        "real_sharpe":      null.real_sharpe,
        "sigma_hat":        null.sigma_hat,
        "kept_columns":     null.kept_columns,
        "n_cols_built":     n_cols_built,
        "n_dropped":        n_dropped,
        "n_bootstrap":      n_replicas,
        "block_size":       block_size,
        "timeframe":        timeframe,
    }

    elapsed = int(time.time() - start)
    #logger.debug(f"FF BOOTSTRAP ── {timeframe} ── elapsed {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")

    return result