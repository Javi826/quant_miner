# core/pipeline/stepM.py
import logging
import numpy as np
from tqdm import tqdm
from setup.config_core import settings
from utils.paralelization import compact_columns_inplace
from utils.reporting import print_stepm_matrix_debug, print_stepm_real_variance_filter_debug, print_stepm_block_starts_debug
from utils.reporting import print_stepm_bootstrap_replicas_debug, print_stepm_se_filter_debug, print_stepm_studentization_debug
from utils.reporting import print_stepm_pvalue_quantile_equivalence_debug, print_stepm_monotonicity_debug, print_stepm_brc_equivalence_debug
logger = logging.getLogger("BOT_batch.pipeline.stepM")

# =============================================================================
# STATISTICAL TEST CONFIG + STEPDOWN / K-FWE CONFIG -ROmano Wolf
# =============================================================================
STEPM_ALPHA         = 0.1              # significance level used inside the Romano-Wolf stepdown search

STEPM_K_MODE        = "kesime"            # "kmaxime", "kesime" or "fdp"
STEPM_K_FWE         = 1
STEPM_K_ESIME_TF    = {
    "1H":     0.01,
    "4H":     0.01,
    "6Hutc":  0.01,
    "12Hutc": 0.01,
}
FDP_GAMMA          = 0.10           # max tolerated false discovery proportion (Romano-Wolf 2007, Algorithm 4.1)
# =============================================================================
# STATISTICAL TEST 
# =============================================================================
FDP_K_MAX          = 8192           # upper bound of the geometric k sweep: 1, 2, 4, ... FDP_K_MAX
CROSS_SECTIONAL_PERCENTILES = np.array([50, 90, 95, 96, 97, 98, 99, 99.9, 99.99, 100])
WHITE_PVALUE_TH      = STEPM_ALPHA
WHITE_N_BOOTSTRAP    = 1000
WHITE_BLOCK_SIZE     = 10            # fixed block length — mirrors montecarlo.py BLOCK_SIZE
STEPM_USE_SPA        = True
STEPM_MAX_ITERATIONS = 500           # safety cap on stepdown iterations
# =============================================================================
# MEMORY-CHUNKING CONFIG — bounds peak RAM without changing any result
# =============================================================================
COLUMN_CHUNK_SIZE    = 5000     # columns processed per chunk for chunked reductions/compaction over dense matrices
PARTITION_ROW_CHUNK  = 50       # bootstrap replicas processed per np.partition call in the stepdown
BOOTSTRAP_CHUNK_SIZE = 8192     # columns processed per GEMM call in the bootstrap moment computation
RANDOM_SEED          = 42

# =============================================================================
# CHUNKED REDUCTIONS — column-wise mean/std without materializing a full-size
# =============================================================================
def _mean_std_by_column_chunks(arr: np.ndarray, ddof: int = 0, chunk_size: int = COLUMN_CHUNK_SIZE):
    n_cols = arr.shape[1]
    means = np.empty(n_cols, dtype=np.float64)
    stds  = np.empty(n_cols, dtype=np.float64)
    for start in range(0, n_cols, chunk_size):
        end = min(start + chunk_size, n_cols)
        chunk = arr[:, start:end]
        means[start:end] = chunk.mean(axis=0, dtype=np.float64)
        stds[start:end]  = chunk.std(axis=0, ddof=ddof, dtype=np.float64)
    return means, stds


def _std_by_column_chunks(arr: np.ndarray, ddof: int = 0, chunk_size: int = COLUMN_CHUNK_SIZE) -> np.ndarray:
    n_cols = arr.shape[1]
    stds = np.empty(n_cols, dtype=np.float64)
    for start in range(0, n_cols, chunk_size):
        end = min(start + chunk_size, n_cols)
        stds[start:end] = arr[:, start:end].std(axis=0, ddof=ddof, dtype=np.float64)
    return stds

# =============================================================================
# STATISTIC — annualized Sharpe per trial column, vectorized over bootstrap replicas
# =============================================================================
def _sharpe_per_column(matrix_arr: np.ndarray) -> np.ndarray:

    means, stds = _mean_std_by_column_chunks(matrix_arr, ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = (means / stds) * np.sqrt(settings.DAYS_PER_YEAR)

    return np.where(stds > 0, sharpe, -np.inf)

# =============================================================================
# MOVING BLOCK BOOTSTRAP — WEIGHT-MATRIX (GEMM) FORMULATION
# =============================================================================
def _generate_block_starts(n_obs: int, block_size: int, n_replicas: int, rng: np.random.Generator):

    n_blocks_needed = int(np.ceil(n_obs / block_size))
    n_block_starts  = n_obs - block_size + 1

    starts = rng.integers(0, n_block_starts, size=(n_replicas, n_blocks_needed), dtype=np.int32)

    len_last = n_obs - (n_blocks_needed - 1) * block_size
    starts_full = starts[:, :-1] if n_blocks_needed > 1 else starts[:, :0]
    starts_last = starts[:, -1]

    return starts_full, starts_last, len_last, n_blocks_needed

def _build_bootstrap_weight_matrix(
    starts_full: np.ndarray,
    starts_last: np.ndarray,
    block_size: int,
    len_last: int,
    n_obs: int,
    n_replicas: int,
) -> np.ndarray:

    diff = np.zeros((n_replicas, n_obs + 1), dtype=np.int32)
    row_idx = np.arange(n_replicas)

    n_blocks_full = starts_full.shape[1]
    if n_blocks_full > 0:
        rows_full   = np.repeat(row_idx, n_blocks_full)
        starts_flat = starts_full.ravel()
        ends_flat   = starts_flat + block_size
        np.add.at(diff, (rows_full, starts_flat), 1)
        np.add.at(diff, (rows_full, ends_flat), -1)

    ends_last = starts_last + len_last
    np.add.at(diff, (row_idx, starts_last), 1)
    np.add.at(diff, (row_idx, ends_last), -1)

    weights = np.cumsum(diff[:, :n_obs], axis=1)
    return weights.astype(np.float32)

def _bootstrap_moments_chunk(
    weight_matrix: np.ndarray,
    batch_values: np.ndarray,
    real_sharpe_batch: np.ndarray,
    n_obs: int,
) -> tuple:

    total_sum   = weight_matrix @ batch_values
    total_sumsq = weight_matrix @ (batch_values * batch_values)

    means = total_sum / n_obs
    var   = (total_sumsq - n_obs * means * means) / (n_obs - 1)
    np.maximum(var, 0.0, out=var)  # guard tiny negative fp error before sqrt
    stds = np.sqrt(var)

    with np.errstate(divide="ignore", invalid="ignore"):
        boot_sharpe = (means / stds) * np.sqrt(settings.DAYS_PER_YEAR)
    boot_sharpe = np.where(stds > 0, boot_sharpe, -np.inf)

    deviations_chunk = boot_sharpe - real_sharpe_batch[None, :]
    sigma_chunk       = deviations_chunk.std(axis=0, ddof=1)

    return deviations_chunk, sigma_chunk

def apply_spa_recentering(studentized_deviations: np.ndarray, z_stat: np.ndarray, n_obs: int) -> tuple:
    """Hansen (2005) SPA_c: recenter clearly-losing columns (z_stat below a
    slowly-growing threshold) to 0 instead of their own negative mean, before
    computing the bootstrap null. 
    """
    threshold = -np.sqrt(2.0 * np.log(np.log(max(n_obs, 3))))
    bad_mask  = z_stat < threshold
    if bad_mask.any():
        studentized_deviations[:, bad_mask] += z_stat[bad_mask][None, :]
    return studentized_deviations, bad_mask, threshold

def compute_deviation_matrix(
    matrix_arr: np.ndarray,
    col_names: list,
    n_bootstrap: int = WHITE_N_BOOTSTRAP,
    block_size: int = WHITE_BLOCK_SIZE,
    seed: int = RANDOM_SEED,
    progress_label: str = "",
) -> dict:

    n_cols_built  = matrix_arr.shape[1]
    col_names_arr = np.asarray(col_names)

    real_sharpe = _sharpe_per_column(matrix_arr)

    if logger.isEnabledFor(logging.DEBUG):
        day_offsets = np.arange(matrix_arr.shape[0])
        print_stepm_matrix_debug(col_names, matrix_arr, matrix_arr.shape[0], day_offsets)

    finite_mask = np.isfinite(real_sharpe)
    if finite_mask.all():
        kept_columns = col_names_arr
    else:
        n_keep       = compact_columns_inplace(finite_mask, matrix_arr, real_sharpe, col_names_arr, chunk_size=COLUMN_CHUNK_SIZE)
        matrix_arr   = matrix_arr[:, :n_keep]
        real_sharpe  = real_sharpe[:n_keep]
        kept_columns = col_names_arr[:n_keep]

    if logger.isEnabledFor(logging.DEBUG):
        print_stepm_real_variance_filter_debug(progress_label, n_cols_built, matrix_arr.shape[1])

    n_obs  = matrix_arr.shape[0]
    n_cols = matrix_arr.shape[1]

    rng = np.random.default_rng(seed)
    starts_full, starts_last, len_last, n_blocks_needed = _generate_block_starts(
        n_obs, block_size, n_bootstrap, rng,
    )

    if logger.isEnabledFor(logging.DEBUG):
        print_stepm_block_starts_debug(progress_label, n_blocks_needed, block_size, len_last, n_obs, n_cols)

    weight_matrix = _build_bootstrap_weight_matrix(
        starts_full, starts_last, block_size, len_last, n_obs, n_bootstrap,
    )

    matrix_arr32 = matrix_arr  # already float32 — no extra copy needed

    n_batches = int(np.ceil(n_cols / BOOTSTRAP_CHUNK_SIZE))
    batch_bounds = [
        (i * BOOTSTRAP_CHUNK_SIZE, min((i + 1) * BOOTSTRAP_CHUNK_SIZE, n_cols))
        for i in range(n_batches)
    ]

    deviations = np.empty((n_bootstrap, n_cols), dtype=np.float32)
    sigma_hat  = np.empty(n_cols, dtype=np.float64)

    desc = f"STEPM BOOTSTRAP {progress_label}".strip()
    for start, end in tqdm(batch_bounds, desc=desc, dynamic_ncols=True):
        dev_chunk, sigma_chunk = _bootstrap_moments_chunk(
            weight_matrix, matrix_arr32[:, start:end], real_sharpe[start:end], n_obs,
        )
        deviations[:, start:end] = dev_chunk
        sigma_hat[start:end]     = sigma_chunk

    if logger.isEnabledFor(logging.DEBUG):
        print_stepm_bootstrap_replicas_debug(progress_label, deviations, n_cols, n_bootstrap)

    valid_se = sigma_hat > 0
    if not valid_se.all():
        n_keep       = compact_columns_inplace(valid_se, deviations, real_sharpe, sigma_hat, kept_columns, chunk_size=COLUMN_CHUNK_SIZE)
        deviations   = deviations[:, :n_keep]
        real_sharpe  = real_sharpe[:n_keep]
        sigma_hat    = sigma_hat[:n_keep]
        kept_columns = kept_columns[:n_keep]

    if logger.isEnabledFor(logging.DEBUG):
        print_stepm_se_filter_debug(progress_label, n_cols, kept_columns.shape[0], sigma_hat)

    deviations /= sigma_hat[None, :]
    studentized_deviations = deviations
    z_stat = real_sharpe / sigma_hat

    if logger.isEnabledFor(logging.DEBUG):
        print_stepm_studentization_debug(
            progress_label, studentized_deviations, z_stat, n_cols_built, n_cols, kept_columns.shape[0],
        )

    if STEPM_USE_SPA:
        studentized_deviations, spa_mask, spa_threshold = apply_spa_recentering(studentized_deviations, z_stat, n_obs)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"SPA RECENTERING {progress_label} ── threshold={spa_threshold:.4f} ── "
                f"{int(spa_mask.sum())}/{spa_mask.shape[0]} columns recentered to 0"
            )

    return {
        "real_sharpe":            real_sharpe,
        "sigma_hat":              sigma_hat,
        "studentized_deviations": studentized_deviations,
        "z_stat":                 z_stat,
        "kept_columns":           kept_columns,
    }

# =============================================================================
# GLOBAL P-VALUE — single number per timeframe, the original White (2000) test.
# =============================================================================
def compute_global_pvalue(deviations: np.ndarray, statistic: np.ndarray) -> dict:

    max_deviation  = np.max(deviations, axis=1)          # (n_bootstrap,)
    best_col_idx   = int(np.argmax(statistic))
    best_statistic = float(statistic[best_col_idx])

    global_p = float(np.mean(max_deviation >= best_statistic))

    return {
        "global_p":       global_p,
        "best_col_idx":    best_col_idx,
        "best_statistic":  best_statistic,
    }

# =============================================================================
# ROW-CHUNKED K-TH LARGEST — same np.partition(...)[:, part_idx] result, but
# =============================================================================
def _kth_largest_by_row_chunks(values: np.ndarray, k_eff: int, chunk_size: int = PARTITION_ROW_CHUNK) -> np.ndarray:
    n_rows, n_cols = values.shape
    part_idx = n_cols - k_eff
    result = np.empty(n_rows, dtype=values.dtype)
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        result[start:end] = np.partition(values[start:end], part_idx, axis=1)[:, part_idx]
    return result

# =============================================================================
# CUT DIAGNOSTIC — where the stepdown cut k lands on the cross-sectional grid
# =============================================================================
def _percentile_from_k(k: int, n_cols: int, grid: np.ndarray = CROSS_SECTIONAL_PERCENTILES) -> float:
    """Snap the exact percentile implied by k to the nearest grid value, so the
    reported row matches the corresponding FF_test row exactly."""
    exact = 100.0 * (n_cols - k) / max(n_cols - 1, 1)
    return float(grid[np.argmin(np.abs(grid - exact))])


def compute_cut_diagnostic(
    studentized_deviations: np.ndarray,
    z_stat: np.ndarray,
    k: int,
    chunk_size: int = PARTITION_ROW_CHUNK,
) -> dict:

    pct  = _percentile_from_k(k, z_stat.shape[0])
    real = float(np.percentile(z_stat, pct))

    below = 0
    for start in range(0, studentized_deviations.shape[0], chunk_size):
        batch  = studentized_deviations[start:start + chunk_size]
        below += int((np.percentile(batch, pct, axis=1) < real).sum())

    return {
        "percentile":  pct,
        "real":        real,
        "pct_below":   100.0 * below / studentized_deviations.shape[0],
    }
# =============================================================================
# STEPM (ROMANO & WOLF, 2005) — stepdown per-rule p-values controlling FWER
# =============================================================================
def stepwise_reality_check_pvalues(
    deviations: np.ndarray,
    statistic: np.ndarray,
    alpha: float = STEPM_ALPHA,
    max_iterations: int = STEPM_MAX_ITERATIONS,
    k: int = STEPM_K_FWE,
    _presorted: tuple = None,   # (order, dev_sorted, stat_sorted), reused across the FDP ladder
    _abort_at: int = None,      # stop early once active_start reaches this count
) -> np.ndarray:

    if k < 1:
        raise ValueError(f"k (k-FWE level) must be >= 1, got {k}.")

    n_bootstrap, n_cols = deviations.shape

    if _presorted is None:
        order       = np.argsort(-statistic)
        dev_sorted  = deviations[:, order]
        stat_sorted = statistic[order]
    else:
        order, dev_sorted, stat_sorted = _presorted

    raw_pval_sorted = np.full(n_cols, np.nan, dtype=np.float64)
    active_start = 0
    aborted = False

    for _iteration in range(max_iterations):
        n_active = n_cols - active_start
        if n_active <= 0:
            break

        if _abort_at is not None and active_start >= _abort_at:
            aborted = True
            break

        buffer_size    = min(active_start, k - 1)
        extended_start = active_start - buffer_size

        extended_view = dev_sorted[:, extended_start:]
        active_stat   = stat_sorted[active_start:]
        n_extended    = extended_view.shape[1]
        k_eff         = min(k, n_extended)

        if k_eff == n_extended and n_extended != k:
            logger.debug(
                f"STEPDOWN iter={_iteration} ── requested k={k} exceeds extended set "
                f"size={n_extended} (active={n_active} + buffer={buffer_size}) ── "
                f"clamping to k_eff={k_eff} "
                f"(k-FWE degenerates toward the global minimum in this iteration)"
            )

        if k_eff == 1:
            kth_dev_extended = extended_view.max(axis=1)
        else:
            kth_dev_extended = _kth_largest_by_row_chunks(extended_view, k_eff)

        sorted_dev  = np.sort(kth_dev_extended)
        insert_pos  = np.searchsorted(sorted_dev, active_stat, side="left")
        candidate_p = (n_bootstrap - insert_pos) / n_bootstrap

        reject_local = candidate_p <= alpha
        n_reject     = int(reject_local.sum())

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"STEPD iter={_iteration} ── k={k} ── active={n_active} ── "
                f"rejected_this_iter={n_reject} ── "
                f"candidate_p range=[{candidate_p.min():.4f}, {candidate_p.max():.4f}]"
            )
            if _iteration == 0:
                print_stepm_pvalue_quantile_equivalence_debug(k, kth_dev_extended, alpha, active_stat, reject_local, n_active)

        if n_reject == 0:
            raw_pval_sorted[active_start:] = candidate_p
            break

        raw_pval_sorted[active_start:active_start + n_reject] = candidate_p[:n_reject]
        active_start += n_reject
    else:
        unresolved_sorted = np.flatnonzero(np.isnan(raw_pval_sorted))
        if unresolved_sorted.size > 0:
            unresolved = order[unresolved_sorted]
            raise RuntimeError(
                f"StepM stepdown did not converge within {max_iterations} "
                f"iterations; unresolved column indices: {unresolved.tolist()}"
            )

    if aborted:
        return None

    non_finite_sorted = ~np.isfinite(raw_pval_sorted)
    if non_finite_sorted.any():
        bad = order[non_finite_sorted]
        raise RuntimeError(
            f"StepM stepdown produced an undefined raw p-value for column "
            f"index(es) {bad.tolist()}; refusing to silently treat it as zero."
        )

    adjusted_pval_sorted = np.maximum.accumulate(raw_pval_sorted)

    adjusted_pval = np.empty(n_cols, dtype=np.float64)
    adjusted_pval[order] = adjusted_pval_sorted

    if logger.isEnabledFor(logging.DEBUG):
        print_stepm_monotonicity_debug(k, adjusted_pval_sorted)

    return adjusted_pval

# =============================================================================
# FDP CONTROL (ROMANO & WOLF, 2007, ALGORITHM 4.1) 
# =============================================================================
def resolve_k_by_fdp(
    deviations: np.ndarray,
    statistic: np.ndarray,
    gamma: float = FDP_GAMMA,
    alpha: float = STEPM_ALPHA,
    k_max: int = FDP_K_MAX,
    timeframe: str = "",
) -> tuple:

    if not 0.0 < gamma < 1.0:
        raise ValueError(f"FDP_GAMMA must lie in (0, 1), got {gamma}.")

    # Sort once, reuse for every k in the ladder below.
    order       = np.argsort(-statistic)
    dev_sorted  = deviations[:, order]
    stat_sorted = statistic[order]
    presorted   = (order, dev_sorted, stat_sorted)

    k        = 1
    last_run = None

    while k <= k_max:
        # Once n_rejected reaches this count, this k already fails the FDP
        # criterion below — no need to finish the stepdown to know that.
        abort_at = int(np.ceil(k / gamma - 1.0)) + 1

        pvals = stepwise_reality_check_pvalues(
            deviations, statistic, alpha=alpha, k=k,
            _presorted=presorted, _abort_at=abort_at,
        )
        aborted    = pvals is None
        n_rejected = abort_at if aborted else int((pvals <= alpha).sum())
        last_run   = (k, pvals, n_rejected)

        k_fmt = f"{k:,}".replace(",", ".")
        if aborted:
            # Exact count was never computed — only known to be >= abort_at.
            n_fmt     = f">{abort_at:,}".replace(",", ".")
            gamma_fmt = f" <{gamma:.4f}"
        else:
            n_fmt         = f"{n_rejected:,}".replace(",", ".")
            gamma_implied = k / n_rejected if n_rejected else float("inf")
            gamma_fmt     = f"{gamma_implied:.4f}"
        logger.info(
            f"FDP SEARCH      {timeframe}: k={k_fmt:>9} ── {n_fmt:>9} columns rejected "
            f"── gamma_implied: {gamma_fmt}"
        )

        if n_rejected < k / gamma - 1.0:
            return last_run

        k *= 2

    logger.warning(
        f"FDP SEARCH     {timeframe}: stop criterion never met up to FDP_K_MAX={k_max} "
        f"── k clamped, realized FDP may exceed gamma={gamma}"
    )
    return last_run
# =============================================================================
# PIPE STEPM — orchestration layer, mirroring dsr.py's pipe_dsr exactly.
# =============================================================================
def empty_stepm_fields() -> dict:
    """Placeholder StepM fields for rules that were never evaluated (pipe skipped)."""
    return {
        "passed_stepm": True,
        "passed_mbias": True,
        "stepm_p":      None,
        "sharpe":       None,
    }

def pipe_stepm(
    raw_results: list,
    matrix_arr: np.ndarray,
    col_names: list,
    stepm_alpha: float = STEPM_ALPHA,
    n_bootstrap: int = WHITE_N_BOOTSTRAP,
    block_size: int = WHITE_BLOCK_SIZE,
    stepm_k_esime: float = None,
    seed: int = RANDOM_SEED,
    timeframe: str = "",
) -> list:

    if stepm_k_esime is None:
        if timeframe not in STEPM_K_ESIME_TF:
            raise ValueError(f"No STEPM_K_ESIME configured for timeframe: {timeframe!r}")
        stepm_k_esime = STEPM_K_ESIME_TF[timeframe]

    if matrix_arr is None:
        logger.warning(f"STEPM ── {timeframe} ── insufficient data — skipping, passing all rules through untouched")
        return [{**r, **empty_stepm_fields()} for r in raw_results]

    if matrix_arr.shape[1] < 2:
        logger.warning(f"STEPM ── {timeframe} ── insufficient columns — skipping, passing all rules through untouched")
        return [{**r, **empty_stepm_fields()} for r in raw_results]
    bootstrap_result = compute_deviation_matrix(
        matrix_arr, col_names, n_bootstrap=n_bootstrap, block_size=block_size,
        seed=seed, progress_label=timeframe,
    )
    kept_columns            = bootstrap_result["kept_columns"]
    real_sharpe             = bootstrap_result["real_sharpe"]
    sigma_hat               = bootstrap_result["sigma_hat"]
    studentized_deviations  = bootstrap_result["studentized_deviations"]
    z_stat                  = bootstrap_result["z_stat"]

    logger.debug(
        f"STEPM ── {timeframe} ── {matrix_arr.shape[1] - len(kept_columns)} degenerate "
        f"columns dropped ── {len(kept_columns)} columns remain"
    )

    global_result = compute_global_pvalue(studentized_deviations, z_stat)
    best_col_idx  = global_result["best_col_idx"]
    best_col_name = str(kept_columns[best_col_idx])

    best_raw_idx  = int(np.argmax(real_sharpe))
    best_raw_name = str(kept_columns[best_raw_idx])
    
    stepm_pvals = None

    if STEPM_K_MODE == "kmaxime":
        k_fwe = STEPM_K_FWE
    elif STEPM_K_MODE == "kesime":
        n_cols_for_k = len(kept_columns)
        k_fwe = max(1, int(np.ceil(stepm_k_esime * n_cols_for_k)))
        k_fwe_fmt = f"{k_fwe:,}".replace(",", ".")
        n_cols_for_k_fmt = f"{n_cols_for_k:,}".replace(",", ".")
        logger.info(
            f"STEPM COLUMNS   {timeframe}: STEPM_K_MODE=kesime ── resolved k={k_fwe_fmt} "
            f"from {stepm_k_esime:.4%} of {n_cols_for_k_fmt} surviving columns"
        )
    elif STEPM_K_MODE == "fdp":
        k_fwe, stepm_pvals, _ = resolve_k_by_fdp(
            studentized_deviations, z_stat, timeframe=timeframe,
        )
    else:
        raise ValueError(f"Unknown STEPM_K_MODE={STEPM_K_MODE!r}; expected 'kmaxime', 'kesime' or 'fdp'.")
    logger.debug(f"\n{'─' * 70}")
    logger.debug(f"  MAX RAW SHARPE (no bootstrap adjustment) ── {timeframe}")
    logger.debug(f"{'─' * 70}")
    logger.debug(f"  best column       : {best_raw_name}")
    logger.debug(f"  best real Sharpe  : {real_sharpe[best_raw_idx]:.4f}")
    logger.debug(f"{'─' * 70}\n")

    logger.info(f"\n{'─' * 70}")
    logger.info(f"  GLOBAL WHITE p-value (studentized) ── {timeframe}")
    logger.info(f"{'─' * 70}")
    logger.info(f"  best column(z) : {best_col_name}")
    logger.info(f"  best Sharpe(z) : {real_sharpe[best_col_idx]:.4f}")
    logger.debug(f" best z-statistic: {global_result['best_statistic']:.4f}  (sigma_hat={sigma_hat[best_col_idx]:.4f})")
    logger.info(f"  global p-value : {global_result['global_p']:.4f}")

    cut = compute_cut_diagnostic(studentized_deviations, z_stat, k_fwe)
    logger.info(
        f"  P{cut['percentile']:<14.6g}: {cut['pct_below']:.2f}% <Real  "
        f"(k={f'{k_fwe:,}'.replace(',', '.')})"
    )
    logger.info(f"{'─' * 70}\n")

    logger.debug(f"STEPM ── {timeframe} ── k-FWE level k={k_fwe}" + (" (strict FWE)" if k_fwe == 1 else " (relaxed control — reasoned extension, see module docstring)"))

    if stepm_pvals is None:
        stepm_pvals = stepwise_reality_check_pvalues(studentized_deviations, z_stat, alpha=stepm_alpha, k=k_fwe)
    stepm_p_by_col = dict(zip(kept_columns, stepm_pvals))
    sharpe_by_col  = dict(zip(kept_columns, real_sharpe))
    z_stat_by_col  = dict(zip(kept_columns, z_stat))
    
    if logger.isEnabledFor(logging.DEBUG):
        print_stepm_brc_equivalence_debug(timeframe, k_fwe, global_result["global_p"], stepm_p_by_col, best_col_name)

    best_col_by_rule: dict[str, str] = {}
    for col_name in kept_columns:
        rule_id = str(col_name).rsplit("__", 1)[0]
        current_best = best_col_by_rule.get(rule_id)
        if current_best is None or stepm_p_by_col[col_name] < stepm_p_by_col[current_best]:
            best_col_by_rule[rule_id] = col_name
    
    n_passed = 0
    results  = []
    for r in raw_results:
        col_name      = best_col_by_rule.get(r["rule_id"])
        best_combo_id = str(col_name).rsplit("__", 1)[1] if col_name else None
        stepm_p       = stepm_p_by_col.get(col_name, float("nan"))
        sharpe_val    = sharpe_by_col.get(col_name, float("nan"))
        z_val         = z_stat_by_col.get(col_name, float("nan"))
        passed        = bool(np.isfinite(stepm_p) and stepm_p <= stepm_alpha)
        n_passed     += int(passed)
    
        results.append({
            **r,
            "best_combo_id": best_combo_id,
            "passed_stepm":  passed,
            "passed_mbias":  passed,
            "stepm_p":       float(stepm_p) if np.isfinite(stepm_p) else None,
            "sharpe":        float(sharpe_val) if np.isfinite(sharpe_val) else None,
            "z_stat":        float(z_val) if np.isfinite(z_val) else None,
        })

    n_cols_rejected = int((stepm_pvals <= stepm_alpha).sum())
    gamma_implied   = k_fwe / n_cols_rejected if n_cols_rejected else float("inf")

    k_fwe_fmt           = f"{k_fwe:,}".replace(",", ".")
    n_passed_fmt        = f"{n_passed:,}".replace(",", ".")
    n_total_fmt         = f"{len(raw_results):,}".replace(",", ".")
    n_cols_rejected_fmt = f"{n_cols_rejected:,}".replace(",", ".")
    n_cols_kept_fmt     = f"{len(kept_columns):,}".replace(",", ".")

    logger.info(
        f"STEPM COLUMNS  {timeframe}: k={k_fwe_fmt} ── {n_cols_rejected_fmt}/{n_cols_kept_fmt} columns rejected "
        f"── gamma_implied={gamma_implied:.4f}"
    )
    logger.info(f"STEPM RULES    {timeframe}: {n_passed_fmt}/{n_total_fmt} rules pass")

    return results
