# core/pipeline/stepM_oos.py
import logging
import numpy as np
import pandas as pd
from pipeline.stepM_is import (
    compute_bootstrap_null,
    compute_global_pvalue,
    WHITE_N_BOOTSTRAP,
    WHITE_BLOCK_SIZE,
    RANDOM_SEED,
)
from utils.batch_metrics import daily_values_from_sell_days, _trading_days_between
logger = logging.getLogger("BOT_batch.pipeline.stepM_oos")

# =============================================================================
# STEPM OOS CONFIG: informative only, one verdict per timeframe (rules are never filtered)
# =============================================================================
STEPM_OOS_ALPHA      = 0.10
STEPM_OOS_MIN_TRADES = 30    # rules with fewer WFO test trades are not evaluated

_TICK = {True: "✅", False: "❌"}


def _fmt(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def _prefix(timeframe: str) -> str:
    return f"{'STEPM OOS':<15}{timeframe}"

# =============================================================================
# WFO TEST RETURNS MATRIX: same convention as the IS matrix (backtest_runner)
# =============================================================================
# One column per rule: its WFO test series, already out-of-sample for everything chosen
# (rule and SELL_AFTER in IS, TP/SL in previous WFO windows). Built with the same helpers
# as run_full_period_search; columns with <= 1 non-zero day skipped, all-zero days dropped.
def _sell_times_ns(trades: pd.DataFrame) -> np.ndarray:
    ts = pd.to_datetime(trades["sell_time"])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert(None)
    return ts.to_numpy("datetime64[ns]")


def _n_trades(rule: dict) -> int:
    trades = rule.get("wfo_test_trades")
    return 0 if trades is None else len(trades)


def build_wfo_test_matrix(rules: list, min_trades: int = STEPM_OOS_MIN_TRADES) -> tuple:
    """(n_days, n_rules) float32 matrix of daily profit by sell day; all-zero days dropped."""
    series = {}
    for r in rules:
        if _n_trades(r) < max(1, min_trades):
            continue
        trades = r["wfo_test_trades"]
        daily_values, n_days, start_day = daily_values_from_sell_days(
            _sell_times_ns(trades), trades["profit"].to_numpy(np.float64),
        )
        if np.count_nonzero(daily_values) > 1:
            series[r["rule_id"]] = (daily_values, n_days, start_day)

    if not series:
        return None, []

    global_start_day = min(start_day for _, _, start_day in series.values())
    offsets = {
        rid: int(_trading_days_between(global_start_day, start_day))
        for rid, (_, _, start_day) in series.items()
    }
    n_days_range = max(offsets[rid] + n_days for rid, (_, n_days, _) in series.items())

    col_names = list(series)
    matrix    = np.zeros((n_days_range, len(col_names)), dtype=np.float32)
    for j, rid in enumerate(col_names):
        daily_values, n_days, _ = series[rid]
        matrix[offsets[rid]:offsets[rid] + n_days, j] = daily_values.astype(np.float32)

    day_mask = np.any(matrix != 0, axis=1)
    return np.ascontiguousarray(matrix[day_mask]), col_names

# =============================================================================
# STATISTIC: raw annualized Sharpe, NOT the studentized z
# =============================================================================
# On sparse trade series the bootstrap sigma_hat is noisy and anti-correlated with
# |Sharpe|, so the studentized z of lucky rules is inflated and the max over the family
# over-rejects. The Sharpe is already scale free with the same null variance for every
# column on a common calendar, so the deviations are multiplied back by sigma_hat
# (this keeps the SPA_c recentering: (dev + z) * sigma_hat = S* - 0).
# The family is small (one column per rule), so the top-M is requested for every kept
# column: that is the full deviation matrix, rebuilt densely here.
def _raw_deviations(null) -> np.ndarray:
    n_boot, n_kept = null.n_bootstrap, null.n_kept
    if null.m != n_kept:
        raise RuntimeError(f"STEPM OOS ── top-M holds {null.m} of {n_kept} columns, expected all of them")
    deviations = np.empty((n_boot, n_kept), dtype=np.float32)
    np.put_along_axis(deviations, null.topm_cols.astype(np.int64), null.topm_vals, axis=1)
    deviations *= null.sigma_hat.astype(np.float32)[None, :]
    return deviations

# =============================================================================
# ONE TIMEFRAME: two verdicts over every rule that entered WFO
# =============================================================================
# Concentrated signal: does the best rule beat the best of N rules without edge?
#   White's global p-value on the max of the null.
# Diffuse signal: are there more individually significant rules than chance gives?
#   A rule is significant when its Sharpe beats the (1 - alpha) quantile of its own null
#   column. The same count is taken in every bootstrap replica (no edge there), so the
#   correlation between rules is already inside the null distribution of the count.
def _evaluate_timeframe(
    rules: list, timeframe: str, alpha: float,
    n_bootstrap: int, block_size: int, seed: int, min_trades: int,
) -> None:
    prefix = _prefix(timeframe)

    matrix, col_names = build_wfo_test_matrix(rules, min_trades)
    if matrix is None or matrix.shape[0] < 2 * block_size:
        logger.warning(f"{prefix}: not enough WFO test data (>= {min_trades} trades per rule) ── no verdict")
        return

    keep      = matrix.std(axis=0) > 0
    matrix    = np.ascontiguousarray(matrix[:, keep])
    col_names = [c for c, k in zip(col_names, keep) if k]
    if not col_names:
        logger.warning(f"{prefix}: every WFO test series is constant ── no verdict")
        return

    null = compute_bootstrap_null(
        matrix, col_names, n_bootstrap=n_bootstrap, block_size=block_size, seed=seed,
        topm_size=matrix.shape[1], progress_label=f"OOS {timeframe}", desc=f"{'STEPM BST OOS':<15}{timeframe}",
    )
    if null.n_kept == 0:
        logger.warning(f"{prefix}: no column with a valid bootstrap sigma ── no verdict")
        return

    deviations = _raw_deviations(null)
    statistic  = null.real_sharpe
    kept       = null.kept_columns
    n_days     = matrix.shape[0]
    del null

    # concentrated signal
    row_max  = deviations.max(axis=1)
    global_p = compute_global_pvalue(row_max, statistic)["global_p"]
    needed   = float(np.quantile(row_max, 1.0 - alpha))
    best_idx = int(np.argmax(statistic))

    # diffuse signal
    crit        = np.quantile(deviations, 1.0 - alpha, axis=0)
    real_count  = int((statistic > crit).sum())
    null_counts = (deviations > crit[None, :]).sum(axis=1)
    diffuse_p   = float(np.mean(null_counts >= real_count))

    pct = f"P{100 * (1 - alpha):g}"
    logger.debug(
        f"{prefix}: {_fmt(n_days)} days ── best Sharpe={statistic[best_idx]:.4f} ({kept[best_idx]}) "
        f"── Sharpe needed={needed:.4f} ({pct} of null max)"
    )
    logger.debug(
        f"{prefix}: gray zone={_fmt(real_count)} rules ── null mean={null_counts.mean():.0f} "
        f"── null {pct}={np.quantile(null_counts, 1.0 - alpha):.0f}"
    )
    logger.info(
        f"{prefix}: {_fmt(len(kept))}/{_fmt(len(rules))} rules evaluable "
        f"── concentrated {_TICK[global_p <= alpha]} p={global_p:.3f} "
        f"── diffuse {_TICK[diffuse_p <= alpha]} p={diffuse_p:.3f}"
    )

# =============================================================================
# PIPE STEPM OOS (informative: logs one verdict per timeframe, returns nothing)
# =============================================================================
def pipe_stepm_oos(
    wfo_results: list,
    alpha: float = STEPM_OOS_ALPHA,
    n_bootstrap: int = WHITE_N_BOOTSTRAP,
    block_size: int = WHITE_BLOCK_SIZE,
    seed: int = RANDOM_SEED,
    min_trades: int = STEPM_OOS_MIN_TRADES,
) -> None:
    rules_by_tf: dict = {}
    for r in wfo_results:
        rules_by_tf.setdefault(r["timeframe"], []).append(r)

    for timeframe, rules in rules_by_tf.items():
        _evaluate_timeframe(rules, timeframe, alpha, n_bootstrap, block_size, seed, min_trades)