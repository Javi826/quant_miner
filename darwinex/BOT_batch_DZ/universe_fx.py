#BOT_batch_BZ/main_universe.py (forex)
import os
import sys
import time
import random
import logging
import itertools
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
LOG_LEVEL = logging.INFO
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger("BOT_batch.experiment.FF_combos")
logger.setLevel(LOG_LEVEL)

MODULE_LOG_LEVELS = {
    "BOT_batch.pipeline.universe":         logging.INFO,
    "BOT_batch.pipeline.FF_test":          logging.INFO,
    "BOT_batch.pipeline.signal_cleaning":  logging.INFO,
    "BOT_batch.pipeline.backtest_runner":  logging.INFO,
}
for module_name, level in MODULE_LOG_LEVELS.items():
    logging.getLogger(module_name).setLevel(level)
for noisy_logger in ("joblib", "matplotlib", "numba"):
    logging.getLogger(noisy_logger).setLevel(logging.INFO)
# -----------------------------------------------------------------------------

from symbols.universe import build_universe
from setup.config_paths import DATA_FOLDER_BY_DATASET
from rule_mining.rule_generator import MAX_DEPTH as RULE_MAX_DEPTH
from rule_mining.rule_runner import _build_rule_dicts
from utils.ohlcv_utils import prepare_ohlcv_arrays
from setup.config_backtest import ORDER_AMOUNT
from pipeline import backtest_runner as backtest_module
from pipeline.signal_cleaning import pipe_signal_cleaning_jaccard
from engines.FF_test import pipe_FF_test
from setup.config_core import settings

# =============================================================================
# GENERAL
# =============================================================================
RANDOM_SEED       = 42
N_JOBS            = -1
SIGNAL_CLEANING   = True
DATASET           = "IS"   # "IS" or "MERGED"

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================
SYMBOL_POOL = [
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD",
    "USDCHF", "NZDUSD", "EURJPY", "GBPJPY", "EURGBP",
    "EURCHF", "AUDJPY", "CADJPY", "CHFJPY", "EURAUD",
    "EURCAD", "AUDCAD", "GBPCAD", "NZDJPY", "GBPCHF",
]

TIMEFRAMES   = ["1H","4H"]
TIMEFRAMES   = ["1H"]
COMBO_SIZES  = [2]

# Sample size per combo size. None = exhaustive (used automatically for N=1).
N_SAMPLES_PER_SIZE = {
     1:  None,
     2:  190,
}

PARAM_GRID_BY_TIMEFRAME = {
    "1H": {
        "SELL_AFTER": [10,100],
        "TP_PCT":     [0.5,1.0,1.5],
        "SL_PCT":     [0.5,1.0,1.5],
    },
    "4H": {
        "SELL_AFTER":[10,100],
        "TP_PCT":    [0.5,1.0,1.5],
        "SL_PCT":    [0.5,1.0,1.5],
    },
}

RANK_PERCENTILES    = [95,96,97,98,99,99.9]
PCT_BELOW_THRESHOLD = 90  

# =============================================================================
# COMBO GENERATION
# =============================================================================
def _generate_combos(pool: list, size: int, n_samples: int | None, seed: int) -> list:
    all_combos = list(itertools.combinations(pool, size))
    if n_samples is None or n_samples >= len(all_combos):
        return all_combos
    rng = random.Random(seed)
    return rng.sample(all_combos, n_samples)


# =============================================================================
# PER-PERCENTILE %<Real — one value per RANK_PERCENTILE, keyed for the report
# =============================================================================
def _pct_below_by_rank(ff_result: dict, rank_percentiles: list) -> dict:
    percentiles      = ff_result["percentiles"]
    pct_below_actual = ff_result["pct_below_actual"]
    return {
        f"pct_below_{p}": float(pct_below_actual[int(np.where(np.isclose(percentiles, p))[0][0])])
        for p in rank_percentiles
    }

# =============================================================================
# BACKTEST UNIVERSE — build the raw universe once, hand it to the FF pipe.
# =============================================================================
def _run_backtest_universe(
    rules: list,
    ohlcv_arr: dict,
    param_grid: dict,
    order_amount: int,
    timeframe: str,
    n_jobs: int = -1,
    apply_signal_cleaning: bool = False,
) -> tuple:
    """Run the brute-force backtest once; shared by any diagnostic that needs the same matrix."""
    if apply_signal_cleaning:
        rules = pipe_signal_cleaning_jaccard(
            rules     = rules,
            ohlcv_arr = ohlcv_arr,
            timeframe = timeframe,
        )

    original_n_jobs = backtest_module.BACKTEST_N_JOBS
    backtest_module.BACKTEST_N_JOBS = n_jobs
    try:
        return backtest_module.pipe_backtesting(
            rules        = rules,
            ohlcv_arr    = ohlcv_arr,
            param_grid   = param_grid,
            order_amount = order_amount,
            timeframe    = timeframe,
        )
    finally:
        backtest_module.BACKTEST_N_JOBS = original_n_jobs

# =============================================================================
# SINGLE COMBO RUN
# =============================================================================
def _run_combo(combo: tuple, ohlcv_is_pool: dict, ohlcv_arr_pool: dict, timeframe: str, param_grid: dict) -> dict | None:
    ohlcv_is_combo  = {sym: ohlcv_is_pool[sym] for sym in combo}
    ohlcv_arr_combo = {sym: ohlcv_arr_pool[sym] for sym in combo}

    combo_key = f"{timeframe}_{'+'.join(combo)}"
    rules = _build_rule_dicts(ohlcv_is_combo, combo_key, timeframe, RULE_MAX_DEPTH)

    try:
        _, _, matrix_arr, col_names = _run_backtest_universe(
            rules                  = rules,
            ohlcv_arr              = ohlcv_arr_combo,
            param_grid             = param_grid,
            order_amount           = ORDER_AMOUNT,
            timeframe              = timeframe,
            n_jobs                 = N_JOBS,
            apply_signal_cleaning  = SIGNAL_CLEANING,
        )
        ff_result = pipe_FF_test(
            matrix_arr = matrix_arr,
            col_names  = col_names,
            timeframe  = timeframe,
        )
    except ValueError as exc:
        logger.warning(f"SKIP ── {timeframe} ── {combo} ── {exc}")
        return None

    if ff_result is None:
        return None

    return {
        "timeframe": timeframe,
        "n_symbols": len(combo),
        "symbols":   "+".join(combo),
        **_pct_below_by_rank(ff_result, RANK_PERCENTILES),
    }

# =============================================================================
# THRESHOLD REPORT — per RANK_PERCENTILE, which combos clear PCT_BELOW_THRESHOLD
# =============================================================================
SYMBOLS_COL_WIDTH  = 20
REPORT_LINE_WIDTH  = 100
REPORT_LEFT_WIDTH  = 58
REPORT_RIGHT_WIDTH = REPORT_LINE_WIDTH - REPORT_LEFT_WIDTH

def _log_threshold_report(subset: pd.DataFrame, rank_percentiles: list, threshold: float) -> None:
    header_right = f"COMBOS ABOVE THRESHOLD ── %<Real > {threshold}"

    for p in rank_percentiles:
        col         = f"pct_below_{p}"
        passing     = subset[subset[col] > threshold].sort_values(col, ascending=False)
        header_left = f"  Pct {p:<6} ── {len(passing)}/{len(subset)} combo(s) pass"

        logger.info(f"\n{'-' * REPORT_LINE_WIDTH}")
        logger.info(f"{header_left:<{REPORT_LEFT_WIDTH}}{header_right:>{REPORT_RIGHT_WIDTH}}")
        logger.info(f"{'-' * REPORT_LINE_WIDTH}")

        if passing.empty:
            logger.info("  No combo(s) passed this threshold.")
            continue

        table = passing[["symbols", "n_symbols", col]].copy()
        table["symbols"] = table["symbols"].str.ljust(SYMBOLS_COL_WIDTH)
        table[col] = table[col].round(2)
        logger.info(table.to_string(index=False))

        # --- NUEVO: la misma lista de "symbols", pero como literal Python ---
        _log_symbols_as_python_list(passing)


def _log_symbols_as_python_list(passing: pd.DataFrame) -> None:
    """Print the passing 'symbols' column as a Python list-of-lists literal, ready to paste."""
    lines = []
    for symbols_str in passing["symbols"].str.strip():
        syms = symbols_str.split("+")
        formatted = ", ".join(f'"{s}"' for s in syms)
        lines.append(f"    [{formatted}],")

    logger.info("[\n" + "\n".join(lines) + "\n]")

# =============================================================================
# CROSS-TIMEFRAME SUMMARY — combos passing the threshold in every timeframe,
# =============================================================================
def _build_common_percentile_summary(
    results_df: pd.DataFrame,
    timeframes: list,
    rank_percentiles: list,
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for symbols, group in results_df.groupby("symbols"):
        if set(group["timeframe"]) != set(timeframes):
            continue

        common_passing = None
        for _, row in group.iterrows():
            passing_here = {p for p in rank_percentiles if row[f"pct_below_{p}"] > threshold}
            common_passing = passing_here if common_passing is None else common_passing & passing_here

        if common_passing:
            rows.append({
                "symbols": symbols,
                "p_min":   min(common_passing),
                "p_max":   max(common_passing),
            })

    return pd.DataFrame(rows).sort_values(["p_min", "symbols"]) if rows else pd.DataFrame(columns=["symbols", "p_min", "p_max"])

def _log_common_percentile_summary(summary_df: pd.DataFrame) -> None:
    logger.info(f"\n{'=' * REPORT_LINE_WIDTH}")
    logger.info("  COMMON PERCENTILE RANGE ── PASSING IN ALL TIMEFRAMES")
    logger.info(f"{'=' * REPORT_LINE_WIDTH}")

    if summary_df.empty:
        logger.info("  No combo(s) passed the threshold in every timeframe.")
        return

    table = summary_df.copy()
    table["symbols"] = table["symbols"].str.ljust(SYMBOLS_COL_WIDTH)
    logger.info(table.to_string(index=False))

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    start = time.time()

    logger.info(f"\n{'─' * 100}")
    logger.info("  FF BOOTSTRAP — SYMBOL COMBINATION EXPERIMENT")
    logger.info(f"{'─' * 100}")
    logger.info(f"  DATASET            : {DATASET} ── {os.path.basename(DATA_FOLDER_BY_DATASET[DATASET])}")
    logger.info(f"  BACKTEST           : {settings.BACKTEST_MODE}")
    logger.info(f"  SYMBOL_POOL        : {SYMBOL_POOL}")
    logger.info(f"  COMBO_SIZES        : {COMBO_SIZES}")
    logger.info(f"  PARAM_GRID_BY_TIMEFRAME : {PARAM_GRID_BY_TIMEFRAME}")
    logger.info(f"  N_SAMPLES_PER_SIZE : {N_SAMPLES_PER_SIZE}")
    logger.info(f"  RANK_PERCENTILES   : {RANK_PERCENTILES}")
    logger.info(f"{'─' * 100}\n")

    ohlcv_data_by_timeframe = build_universe(
        DATA_FOLDER_BY_DATASET[DATASET], {tf: SYMBOL_POOL for tf in TIMEFRAMES},
        dataset=DATASET,
    )
    ohlcv_arr_by_timeframe  = {
        timeframe: prepare_ohlcv_arrays(ohlcv_is)
        for timeframe, ohlcv_is in ohlcv_data_by_timeframe.items()
    }

    all_rows = []

    for timeframe in TIMEFRAMES:
        ohlcv_is_pool  = ohlcv_data_by_timeframe[timeframe]
        ohlcv_arr_pool = ohlcv_arr_by_timeframe[timeframe]
        param_grid     = PARAM_GRID_BY_TIMEFRAME[timeframe]

        for size in COMBO_SIZES:
            combos   = _generate_combos(SYMBOL_POOL, size, N_SAMPLES_PER_SIZE.get(size), RANDOM_SEED)
            n_combos = len(combos)
            logger.info(f"\n{'=' * 100}")
            logger.info(f"  {timeframe.upper()} ── N={size} ── RUNNING {n_combos} COMBO(S)")
            logger.info(f"{'=' * 100}\n")

            for combo_idx, combo in enumerate(combos, start=1):
                logger.info(f"{'-' * 100}")
                logger.info(f"Testing: {' + '.join(combo)} <{combo_idx}/{n_combos}>")
                logger.info(f"{'-' * 100}")
                row = _run_combo(combo, ohlcv_is_pool, ohlcv_arr_pool, timeframe, param_grid)
                if row is not None:
                    all_rows.append(row)

    results_df = pd.DataFrame(all_rows)

    for timeframe in TIMEFRAMES:
        subset = results_df[results_df["timeframe"] == timeframe]
        logger.info(f"\n{'=' * REPORT_LINE_WIDTH}")
        logger.info(f"  TIMEFRAME: {timeframe.upper()}")
        logger.info(f"{'=' * REPORT_LINE_WIDTH}")
        _log_threshold_report(subset, RANK_PERCENTILES, PCT_BELOW_THRESHOLD)

    summary_df = _build_common_percentile_summary(results_df, TIMEFRAMES, RANK_PERCENTILES, PCT_BELOW_THRESHOLD)
    _log_common_percentile_summary(summary_df)

    elapsed = int(time.time() - start)
    logger.info(f"\n🏁 TOTAL — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")