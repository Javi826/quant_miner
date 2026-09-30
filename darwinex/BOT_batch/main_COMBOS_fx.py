#BOT_batch_BZ/main_symb_fx.py (forex)
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
    "BOT_batch.pipeline.stepM_is":         logging.INFO,
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
from pipeline.stepM_is import pipe_stepm
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
SYMBOL_POOL_BY_TIMEFRAME = {
    "1H": [
    "USDJPY",
    "EURAUD",
    "NZDJPY",
    "EURJPY",
    "AUDJPY",
    "EURCHF",
    "GBPCHF",
    "GBPJPY",
    ],
    "4H": [
    "EURJPY",
    "USDCAD",
    "GBPJPY",
    "AUDJPY",
    "CHFJPY",
    ],
}

TIMEFRAMES   = ["1H","4H"]
COMBO_SIZES  = [1,2]

# Sample size per combo size. None = exhaustive (used automatically for N=1).
N_SAMPLES_PER_SIZE = {
     1:  None,
     99: 99 ,
}

PARAM_GRID_BY_TIMEFRAME = {
    "1H": {
        "SELL_AFTER": [100],
        "TP_PCT":     [1.5],
        "SL_PCT":     [0.5],
    },
    "4H": {
        "SELL_AFTER": [40],
        "TP_PCT":     [1.5],
        "SL_PCT":     [0.5],
    },
}

# Ranking: rules passing StepM IS (as main_back_fx keeps them), then %<Real of the FF at RANK_PERCENTILES, in this
# order (P100 = 1 - global White p). Combos with no rule passing are left out
RANK_PERCENTILES = [100, 99.9]     # also the only percentiles the FF computes (its top-M covers down to the lowest)
RANK_TOP_N       = 30

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
# PER-PERCENTILE %<Real — one value per percentile, keyed for the report
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
def _count_stepm_rules(raw_results: list, matrix_arr: np.ndarray, col_names: list, timeframe: str) -> int:
    """Rules passing StepM IS, as main_back_fx keeps them (POST-MBIAS). If StepM skips (fewer than 2 columns) it
    lets every rule through untouched: that counts 0, not all of them."""
    res = pipe_stepm(raw_results=raw_results, matrix_arr=matrix_arr, col_names=col_names, timeframe=timeframe)
    if any(r["passed_mbias"] and r["stepm_p"] is None for r in res):
        logger.warning(f"STEPM SKIPPED  {timeframe}: counted as 0 rules")
        return 0
    return sum(bool(r["passed_mbias"]) for r in res)


def _run_combo(combo: tuple, combo_idx: int, ohlcv_is_pool: dict, ohlcv_arr_pool: dict, timeframe: str,
               param_grid: dict) -> dict | None:
    ohlcv_is_combo  = {sym: ohlcv_is_pool[sym] for sym in combo}
    ohlcv_arr_combo = {sym: ohlcv_arr_pool[sym] for sym in combo}

    combo_key = f"{timeframe}_c{combo_idx:02d}"            # as main_back_fx: the index shown in "Testing: ... <i/n>"
    rules = _build_rule_dicts(ohlcv_is_combo, combo_key, timeframe, RULE_MAX_DEPTH)

    try:
        raw_results, _, matrix_arr, col_names = _run_backtest_universe(
            rules                  = rules,
            ohlcv_arr              = ohlcv_arr_combo,
            param_grid             = param_grid,
            order_amount           = ORDER_AMOUNT,
            timeframe              = timeframe,
            n_jobs                 = N_JOBS,
            apply_signal_cleaning  = SIGNAL_CLEANING,
        )

        ff_result = pipe_FF_test(
            matrix_arr  = np.array(matrix_arr, copy=True),
            col_names   = col_names,
            percentiles = RANK_PERCENTILES,
            timeframe   = timeframe,
        )
        if ff_result is None:
            return None
        n_rules = _count_stepm_rules(raw_results, matrix_arr, col_names, timeframe)
    except ValueError as exc:
        logger.warning(f"SKIP ── {timeframe} ── {combo} ── {exc}")
        return None

    return {
        "timeframe": timeframe,
        "n_symbols": len(combo),
        "symbols":   "+".join(combo),
        "rules":     n_rules,
        **_pct_below_by_rank(ff_result, RANK_PERCENTILES),
    }

# =============================================================================
# REPORT HELPERS
# =============================================================================
SYMBOLS_COL_WIDTH  = 20
REPORT_LINE_WIDTH  = 100
HEADER_LABEL_WIDTH = 24
HEADER_INDENT       = " " * (2 + HEADER_LABEL_WIDTH + 3)   # aligns continuation lines under the value


def _pct_col(p) -> str:
    return f"pct_below_{p}"


def _header_line(label: str, value: str) -> str:
    """One header row, label padded to HEADER_LABEL_WIDTH; value's own newlines get HEADER_INDENT."""
    value = value.replace("\n", "\n" + HEADER_INDENT)
    return f"  {label:<{HEADER_LABEL_WIDTH}} : {value}"


def _format_param_grid(grid: dict) -> str:
    """PARAM_GRID_BY_TIMEFRAME as a multi-line list for the run header (one timeframe per line)."""
    lines = [f"'{tf}': {params}," for tf, params in grid.items()]
    return "\n".join(lines)


def _format_symbol_pool(pool: list, per_line: int = 5) -> str:
    """SYMBOL_POOL as a multi-line list for the run header (max `per_line` symbols per line)."""
    lines = []
    for i in range(0, len(pool), per_line):
        chunk = ", ".join(f"'{s}'" for s in pool[i:i + per_line])
        lines.append(chunk + ",")
    return "\n".join(lines)

def _log_symbols_as_python_list(passing: pd.DataFrame, level: int = logging.INFO) -> None:
    """Print the passing 'symbols' column as a Python list-of-lists literal, ready to paste."""
    lines = []
    for symbols_str in passing["symbols"].str.strip():
        syms = symbols_str.split("+")
        formatted = ", ".join(f'"{s}"' for s in syms)
        lines.append(f"    [{formatted}],")

    logger.log(level, "[\n" + "\n".join(lines) + "\n]")

# =============================================================================
# RANKING ── rules passing StepM IS, then %<Real at RANK_PERCENTILES
# =============================================================================
def _build_ranking(subset: pd.DataFrame) -> pd.DataFrame:
    """One timeframe: the combos with rules passing StepM IS, by rules, then %<Real at RANK_PERCENTILES in order."""
    df = subset[subset["rules"] > 0]
    keys = ["rules", *[_pct_col(p) for p in RANK_PERCENTILES]]
    return df.sort_values(keys, ascending=False, kind="stable").reset_index(drop=True)


def _log_ranking(ranking: pd.DataFrame, n_tested: int, n_run: int, timeframe: str, top_n: int) -> None:
    """Top-N combos of the ranking (combos with no rule passing StepM IS are already out). n_run: combos with a
    result (the rest were skipped)."""
    shortlist = ranking.head(top_n)

    logger.info(f"\n{'=' * REPORT_LINE_WIDTH}")
    logger.info(f"  RANKING {timeframe.upper()} ── {n_tested} combos tested: {len(ranking)} with rules passing StepM IS, "
                f"{n_run - len(ranking)} with 0 rules, {n_tested - n_run} skipped ── showing the top {len(shortlist)}")
    logger.info(f"{'=' * REPORT_LINE_WIDTH}")

    if shortlist.empty:
        logger.info("  No combo(s) to rank.")
        return

    rename = {_pct_col(p): f"p{p}" for p in RANK_PERCENTILES}
    table  = shortlist[["symbols", "n_symbols", "rules", *rename]].rename(columns=rename)
    table["symbols"] = table["symbols"].str.ljust(SYMBOLS_COL_WIDTH)
    logger.info(table.round(2).to_string(index=False))

    _log_symbols_as_python_list(shortlist)
# =============================================================================
# CROSS RANKING ── combos ranked in every timeframe (TOP BOTH)
# =============================================================================
def _build_cross_ranking(rankings_by_tf: dict) -> pd.DataFrame:
    """Combos with rules in every timeframe: by the fewest rules across timeframes, then the lowest %<Real at each
    RANK_PERCENTILE across timeframes."""
    per_tf = []
    for timeframe, ranking in rankings_by_tf.items():
        cols = ["symbols", "n_symbols", "rules", *[_pct_col(p) for p in RANK_PERCENTILES]]
        per_tf.append(ranking[cols].rename(columns={
            "rules": f"rules_{timeframe}", **{_pct_col(p): f"p{p}_{timeframe}" for p in RANK_PERCENTILES},
        }))

    merged = per_tf[0]
    for other in per_tf[1:]:
        merged = merged.merge(other, on=["symbols", "n_symbols"], how="inner")

    merged["min_rules"] = merged[[f"rules_{tf}" for tf in rankings_by_tf]].min(axis=1)
    for p in RANK_PERCENTILES:
        merged[f"min_p{p}"] = merged[[f"p{p}_{tf}" for tf in rankings_by_tf]].min(axis=1)

    keys = ["min_rules", *[f"min_p{p}" for p in RANK_PERCENTILES]]
    return merged.sort_values(keys, ascending=False, kind="stable").reset_index(drop=True)


def _log_cross_ranking(cross_ranking: pd.DataFrame, timeframes: list, top_n: int) -> None:
    shortlist = cross_ranking.head(top_n)
    label     = " + ".join(tf.upper() for tf in timeframes)

    logger.info(f"\n{'=' * REPORT_LINE_WIDTH}")
    logger.info(f"  RANKING TOP BOTH ({label}) ── TOP {len(shortlist)} of {len(cross_ranking)} common candidate(s)")
    logger.info(f"{'=' * REPORT_LINE_WIDTH}")

    if shortlist.empty:
        logger.info("  No common combo(s) to rank.")
        return

    cols = [
        "symbols", "n_symbols", "min_rules", *[f"min_p{p}" for p in RANK_PERCENTILES],
        *[f"rules_{tf}" for tf in timeframes],
        *[f"p{p}_{tf}" for p in RANK_PERCENTILES for tf in timeframes],
    ]
    table = shortlist[cols].copy()
    table["symbols"] = table["symbols"].str.ljust(SYMBOLS_COL_WIDTH)
    logger.info(table.round(2).to_string(index=False))

    _log_symbols_as_python_list(shortlist)
# =============================================================================
# MAIN
# =============================================================================
def main():
    missing_pool = [tf for tf in TIMEFRAMES if not SYMBOL_POOL_BY_TIMEFRAME.get(tf)]
    if missing_pool:
        raise ValueError(f"SYMBOL_POOL_BY_TIMEFRAME has no symbols for timeframes: {missing_pool}")

    logger.info(f"\n{'─' * 100}")
    logger.info("  FF BOOTSTRAP — SYMBOL COMBINATION EXPERIMENT")
    logger.info(f"{'─' * 100}")
    logger.info(_header_line("DATASET", f"{DATASET} ── {os.path.basename(DATA_FOLDER_BY_DATASET[DATASET])}"))
    logger.info(_header_line("BACKTEST", str(settings.BACKTEST_MODE)))
    for tf in TIMEFRAMES:
        logger.info(_header_line(f"SYMBOL_POOL {tf}", _format_symbol_pool(SYMBOL_POOL_BY_TIMEFRAME[tf])))
    logger.info(_header_line("COMBO_SIZES", str(COMBO_SIZES)))
    logger.info(_header_line("PARAM_GRID_BY_TIMEFRAME", _format_param_grid(PARAM_GRID_BY_TIMEFRAME)))
    logger.info(_header_line("N_SAMPLES_PER_SIZE", str(N_SAMPLES_PER_SIZE)))
    logger.info(_header_line("RANKING", f"rules passing StepM IS, then %<Real at {RANK_PERCENTILES}"))
    logger.info(f"{'─' * 100}\n")

    ohlcv_data_by_timeframe = build_universe(
        DATA_FOLDER_BY_DATASET[DATASET], {tf: SYMBOL_POOL_BY_TIMEFRAME[tf] for tf in TIMEFRAMES},
        dataset=DATASET,
    )
    ohlcv_arr_by_timeframe  = {
        timeframe: prepare_ohlcv_arrays(ohlcv_is)
        for timeframe, ohlcv_is in ohlcv_data_by_timeframe.items()
    }

    all_rows = []
    n_tested = {}

    for timeframe in TIMEFRAMES:
        ohlcv_is_pool  = ohlcv_data_by_timeframe[timeframe]
        ohlcv_arr_pool = ohlcv_arr_by_timeframe[timeframe]
        param_grid     = PARAM_GRID_BY_TIMEFRAME[timeframe]

        for size in COMBO_SIZES:
            combos   = _generate_combos(SYMBOL_POOL_BY_TIMEFRAME[timeframe], size, N_SAMPLES_PER_SIZE.get(size), RANDOM_SEED)
            n_combos = len(combos)
            n_tested[timeframe] = n_tested.get(timeframe, 0) + n_combos
            logger.info(f"\n{'=' * 100}")
            logger.info(f"  {timeframe.upper()} ── N={size} ── RUNNING {n_combos} COMBO(S)")
            logger.info(f"{'=' * 100}\n")

            for combo_idx, combo in enumerate(combos, start=1):
                logger.info(f"{'-' * 100}")
                logger.info(f"Testing: {' + '.join(combo)} <{combo_idx}/{n_combos}>")
                logger.info(f"{'-' * 100}")
                row = _run_combo(combo, combo_idx, ohlcv_is_pool, ohlcv_arr_pool, timeframe, param_grid)
                if row is not None:
                    all_rows.append(row)

    columns    = ["timeframe", "n_symbols", "symbols", "rules", *[_pct_col(p) for p in RANK_PERCENTILES]]
    results_df = pd.DataFrame(all_rows, columns=columns)

    rankings_by_tf = {}
    for timeframe in TIMEFRAMES:
        subset  = results_df[results_df["timeframe"] == timeframe]
        ranking = _build_ranking(subset)
        _log_ranking(ranking, n_tested.get(timeframe, 0), len(subset), timeframe, RANK_TOP_N)
        rankings_by_tf[timeframe] = ranking

    if len(rankings_by_tf) > 1:
        cross_ranking = _build_cross_ranking(rankings_by_tf)
        _log_cross_ranking(cross_ranking, list(rankings_by_tf), RANK_TOP_N)


if __name__ == "__main__":
    start = time.time()
    try:
        main()
        elapsed = int(time.time() - start)
        logger.info(f"\n🏁 TOTAL — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")
    except KeyboardInterrupt:
        elapsed = int(time.time() - start)
        logger.info(f"\n⛔  INTERRUPTED BY USER — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")
        sys.exit(0)