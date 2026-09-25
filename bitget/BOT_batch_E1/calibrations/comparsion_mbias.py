#BOT_batch_E1/research/comparison_mbias.py

import os
import sys
import time
import logging
import numpy as np
from itertools import combinations
from scipy.stats import pearsonr, spearmanr
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
LOG_LEVEL = logging.INFO
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger("BOT_batch.main_comp")
logger.setLevel(LOG_LEVEL)

DSR_LOG_LEVEL = logging.INFO
logging.getLogger("BOT_batch.pipeline.dsr").setLevel(DSR_LOG_LEVEL)
logging.getLogger("BOT_batch.pipeline.backtest_runner").setLevel(DSR_LOG_LEVEL)

STEPM_LOG_LEVEL = logging.INFO
logging.getLogger("BOT_batch.pipeline.stepM_is").setLevel(STEPM_LOG_LEVEL)
#------------------------------------------------------------------------------
REPORTING_LOG_LEVEL = logging.INFO
logging.getLogger("BOT_batch.utils.reporting").setLevel(REPORTING_LOG_LEVEL)

logging.getLogger("joblib").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

from symbols.universe import build_universe
from setup.config_paths import DATA_FOLDER_BY_DATASET
from rule_mining.rule_generator import MAX_DEPTH as RULE_MAX_DEPTH
from rule_mining.rule_runner import _build_rule_dicts
from utils.ohlcv_utils import prepare_ohlcv_arrays
from setup.config_backtest import ORDER_AMOUNT
from pipeline import backtest_runner as backtest_module
from dsr import pipe_dsr
from pipeline.stepM_is import pipe_stepm, STEPM_ALPHA, WHITE_PVALUE_TH, WHITE_N_BOOTSTRAP, WHITE_BLOCK_SIZE
from pipeline.signal_cleaning import pipe_signal_cleaning_jaccard
# =============================================================================
# UNIVERSE / SEARCH SPACE CONFIGURATION
# =============================================================================
DATASET = "MERGED"   # "IS", "OOS" or "MERGED"
N_JOBS  = -1         # -1 = use all available cores, for both the backtest search and the StepM bootstrap

TIMEFRAMES = ["1H","4H", "6Hutc", "12Hutc"]
SYMBOLS_BY_TIMEFRAME = {
    "1H":     ["ADAUSDT","AVAXUSDT","BCHUSDT","BNBUSDT","DOGEUSDT","LINKUSDT","NEARUSDT","SOLUSDT","UNIUSDT","XRPUSDT"],
    "4H":     ["ADAUSDT","AVAXUSDT","BCHUSDT","BNBUSDT","DOGEUSDT","LINKUSDT","NEARUSDT","SOLUSDT","UNIUSDT","XRPUSDT"],
    "6Hutc":  ["ADAUSDT","AVAXUSDT","BCHUSDT","BNBUSDT","DOGEUSDT","LINKUSDT","NEARUSDT","SOLUSDT","UNIUSDT","XRPUSDT"],
    "12Hutc": ["ADAUSDT","AVAXUSDT","BCHUSDT","BNBUSDT","DOGEUSDT","LINKUSDT","NEARUSDT","SOLUSDT","UNIUSDT","XRPUSDT"],
}

PARAM_GRID = {
    "SELL_AFTER": [50],
    "TP_PCT":     [6,8,10],
    "SL_PCT":     [6,8],
}

# =============================================================================
# DSR vs STEPM — full brute universe comparison, no other pipeline stages
# =============================================================================
DSR_TH              = 0.95
STEPM_K_ESIME       = 0.001

# =============================================================================
# METHOD REGISTRY — adding a new data-snooping test only requires a new entry
# here plus a new "pairs" tuple in _print_correlation; every report function
# below iterates over this registry instead of hardcoding method names.
# =============================================================================
DSR_TEST   = True
STEPM_TEST = True
METHOD_SPECS = {
    "dsr":   {"value_key": "dsr",     "ok_key": "passed_dsr",   "label": "DSR"},
    "stepm": {"value_key": "stepm_p", "ok_key": "passed_stepm", "label": "STEPM"},
}
_METHOD_FLAGS = {"dsr": DSR_TEST, "stepm": STEPM_TEST}
METHOD_ORDER  = [m for m in ["dsr", "stepm"] if _METHOD_FLAGS[m]]

# =============================================================================
# SIGNAL CLEANING — dedupe near-identical rules before the brute backtest
# =============================================================================
SIGNAL_CLEANING_TEST = True

# =============================================================================
# COMPARISON — build the raw universe once, hand it to all pipes, compare.
# =============================================================================
def compare_dsr_vs_stepm_from_raw(
    raw_results: list,
    matrix_arr,
    col_names: list,
    dsr_th: float,
    n_combos: int,
    stepm_k_esime: float | None = None,
    timeframe: str = "",
    n_jobs: int = -1,
) -> dict:

    if not METHOD_ORDER:
        logger.warning(f"COMPARE ── {timeframe} ── no method enabled (DSR/STEPM both off) — skipping comparison")
        return {}

    results_by_method = {}

    if "dsr" in METHOD_ORDER:
        dsr_results = pipe_dsr(
            raw_results = raw_results,
            matrix_arr  = matrix_arr,
            dsr_th      = dsr_th,
            n_combos    = n_combos,
            timeframe   = timeframe,
        )
        results_by_method["dsr"] = {r["rule_id"]: r for r in dsr_results}

    if "stepm" in METHOD_ORDER:
        stepm_results = pipe_stepm(
            raw_results    = raw_results,
            matrix_arr     = matrix_arr,
            col_names      = col_names,
            stepm_k_esime  = stepm_k_esime,
            timeframe      = timeframe,
        )
        results_by_method["stepm"] = {r["rule_id"]: r for r in stepm_results}

    _print_comparison(raw_results, results_by_method, timeframe)

    return results_by_method


def _run_backtest_universe(
    rules: list,
    ohlcv_arr: dict,
    param_grid: dict,
    order_amount: int,
    timeframe: str,
    n_jobs: int = -1,
    apply_signal_cleaning: bool = False,
) -> tuple:
    """Run the brute-force backtest once; shared by the method comparison
    and any standalone diagnostic (e.g. FF) that needs the same matrix."""
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
# REPORTING
# =============================================================================
_SORT_METHOD = "stepm" if "stepm" in METHOD_ORDER else (METHOD_ORDER[0] if METHOD_ORDER else None)

def _sort_key(row: dict):
    if _SORT_METHOD is None:
        return 0
    val = row.get(f"{_SORT_METHOD}_value")
    return val if val is not None else float("inf")
def _build_comparison_rows(raw_results: list, results_by_method: dict) -> list:
    """One dict per rule, with each method's value/ok, restricted to rules
    where STEPM produced a p-value (same gate as before)."""
    rows = []
    for r in raw_results:
        rid = r["rule_id"]
        if "stepm" in results_by_method and results_by_method["stepm"].get(rid, {}).get("stepm_p") is None:
            continue

        row = {"rule_id": rid, "side": r.get("side")}
        for method in METHOD_ORDER:
            spec      = METHOD_SPECS[method]
            method_r  = results_by_method[method].get(rid, {})
            row[f"{method}_value"] = method_r.get(spec["value_key"])
            row[f"{method}_ok"]    = bool(method_r.get(spec["ok_key"], False))
        rows.append(row)
    return rows


def _format_row(row: dict, id_width: int) -> str:
    cells = [f"{row['rule_id']:<{id_width}}"]
    for method in METHOD_ORDER:
        value = row[f"{method}_value"]
        value = value if value is not None else float("nan")
        mark  = "✅" if row[f"{method}_ok"] else "❌"
        cells.append(f"{value:<10.4f}{mark:<10}")
    return "".join(cells)


def _row_header(id_width: int) -> str:
    cells = [f"{'RULE_ID':<{id_width}}"]
    for method in METHOD_ORDER:
        label     = METHOD_SPECS[method]["label"]
        value_col = "DSR" if method == "dsr" else f"{label}_p"
        cells.append(f"{value_col:<10}{label + '_OK':<10}")
    return "".join(cells)


def _print_comparison(raw_results: list, results_by_method: dict, timeframe: str) -> None:

    n_total = len(raw_results)

    ok_by_method = {method: {} for method in METHOD_ORDER}
    for method in METHOD_ORDER:
        spec  = METHOD_SPECS[method]
        by_id = results_by_method[method]
        for r in raw_results:
            rid = r["rule_id"]
            ok_by_method[method][rid] = bool(by_id.get(rid, {}).get(spec["ok_key"], False))

    n_passed = {method: sum(ok_by_method[method].values()) for method in METHOD_ORDER}

    n_pairwise_both_passed = {
        (m1, m2): sum(
            ok_by_method[m1][r["rule_id"]] and ok_by_method[m2][r["rule_id"]]
            for r in raw_results
        )
        for m1, m2 in combinations(METHOD_ORDER, 2)
    }

    n_all_passed = sum(
        all(ok_by_method[m][r["rule_id"]] for m in METHOD_ORDER)
        for r in raw_results
    )

    rows = _build_comparison_rows(raw_results, results_by_method)
    rows.sort(key=_sort_key)

    logger.info(f"\n{'─' * 90}")
    logger.info(f"  DSR vs STEPM ── {timeframe}")
    logger.info(f"{'─' * 90}")
    _print_side_breakdown(raw_results, results_by_method)
    id_width = max((len(row["rule_id"]) for row in rows), default=8) + 2
    logger.info(f"  top {min(20, len(rows))} by STEPM p-value")
    logger.info(_row_header(id_width))
    logger.info(f"{'─' * 90}")
    for row in rows[:10]:
        logger.info(_format_row(row, id_width))
    _print_correlation(rows, timeframe)
    _print_disagreement_breakdown(rows, timeframe)
    logger.info(f"{'─' * 90}")
    single_labels = [METHOD_SPECS[m]["label"] for m in METHOD_ORDER]
    pair_labels   = [
        f"{METHOD_SPECS[m1]['label']}-{METHOD_SPECS[m2]['label']} BOTH-PASS"
        for m1, m2 in n_pairwise_both_passed
    ]
    all_labels    = single_labels + pair_labels + ["ALL-PASS"]
    label_width   = max(len(label) for label in all_labels) + 2
    for method in METHOD_ORDER:
        label = METHOD_SPECS[method]["label"]
        pct   = n_passed[method] / n_total * 100.0 if n_total else 0.0
        logger.info(f"  {label:<{label_width}}── {n_passed[method]}/{n_total} passed ({pct:.2f}%)")
    for (m1, m2), n_both in n_pairwise_both_passed.items():
        label        = f"{METHOD_SPECS[m1]['label']}-{METHOD_SPECS[m2]['label']} BOTH-PASS"
        n_union_pair = n_passed[m1] + n_passed[m2] - n_both
        jaccard_pct  = n_both / n_union_pair * 100.0 if n_union_pair else 0.0
        logger.info(f"  {label:<{label_width}}── {n_both}/{n_union_pair}  │  Jaccard: {jaccard_pct:.1f}%")
    n_union_all       = len({r["rule_id"] for r in raw_results if any(ok_by_method[m][r["rule_id"]] for m in METHOD_ORDER)})
    jaccard_all_pct   = n_all_passed / n_union_all * 100.0 if n_union_all else 0.0
    logger.info(f"  {'ALL-PASS':<{label_width}}── {n_all_passed}/{n_union_all}  │  Jaccard: {jaccard_all_pct:.1f}%")
    logger.info(f"{'─' * 90}\n")


def _print_disagreement_breakdown(rows: list, timeframe: str) -> None:
    """
    rows: dicts from _build_comparison_rows, already restricted to rules where
    StepM produced a p-value. Splits into ALL PASS, ALL FAIL, and MIXED
    (methods disagree on this rule) — MIXED is where conclusions differ and
    avoids the combinatorics that per-method-pair tables would need.
    """
    all_pass, all_fail, mixed = [], [], []
    for row in rows:
        oks = tuple(row[f"{method}_ok"] for method in METHOD_ORDER)
        if all(oks):
            all_pass.append(row)
        elif not any(oks):
            all_fail.append(row)
        else:
            mixed.append(row)

    id_width = max((len(row["rule_id"]) for row in rows), default=8) + 2
    header   = _row_header(id_width)

    def _print_table(title: str, table_rows: list) -> None:
        logger.info(f"\n{'─' * 90}")
        logger.info(f"  {title} ── {timeframe} ── n={len(table_rows)}")
        logger.info(f"{'─' * 90}")
        if not table_rows:
            logger.info("  (none)")
            return
        logger.info(header)
        logger.info(f"{'─' * 90}")
        sort_key = _sort_key
        for row in sorted(table_rows, key=sort_key):
            logger.info(_format_row(row, id_width))

    _print_table(f"ALL PASS ({' + '.join(METHOD_SPECS[m]['label'] for m in METHOD_ORDER)})", all_pass)
    _print_table("MIXED ── methods disagree", mixed)


def _print_side_breakdown(raw_results: list, results_by_method: dict) -> None:
    """Counts long/short rules among those that PASSED each method."""
    counts = {method: {"long": 0, "short": 0} for method in METHOD_ORDER}

    for r in raw_results:
        rid  = r["rule_id"]
        side = r.get("side")
        if side not in ("long", "short"):
            continue
        for method in METHOD_ORDER:
            spec  = METHOD_SPECS[method]
            by_id = results_by_method[method]
            if by_id.get(rid, {}).get(spec["ok_key"], False):
                counts[method][side] += 1

    logger.info(f"  {'METHOD':<10}{'LONG':<8}{'SHORT':<8}{'TOTAL':<8}")
    for method in METHOD_ORDER:
        label            = METHOD_SPECS[method]["label"]
        long_n, short_n  = counts[method]["long"], counts[method]["short"]
        logger.info(f"  {label:<10}{long_n:<8}{short_n:<8}{long_n + short_n:<8}")


def _print_market_bias(ohlcv_arr: dict, timeframe: str) -> None:
    """Equal-weight average buy&hold return across all symbols in the universe,
    to check whether the test period itself was directionally biased."""
    returns = []
    for arr in ohlcv_arr.values():
        close = arr["close"]
        if len(close) < 2:
            continue
        returns.append((float(close[-1]) / float(close[0]) - 1.0) * 100.0)

    if not returns:
        logger.warning(f"MARKET BIAS ── {timeframe} ── no data")
        return

    avg_ret      = float(np.mean(returns))
    pct_positive = float(np.mean([r > 0 for r in returns])) * 100.0

    logger.info(f"{'─' * 70}")
    logger.info(f"  MARKET BIAS (buy&hold, equal-weight) ── {timeframe} ── n_symbols={len(returns)}")
    logger.info(f"    Avg return        : {avg_ret:+.2f}%")
    logger.info(f"    Symbols positive  : {pct_positive:.1f}%")
    logger.info(f"{'─' * 70}\n")


def _print_correlation(rows: list, timeframe: str) -> None:
    if len(rows) < 3:
        logger.warning(f"COMPARE ── {timeframe} ── not enough rules to compute correlation")
        return

    _all_pairs = [
        ("dsr_value", "stepm_value", "DSR", "STEPM_p"),
    ]
    pairs = [
        p for p in _all_pairs
        if p[0].replace("_value", "") in METHOD_ORDER and p[1].replace("_value", "") in METHOD_ORDER
    ]

    logger.info(f"{'─' * 90}")
    logger.info(f"  CROSS-METHOD CORRELATION ── {timeframe}")

    for key_a, key_b, label_a, label_b in pairs:
        valid = [row for row in rows if row[key_a] is not None and row[key_b] is not None]
        if len(valid) < 3:
            logger.info(f"  [{label_a} vs {label_b}] not enough rules (n={len(valid)})")
            continue

        arr_a = np.asarray([row[key_a] for row in valid])
        arr_b = np.asarray([row[key_b] for row in valid])

        pearson_r, pearson_p   = pearsonr(arr_a, arr_b)
        spearman_r, spearman_p = spearmanr(arr_a, arr_b)
        logger.info(f"  [{label_a} vs {label_b}]  n={len(valid)}")
        logger.info(f"    Pearson  r = {pearson_r:.4f}  (p={pearson_p:.4g})")
        logger.info(f"    Spearman r = {spearman_r:.4f}  (p={spearman_p:.4g})")

        # StepM's stepdown produces a saturated tail of p=1.0 ties by
        # construction — excluding them isolates the region where ranking
        # signal, if any, still discriminates.
        saturatable_keys = [
            key for key, label in ((key_a, label_a), (key_b, label_b))
            if label in ("STEPM_p",)
        ]
        for sat_key in saturatable_keys:
            sat_label     = label_a if sat_key == key_a else label_b
            n_saturated   = sum(1 for row in valid if row[sat_key] >= 1.0)
            pct_saturated = n_saturated / len(valid) * 100.0 if valid else 0.0
            logger.info(f"    {sat_label}=1.0 saturation : {n_saturated}/{len(valid)} ({pct_saturated:.1f}%)")

        if saturatable_keys:
            not_saturated = [
                row for row in valid
                if all(row[sat_key] < 1.0 for sat_key in saturatable_keys)
            ]
            excl_label = " & ".join(
                f"{(label_a if k == key_a else label_b)}=1.0" for k in saturatable_keys
            )
            if len(not_saturated) >= 3:
                a2 = np.asarray([row[key_a] for row in not_saturated])
                b2 = np.asarray([row[key_b] for row in not_saturated])
                pr2, pp2 = pearsonr(a2, b2)
                sr2, sp2 = spearmanr(a2, b2)
                logger.info(f"  [excluding {excl_label}]  n={len(not_saturated)}")
                logger.info(f"    Pearson  r = {pr2:.4f}  (p={pp2:.4g})")
                logger.info(f"    Spearman r = {sr2:.4f}  (p={sp2:.4g})")
            else:
                logger.info(f"  [excluding {excl_label}]  not enough rules (n={len(not_saturated)})")

    logger.info(f"{'─' * 90}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    start = time.time()

    logger.info(f"\n{'─' * 115}")
    logger.info(f"  DSR vs STEPM — FULL BRUTE UNIVERSE COMPARISON")
    logger.info(f"{'─' * 115}")
    logger.info(f"  DATASET        : {DATASET} ── {os.path.basename(DATA_FOLDER_BY_DATASET[DATASET])}")
    logger.info(f"  TIMEFRAMES     : {TIMEFRAMES}")
    logger.info(f"  SYMBOLS_BY_TF  : {{tf: len(s) for tf, s in SYMBOLS_BY_TIMEFRAME.items()}}")
    logger.debug(f"  MAX_DEPTH      : {RULE_MAX_DEPTH}")
    logger.info(f"  PARAM_GRID     : {PARAM_GRID}")
    logger.info(f"  DSR_TH         : {DSR_TH}")
    logger.info(f"  STEPM_ALPHA    : {STEPM_ALPHA}")
    logger.info(f"  STEPM_PVALUE_TH: {WHITE_PVALUE_TH}")
    logger.info(f"  N_BOOTSTRAP    : {WHITE_N_BOOTSTRAP}")
    logger.info(f"  BLOCK_SIZE     : {WHITE_BLOCK_SIZE}")
    logger.info(f"  K_ESIME        : {STEPM_K_ESIME}")
    logger.info(f"  SIGNAL_CLEANING: {SIGNAL_CLEANING_TEST}")
    logger.info(f"{'─' * 115}\n")

    # -------------------------------------------------------------------
    # DATA LOADING — cheap, sequential across timeframes.
    # -------------------------------------------------------------------
    ohlcv_data_by_timeframe = build_universe(
        DATA_FOLDER_BY_DATASET[DATASET], SYMBOLS_BY_TIMEFRAME,
        dataset=DATASET,
    )
    ohlcv_arr_by_timeframe  = {
        timeframe: prepare_ohlcv_arrays(ohlcv_is)
        for timeframe, ohlcv_is in ohlcv_data_by_timeframe.items()
    }

    # -------------------------------------------------------------------
    # DSR vs STEPM — one comparison per timeframe, on the full brute
    # -------------------------------------------------------------------
    comparisons_by_timeframe = {}

    for timeframe in TIMEFRAMES:
        _print_market_bias(ohlcv_arr_by_timeframe[timeframe], timeframe)

        rules_for_timeframe = _build_rule_dicts(
            ohlcv_data_by_timeframe[timeframe], timeframe, RULE_MAX_DEPTH,
        )

        raw_results, n_combos, matrix_arr, col_names = _run_backtest_universe(
            rules                  = rules_for_timeframe,
            ohlcv_arr              = ohlcv_arr_by_timeframe[timeframe],
            param_grid              = PARAM_GRID,
            order_amount           = ORDER_AMOUNT,
            timeframe               = timeframe,
            n_jobs                  = N_JOBS,
            apply_signal_cleaning   = SIGNAL_CLEANING_TEST,
        )

        comparisons_by_timeframe[timeframe] = compare_dsr_vs_stepm_from_raw(
            raw_results     = raw_results,
            matrix_arr      = matrix_arr,
            col_names       = col_names,
            dsr_th          = DSR_TH,
            n_combos        = n_combos,
            stepm_k_esime   = STEPM_K_ESIME,
            timeframe       = timeframe,
            n_jobs          = N_JOBS,
        )

    elapsed = int(time.time() - start)
    logger.info(f"\n🏁 TOTAL — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")