# darwinex/BOT_batch/compare_backtest_runners.py
import os
import sys
import math
import time
import pstats
import cProfile
import logging
import importlib
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))

import numpy as np
import pandas as pd
from main_BACKT_fx import build_combos, load_ohlcv_by_combo, PARAM_GRID_BY_TIMEFRAME, DATASET_IS
from setup.config_paths import DATA_FOLDER_BY_DATASET
from setup.config_backtest import ORDER_AMOUNT
from rule_mining.rule_generator import MAX_DEPTH
from rule_mining.rule_runner import _build_rule_dicts
from pipeline.signal_cleaning import pipe_signal_cleaning_jaccard
from pipeline.backtest_runner import _combo_grid
from pipeline import backtest_runner as runner_new
from setup.config_core import settings

REFERENCE_SUFFIX = "_OLD"   # frozen reference: pipeline/backtest_runner_OLD.py + backtesters/ZX_compute_BT_<MODE>_OLD

runner_old = importlib.import_module(f"pipeline.backtest_runner{REFERENCE_SUFFIX}")
bt_old     = importlib.import_module(f"backtesters.ZX_compute_BT_{settings.BACKTEST_MODE}{REFERENCE_SUFFIX}")
bt_new     = importlib.import_module(f"backtesters.ZX_compute_BT_{settings.BACKTEST_MODE}")

logger = logging.getLogger("BOT_batch.compare_backtest_runners")
logger.setLevel(logging.INFO)

# =============================================================================
# RUN CONFIG
# =============================================================================
RUN_MODE      = "single"                        # "compare" | "single" | "profile"
COMBOS_TO_RUN = ["1H_c01", "1H_c02", "4H_c01"]   # None = all combos from main_backt_fx
MAX_RULES     = None                             # None = all rules after Jaccard cleaning
MAX_REPORTED  = 10

# single mode: run_backtest_from_prepared / _light (WFO path) on sampled rules, windows and every param combo
SINGLE_MAX_RULES = 200
SINGLE_WINDOWS   = ((0.0, 1.0), (0.0, 0.5), (0.3, 0.7), (0.75, 1.0))   # fractions of the data period

# profile mode: new runner only, evenly sampled rules
# n_jobs=1 profiles the per-rule work; n_jobs=-1 with all rules profiles the fixed main-process part
PROFILE_COMBO     = "1H_c01"
PROFILE_MAX_RULES = None
PROFILE_N_JOBS    = -1
PROFILE_TOP_N     = 30


def _prepare_rules(combo: dict, max_rules: int | None) -> tuple:
    ohlcv_data_by_combo, ohlcv_arr_by_combo = load_ohlcv_by_combo([combo], DATA_FOLDER_BY_DATASET[DATASET_IS], DATASET_IS)
    combo_key, timeframe = combo["combo_key"], combo["timeframe"]
    ohlcv_arr = ohlcv_arr_by_combo[combo_key]
    rules = _build_rule_dicts(ohlcv_data_by_combo[combo_key], combo_key, timeframe, MAX_DEPTH)
    rules = pipe_signal_cleaning_jaccard(rules=rules, ohlcv_arr=ohlcv_arr, timeframe=timeframe)
    if max_rules is not None:
        rules = rules[:max_rules]
    return rules, ohlcv_arr


def _run_runner(runner, rules: list, ohlcv_arr: dict, timeframe: str) -> tuple:
    start  = time.perf_counter()
    output = runner.pipe_backtesting(
        rules        = rules,
        ohlcv_arr    = ohlcv_arr,
        param_grid   = PARAM_GRID_BY_TIMEFRAME[timeframe],
        order_amount = ORDER_AMOUNT,
        timeframe    = timeframe,
    )
    return output, time.perf_counter() - start


def _values_equal(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_values_equal(a[k], b[k]) for k in a)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a_arr, b_arr = np.asarray(a), np.asarray(b)
        if a_arr.shape != b_arr.shape or a_arr.dtype != b_arr.dtype:
            return False
        return bool(np.array_equal(a_arr, b_arr, equal_nan=a_arr.dtype.kind in "fc"))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_values_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
        return (math.isnan(a) and math.isnan(b)) or a == b
    return bool(a == b)


def _bitwise_equal(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    a_c, b_c = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return bool(np.array_equal(a_c.view(np.uint8), b_c.view(np.uint8)))


def _compare_outputs(old: tuple, new: tuple) -> list:
    old_results, old_n_combos, old_matrix, old_cols = old
    new_results, new_n_combos, new_matrix, new_cols = new
    issues = []

    if old_n_combos != new_n_combos:
        issues.append(f"n_combos: {old_n_combos} vs {new_n_combos}")

    if old_cols != new_cols:
        first = next((i for i, (a, b) in enumerate(zip(old_cols, new_cols)) if a != b), min(len(old_cols), len(new_cols)))
        issues.append(f"col_names: len {len(old_cols)} vs {len(new_cols)}, first mismatch at index {first}")

    if old_matrix.shape != new_matrix.shape or old_matrix.dtype != new_matrix.dtype:
        issues.append(f"matrix: {old_matrix.shape} {old_matrix.dtype} vs {new_matrix.shape} {new_matrix.dtype}")
    elif not _bitwise_equal(old_matrix, new_matrix):
        diff_cols = np.flatnonzero(np.any(old_matrix != new_matrix, axis=0))
        names     = [old_cols[i] for i in diff_cols[:MAX_REPORTED].tolist()] if old_cols == new_cols else diff_cols[:MAX_REPORTED].tolist()
        issues.append(f"matrix: {diff_cols.size} columns differ (bitwise), e.g. {names}")

    if len(old_results) != len(new_results):
        issues.append(f"raw_results: len {len(old_results)} vs {len(new_results)}")
        return issues

    n_bad = 0
    for idx, (old_row, new_row) in enumerate(zip(old_results, new_results)):
        keys     = old_row.keys() | new_row.keys()
        bad_keys = sorted(k for k in keys if k not in old_row or k not in new_row or not _values_equal(old_row[k], new_row[k]))
        if not bad_keys:
            continue
        n_bad += 1
        if n_bad <= MAX_REPORTED:
            details = ", ".join(f"{k}: {old_row.get(k)!r} vs {new_row.get(k)!r}" for k in bad_keys)
            issues.append(f"rule {old_row.get('rule_id', idx)} -> {details}")
    if n_bad:
        issues.append(f"raw_results: {n_bad}/{len(old_results)} rules differ")

    return issues


def _select_combos(combo_keys: list | None) -> list:
    combos = build_combos()
    if not combo_keys:
        return combos
    wanted  = set(combo_keys)
    combos  = [c for c in combos if c["combo_key"] in wanted]
    missing = wanted - {c["combo_key"] for c in combos}
    if missing:
        raise ValueError(f"Unknown combo keys: {sorted(missing)}")
    return combos


def _sample_rules(rules: list, max_rules: int | None) -> list:
    if max_rules is None or len(rules) <= max_rules:
        return rules
    idx = np.linspace(0, len(rules) - 1, max_rules).astype(np.int64)
    return [rules[i] for i in idx.tolist()]


def _run_compare() -> bool:
    combos = _select_combos(COMBOS_TO_RUN)

    total_old, total_new, n_identical = 0.0, 0.0, 0
    for combo in combos:
        combo_key, timeframe = combo["combo_key"], combo["timeframe"]
        rules, ohlcv_arr = _prepare_rules(combo, MAX_RULES)

        old_output, old_s = _run_runner(runner_old, rules, ohlcv_arr, timeframe)
        new_output, new_s = _run_runner(runner_new, rules, ohlcv_arr, timeframe)
        issues = _compare_outputs(old_output, new_output)
        del old_output, new_output

        total_old += old_s
        total_new += new_s
        n_identical += not issues
        speedup = old_s / new_s if new_s > 0 else math.inf
        status  = "IDENTICAL" if not issues else "DIFFERENT"
        logger.info(f"{combo_key:<10} rules={len(rules):<7} {status:<9} old {old_s:8.2f} s | new {new_s:8.2f} s | x{speedup:.2f}")
        for issue in issues:
            logger.info(f"    {issue}")

    total_speedup = total_old / total_new if total_new > 0 else math.inf
    logger.info(f"TOTAL      {n_identical}/{len(combos)} identical | old {total_old:.2f} s | new {total_new:.2f} s | x{total_speedup:.2f}")
    return n_identical == len(combos)


def _slice_window(ohlcv_arr: dict, start_frac: float, end_frac: float) -> dict:
    all_ts   = np.concatenate([arr["ts"] for arr in ohlcv_arr.values()])
    first    = all_ts.min().astype(np.int64)
    span     = all_ts.max().astype(np.int64) - first
    start_ts = np.int64(first + span * start_frac).astype("datetime64[ns]")
    end_ts   = np.int64(first + span * end_frac).astype("datetime64[ns]")
    window   = {}
    for sym, arr in ohlcv_arr.items():
        lo = int(np.searchsorted(arr["ts"], start_ts, side="left"))
        hi = int(np.searchsorted(arr["ts"], end_ts, side="right"))
        window[sym] = {key: values[lo:hi] for key, values in arr.items()}
    return window


def _frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if list(a.columns) != list(b.columns) or len(a) != len(b) or not a.index.equals(b.index):
        return False
    for column in a.columns:
        x, y = a[column].to_numpy(), b[column].to_numpy()
        if x.dtype != y.dtype:
            return False
        if x.dtype.kind == "f":
            if not _bitwise_equal(x, y):
                return False
        elif not np.array_equal(x, y):
            return False
    return True


def _single_outputs_equal(old_prepared, new_prepared, params: dict) -> bool:
    args = (params["SELL_AFTER"], params["TP_PCT"], params["SL_PCT"], ORDER_AMOUNT)
    old_full  = bt_old.run_backtest_from_prepared(old_prepared, *args)["__PORTFOLIO__"]
    new_full  = bt_new.run_backtest_from_prepared(new_prepared, *args)["__PORTFOLIO__"]
    old_light = bt_old.run_backtest_from_prepared_light(old_prepared, *args)["__PORTFOLIO__"]
    new_light = bt_new.run_backtest_from_prepared_light(new_prepared, *args)["__PORTFOLIO__"]
    return (
        old_full.keys() == new_full.keys() and old_light.keys() == new_light.keys()
        and old_full["trades"] == new_full["trades"]
        and _frames_equal(old_full["trade_log"], new_full["trade_log"])
        and _frames_equal(old_light["trade_log"], new_light["trade_log"])
    )


def _run_single() -> bool:
    combos      = _select_combos(COMBOS_TO_RUN)
    n_identical = 0
    for combo in combos:
        combo_key, timeframe = combo["combo_key"], combo["timeframe"]
        rules, ohlcv_arr = _prepare_rules(combo, None)
        rules  = _sample_rules(rules, SINGLE_MAX_RULES)
        params = _combo_grid(PARAM_GRID_BY_TIMEFRAME[timeframe])

        n_cases, bad_cases, old_s, new_s = 0, [], 0.0, 0.0
        for start_frac, end_frac in SINGLE_WINDOWS:
            window = _slice_window(ohlcv_arr, start_frac, end_frac)
            for rule in rules:
                ohlcv_arrays = {
                    sym: {**arr, "signal": np.asarray(rule["signal_fn"](arr, live_trading=False), dtype=np.float32)}
                    for sym, arr in window.items()
                }
                start = time.perf_counter()
                old_prepared = bt_old.prepare_backtest_data(ohlcv_arrays)
                mid = time.perf_counter()
                new_prepared = bt_new.prepare_backtest_data(ohlcv_arrays)
                old_s += mid - start
                new_s += time.perf_counter() - mid
                for p in params:
                    n_cases += 1
                    if not _single_outputs_equal(old_prepared, new_prepared, p):
                        bad_cases.append((rule["rule_id"], (start_frac, end_frac), p))

        identical = not bad_cases
        n_identical += identical
        logger.info(f"{combo_key:<10} single cases={n_cases:<7} {'IDENTICAL' if identical else 'DIFFERENT'} | prepare old {old_s:.2f} s new {new_s:.2f} s")
        for rule_id, window_frac, p in bad_cases[:MAX_REPORTED]:
            logger.info(f"    rule {rule_id} window {window_frac} params {p}")
        if bad_cases:
            logger.info(f"    {len(bad_cases)}/{n_cases} cases differ")

    logger.info(f"TOTAL      {n_identical}/{len(combos)} identical (single-backtest API)")
    return n_identical == len(combos)


def _profiled_entry_points() -> dict:
    # Cython functions are invisible to cProfile: Python wrappers give them their own entry.
    backtest_grid     = runner_new.backtest_grid
    build_rule_events = runner_new.build_rule_events

    def cy_backtest_grid(*args, **kwargs):
        return backtest_grid(*args, **kwargs)

    def cy_build_rule_events(*args, **kwargs):
        return build_rule_events(*args, **kwargs)

    return {"backtest_grid": cy_backtest_grid, "build_rule_events": cy_build_rule_events}


def _run_profile() -> None:
    combo = _select_combos([PROFILE_COMBO])[0]
    rules, ohlcv_arr = _prepare_rules(combo, None)
    rules = _sample_rules(rules, PROFILE_MAX_RULES)

    patches  = {"BACKTEST_N_JOBS": PROFILE_N_JOBS, **_profiled_entry_points()}
    backup   = {name: getattr(runner_new, name) for name in patches}
    profiler = cProfile.Profile()
    try:
        for name, value in patches.items():
            setattr(runner_new, name, value)
        profiler.enable()
        _, elapsed = _run_runner(runner_new, rules, ohlcv_arr, combo["timeframe"])
    finally:
        profiler.disable()
        for name, value in backup.items():
            setattr(runner_new, name, value)

    logger.info(f"PROFILE    {combo['combo_key']} rules={len(rules)} n_jobs={PROFILE_N_JOBS} | {elapsed:.2f} s")
    stats = pstats.Stats(profiler, stream=sys.stdout)
    stats.sort_stats("tottime").print_stats(PROFILE_TOP_N)
    stats.sort_stats("cumulative").print_stats(PROFILE_TOP_N)


def main() -> None:
    if RUN_MODE == "compare":
        _run_compare()
    elif RUN_MODE == "single":
        _run_single()
    elif RUN_MODE == "profile":
        _run_profile()
    else:
        raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}")


if __name__ == "__main__":
    main()