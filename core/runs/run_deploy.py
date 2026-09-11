# core/runs/run_deploy.py
import itertools
import logging
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import importlib
from setup.config_core import settings
_bt = importlib.import_module(f"backtesters.ZX_compute_BT_{settings.BACKTEST_MODE}")
run_grid_backtest = _bt.run_grid_backtest
from pipeline.wfo import EMA_ALPHA, WFO_WINDOW_CONFIG, build_ohlcv_with_signal, compute_metric
from engines.wfo_WF import WARMUP_BARS, MIN_COOLDOWN_BARS, update_ema_state, round_params_dict
from utils.ohlcv_utils import prepare_ohlcv_arrays, get_bars_per_year
from setup.config_core import settings
logger = logging.getLogger("BOT_batch.runs.run_deploy")

# =============================================================================
# PRIVATE HELPERS — 
# =============================================================================

def _backward_frompresent_window_bounds(max_length: int, length_train_set: int, length_test: int) -> list:

    windows       = []
    train_end_idx = max_length - 1
    while True:
        train_start_idx = train_end_idx - length_train_set + 1
        if train_start_idx < 0:
            break
        windows.append((train_start_idx, train_end_idx))
        train_end_idx -= length_test
    return windows

def run_wfo_deploy_ema(
    ohlcv_is: dict,
    timeframe: str,
    param_grid: dict,
    signal_fn: callable,
    order_amount: int,
    n_jobs: int,
) -> tuple:

    _wfo_cfg = WFO_WINDOW_CONFIG.get(timeframe)
    if _wfo_cfg is None:
        raise ValueError(f"No WFO window config for timeframe: {timeframe}")

    bars_per_month   = get_bars_per_year(timeframe) / 12
    length_train_set = int(_wfo_cfg["train_months"] * bars_per_month)
    pct_train_set    = _wfo_cfg["train_months"] / (_wfo_cfg["train_months"] + _wfo_cfg["test_months"])
    length_test      = int(length_train_set / pct_train_set - length_train_set)

    ohlcv_arr  = prepare_ohlcv_arrays(ohlcv_is)
    ref_sym    = max(ohlcv_arr.keys(), key=lambda k: len(ohlcv_arr[k]["ts"]))
    ref_ts     = ohlcv_arr[ref_sym]["ts"]
    max_length = len(ref_ts)

    param_names       = list(param_grid.keys())
    lists_for_grid    = [param_grid[k] for k in param_names]
    param_ranges      = dict(zip(param_names, lists_for_grid))
    dict_combinations = [dict(zip(param_names, comb)) for comb in itertools.product(*lists_for_grid)]

    EDGE_BUFFER_BARS = max(param_ranges.get("SELL_AFTER", [WARMUP_BARS]) + [MIN_COOLDOWN_BARS])


    def _evaluate(params, base_arrays, train_start_ts, train_edge_ts):
        arrays      = build_ohlcv_with_signal(base_arrays, signal_fn, [], params)
        results     = run_grid_backtest(
            arrays,
            sell_after   = params["SELL_AFTER"],
            tp_pct       = params["TP_PCT"],
            sl_pct       = params["SL_PCT"],
            order_amount = order_amount,
        )
        trade_log   = results["__PORTFOLIO__"]["trade_log"]
        n_before    = len(trade_log)
        if not trade_log.empty:
            truncated_mask = (
                trade_log["exit_reason"].isin(["SELL_AFTER", "END_OF_DATA"]) &
                (trade_log["buy_time"] >= pd.Timestamp(train_start_ts)) &
                (trade_log["buy_time"] > pd.Timestamp(train_edge_ts))
            )
            trade_log = trade_log[~truncated_mask]
            results   = {"__PORTFOLIO__": {"trade_log": trade_log}}
        n_after = len(trade_log)
        return compute_metric(results), params, n_before, n_after

    windows = _backward_frompresent_window_bounds(max_length, length_train_set, length_test)
    if not windows:
        raise ValueError("Not enough history to build even one deploy train window")

    ema_raw            = None
    deploy_symbols      = []
    present_train_start = None
    present_train_end   = None

    # OLDEST -> NEWEST, so the EMA chain is built in correct chronological order.
    for train_start_idx, train_end_idx in reversed(windows):
        train_start_ts = ref_ts[train_start_idx]
        train_end_ts   = ref_ts[train_end_idx]
        train_edge_ts  = ref_ts[max(train_start_idx, train_end_idx - EDGE_BUFFER_BARS)]

        candidate_indices = {}
        for sym, arr_dict in ohlcv_arr.items():
            sym_ts = arr_dict["ts"]
            if sym_ts[0] > train_start_ts or sym_ts[-1] < train_end_ts:
                continue
            t0 = int(np.searchsorted(sym_ts, train_start_ts, side="left"))
            t1 = int(np.searchsorted(sym_ts, train_end_ts,   side="right"))
            if t1 > t0:
                candidate_indices[sym] = (t0, t1, t0, t1)

        selected = candidate_indices
        if not selected:
            logger.debug(f"DEPLOY EMA ── window [{train_start_ts}..{train_end_ts}] skipped: no symbols available")
            continue

        base_arrays = {}
        for sym, (t0, t1, _, _) in selected.items():
            arr_dict   = ohlcv_arr[sym]
            warm_start = max(0, t0 - WARMUP_BARS)
            base_arrays[sym] = {
                "ts":        arr_dict["ts"][warm_start:t1],
                "open":      arr_dict["open"][warm_start:t1],
                "high":      arr_dict["high"][warm_start:t1],
                "low":       arr_dict["low"][warm_start:t1],
                "close":     arr_dict["close"][warm_start:t1],
                "volume":    arr_dict.get("volume", arr_dict["close"] * 0)[warm_start:t1],
                "low_time":  arr_dict["low_time"][warm_start:t1],
                "high_time": arr_dict["high_time"][warm_start:t1],
            }

        results = Parallel(n_jobs=n_jobs)(
            delayed(_evaluate)(p, base_arrays, train_start_ts, train_edge_ts) for p in dict_combinations
        )
        _, raw_best_params, best_n_before, best_n_after = max(results, key=lambda x: x[0])

        ema_raw = update_ema_state(ema_raw, raw_best_params, alpha=EMA_ALPHA)

        present_train_start, present_train_end = train_start_ts, train_end_ts
        deploy_symbols = sorted(selected.keys())

        logger.debug(
            f"DEPLOY EMA ── window [{pd.Timestamp(train_start_ts).date()} .. {pd.Timestamp(train_end_ts).date()}] "
            f"| symbols={len(deploy_symbols)} "
            f"| trades={best_n_before}->{best_n_after} "
            f"| raw_best={raw_best_params} "
            f"| ema_state={round_params_dict(ema_raw, param_ranges)}"
        )

    if ema_raw is None:
        raise ValueError("No deploy window produced a valid train optimum (no symbols available in any window)")

    deploy_params = round_params_dict(ema_raw, param_ranges)
    return deploy_params, deploy_symbols, present_train_start, present_train_end
