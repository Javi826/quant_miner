# core/runs/run_deploy.py
import itertools
import logging
import numpy as np
import pandas as pd
from functools import partial
from joblib import Parallel, delayed
from pipeline.wfo import EMA_ALPHA, wfo_window_lengths, _evaluate_fn
from engines.wfo_WF import WARMUP_BARS, update_ema_state, round_params_dict, slice_window_arrays
from utils.ohlcv_utils import prepare_ohlcv_arrays
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
    ohlcv_oos: dict,
    timeframe: str,
    param_grid: dict,
    signal_fn: callable,
    order_amount: int,
    n_jobs: int,
) -> tuple:

    length_train_set, _, length_test = wfo_window_lengths(timeframe)
    ohlcv_arr  = prepare_ohlcv_arrays(ohlcv_oos)
    ref_sym    = max(ohlcv_arr.keys(), key=lambda k: len(ohlcv_arr[k]["ts"]))
    ref_ts     = ohlcv_arr[ref_sym]["ts"]
    max_length = len(ref_ts)

    param_names       = list(param_grid.keys())
    lists_for_grid    = [param_grid[k] for k in param_names]
    param_ranges      = dict(zip(param_names, lists_for_grid))
    dict_combinations = [dict(zip(param_names, comb)) for comb in itertools.product(*lists_for_grid)]

    evaluate_fn = partial(
        _evaluate_fn,
        signal_fn          = signal_fn,
        signal_params_keys = [],
        order_amount       = order_amount,
        _signal_cache      = {},
        _prepared_cache    = {},
    )

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
            warm_start       = max(0, t0 - WARMUP_BARS)
            base_arrays[sym] = slice_window_arrays(ohlcv_arr[sym], warm_start, t1)

        results = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_fn)(p, base_arrays, train_start_ts) for p in dict_combinations
        )
        _, raw_best_params = max(results, key=lambda x: x[0])

        ema_raw = update_ema_state(ema_raw, raw_best_params, alpha=EMA_ALPHA)

        present_train_start, present_train_end = train_start_ts, train_end_ts
        deploy_symbols = sorted(selected.keys())

        logger.debug(
            f"DEPLOY EMA ── window [{pd.Timestamp(train_start_ts).date()} .. {pd.Timestamp(train_end_ts).date()}] "
            f"| symbols={len(deploy_symbols)} "
            f"| raw_best={raw_best_params} "
            f"| ema_state={round_params_dict(ema_raw, param_ranges)}"
        )

    if ema_raw is None:
        raise ValueError("No deploy window produced a valid train optimum (no symbols available in any window)")

    deploy_params = round_params_dict(ema_raw, param_ranges)
    return deploy_params, deploy_symbols, present_train_start, present_train_end
