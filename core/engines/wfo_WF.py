# core/engines/wfo_WF.py
import logging
import contextlib
import numpy as np
import pandas as pd
import itertools
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from utils.paralelization import arrays_to_shared_memory, arrays_from_shared_memory
from setup.config_backtest import INITIAL_BALANCE
from utils.batch_metrics import compute_metrics
logger = logging.getLogger("BOT_batch.engines.wfo_WF")

WARMUP_BARS       = 100
MIN_COOLDOWN_BARS = 100

# =============================================================================
# PARAM ROUNDING HELPERS
# =============================================================================

def _decimals_for_values(values) -> int:

    max_decimals = 0
    for v in values:
        s = f"{float(v):.10f}".rstrip("0")
        if "." in s:
            max_decimals = max(max_decimals, len(s.split(".")[1]))
    return max_decimals

def _snap_to_grid(value: float, grid_values: list):

    return min(grid_values, key=lambda g: abs(g - value))


def round_params_dict(params: dict, param_ranges: dict) -> dict:
    return {
        k: _snap_to_grid(v, param_ranges[k])
        for k, v in params.items()
    }
# =============================================================================
# EMA STATE — running exponential moving average of per-window optimal params
# =============================================================================

def update_ema_state(ema_raw: dict | None, new_best: dict, alpha: float) -> dict:

    if ema_raw is None:
        return dict(new_best)
    return {k: alpha * new_best[k] + (1.0 - alpha) * ema_raw[k] for k in new_best}

# =============================================================================
# WINDOW SYMBOL SELECTION
# =============================================================================
def _find_window_indices(
    sym_ts: np.ndarray,
    train_start_ts,
    test_start_ts,
    test_end_ts,
) -> tuple | None:

    if sym_ts[0] > train_start_ts or sym_ts[-1] < test_end_ts:
        return None

    t0    = int(np.searchsorted(sym_ts, train_start_ts, side="left"))
    t1    = int(np.searchsorted(sym_ts, test_start_ts,  side="left"))
    test0 = t1
    test1 = int(np.searchsorted(sym_ts, test_end_ts,    side="right"))

    if t1 <= t0 or test1 <= test0:
        return None
    return t0, t1, test0, test1

def _evaluate_with_shm(params: dict, shm_metadata: dict, evaluate_fn, train_start_ts) -> tuple:

    base_arrays, shm_handles = arrays_from_shared_memory(shm_metadata)
    try:
        return evaluate_fn(params, base_arrays, train_start_ts)
    finally:
        for shm in shm_handles:
            shm.close()
# =============================================================================
# WINDOW ARRAY SLICING
# =============================================================================
def slice_window_arrays(arr_dict: dict, start: int, end: int) -> dict:

    return {
        "ts":        arr_dict["ts"][start:end],
        "open":      arr_dict["open"][start:end],
        "high":      arr_dict["high"][start:end],
        "low":       arr_dict["low"][start:end],
        "close":     arr_dict["close"][start:end],
        "volume":    arr_dict.get("volume", arr_dict["close"] * 0)[start:end],
        "low_time":  arr_dict["low_time"][start:end],
        "high_time": arr_dict["high_time"][start:end],
    }

        
# =============================================================================
# WALK FORWARD OPTIMIZATION
# =============================================================================
def walk_forward_optimization(
    ohlcv_arr,
    param_ranges,
    length_train_set,
    pct_train_set,
    anchored,
    evaluate_fn,
    ema_alpha,
    n_jobs=-1,
    show_progress=False,
    collect_train_trades_fn=None,
    collect_test_trades_fn=None,
    inherit_entry_block=False,
):
    if evaluate_fn is None:
        raise ValueError("You must pass an evaluate_fn(params, base_arrays) function")

    COOLDOWN_BARS    = max(param_ranges.get("SELL_AFTER", [WARMUP_BARS]) + [MIN_COOLDOWN_BARS])

    keys               = list(param_ranges.keys())
    all_combinations   = list(itertools.product(*[param_ranges[k] for k in keys]))
    dict_combinations  = [dict(zip(keys, comb)) for comb in all_combinations]

    length_test        = int(length_train_set / pct_train_set - length_train_set)
    best_params_list   = []
    best_criteria_list = []
    window_idx         = 1

    train_start_dates  = []
    train_end_dates    = []
    test_start_dates   = []
    test_end_dates     = []
    train_symbols_list = []
    test_symbols_list  = []

    # Trade accumulators per window
    train_trades_list   = []
    test_trades_list    = []
    test_n_trades_list  = []
    train_criteria_list = [] 

    ema_raw        = None
    prev_last_exit = None  # sell_time of last kept test trade of previous window (NPY only)

    ref_sym    = max(ohlcv_arr.keys(), key=lambda k: len(ohlcv_arr[k]['ts']))
    ref_ts     = ohlcv_arr[ref_sym]['ts']
    max_length = len(ref_ts)
    
    offset            = 0 if anchored else (max_length - length_train_set) % length_test
    start             = offset
    end               = offset + length_train_set
    last_test_end_ref = end + length_test

    while start < max_length:
        remaining_data = max_length - (end if anchored else start)
        is_last_window = remaining_data < (length_train_set + length_test)

        # -----------------------------------------------------------------
        # Define window date boundaries from ref_sym
        # -----------------------------------------------------------------
        if is_last_window:
            remaining_from_last = max_length - last_test_end_ref
            if remaining_from_last < int(length_test * 0.5):
                break
            t0_ref    = 0 if anchored else start
            t1_ref    = last_test_end_ref
            test0_ref = t1_ref
            test1_ref = max_length
        else:
            t0_ref    = 0 if anchored else start
            t1_ref    = end if anchored else start + length_train_set
            test0_ref = t1_ref
            test1_ref = min(t1_ref + length_test, max_length)
            last_test_end_ref = test1_ref

        train_start_ts = ref_ts[t0_ref]
        test_start_ts  = ref_ts[t1_ref] if t1_ref < max_length else ref_ts[-1]
        test_end_ts    = ref_ts[test1_ref - 1]

        # -----------------------------------------------------------------
        # Collect all valid candidates using date-aligned indices
        # -----------------------------------------------------------------
        candidate_indices = {}
        for sym, arr_dict in ohlcv_arr.items():
            indices = _find_window_indices(
                arr_dict["ts"], train_start_ts, test_start_ts, test_end_ts
            )
            if indices is not None:
                candidate_indices[sym] = indices

        # -----------------------------------------------------------------
        # Every symbol with valid data for this window is used — the symbol
        # -----------------------------------------------------------------
        selected_indices = candidate_indices

        train_indices = {sym: (t0, t1)       for sym, (t0, t1, _,     _    ) in selected_indices.items()}
        test_indices  = {sym: (test0, test1) for sym, (_,  _,  test0, test1) in selected_indices.items()}

        train_symbols_list.append(sorted(train_indices.keys()))
        test_symbols_list.append(sorted(test_indices.keys()))

        if not train_indices:
            break

        if ref_sym in selected_indices:
            t0, t1, test0, test1 = selected_indices[ref_sym]
        else:
            t0, t1, test0, test1 = t0_ref, t1_ref, test0_ref, test1_ref

        # -----------------------------------------------------------
        # Prepare base arrays (train + test) with warmup prefix
        # -----------------------------------------------------------
        base_arrays = {}
        for sym, (t0_sym, t1_sym) in train_indices.items():
            warm_start       = max(0, t0_sym - WARMUP_BARS)
            base_arrays[sym] = slice_window_arrays(ohlcv_arr[sym], warm_start, t1_sym)

        base_arrays_test = {}
        for sym, (t0_sym, t1_sym) in test_indices.items():
            arr_dict              = ohlcv_arr[sym]
            warm_start            = max(0, t0_sym - WARMUP_BARS)
            cool_end              = min(len(arr_dict["ts"]), t1_sym + COOLDOWN_BARS + 1)
            base_arrays_test[sym] = slice_window_arrays(arr_dict, warm_start, cool_end)
        # -----------------------------------------------------------
        # Parallel evaluation via shared memory
        # -----------------------------------------------------------
        if n_jobs == 1:
            results = [
                evaluate_fn(params, base_arrays, train_start_ts)
                for params in dict_combinations
            ]
        else:
            shm_list, shm_metadata = arrays_to_shared_memory(base_arrays)
            try:
                with (tqdm_joblib(
                    tqdm(desc=f"🔁 WFO Window {window_idx}", total=len(dict_combinations), dynamic_ncols=True)
                ) if show_progress else contextlib.nullcontext()):
                    results = Parallel(n_jobs=n_jobs)(
                        delayed(_evaluate_with_shm)(params, shm_metadata, evaluate_fn, train_start_ts)
                        for params in dict_combinations
                    )
            finally:
                for shm in shm_list:
                    shm.close()
                    shm.unlink()

        # -----------------------------------------------------------
        # Select best result (on train)
        # -----------------------------------------------------------
        _, raw_best_params = max(results, key=lambda x: x[0])

        ema_raw          = update_ema_state(ema_raw, raw_best_params, alpha=ema_alpha)
        effective_params = round_params_dict(ema_raw, param_ranges)

        best_params_list.append(effective_params)

        window_test_n_trades = 0
        test_criterion       = np.nan
        train_criterion      = np.nan

        df_train = None
        if collect_train_trades_fn is not None and base_arrays:
            df_train = collect_train_trades_fn(effective_params, base_arrays)
            if df_train is not None and not df_train.empty:
                n_before = len(df_train)

                truncated_mask    = df_train["exit_reason"] == "END_OF_DATA"
                below_warmup_mask = df_train["buy_time"] < pd.Timestamp(train_start_ts)
                df_train = df_train[~truncated_mask & ~below_warmup_mask].copy()
                n_after = len(df_train)
                #logger.debug(f"WFO window {window_idx} ── train trades={n_before}->{n_after}")

        df_test = None
        if collect_test_trades_fn is not None and base_arrays_test:
            entry_from = pd.Timestamp(test_start_ts)
            if inherit_entry_block and prev_last_exit is not None:
                entry_from = max(entry_from, prev_last_exit)
            df_test = collect_test_trades_fn(effective_params, base_arrays_test, entry_from_ts=entry_from)
            if df_test is not None and not df_test.empty:
                df_test = df_test[
                    (df_test["buy_time"] >= pd.Timestamp(test_start_ts)) &
                    (df_test["buy_time"] <= pd.Timestamp(test_end_ts)) &
                    (df_test["exit_reason"] != "END_OF_DATA")
                ].copy()
        prev_last_exit = None

        train_has_trades = df_train is not None and not df_train.empty
        test_has_trades  = df_test is not None and not df_test.empty

        if train_has_trades and test_has_trades:
            df_train["wfo_window"] = window_idx
            train_trades_list.append(df_train)

            df_test["wfo_window"] = window_idx
            test_trades_list.append(df_test)
            prev_last_exit = df_test["sell_time"].max()
            window_test_n_trades = len(df_test)
            test_criterion       = float(df_test["profit"].sum()) / INITIAL_BALANCE * 100

            m_train         = compute_metrics(df_train, capital=INITIAL_BALANCE, name="", include_weekly=False)
            train_criterion = m_train["Net_Gain_pct"]
        else:
            logger.debug(
                f"WFO window {window_idx} ── dropped from train/test WFR pool "
                f"(train_has_trades={train_has_trades}, test_has_trades={test_has_trades})"
            )

        best_criteria_list.append(test_criterion)
        test_n_trades_list.append(window_test_n_trades)
        train_criteria_list.append(train_criterion)

        train_start_dates.append(ref_ts[t0]       if t0       < len(ref_ts) else None)
        train_end_dates.append(ref_ts[t1 - 1]     if t1 - 1   < len(ref_ts) else None)
        test_start_dates.append(ref_ts[test0]     if test0    < len(ref_ts) else None)
        test_end_dates.append(ref_ts[test1 - 1]   if test1 - 1 < len(ref_ts) else None)

        window_idx += 1
        if is_last_window:
            break

        if anchored:
            end += length_test
        else:
            start += length_test
            end = start + length_train_set


    final_params = best_params_list[-1] if best_params_list else {}

    # -----------------------------------------------------------
    # Final DataFrame with train/test dates, params, and criterion
    # -----------------------------------------------------------
    # Raw timestamps for exact alignment (used by debug)
    train_start_ts_raw = list(train_start_dates)
    test_start_ts_raw  = list(test_start_dates)
    test_end_ts_raw    = list(test_end_dates)

    train_start_dates  = [pd.to_datetime(d).date() if d is not None else None for d in train_start_dates]
    train_end_dates    = [pd.to_datetime(d).date() if d is not None else None for d in train_end_dates]
    test_start_dates   = [pd.to_datetime(d).date() if d is not None else None for d in test_start_dates]
    test_end_dates     = [pd.to_datetime(d).date() if d is not None else None for d in test_end_dates]

    df_results = pd.DataFrame(best_params_list)
    df_results.insert(0, 'train_start', train_start_dates)
    df_results.insert(1, 'train_end',   train_end_dates)
    df_results.insert(2, 'test_start',  test_start_dates)
    df_results.insert(3, 'test_end',    test_end_dates)
    df_results['best_crite']      = best_criteria_list
    df_results['ts_trades']       = test_n_trades_list
    df_results['tr_symbols']      = [len(s) for s in train_symbols_list]
    df_results['ts_symbols']      = [len(s) for s in test_symbols_list]
    df_results['tr_syms']         = train_symbols_list
    df_results['ts_syms']         = test_symbols_list
    df_results['_train_start_ts'] = train_start_ts_raw
    df_results['_test_start_ts']  = test_start_ts_raw
    df_results['_test_end_ts']    = test_end_ts_raw

    summary_row = dict(final_params)
    summary_row['train_start'] = 'wfo_ema'
    summary_row['train_end']   = ''
    summary_row['test_start']  = ''
    summary_row['test_end']    = ''
    summary_row['best_crite']  = df_results['best_crite'].mean() if 'best_crite' in df_results else None

    df_results = pd.concat([df_results, pd.DataFrame([summary_row])], ignore_index=True)


    sep_row = {col: "·" * min(8, len(str(col))) for col in df_results.columns}
    sep_row["train_start"] = "·" * 10
    df_display   = pd.concat([df_results.iloc[:-1], pd.DataFrame([sep_row]), df_results.iloc[[-1]]], ignore_index=True)
    display_cols = [c for c in df_display.columns if not c.startswith("_") and c not in ("tr_syms", "ts_syms")]
    logger.debug(f"WFO Final summary — parameters, criterion, and train/test dates per window:\n{df_display[display_cols].to_string()}\n{'─'*115}")

    # -----------------------------------------------------------
    # Concatenate per-window trade logs
    # -----------------------------------------------------------
    wfo_train_trades = pd.concat(train_trades_list, ignore_index=True) if train_trades_list else pd.DataFrame()
    wfo_test_trades  = pd.concat(test_trades_list,  ignore_index=True) if test_trades_list  else pd.DataFrame()

    valid_train_criteria  = [c for c in train_criteria_list if np.isfinite(c)]
    train_net_gain_avg    = float(np.mean(valid_train_criteria)) if valid_train_criteria else 0.0

    valid_test_criteria   = [c for c in best_criteria_list if np.isfinite(c)]
    test_net_gain_avg     = float(np.mean(valid_test_criteria)) if valid_test_criteria else 0.0

    return final_params, df_results, wfo_train_trades, wfo_test_trades, window_idx, train_net_gain_avg, test_net_gain_avg
