#core/pipeline/backtest_runner.py
import os
import logging
import math
import itertools
import importlib
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from joblib import Parallel, delayed, effective_n_jobs
from multiprocessing.shared_memory import SharedMemory
from setup.config_backtest import INITIAL_BALANCE, COMISION
from setup.config_core import settings
from utils.batch_metrics import sharpe_from_daily_values, skew_kurtosis_from_daily_values, daily_values_from_sell_days, equity_from_daily_values, _to_trading_days, _trading_days_between
from utils.paralelization import arrays_to_shared_memory, arrays_from_shared_memory
from signals.indicators_bank import ConditionBank
_bt = importlib.import_module(f"backtesters.ZX_compute_BT_{settings.BACKTEST_MODE}")
backtest_core         = _bt.backtest_core
prepare_static_arrays = _bt.prepare_static_arrays
prepare_signal_arrays = _bt.prepare_signal_arrays
logger = logging.getLogger("BOT_batch.pipeline.backtest_runner")
DTYPE  = np.float32
# =============================================================================
# BACKTEST EXECUTION CONFIG
# =============================================================================
BACKTEST_N_JOBS             = -1
BACKTEST_MIN_TRADES         = 150
BACKTEST_TASKS_PER_WORKER   = 8      # load balancing: minimum number of tasks per worker
BACKTEST_MAX_RULES_PER_TASK = 64     # rules per task (amortizes dispatch)
MATRIX_GATHER_ROWS          = 512    # matrix rows copied per block in the final gather
MATRIX_GATHER_THREADS       = 8      # threads for the final gather (numpy copies release the GIL)

# =============================================================================
# FULL-PERIOD GRID SEARCH
#   - OHLCV SHM views, static bundle and condition banks cached per worker.
#   - One task = block of consecutive rules with its own SHM segment: valid
#     rows are appended there (no shared tmpfs file written by every worker,
#     no zero-fill of the full matrix).
#   - Engine timeline reduced to event ticks + the tick before each one + the
#     last tick (the engine only pops positions elsewhere; popping them before
#     the next event tick yields the same trades, order and cash).
#   - Duration computed only for the winning combo.
#   - Final matrix gathered from the valid columns only.
# =============================================================================

def _combo_id(params: dict) -> str:
    return "_".join(f"{k}{v}" for k, v in sorted(params.items()))


def _combo_grid(param_grid: dict) -> list:
    keys = list(param_grid.keys())
    return [dict(zip(keys, c)) for c in itertools.product(*[param_grid[k] for k in keys])]


def _global_day_grid(ohlcv_arr: dict) -> tuple:

    all_ts   = np.concatenate([arr["ts"] for arr in ohlcv_arr.values()])
    all_days = all_ts.astype("datetime64[D]")
    global_start_day = _to_trading_days(all_days.min())
    global_end_day   = _to_trading_days(all_days.max())
    n_days_range = int(_trading_days_between(global_start_day, global_end_day)) + 1
    return global_start_day, n_days_range


def _compress_timeline(all_timestamps_int: np.ndarray, ev_col0: np.ndarray) -> np.ndarray:
    n_ticks = all_timestamps_int.shape[0]
    if n_ticks == 0 or ev_col0.shape[0] == 0:
        return all_timestamps_int
    pos = np.searchsorted(all_timestamps_int, ev_col0)
    if pos[-1] >= n_ticks or not np.array_equal(all_timestamps_int[pos], ev_col0):
        return all_timestamps_int
    keep = np.zeros(n_ticks, dtype=bool)
    keep[pos] = True
    keep[pos[pos > 0] - 1] = True
    keep[-1] = True
    return all_timestamps_int[keep]


def _build_full_period_ohlcv(ohlcv_arr: dict, signal_fn: callable, condition_banks: dict | None = None) -> dict:
    ohlcv_arrays = {}
    for sym, arr in ohlcv_arr.items():
        bank    = condition_banks.get(sym) if condition_banks else None
        signals = signal_fn(arr, live_trading=False, bank=bank)
        ohlcv_arrays[sym] = {**arr, "signal": np.asarray(signals, dtype=DTYPE)}
    return ohlcv_arrays


def _winner_metrics_from_daily_values(daily_values: np.ndarray, n_days: int, sharpe: float, duration_train: float) -> dict:
    _, max_dd, net_gain = equity_from_daily_values(daily_values, INITIAL_BALANCE)

    if n_days > 2:
        skew_val, kurt_val = skew_kurtosis_from_daily_values(daily_values)
    else:
        skew_val, kurt_val = np.nan, np.nan

    return {
        "sharpe_train":   sharpe,
        "skew_train":     skew_val,
        "kurtosis_train": kurt_val,
        "n_days_train":   n_days,
        "net_gain_train": round(float(net_gain), 2),
        "max_dd_train":   round(float(max_dd), 2),
        "duration_train": round(float(duration_train), 2),
    }


def _empty_winner_metrics() -> dict:
    return {
        "sharpe_train":   np.nan,
        "skew_train":     np.nan,
        "kurtosis_train": np.nan,
        "n_days_train":   0,
        "net_gain_train": np.nan,
        "max_dd_train":   np.nan,
        "duration_train": np.nan,
    }

# =============================================================================
# WORKER
# =============================================================================
_WORKER_CTX: dict = {"key": None, "ctx": None}


def _static_bundle_cache_key(shm_metadata: dict):
    try:
        return tuple(
            (sym, key, info.get("name", info.get("value")))
            for sym in sorted(shm_metadata)
            for key, info in sorted(shm_metadata[sym].items())
        )
    except TypeError:
        return None


def _get_worker_ctx(shm_metadata: dict) -> tuple:
    """(ohlcv_arr, static_bundle, condition_banks, shm_handles, cached)"""
    cache_key = _static_bundle_cache_key(shm_metadata)
    if cache_key is not None and _WORKER_CTX["ctx"] is not None and _WORKER_CTX["key"] == cache_key:
        return _WORKER_CTX["ctx"]

    ohlcv_arr, shm_handles = arrays_from_shared_memory(shm_metadata)
    static_bundle   = prepare_static_arrays(ohlcv_arr)
    condition_banks = {sym: ConditionBank(arr) for sym, arr in ohlcv_arr.items()}

    if cache_key is None:
        return ohlcv_arr, static_bundle, condition_banks, shm_handles, False

    old_ctx = _WORKER_CTX["ctx"]
    _WORKER_CTX["key"], _WORKER_CTX["ctx"] = None, None
    if old_ctx is not None:
        old_handles = old_ctx[3]
        del old_ctx
        for shm in old_handles:
            shm.close()

    ctx = (ohlcv_arr, static_bundle, condition_banks, shm_handles, True)
    _WORKER_CTX["key"], _WORKER_CTX["ctx"] = cache_key, ctx
    return ctx


def _run_full_period_for_rule(
    rule_id: str,
    rule_idx: int,
    signal_fn: callable,
    ctx: tuple,
    engine_params: list,
    combo_ids: list,
    order_amount: float,
    seg_rows: np.ndarray,
    seg_cols: list,
    global_start_day: np.datetime64,
    n_combos: int,
) -> tuple:

    ohlcv_arr, static_bundle, condition_banks = ctx[0], ctx[1], ctx[2]

    ohlcv_arrays        = _build_full_period_ohlcv(ohlcv_arr, signal_fn, condition_banks)
    max_possible_trades = sum(int(np.count_nonzero(arr["signal"])) for arr in ohlcv_arrays.values())

    if max_possible_trades < BACKTEST_MIN_TRADES:
        return rule_id, {**_empty_winner_metrics(), "best_combo_id": combo_ids[0]}

    engine_arrays = tuple(prepare_signal_arrays(static_bundle, ohlcv_arrays)[7])
    engine_arrays = engine_arrays[:10] + (_compress_timeline(engine_arrays[10], engine_arrays[11]),) + engine_arrays[11:]

    initial_balance = float(INITIAL_BALANCE)
    comi_factor     = float(COMISION) / 100.0
    col_base        = rule_idx * n_combos
    neg_inf         = -np.inf

    best_rank = None
    best_idx  = 0
    best      = None

    for combo_idx, (sell_after, tp_pct, sl_pct) in enumerate(engine_params):
        core_output = backtest_core(*engine_arrays, initial_balance, comi_factor, order_amount, sell_after, tp_pct, sl_pct)

        n_trades = core_output[0]
        if n_trades == 0 or n_trades < BACKTEST_MIN_TRADES:
            rank, bundle = neg_inf, None
        else:
            sell_time_int = core_output[4]
            profits       = core_output[7]

            daily_values, n_days, start_day = daily_values_from_sell_days(sell_time_int.view("datetime64[ns]"), profits)

            sharpe_metric = sharpe_from_daily_values(daily_values)
            rank = sharpe_metric if math.isfinite(sharpe_metric) else neg_inf

            if np.count_nonzero(daily_values) > 1:
                # New segment pages are zero-filled by the OS: only the traded span is written.
                row_offset = int(_trading_days_between(global_start_day, start_day))
                seg_rows[len(seg_cols), row_offset:row_offset + n_days] = daily_values.astype(np.float32)
                seg_cols.append(col_base + combo_idx)

            bundle = (daily_values, n_days, sharpe_metric, core_output[2], sell_time_int)

        if best_rank is None or rank > best_rank:
            best_rank, best_idx, best = rank, combo_idx, bundle

    if best is None:
        winner_metrics = _empty_winner_metrics()
    else:
        best_daily_values, best_n_days, best_sharpe_metric, best_buy, best_sell = best
        best_duration_train = float(np.mean(best_sell - best_buy)) / 1e9 / 86400.0
        winner_metrics = _winner_metrics_from_daily_values(
            best_daily_values, best_n_days, best_sharpe_metric, best_duration_train,
        )

    return rule_id, {**winner_metrics, "best_combo_id": combo_ids[best_idx]}


def _run_rules_block_shm(
    block: list,
    shm_metadata: dict,
    engine_params: list,
    combo_ids: list,
    order_amount: float,
    seg_name: str,
    n_days_range: int,
    global_start_day: np.datetime64,
    n_combos: int,
) -> tuple:
    """Returns (rule results, valid column indices in segment order, non-zero day mask)."""

    ctx = _get_worker_ctx(shm_metadata)
    seg = SharedMemory(name=seg_name, create=False)
    try:
        seg_rows = np.ndarray((len(block) * n_combos, n_days_range), dtype=np.float32, buffer=seg.buf)
        seg_cols = []
        results = [
            _run_full_period_for_rule(
                rule_id, rule_idx, signal_fn, ctx, engine_params, combo_ids, order_amount,
                seg_rows, seg_cols, global_start_day, n_combos,
            )
            for rule_id, rule_idx, signal_fn in block
        ]
        n_valid  = len(seg_cols)
        day_mask = np.zeros(n_days_range, dtype=bool)
        for start in range(0, n_valid, MATRIX_GATHER_ROWS):
            day_mask |= np.any(seg_rows[start:min(start + MATRIX_GATHER_ROWS, n_valid)] != 0, axis=0)
        del seg_rows
        return results, np.asarray(seg_cols, dtype=np.int64), day_mask
    finally:
        seg.close()
        if not ctx[4]:
            handles = ctx[3]
            del ctx
            for shm in handles:
                shm.close()

# =============================================================================
# SEARCH + MATRIX
# =============================================================================
def _create_segment(n_bytes: int) -> str:
    shm = SharedMemory(create=True, size=max(n_bytes, 1))
    name = shm.name
    shm.close()
    return name


def _unlink_segment(name: str) -> None:
    try:
        shm = SharedMemory(name=name, create=False)
    except FileNotFoundError:
        return
    shm.close()
    shm.unlink()


def _gather_segment(name: str, n_valid: int, n_days_range: int, days: np.ndarray, out: np.ndarray, col_start: int) -> None:
    # Copies the segment's valid rows, transposed, into out[:, col_start:col_start + n_valid].
    if n_valid == 0:
        _unlink_segment(name)
        return
    shm = SharedMemory(name=name, create=False)
    try:
        rows     = np.ndarray((n_valid, n_days_range), dtype=np.float32, buffer=shm.buf)
        all_days = days.shape[0] == n_days_range
        for start in range(0, n_valid, MATRIX_GATHER_ROWS):
            block = rows[start:start + MATRIX_GATHER_ROWS]
            if not all_days:
                block = block[:, days]
            out[:, col_start + start:col_start + start + block.shape[0]] = block.T
        del rows, block
    finally:
        shm.close()
        shm.unlink()


def run_full_period_search(
    rules: list,
    ohlcv_arr: dict,
    param_grid: dict,
    order_amount: int,
    global_start_day: np.datetime64,
    n_days_range: int,
    progress_label: str = "",
) -> tuple:

    desc = f"BACKTEST FULL   {progress_label}".strip()

    combos        = _combo_grid(param_grid)
    n_combos      = len(combos)
    combo_ids     = [_combo_id(p) for p in combos]
    engine_params = [(int(p["SELL_AFTER"]), float(p["TP_PCT"]), float(p["SL_PCT"])) for p in combos]

    n_rules  = len(rules)
    row_size = n_days_range * np.dtype(np.float32).itemsize

    n_workers      = max(1, effective_n_jobs(BACKTEST_N_JOBS))
    rules_per_task = max(1, min(BACKTEST_MAX_RULES_PER_TASK, -(-n_rules // (n_workers * BACKTEST_TASKS_PER_WORKER))))
    items          = [(r["rule_id"], rule_idx, r["signal_fn"]) for rule_idx, r in enumerate(rules)]
    blocks         = [items[i:i + rules_per_task] for i in range(0, n_rules, rules_per_task)]

    pending   = []
    results   = []
    seg_cols  = []
    day_mask  = np.zeros(n_days_range, dtype=bool)
    try:
        for block in blocks:
            pending.append(_create_segment(len(block) * n_combos * row_size))

        shm_list, ohlcv_metadata = arrays_to_shared_memory(ohlcv_arr)
        try:
            with tqdm(total=n_rules, desc=desc, dynamic_ncols=True) as pbar:
                for block_results, block_cols, block_mask in Parallel(n_jobs=BACKTEST_N_JOBS, batch_size=1, pre_dispatch="all", return_as="generator")(
                    delayed(_run_rules_block_shm)(
                        block, ohlcv_metadata, engine_params, combo_ids, float(order_amount),
                        seg_name, n_days_range, global_start_day, n_combos,
                    )
                    for block, seg_name in zip(blocks, pending)
                ):
                    results.extend(block_results)
                    seg_cols.append(block_cols)
                    day_mask |= block_mask
                    pbar.update(len(block_results))
        finally:
            for shm in shm_list:
                shm.close()
                shm.unlink()

        valid_cols = np.concatenate(seg_cols) if seg_cols else np.empty(0, dtype=np.int64)
        n_valid    = valid_cols.shape[0]
        if n_valid == 0:
            matrix_arr = np.empty((n_days_range, 0), dtype=np.float32)
            for name in pending:
                _unlink_segment(name)
        else:
            days       = np.flatnonzero(day_mask)
            matrix_arr = np.empty((days.shape[0], n_valid), dtype=np.float32)
            col_starts = np.concatenate(([0], np.cumsum([c.shape[0] for c in seg_cols])[:-1]))
            n_threads  = max(1, min(MATRIX_GATHER_THREADS, os.cpu_count() or 1))
            with ThreadPoolExecutor(max_workers=n_threads) as pool:
                list(pool.map(
                    lambda args: _gather_segment(args[0], args[1], n_days_range, days, matrix_arr, args[2]),
                    [(name, c.shape[0], int(k0)) for name, c, k0 in zip(pending, seg_cols, col_starts)],
                ))
        pending = []
    finally:
        for name in pending:
            _unlink_segment(name)

    full_period_by_rule = dict(results)
    return full_period_by_rule, matrix_arr, valid_cols, combo_ids

# =============================================================================
# PIPE BACKTESTING
# =============================================================================
def pipe_backtesting(
    rules: list,
    ohlcv_arr: dict,
    param_grid: dict,
    order_amount: int,
    timeframe: str = "",
) -> tuple:

    n_combos = 1
    for _values in param_grid.values():
        n_combos *= len(_values)

    _all_ts_dbg = np.concatenate([arr["ts"] for arr in ohlcv_arr.values()])
    logger.debug(f"BACKTEST FULL INPUT {timeframe}: date range [{_all_ts_dbg.min()} .. {_all_ts_dbg.max()}] over {len(ohlcv_arr)} symbol(s)")

    global_start_day, n_days_range = _global_day_grid(ohlcv_arr)

    if not rules:
        return [], n_combos, np.empty((0, 0), dtype=np.float32), []

    full_period_by_rule, matrix_arr, valid_cols, combo_ids = run_full_period_search(
        rules            = rules,
        ohlcv_arr        = ohlcv_arr,
        param_grid       = param_grid,
        order_amount     = order_amount,
        global_start_day = global_start_day,
        n_days_range     = n_days_range,
        progress_label   = timeframe,
    )

    raw_results = [
        {**r, **full_period_by_rule[r["rule_id"]]}
        for r in rules
    ]

    col_names = [f"{rules[c // n_combos]['rule_id']}__{combo_ids[c % n_combos]}" for c in valid_cols.tolist()]

    return raw_results, n_combos, matrix_arr, col_names