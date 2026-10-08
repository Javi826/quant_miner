#core/pipeline/backtest_runner.py NEW
import os
import logging
import math
import itertools
import importlib
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple
from tqdm import tqdm
from joblib import Parallel, delayed, effective_n_jobs
from multiprocessing.shared_memory import SharedMemory
from setup.config_backtest import INITIAL_BALANCE, COMISION
from setup.config_core import settings
from utils.batch_metrics import sharpes_from_daily_rows, skew_kurtosis_from_daily_values, equity_from_daily_values, _to_trading_days, _trading_days_between, _trading_day_table
from utils.paralelization import arrays_to_shared_memory, arrays_from_shared_memory
from pipeline.spec_table import SpecTable, build_spec_table, rule_spec_index
_bt = importlib.import_module(f"backtesters.ZX_compute_BT_{settings.BACKTEST_MODE}")
backtest_grid                = getattr(_bt, "backtest_grid", None)
build_rule_events_from_words = getattr(_bt, "build_rule_events_from_words", None)
market_arrays                = getattr(_bt, "market_arrays", None)
prepare_static_arrays        = _bt.prepare_static_arrays
logger = logging.getLogger("BOT_batch.pipeline.backtest_runner")
# =============================================================================
# BACKTEST EXECUTION CONFIG
# =============================================================================
BACKTEST_N_JOBS             = -1
BACKTEST_MIN_TRADES         = 150
BACKTEST_TASKS_PER_WORKER   = 8      # load balancing: minimum number of tasks per worker
BACKTEST_MAX_RULES_PER_TASK = 512    # rules per task (amortizes dispatch, pickling and shared-memory segments)
MATRIX_GATHER_ROWS          = 512    # matrix rows copied per block in the final gather
MATRIX_GATHER_THREADS       = 8      # threads for the final gather (numpy copies release the GIL)

# =============================================================================
# FULL-PERIOD GRID SEARCH
# =============================================================================

def _combo_id(params: dict) -> str:
    return "_".join(f"{k}{v}" for k, v in sorted(params.items()))


def _combo_grid(param_grid: dict) -> list:
    keys = list(param_grid.keys())
    return [dict(zip(keys, c)) for c in itertools.product(*[param_grid[k] for k in keys])]


def _engine_grid(combos: list) -> tuple:
    return (
        np.array([int(p["SELL_AFTER"]) for p in combos], dtype=np.int64),
        np.array([float(p["TP_PCT"]) for p in combos], dtype=np.float64),
        np.array([float(p["SL_PCT"]) for p in combos], dtype=np.float64),
    )


def _all_days(ohlcv_arr: dict) -> np.ndarray:
    return np.concatenate([arr["ts"] for arr in ohlcv_arr.values()]).astype("datetime64[D]")


def _global_day_grid(ohlcv_arr: dict) -> tuple:

    all_days = _all_days(ohlcv_arr)
    global_start_day = _to_trading_days(all_days.min())
    global_end_day   = _to_trading_days(all_days.max())
    n_days_range = int(_trading_days_between(global_start_day, global_end_day)) + 1
    return global_start_day, n_days_range


def _trading_calendar(ohlcv_arr: dict, global_start_day: np.datetime64) -> tuple:
    # (trading index per calendar day, first calendar day of the table, trading index of the global start day)
    day_ints = _all_days(ohlcv_arr).view(np.int64)
    table_first, _, trading_index = _trading_day_table(int(day_ints.min()), int(day_ints.max()))
    trading_index = np.ascontiguousarray(trading_index, dtype=np.int64)
    start_int     = int(np.datetime64(global_start_day, "D").astype(np.int64))
    return trading_index, int(table_first), int(trading_index[start_int - table_first])


def check_spec_table_layout(symbols: tuple, n_bars: np.ndarray, static_bundle: dict) -> None:
    if tuple(symbols) != tuple(static_bundle["symbols_by_sid"]):
        raise ValueError(f"spec table symbols {tuple(symbols)} differ from the backtest symbols {static_bundle['symbols_by_sid']}")
    if not np.array_equal(np.asarray(n_bars), static_bundle["sym_len"]):
        raise ValueError("spec table bar counts differ from the backtest symbol lengths")


def _winner_metrics_from_daily_values(daily_values: np.ndarray, n_days: int, sharpe: float, duration_is: float) -> dict:
    _, max_dd, net_gain = equity_from_daily_values(daily_values, INITIAL_BALANCE)

    if n_days > 2:
        skew_val, kurt_val = skew_kurtosis_from_daily_values(daily_values)
    else:
        skew_val, kurt_val = np.nan, np.nan

    return {
        "sharpe_is":   sharpe,
        "skew_is":     skew_val,
        "kurtosis_is": kurt_val,
        "n_days_is":   n_days,
        "net_gain_is": round(float(net_gain), 2),
        "max_dd_is":   round(float(max_dd), 2),
        "duration_is": round(float(duration_is), 2),
    }


def _empty_winner_metrics() -> dict:
    return {
        "sharpe_is":   np.nan,
        "skew_is":     np.nan,
        "kurtosis_is": np.nan,
        "n_days_is":   0,
        "net_gain_is": np.nan,
        "max_dd_is":   np.nan,
        "duration_is": np.nan,
    }

# =============================================================================
# WORKER
# =============================================================================
_SPEC_TABLE_KEY = "spec_table"
_WORKER_CTX: dict = {"key": None, "ctx": None}


class _WorkerCtx(NamedTuple):
    static_bundle: dict
    spec_words: np.ndarray
    word_offsets: np.ndarray
    shm_handles: list
    cached: bool


def _static_bundle_cache_key(shm_metadata: dict):
    try:
        return tuple(
            (sym, key, info.get("name", info.get("value")))
            for sym in sorted(shm_metadata)
            for key, info in sorted(shm_metadata[sym].items())
        )
    except TypeError:
        return None


def _get_worker_ctx(shm_metadata: dict, table_metadata: dict) -> _WorkerCtx:
    ohlcv_key = _static_bundle_cache_key(shm_metadata)
    table_key = _static_bundle_cache_key(table_metadata)
    cache_key = None if ohlcv_key is None or table_key is None else (ohlcv_key, table_key)
    if cache_key is not None and _WORKER_CTX["ctx"] is not None and _WORKER_CTX["key"] == cache_key:
        return _WORKER_CTX["ctx"]

    ohlcv_arr, shm_handles     = arrays_from_shared_memory(shm_metadata)
    table_arrays, table_handles = arrays_from_shared_memory(table_metadata)
    table         = table_arrays[_SPEC_TABLE_KEY]
    static_bundle = prepare_static_arrays(ohlcv_arr)
    static_bundle["calendar"] = _trading_calendar(ohlcv_arr, _global_day_grid(ohlcv_arr)[0])
    check_spec_table_layout(table["symbols"], table["n_bars"], static_bundle)

    ctx = _WorkerCtx(
        static_bundle = static_bundle,
        spec_words    = table["words"],
        word_offsets  = table["word_offsets"],
        shm_handles   = shm_handles + table_handles,
        cached        = cache_key is not None,
    )
    del ohlcv_arr, table, table_arrays
    if cache_key is None:
        return ctx

    old_ctx = _WORKER_CTX["ctx"]
    _WORKER_CTX["key"], _WORKER_CTX["ctx"] = None, None
    if old_ctx is not None:
        old_handles = old_ctx.shm_handles
        del old_ctx
        for shm in old_handles:
            shm.close()

    _WORKER_CTX["key"], _WORKER_CTX["ctx"] = cache_key, ctx
    return ctx


def _run_full_period_for_rule(
    rule_idx: int,
    spec_idx: np.ndarray,
    is_short: bool,
    ctx: _WorkerCtx,
    engine_grid: tuple,
    combo_ids: list,
    order_amount: float,
    seg_rows: np.ndarray,
    seg_cols: list,
    n_days_range: int,
) -> tuple:

    static_bundle = ctx.static_bundle
    n_combos = len(combo_ids)

    signal_events, ev_short, timeline = build_rule_events_from_words(
        ctx.spec_words, ctx.word_offsets, spec_idx, is_short,
        static_bundle["ts_int_2d"], static_bundle["sym_len"],
        static_bundle["tick_pos_2d"], static_bundle["all_timestamps_int"], static_bundle["idx_workspace"],
    )
    if signal_events.shape[0] < BACKTEST_MIN_TRADES:
        return rule_idx, {**_empty_winner_metrics(), "best_combo_id": combo_ids[0]}

    n_trades, day_start, n_days, n_nonzero, daily, duration = backtest_grid(
        market_arrays(static_bundle), signal_events, ev_short, timeline,
        *engine_grid,
        float(INITIAL_BALANCE), float(COMISION) / 100.0, order_amount, BACKTEST_MIN_TRADES,
        *static_bundle["calendar"], n_days_range,
    )

    col_base = rule_idx * n_combos
    neg_inf  = -np.inf
    sharpes  = sharpes_from_daily_rows(daily, day_start, n_days)

    best_rank   = None
    best_idx    = 0
    best_valid  = False
    best_sharpe = np.nan

    for combo_idx in range(n_combos):
        combo_trades = int(n_trades[combo_idx])
        if combo_trades == 0 or combo_trades < BACKTEST_MIN_TRADES:
            rank, valid, sharpe_metric = neg_inf, False, np.nan
        else:
            start        = int(day_start[combo_idx])
            stop         = start + int(n_days[combo_idx])
            daily_values = daily[combo_idx, start:stop]

            sharpe_metric = sharpes[combo_idx]
            rank  = sharpe_metric if math.isfinite(sharpe_metric) else neg_inf
            valid = True

            if n_nonzero[combo_idx] > 1:
                # New segment pages are zero-filled by the OS: only the traded span is written.
                seg_rows[len(seg_cols), start:stop] = daily_values
                seg_cols.append(col_base + combo_idx)

        if best_rank is None or rank > best_rank:
            best_rank, best_idx, best_valid, best_sharpe = rank, combo_idx, valid, sharpe_metric

    if not best_valid:
        winner_metrics = _empty_winner_metrics()
    else:
        best_trades      = int(n_trades[best_idx])
        best_start       = int(day_start[best_idx])
        best_n_days      = int(n_days[best_idx])
        best_duration_is = float(np.mean(duration[best_idx, :best_trades])) / 1e9 / 86400.0
        winner_metrics = _winner_metrics_from_daily_values(
            daily[best_idx, best_start:best_start + best_n_days], best_n_days, best_sharpe, best_duration_is,
        )

    return rule_idx, {**winner_metrics, "best_combo_id": combo_ids[best_idx]}


def _run_rules_block_shm(
    block: tuple,
    shm_metadata: dict,
    table_metadata: dict,
    engine_grid: tuple,
    combo_ids: list,
    order_amount: float,
    seg_name: str,
    n_days_range: int,
) -> tuple:
    # Returns (rule results, valid column indices in segment order, non-zero day mask).
    rule_start, block_spec_idx, block_is_short = block
    n_block  = block_spec_idx.shape[0]
    n_combos = len(combo_ids)

    ctx = _get_worker_ctx(shm_metadata, table_metadata)
    seg = SharedMemory(name=seg_name, create=False)
    try:
        seg_rows = np.ndarray((n_block * n_combos, n_days_range), dtype=np.float32, buffer=seg.buf)
        seg_cols = []
        results = [
            _run_full_period_for_rule(
                rule_start + offset, block_spec_idx[offset], bool(block_is_short[offset]), ctx,
                engine_grid, combo_ids, order_amount, seg_rows, seg_cols, n_days_range,
            )
            for offset in range(n_block)
        ]
        n_valid  = len(seg_cols)
        day_mask = np.zeros(n_days_range, dtype=bool)
        for start in range(0, n_valid, MATRIX_GATHER_ROWS):
            day_mask |= np.any(seg_rows[start:min(start + MATRIX_GATHER_ROWS, n_valid)] != 0, axis=0)
        del seg_rows
        return results, np.asarray(seg_cols, dtype=np.int64), day_mask
    finally:
        seg.close()
        if not ctx.cached:
            handles = ctx.shm_handles
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
    n_days_range: int,
    progress_label: str = "",
    spec_table: SpecTable = None,
) -> tuple:

    desc = f"BACKTEST FULL   {progress_label}".strip()
    if spec_table is None:
        spec_table = build_spec_table(rules, ohlcv_arr, progress_label, show_progress=False)

    combos      = _combo_grid(param_grid)
    n_combos    = len(combos)
    combo_ids   = [_combo_id(p) for p in combos]
    engine_grid = _engine_grid(combos)

    n_rules  = len(rules)
    row_size = n_days_range * np.dtype(np.float32).itemsize

    n_workers      = max(1, effective_n_jobs(BACKTEST_N_JOBS))
    rules_per_task = max(1, min(BACKTEST_MAX_RULES_PER_TASK, -(-n_rules // (n_workers * BACKTEST_TASKS_PER_WORKER))))
    spec_idx, sides = rule_spec_index(rules, spec_table.index_by_identity)
    is_short = sides != "long"   # signal_fn: "long" -> +1, any other side -> -1
    blocks   = [(i, spec_idx[i:i + rules_per_task], is_short[i:i + rules_per_task]) for i in range(0, n_rules, rules_per_task)]

    pending   = []
    results   = []
    seg_cols  = []
    day_mask  = np.zeros(n_days_range, dtype=bool)
    try:
        for _, block_spec_idx, _ in blocks:
            pending.append(_create_segment(block_spec_idx.shape[0] * n_combos * row_size))

        shm_list, ohlcv_metadata = arrays_to_shared_memory(ohlcv_arr)
        try:
            table_shm, table_metadata = arrays_to_shared_memory({_SPEC_TABLE_KEY: spec_table.shared_arrays()})
            shm_list += table_shm
            with tqdm(total=n_rules, desc=desc, dynamic_ncols=True) as pbar:
                for block_results, block_cols, block_mask in Parallel(n_jobs=BACKTEST_N_JOBS, batch_size=1, pre_dispatch="all", return_as="generator")(
                    delayed(_run_rules_block_shm)(
                        block, ohlcv_metadata, table_metadata, engine_grid, combo_ids, float(order_amount),
                        seg_name, n_days_range,
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

    full_period_by_rule = {rules[rule_idx]["rule_id"]: metrics for rule_idx, metrics in results}
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
    spec_table: SpecTable = None,
) -> tuple:

    n_combos = 1
    for _values in param_grid.values():
        n_combos *= len(_values)

    _all_ts_dbg = np.concatenate([arr["ts"] for arr in ohlcv_arr.values()])
    logger.debug(f"BACKTEST FULL INPUT {timeframe}: date range [{_all_ts_dbg.min()} .. {_all_ts_dbg.max()}] over {len(ohlcv_arr)} symbol(s)")

    _, n_days_range = _global_day_grid(ohlcv_arr)

    if backtest_grid is None:
        raise NotImplementedError(f"Backtester ZX_compute_BT_{settings.BACKTEST_MODE} does not implement the grid API (backtest_grid)")
    if build_rule_events_from_words is None:
        raise NotImplementedError(f"Backtester ZX_compute_BT_{settings.BACKTEST_MODE} does not implement build_rule_events_from_words")

    if not rules:
        return [], n_combos, np.empty((0, 0), dtype=np.float32), []

    full_period_by_rule, matrix_arr, valid_cols, combo_ids = run_full_period_search(
        rules            = rules,
        ohlcv_arr        = ohlcv_arr,
        param_grid       = param_grid,
        order_amount     = order_amount,
        n_days_range     = n_days_range,
        progress_label   = timeframe,
        spec_table       = spec_table,
    )

    raw_results = [
        {**r, **full_period_by_rule[r["rule_id"]]}
        for r in rules
    ]

    col_names = [f"{rules[c // n_combos]['rule_id']}__{combo_ids[c % n_combos]}" for c in valid_cols.tolist()]

    return raw_results, n_combos, matrix_arr, col_names